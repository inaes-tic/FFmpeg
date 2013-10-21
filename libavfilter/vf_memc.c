/*
 * Copyright (c) 2006 Michael Niedermayer <michaelni@gmx.at>
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/**
 * @file
 * Motion Compensation Deinterlacer
 * Ported from MPlayer libmpcodecs/vf_mcdeint.c.
 *
 * Known Issues:
 *
 * The motion estimation is somewhat at the mercy of the input, if the
 * input frames are created purely based on spatial interpolation then
 * for example a thin black line or another random and not
 * interpolateable pattern will cause problems.
 * Note: completely ignoring the "unavailable" lines during motion
 * estimation did not look any better, so the most obvious solution
 * would be to improve tfields or penalize problematic motion vectors.
 *
 * If non iterative ME is used then snow currently ignores the OBMC
 * window and as a result sometimes creates artifacts.
 *
 * Only past frames are used, we should ideally use future frames too,
 * something like filtering the whole movie in forward and then
 * backward direction seems like a interesting idea but the current
 * filter framework is FAR from supporting such things.
 *
 * Combining the motion compensated image with the input image also is
 * not as trivial as it seems, simple blindly taking even lines from
 * one and odd ones from the other does not work at all as ME/MC
 * sometimes has nothing in the previous frames which matches the
 * current. The current algorithm has been found by trial and error
 * and almost certainly can be improved...
 */

#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavcodec/avcodec.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"

enum MEMCMode {
    MODE_FAST = 0,
    MODE_MEDIUM,
    MODE_SLOW,
    MODE_EXTRA_SLOW,
    MODE_NB,
};

enum MEMCParity {
    PARITY_TFF  =  0, ///< top field first
    PARITY_BFF  =  1, ///< bottom field first
};

typedef struct {
    const AVClass *class;
    enum MEMCMode mode;
    enum MEMCParity parity;
    int qp;
    AVCodecContext *enc_ctx;
} MEMCContext;

#define OFFSET(x) offsetof(MEMCContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define CONST(name, help, val, unit) { name, help, 0, AV_OPT_TYPE_CONST, {.i64=val}, INT_MIN, INT_MAX, FLAGS, unit }

static const AVOption memc_options[] = {
    { "mode", "set mode", OFFSET(mode), AV_OPT_TYPE_INT, {.i64=MODE_FAST}, 0, MODE_NB-1, FLAGS, .unit="mode" },
    CONST("fast",       NULL, MODE_FAST,       "mode"),
    CONST("medium",     NULL, MODE_MEDIUM,     "mode"),
    CONST("slow",       NULL, MODE_SLOW,       "mode"),
    CONST("extra_slow", NULL, MODE_EXTRA_SLOW, "mode"),

    { "parity", "set the assumed picture field parity", OFFSET(parity), AV_OPT_TYPE_INT, {.i64=PARITY_BFF}, -1, 1, FLAGS, "parity" },
    CONST("tff", "assume top field first",    PARITY_TFF, "parity"),
    CONST("bff", "assume bottom field first", PARITY_BFF, "parity"),

    { "qp", "set qp", OFFSET(qp), AV_OPT_TYPE_INT, {.i64=1}, INT_MIN, INT_MAX, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(memc);

static int config_props(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    MEMCContext *memc = ctx->priv;
    AVCodec *enc;
    AVCodecContext *enc_ctx;
    AVDictionary *opts = NULL;
    int ret;

    if (!(enc = avcodec_find_encoder(AV_CODEC_ID_SNOW))) {
        av_log(ctx, AV_LOG_ERROR, "Snow encoder is not enabled in libavcodec\n");
        return AVERROR(EINVAL);
    }

    memc->enc_ctx = avcodec_alloc_context3(enc);
    if (!memc->enc_ctx)
        return AVERROR(ENOMEM);
    enc_ctx = memc->enc_ctx;
    enc_ctx->width  = inlink->w;
    enc_ctx->height = inlink->h;
    enc_ctx->time_base = (AVRational){1,25};  // meaningless
    enc_ctx->gop_size = 300;
    enc_ctx->max_b_frames = 0;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    enc_ctx->flags = CODEC_FLAG_QSCALE | CODEC_FLAG_LOW_DELAY;
    enc_ctx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
    enc_ctx->global_quality = 1;
    enc_ctx->me_cmp = enc_ctx->me_sub_cmp = FF_CMP_SAD;
    enc_ctx->mb_cmp = FF_CMP_SSE;
    av_dict_set(&opts, "memc_only", "1", 0);

    switch (memc->mode) {
    case MODE_EXTRA_SLOW:
        enc_ctx->refs = 3;
    case MODE_SLOW:
        enc_ctx->me_method = ME_ITER;
    case MODE_MEDIUM:
        enc_ctx->flags |= CODEC_FLAG_4MV;
        enc_ctx->dia_size = 2;
    case MODE_FAST:
        enc_ctx->flags |= CODEC_FLAG_QPEL;
    }

    ret = avcodec_open2(enc_ctx, enc, &opts);
    av_dict_free(&opts);
    if (ret < 0)
        return ret;

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    MEMCContext *memc = ctx->priv;

    if (memc->enc_ctx) {
        avcodec_close(memc->enc_ctx);
        av_freep(&memc->enc_ctx);
    }
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum PixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE
    };

    ff_set_common_formats(ctx, ff_make_format_list(pix_fmts));

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *inpic)
{
    MEMCContext *memc = inlink->dst->priv;
    AVFilterLink *outlink = inlink->dst->outputs[0];
    AVFrame *frame_dec, *outpic;
    AVPacket pkt;
    int x, y, i, ret, got_frame = 0;

    outpic = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!outpic) {
        av_frame_free(&inpic);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(outpic, inpic);
    inpic->quality = memc->qp * FF_QP2LAMBDA;

    av_init_packet(&pkt);
    pkt.data = NULL;    // packet data will be allocated by the encoder
    pkt.size = 0;

    ret = avcodec_encode_video2(memc->enc_ctx, &pkt, inpic, &got_frame);
    if (ret < 0)
        goto end;

    frame_dec = memc->enc_ctx->coded_frame;

    if (frame_dec->pict_type != AV_PICTURE_TYPE_I)
        av_log (inlink->dst, AV_LOG_WARNING, "MC\n");
    else
        av_log (inlink->dst, AV_LOG_WARNING, "noMC\n");

    for (i = 0; i < 3; i++) {
        int is_chroma = !!i;
        int w = FF_CEIL_RSHIFT(inlink->w, is_chroma);
        int h = FF_CEIL_RSHIFT(inlink->h, is_chroma);
        int fils = frame_dec->linesize[i];
        int dsts = outpic   ->linesize[i];

        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                for (x = 0; x < w; x++) {
                    outpic->data[i][x + y*dsts] = frame_dec->data[i][x + y*fils];
                    if (frame_dec->pict_type == AV_PICTURE_TYPE_I)
                        frame_dec->data[i][x + y*fils] = 0;
                }
            }
        }
    }


end:
    av_free_packet(&pkt);
    av_frame_free(&inpic);

    if (ret < 0) {
        return ret;
    }
    return ff_filter_frame(outlink, outpic);
}

static const AVFilterPad memc_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_props,
    },
    { NULL }
};

static const AVFilterPad memc_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter avfilter_vf_memc = {
    .name          = "memc",
    .description   = NULL_IF_CONFIG_SMALL("Apply motion compensating deinterlacing."),
    .priv_size     = sizeof(MEMCContext),
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = memc_inputs,
    .outputs       = memc_outputs,
    .priv_class    = &memc_class,
};
