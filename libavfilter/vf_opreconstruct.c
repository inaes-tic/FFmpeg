/*
 * Copyright (C) 2013 Cooperativa de trabajo OpCode Limitada,
 * Copyright (C) 2013 Instituto Nacional de Asociativismo y Economia Social,
 * Copyright (C) 2013 Niv Sardi <xaiki@inaes.gob.ar>, All Rights Reserved
 *
 * Author of de-interlace algorithm: Jim Easterbrook for BBC R&D
 * Based on the process described by Martin Weston for BBC R&D
 * Author of FFmpeg filter: Niv Sardi for INAES & OpCode.
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "video.h"


enum { /* XXX: this should be included in cv.h */
    OPTFLOW_USE_INITIAL_FLOW = CV_LKFLOW_INITIAL_GUESSES,
    OPTFLOW_LK_GET_MIN_EIGENVALS = CV_LKFLOW_GET_MIN_EIGENVALS,
    OPTFLOW_FARNEBACK_GAUSSIAN = 256
};

enum { R = 0, G, B, A };

#define MAX_HEIGHT 1080
#define MAX_WIDTH  1920
#define MAX_DEPTH  9

#define CV_ELEM_PTR_FAST( frame, row, col, pix_size )  \
    (assert( (unsigned)(row) < (unsigned)(mat).rows &&   \
             (unsigned)(col) < (unsigned)(mat).cols ),   \
     (frame)->data[0] + (size_t)(frame)->linesize*(row) + (pix_size)*(col))

#define CV_ELEM( frame, elemtype, row, col )           \
    (*(elemtype*)CV_ELEM_PTR_FAST( frame, row, col, sizeof(elemtype)))

#define PIXEL(frame, plane, x, y, def)                                  \
    PIXEL_FULL(frame->data, plane, x, y, frame->width, frame->height, frame->linesize[plane], def)

#define BUFFER_PIXEL(buf, plane, x, y, def)                            \
    PIXEL_FULL(buf, plane, x, y, MAX_WIDTH, MAX_HEIGHT, MAX_WIDTH, def)

#define PIX(data, plane, x, y, stride) data[plane][(x) + (y) * (stride)]
#define FPIX(frame, plane, x, y) PIX((frame)->data, plane, x, y, (frame)->linesize[(plane)])

#define PIXEL_FULL(data, plane, x, y, w, h, stride, def) \
    ((x) < 0 || (y) < 0) ? (def) : \
    (((x) >= (w) || (y) >= (h)) ? (def) : \
    PIX(data, plane, x, y, stride))

#define AMP(x, y) (sqrt ((x*x) + (y*y)))

#define PUSH_ARRAY(array, v)                         \
         if (! array[0]) array[0] = v;               \
    else if (! array[1]) array[1] = v;               \
    else if (! array[2]) array[2] = v;               \
    else if (! array[3]) array[3] = v;               \
    else if (! array[4]) array[4] = v;               \
    else if (! array[5]) array[5] = v;               \
    else if (! array[6]) array[6] = v;               \
    else if (! array[7]) array[7] = v;               \
         //else fprintf (stderr, "error, array: %p out of bounds\n", array);

typedef struct OpRFarnebackContext {
    double pyrScale;
    int levels;
    int winsize;
    int iterations;
    int polyN;
    double polySigma;
    double treshold;
    int flags;
} OpRFarnebackContext;

typedef struct OpRInterpolateContext {
    int interpolate;
} OpRInterpolateContext;

typedef struct OpRContext {
    const AVClass *class;
    OpRFarnebackContext fctx;
    OpRInterpolateContext ictx;
    int linesize[4];      ///< bytes of pixel data per line for each plane
    int planeheight[4];   ///< height of each plane
    int eof;
    int nb_planes;
    double ts_unit;
    CvMat *gray, *prevgray, *flow, *cflow;
    IplImage *prev, *cur, *next;
    uint8_t rgba_map[4];
    int step;
} OpRContext;

uint64_t buffer[MAX_WIDTH][MAX_HEIGHT][MAX_DEPTH];
double fbuffer[MAX_WIDTH][MAX_HEIGHT][MAX_DEPTH];

#define OFFSET(x) offsetof(OpRContext, fctx.x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define FARNEOPT(args...) OPROPT("Farneback Algorithm", fctx, args)
#define INTEROPT(args...) OPROPT("Interpolation", ictx, args)
#define OPROPT(sec, s, n, h, t, d, m, x) {                              \
            .name	 = #n,                                          \
            .help	 = sec " Option '" #n "': " h,                  \
            .default_val = d,                                           \
            .type        = t,                                           \
            .min	 = m,                                           \
            .max	 = x,                                           \
            .flags	 = FLAGS,                                       \
            .offset	 = offsetof(OpRContext, s.n),                   \
            }                                                           \

static const AVOption opreconstruct_options[] = {
    FARNEOPT(pyrScale,
             "Specifies the image scale (<1) to build the pyramids for each" \
             "image. pyrScale=0.5 means the classical pyramid, where each next" \
             "layer is twice smaller than the previous.",
             AV_OPT_TYPE_DOUBLE, {.dbl=.5}, 0, 1),
    FARNEOPT(levels,
             "The number of pyramid layers, including the initial image. levels=1"
             "means that no extra layers are created and only the original images"
             "are used",
             AV_OPT_TYPE_INT, {.i64=3}, 1, 100),
    FARNEOPT(winsize,
             "The averaging window size; The larger values increase the algorithm"
             "robustness to image noise and give more chances for fast motion"
             "detection, but yield more blurred motion field",
             AV_OPT_TYPE_INT, {.i64=15}, 1, 1000),
    FARNEOPT(iterations,
             "The number of iterations the algorithm does at each pyramid level",
             AV_OPT_TYPE_INT, {.i64=3}, 1, 100),
    FARNEOPT(polyN,
             "Size of the pixel neighborhood used to find polynomial expansion in"
             "each pixel. The larger values mean that the image will be"
             "approximated with smoother surfaces, yielding more robust algorithm"
             "and more blurred motion field. Typically, polyN =5 or 7",
             AV_OPT_TYPE_INT, {.i64=5}, 1, 20),
    FARNEOPT(polySigma,
             "Standard deviation of the Gaussian that is used to smooth derivatives"
             "that are used as a basis for the polynomial expansion. For polyN=5"
             "you can set polySigma=1.1 , for polyN=7 a good value would be"
             "polySigma=1.5",
             AV_OPT_TYPE_DOUBLE, {.dbl=1.2}, 1, 2),
    FARNEOPT(treshold,
             "noise treshold. When computing the optical flow, we can cut-off low movement"
             "as noise treshold.",
             AV_OPT_TYPE_DOUBLE, {.dbl=0}, 0, 255),
    FARNEOPT(flags,
             "1: OPTFLOW_USE_INITIAL_FLOW Use the input flow as the initial flow"
             "approximation"
             "2: OPTFLOW_FARNEBACK_GAUSSIAN Use a Gaussian winsize x winsize filter"
             "instead of box filter of the same size for optical flow estimation. Usually,"
             "this option gives more accurate flow than with a box filter, at the cost of"
             "lower speed (and normally winsize for a Gaussian window should be set to a"
             "larger value to achieve the same level of robustness",
             AV_OPT_TYPE_FLAGS, {.i64=0}, 0, 3),
    {NULL}
};

AVFILTER_DEFINE_CLASS(opreconstruct);

static int query_formats(AVFilterContext *ctx)
{
/*    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P,
        AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUVJ411P,
        AV_PIX_FMT_YUVA420P, AV_PIX_FMT_YUVA422P, AV_PIX_FMT_YUVA444P,
        AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRAP,
        AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_NONE
    }; */

    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_BGR24, AV_PIX_FMT_BGRA, AV_PIX_FMT_GRAY8, AV_PIX_FMT_NONE
    };

    ff_set_common_formats(ctx, ff_make_format_list(pix_fmts));

    return 0;
}

static void fill_iplimage_from_frame(IplImage *img, const AVFrame *frame, enum AVPixelFormat pixfmt)
{
    IplImage *tmpimg;
    int depth, channels_nb;

    if      (pixfmt == AV_PIX_FMT_GRAY8) { depth = IPL_DEPTH_8U;  channels_nb = 1; }
    else if (pixfmt == AV_PIX_FMT_BGRA)  { depth = IPL_DEPTH_8U;  channels_nb = 4; }
    else if (pixfmt == AV_PIX_FMT_BGR24) { depth = IPL_DEPTH_8U;  channels_nb = 3; }
    else return;

    tmpimg = cvCreateImageHeader((CvSize){frame->width, frame->height}, depth, channels_nb);
    *img = *tmpimg;
    img->imageData = img->imageDataOrigin = frame->data[0];
    img->dataOrder = IPL_DATA_ORDER_PIXEL;
    img->origin    = IPL_ORIGIN_TL;
    img->widthStep = frame->linesize[0];
}

static void fill_frame_from_mat(AVFrame *frame, const CvMat *mat, enum AVPixelFormat pixfmt)
{
    frame->linesize[0] = mat->step;
    frame->data[0] = mat->data.ptr;
}





static float drawOptFlowMap(const CvMat* flow, CvMat* cflowmap, int step,
                    double scale, CvScalar color)
{
    int x, y;
    //    const float scale = 64.0;
    const float offset = 128.0;
    const float treshold = 1.0;

    float max_flow = 0.0;
    //    (void)scale;
    for( y = 0; y < cflowmap->rows; y += step)
        for( x = 0; x < cflowmap->cols; x += step)
        {
            CvPoint2D32f fxy = CV_MAT_ELEM(*flow, CvPoint2D32f, y, x);
            CvScalar col;

            if (fabs(fxy.x) < treshold) fxy.x = 0;
            if (fabs(fxy.y) < treshold) fxy.y = 0;

            col = CV_RGB (offset + fxy.x*scale, offset + fxy.y*scale, offset);

            /*            cvLine(cflowmap, cvPoint(x,y), cvPoint(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color, 1, 8, 0);
            cvCircle(cflowmap, cvPoint(x,y), 2, color, -1, 8, 0);
            */

            cvCircle(cflowmap, cvPoint(x,y), 2, col, -1, 8, 0);

            if (fabs(fxy.x) > max_flow) max_flow = fabs(fxy.x);
            if (fabs(fxy.y) > max_flow) max_flow = fabs(fxy.y);
        }

    return max_flow;
}

static int filter_interpolate (AVFilterContext *ctx, AVFrame *in, const CvMat *flow, float rate) {
    OpRContext *s = ctx->priv;
    OpRInterpolateContext ic = s->ictx;
    AVFilterLink *outlink = ctx->outputs[2];

    int i = 0, x, y;
    AVFrame *out;

    uint8_t *dst;
    const uint8_t *src;

    const int step = s->step;
    const uint8_t r = s->rgba_map[R];
    const uint8_t g = s->rgba_map[G];
    const uint8_t b = s->rgba_map[B];
    const uint8_t a = s->rgba_map[A];

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);

    if (!out)
        return AVERROR(ENOMEM);

    in->pts  = outlink->frame_count * s->ts_unit;

    av_frame_copy_props(out, in);
    out->pts = (outlink->frame_count + 1) * s->ts_unit;
    /*    if (step == 1)
          return ff_filter_frame(outlink, in);*/

    bzero (buffer,  MAX_WIDTH*MAX_HEIGHT*MAX_DEPTH*sizeof(uint64_t));
    bzero (fbuffer, MAX_WIDTH*MAX_HEIGHT*MAX_DEPTH*sizeof(double));

    av_log (ctx, AV_LOG_WARNING, "interpolate: %p:[%d]%dx%d\n", in, s->nb_planes, flow->rows, flow->cols);
    av_log (ctx, AV_LOG_WARNING, "interpolate: %dx%d\n", in->width, in->height);
    /* first we write down all possible movements */

    src = in ->data[0];
    dst = out->data[0];

    for (y = 0; y < flow->rows; y++) {
        for (x = 0; x < flow->cols; x++) {
            CvPoint2D32f fxy = CV_MAT_ELEM(*flow, CvPoint2D32f, y, x);
            int   x_c,  x_f,  y_c,  y_f;
            float x_ce, x_fe, y_ce, y_fe;
            union  {
                uint64_t i64;
                uint8_t  i8[4];
            } pix;

            x_f  = x + (int) fxy.x*rate;
            x_fe = x_f - fxy.x*rate;
            x_c  = x_f + 1;
            x_ce = 1 - x_fe;

            y_f  = y + (int) fxy.y*rate;
            y_fe = y_f - fxy.y*rate;
            y_c  = y_f + 1;
            y_ce = 1 - y_fe;

            x_f = (x_f < 0)?0:x_f;
            x_f = (x_f > flow->cols)?flow->cols:x_f;

            y_f = (y_f < 0)?0:y_f;
            y_f = (y_f > flow->rows)?flow->rows:y_f;

            x_c = (x_c < 0)?0:x_c;
            x_c = (x_c > flow->cols)?flow->cols:x_c;

            y_c = (y_c < 0)?0:y_c;
            y_c = (y_c > flow->rows)?flow->rows:y_c;


            out->data[0][x_f*step + r + y_f*out->linesize[0]] = src[x*step + r];
            out->data[0][x_f*step + g + y_f*out->linesize[0]] = src[x*step + g];
            out->data[0][x_f*step + b + y_f*out->linesize[0]] = src[x*step + b];

            out->data[0][x_c*step + r + y_f*out->linesize[0]] = src[x*step + r];
            out->data[0][x_c*step + g + y_f*out->linesize[0]] = src[x*step + g];
            out->data[0][x_c*step + b + y_f*out->linesize[0]] = src[x*step + b];

            out->data[0][x_f*step + r + y_c*out->linesize[0]] = src[x*step + r];
            out->data[0][x_f*step + g + y_c*out->linesize[0]] = src[x*step + g];
            out->data[0][x_f*step + b + y_c*out->linesize[0]] = src[x*step + b];

            out->data[0][x_c*step + r + y_c*out->linesize[0]] = src[x*step + r];
            out->data[0][x_c*step + g + y_c*out->linesize[0]] = src[x*step + g];
            out->data[0][x_c*step + b + y_c*out->linesize[0]] = src[x*step + b];

        }
        src += in ->linesize[0];
        dst += out->linesize[0];
    }

            /*
            pix.i64 = in->data[0][x*step + y+in->linesize[0]];

            PUSH_ARRAY(buffer [x_c][y_c], pix.i64);
            PUSH_ARRAY(buffer [x_c][y_f], pix.i64);
            PUSH_ARRAY(buffer [x_f][y_c], pix.i64);
            PUSH_ARRAY(buffer [x_f][y_f], pix.i64);

            PUSH_ARRAY(fbuffer [x_c][y_c], x_ce * y_ce);
            PUSH_ARRAY(fbuffer [x_c][y_f], x_ce * y_fe);
            PUSH_ARRAY(fbuffer [x_f][y_c], x_fe * y_ce);
            PUSH_ARRAY(fbuffer [x_f][y_f], x_fe * y_fe);
        }
        src += in ->linesize[0];
        dst += out->linesize[0];
    }
            */
    /* then we interpolate the actual pixels */
            /*
    dst = out->data[0];
    src = in ->data[0];

    for (y = 0; y < out->height; y++) {
        for (x = 0; x < out->width; x++) {
            //            fprintf (stderr, "x: %d, y: %d, w: %d, h: %d, s: %d\n", x, y, out->width, out->height, step);
            int j;
            uint64_t *buf =  buffer[x][y];
            double  *fbuf = fbuffer[x][y];
            union  {
                uint64_t i64;
                uint8_t  i8[4];
            } pix;

            double sum = 0;

            pix.i64 = 0;
            for (j=0; fbuf[j] && j < MAX_DEPTH ; j++)  {
                pix.i64 += fbuf[j]*fbuf[j]*buf[j];
                sum += fbuf[j]*fbuf[j];
            }

            dst[x*step] = pix.i64;
        }
        dst += out->linesize[0];
        src += in ->linesize[0];
    }
            */
    ff_filter_frame(outlink, in);
    return ff_filter_frame(outlink, out);
}

static int filter_farneback (AVFilterContext *ctx, AVFrame *in) {
    OpRContext *s = ctx->priv;
    OpRFarnebackContext f = s->fctx;
    AVFilterLink **outlink = ctx->outputs;
    int flags = 0;
    int i, ret = 0;

    AVFrame *out[2];
    float mf;
    static int count = 0;
    count++;

    if (f.flags & 1)
        flags |= OPTFLOW_USE_INITIAL_FLOW;
    if (f.flags & 2 )
        flags |= OPTFLOW_FARNEBACK_GAUSSIAN;


    for (i = 0; i<2; i++)  {
        out[i] = ff_get_video_buffer(outlink[i], outlink[i]->w, outlink[i]->h);

        if (!out[i])
            return AVERROR(ENOMEM);

        av_frame_copy_props(out[i], in);
    }

    cvCalcOpticalFlowFarneback(s->prevgray,  s->gray, s->flow,
                               f.pyrScale,   f.levels, f.winsize,
                               f.iterations, f.polyN,  f.polySigma,
                               flags);
    cvCvtColor(s->prevgray, s->cflow, CV_GRAY2BGR);
    mf = drawOptFlowMap(s->flow, s->cflow, 1, 128, CV_RGB(0, 255, 0));
    av_log (ctx, AV_LOG_WARNING, "frame: %d:%p, maxflow: %f\n", count, in, mf);

    fill_frame_from_mat (out[0], s->flow,  outlink[0]->format);
    fill_frame_from_mat (out[1], s->cflow, outlink[1]->format);

    //    av_log(ctx, AV_LOG_WARNING, "filter farneback %lld\n", out[0]->pts);

    ret  = filter_interpolate (ctx, in, s->flow, 0.5);
    //    ret &= filter_interpolate (ctx, in, s->flow, 2)<<4;

    //    av_frame_free (&in);

    ret &= ff_filter_frame(outlink[0], out[0])<<8;
    ret &= ff_filter_frame(outlink[1], out[1])<<16;

    av_log (ctx, AV_LOG_WARNING, "ret: %d\n", ret);
    return 0;
}

static inline IplImage *clone_frame (AVFrame *frame, int format) {
    IplImage outimg;

    fill_iplimage_from_frame(&outimg, frame, format);
    return cvCloneImage(&outimg);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    OpRContext *s = ctx->priv;
    int format = inlink->format;


    //    av_log(ctx, AV_LOG_WARNING, "filter frame %p\n", frame);
    if(!s->gray)
    {
        CvMat *gray = cvCreateMat(frame->height, frame->width, CV_8UC1);
        s->prevgray = cvCreateMat(gray->rows, gray->cols, gray->type);
        s->flow     = cvCreateMat(gray->rows, gray->cols, CV_32FC2);
        s->cflow    = cvCreateMat(gray->rows, gray->cols, CV_8UC3);
        s->gray = gray;
    }

    cvReleaseImage(&s->prev);
    cvReleaseMat(&s->prevgray);
    s->prev     = s->cur;
    s->cur      = s->next;
    s->prevgray = s->gray;
    s->next     = clone_frame (frame, format);
    s->gray	= cvCreateMat(s->next->height, s->next->width, CV_8UC1);
    cvCvtColor(s->next, s->gray, CV_RGB2GRAY);

    if (!s->cur) {
        s->cur = cvCloneImage(s->next);
        if (!s->cur)
            return AVERROR(ENOMEM);
    }

    if (!s->prev)
        return 0;

    return filter_farneback(ctx, frame);
}

static int request_frame(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    OpRContext *s = ctx->priv;

    do {
        int ret;

        if (s->eof)
            return AVERROR_EOF;

        ret = ff_request_frame(ctx->inputs[0]);

        if (ret == AVERROR_EOF && s->cur) {
            filter_frame(ctx->inputs[0], NULL);
            s->eof = 1;
        } else if (ret < 0) {
            return ret;
        }
    } while (!s->cur);

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterLink *inlink = outlink->src->inputs[0];
    OpRContext *s = outlink->src->priv;

    outlink->time_base.num = inlink->time_base.num;
    outlink->time_base.den = inlink->time_base.den * 2;
    outlink->frame_rate.num = inlink->frame_rate.num * 2;
    outlink->frame_rate.den = inlink->frame_rate.den;
    outlink->flags |= FF_LINK_FLAG_REQUEST_LOOP;
    s->ts_unit = av_q2d(av_inv_q(av_mul_q(outlink->frame_rate, outlink->time_base)));

    return 0;

}

static int config_input(AVFilterLink *inlink)
{
    OpRContext *s = inlink->dst->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    int ret;

    if ((ret = av_image_fill_linesizes(s->linesize, inlink->format, inlink->w)) < 0)
        return ret;

    s->planeheight[1] = s->planeheight[2] = FF_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;

    s->nb_planes = av_pix_fmt_count_planes(inlink->format);

    ff_fill_rgba_map(s->rgba_map, inlink->format);
    s->step = av_get_padded_bits_per_pixel(desc) >> 3;

    return 0;
}

static av_cold int init(AVFilterContext *ctx)
{
    av_log(ctx, AV_LOG_WARNING, "initing");
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    OpRContext *s = ctx->priv;

    cvReleaseImage(&s->prev);
    cvReleaseImage(&s->cur );
    cvReleaseImage(&s->next);
}

static const AVFilterPad opr_inputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .filter_frame  = filter_frame,
        .config_props  = config_input,
    },
    { NULL }
};

static const AVFilterPad opr_outputs[] = {
    {
        .name          = "flow",
        .type          = AVMEDIA_TYPE_VIDEO,
        .request_frame = request_frame,
    },
    {
        .name          = "cflow",
        .type          = AVMEDIA_TYPE_VIDEO,
    },
    {
        .name          = "render",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter avfilter_vf_opreconstruct = {
    .name          = "opreconstruct",
    .description   = NULL_IF_CONFIG_SMALL("Reconstruct missing frames in 30p->25p translation"),
    .priv_size     = sizeof(OpRContext),
    .priv_class    = &opreconstruct_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = opr_inputs,
    .outputs       = opr_outputs,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
};
