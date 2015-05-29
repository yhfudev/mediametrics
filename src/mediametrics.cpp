// mediamatrics.cpp
//
// calculate media matrics, such as PSNR, SSIM or MS-SSIM values of the video frames.
//
// Author: Yunhui Fu <yhfudev@gmail.com>
//
// Compile:
//   indent -kr -nut mediamatrics.cpp
//   sudo apt-get install libopencv-dev libopencv-gpu-dev x264 v4l-utils ffmpeg
//   # if you use intel graphics card
//   sudo apt-get install ocl-icd-libopencl1
//   sudo apt-get autoremove
//   rm mediamatrics; g++ -g $(pkg-config --cflags opencv) -o mediamatrics mediamatrics.cpp $(pkg-config --libs opencv) -lopencv_gpu -DUSEGPU=0 -DMYDEBUG=0
//
// Install:
//   sudo cp mediamatrics /usr/bin/
//
// Test:
//   ./mediamatrics -m -p ts4k-v315k-320x180.webm -r 853x480,1280*720 /home/var-ubuntu-old/www/dash/media-contents/ts4k-webm/ts4k-v7000k-3840x1714.webm /home/var-ubuntu-old/www/dash/media-contents/ts4k-webm/ts4k-v315k-320x180.webm
//
// Ref:
//    http://docs.opencv.org/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
//    https://github.com/moravianlibrary/differ/tree/master/metric
//
// Runtime errors:
//   1. [swscaler @ 0x1339c30] No accelerated colorspace conversion found from yuv420p to bgr24.
//   According to this post, http://alexsleat.co.uk/2011/01/09/how-to-fix-no-accelerated-colorspace-conversion-found-from-yuv420p-to-bgr24-opencv-2-2-0-ubuntu-10-10/
//      This problem is basically an issue converting YUV to RGB using ffmpeg, in order for it to work ffmpeg needs to be recompiled with x264. To get around it use the following to recompile ffmpeg and OpenCV 2.1/2.2:
//          Follow steps 1-to-4 of FakeOutdoorsman’s guide on ubuntuforums.org – here(https://trac.ffmpeg.org/wiki/UbuntuCompilationGuide)
//          OpenCV 2.1/2.2 Install Guide by Sebastian Montabone - here(http://www.samontab.com/web/2010/04/installing-opencv-2-1-in-ubuntu/)
//   2. cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT) return 0
//      some of the video file don't support such paremeter?
//      ignore 0 in the source code


#define PROG_VERSION_CSTR "0.2.0"

//#define MYDEBUG 0
//#define USEGPU 1

#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <getopt.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string.h>

#include <iomanip>  // for controlling float print precision
//#include <cstdarg>
#include <string>

#include <algorithm>            // std::min

#if 0
#include <cv.h>
#include <highgui.h>
#else
#include <opencv2/opencv.hpp>

#include <vector>
using namespace std;
#include <opencv2/gpu/gpu.hpp>        // GPU structures and methods

#endif

#ifdef __WIN32__                // or whatever
#define PRIiSZ "ld"
#define PRIuSZ "Iu"
#else
#define PRIiSZ "zd"
#define PRIuSZ "zu"
#endif

extern char flg_use_gpu;
#define IS_USEGPU() (flg_use_gpu)

int get_desktop_resolution (int &w, int &h);
void cvShowManyImages(const char* title, int maxw, int maxh, int nArgs, ...);

double calculate_vifp_1 (IplImage * img_orig, IplImage * img_compared);
#define calculate_vifp_realval calculate_vifp_1

CvScalar calculate_mse_scalar(IplImage * img_orig, IplImage * img_compared);
double
calculate_mse_realval(IplImage * img_orig, IplImage * img_compared)
{
    CvScalar s = calculate_mse_scalar (img_orig, img_compared);
    return s.val[0];
}

typedef struct _ssim_pameters_t {
    double K1;
    double K2;
    int L;
    int gaussian_window;
    double gaussian_sigma;
} ssim_pameters_t;

double calculate_psnr_1 (IplImage * img_orig, IplImage * img_compared);
int calculate_ssim_1 ( ssim_pameters_t * param, IplImage * img_orig, IplImage * img_compared, CvScalar * ret_mssim, CvScalar * ret_mean_cs);

#if USEGPU
struct BufferPSNR                                     // Optimized GPU versions
{   // Data allocations are very expensive on GPU. Use a buffer to solve: allocate once reuse later.
    cv::gpu::GpuMat gI1, gI2, gs, t1,t2;
    cv::gpu::GpuMat buf;
};
double calculate_psnr_gpu_optimized (IplImage * img_orig, IplImage * img_compared, BufferPSNR& b);

struct BufferMSSIM                                     // Optimized GPU versions
{   // Data allocations are very expensive on GPU. Use a buffer to solve: allocate once reuse later.
    cv::gpu::GpuMat gI1, gI2, gs, t1,t2;

    cv::gpu::GpuMat I1_2, I2_2, I1_I2;
    std::vector<cv::gpu::GpuMat> vI1, vI2;

    cv::gpu::GpuMat mu1, mu2;
    cv::gpu::GpuMat mu1_2, mu2_2, mu1_mu2;

    cv::gpu::GpuMat sigma1_2, sigma2_2, sigma12;
    cv::gpu::GpuMat t3;
    cv::gpu::GpuMat t4;

    cv::gpu::GpuMat ssim_map;
    cv::gpu::GpuMat cs_map;

    cv::gpu::GpuMat buf;
};
int calculate_ssim_gpu_optimized ( ssim_pameters_t * param, IplImage * img_orig, IplImage * img_compared, CvScalar * ret_mssim, CvScalar * ret_mean_cs, struct BufferMSSIM& b);

BufferPSNR bufferPSNR;
//#define calculate_psnr_gpu(i1,i2) calculate_psnr_gpu_optimized(i1,i2,bufferPSNR)
#define calculate_psnr_gpu(i1,i2) ((IS_USEGPU() && cv::gpu::getCudaEnabledDeviceCount())?calculate_psnr_gpu_optimized(i1,i2,bufferPSNR):calculate_psnr_1(i1,i2))
BufferMSSIM bufferMSSIM;
//#define calculate_ssim_gpu(param,iplimg1,iplimg2,ret_mssim,ret_mean_cs) calculate_ssim_gpu_optimized(param,iplimg1,iplimg2,ret_mssim,ret_mean_cs,bufferMSSIM)
#define calculate_ssim_gpu(param,iplimg1,iplimg2,ret_mssim,ret_mean_cs) ((IS_USEGPU() && cv::gpu::getCudaEnabledDeviceCount())?calculate_ssim_gpu_optimized(param,iplimg1,iplimg2,ret_mssim,ret_mean_cs,bufferMSSIM):calculate_ssim_1(param,iplimg1,iplimg2,ret_mssim,ret_mean_cs))

#else
#define calculate_psnr_gpu(i1,i2) calculate_psnr_1(i1,i2)
#define calculate_ssim_gpu(param,iplimg1,iplimg2,ret_mssim,ret_mean_cs) calculate_ssim_1(param,iplimg1,iplimg2,ret_mssim,ret_mean_cs)
#endif // USEGPU

#define calculate_psnr_realval calculate_psnr_gpu

int
calculate_ssim_wang (IplImage * img_orig, IplImage * img_compared, CvScalar * ret_mssim, CvScalar * ret_mean_cs)
{
    ssim_pameters_t p;
    p.K1 = 0.01f;
    p.K2 = 0.03f;
    p.L = 255;
    p.gaussian_window = 11;
    p.gaussian_sigma = 1.5;
    return calculate_ssim_gpu (&p, img_orig, img_compared, ret_mssim, ret_mean_cs);
}

CvScalar calculate_msssim_scalar(ssim_pameters_t * param, IplImage * source1, IplImage * source2, int level, double *beta);

// MS-SSIM (Wang)
CvScalar
calculate_msssim_wang (IplImage * img_orig, IplImage * img_compared)
{
    double beta[] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
    ssim_pameters_t p;
    p.K1 = 0.01f;
    p.K2 = 0.03f;
    p.L = 255;
    p.gaussian_window = 11;
    p.gaussian_sigma = 1.5;
    return calculate_msssim_scalar (&p, img_orig, img_compared, 5, beta);
}
// MS-SSIM* (Rouse/Hemami)
CvScalar
calculate_msssim_rh (IplImage * img_orig, IplImage * img_compared)
{
    double beta[] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
    ssim_pameters_t p;
    p.K1 = 0.00f;
    p.K2 = 0.00f;
    p.L = 255;
    p.gaussian_window = 11;
    p.gaussian_sigma = 1.5;
    return calculate_msssim_scalar (&p, img_orig, img_compared, 5, beta);
}

double
calculate_ssim_realval(IplImage * img_orig, IplImage * img_compared)
{
    CvScalar mssim;
    CvScalar meancs;
    calculate_ssim_wang (img_orig, img_compared, &mssim, &meancs);
#if MYDEBUG
    std::cerr << "SSIM result of B" << mssim.val[0] << " G" << mssim.val[1] << " R" << mssim.val[2] << std::endl;
#endif // MYDEBUG
    double ssim = mssim.val[0];
    for (unsigned int i = 1; i < 3; i++) {
        ssim = std::min(ssim, mssim.val[i]);
    }
    return ssim;
}

double
calculate_msssim_realval(IplImage * img_orig, IplImage * img_compared)
{
    CvScalar msssim = calculate_msssim_wang (img_orig, img_compared);
#if MYDEBUG
    std::cerr << "MS-SSIM result of B" << msssim.val[0] << " G" << msssim.val[1] << " R" << msssim.val[2] << std::endl;
#endif // MYDEBUG
    double ssim = msssim.val[0];
    for (unsigned int i = 1; i < 3; i++) {
        ssim = std::min(ssim, msssim.val[i]);
    }
    return ssim;
}

typedef struct _target_resolution_t {
    size_t width;
    size_t height;
} target_resolution_t;

class my_capture_t {
public:
    //my_capture_base_t ();
    virtual IplImage * peek_next_frame () = 0;
    virtual size_t get_width () = 0;
    virtual size_t get_height () = 0;
    virtual size_t get_fps () = 0;
    virtual size_t get_frames () = 0;
    virtual void set_frame_position (size_t pos) = 0;
private:
};

class my_capture_onepic_t : public my_capture_t {
public:
    my_capture_onepic_t(char *name): finished(false), img(NULL) { fmt = name; std::cerr << "[DBG] my_capture_onepic_t construct" << std::endl; }
    ~my_capture_onepic_t() { if (NULL != img) { cvReleaseImage(&img); } }
    IplImage * peek_next_frame () { if (finished) return NULL; if (NULL == img) { load_image (); } return img; }
    size_t get_width () { if (NULL == img) { load_image (); }  if (NULL != img) return img->width; return 0; }
    size_t get_height () { if (NULL == img) { load_image (); }  if (NULL != img) return img->height; return 0; }
    size_t get_fps () { return 1; }
    size_t get_frames () { return 1; }
    void set_frame_position (size_t pos) {}

protected:
    void load_image () { img = cvLoadImage(fmt.c_str(), CV_LOAD_IMAGE_ANYCOLOR); }
    bool finished;
    IplImage * img;
    std::string fmt;
};

int
file_exist (char *fn)
{
//int stat(const char *path, struct stat *buf);
    struct stat st;
    if (0 == stat (fn, &st)) {
        return 0;
    }
    return 1;
}

class my_capture_png_var_t : public my_capture_t {
public:
    my_capture_png_var_t (size_t start_idx, char *fmt1): img(NULL), idx(start_idx), frames(0) { fmt = fmt1; std::cerr << "[DBG] my_capture_png_var_t construct" << std::endl; }
    IplImage * peek_next_frame () {
        if (NULL != img) {
            cvReleaseImage(&img);
        }
        load_image ();
        idx ++;
        return img;
    }
    size_t get_width () { if (NULL == img) load_image (); if (NULL != img) { return img->width; } return 0; }
    size_t get_height () { if (NULL == img) load_image (); if (NULL != img) { return img->height; } return 0; }
    size_t get_fps () { return 25; }
    size_t get_frames () {
        if (0 == this->frames) {
            char filename[300];
            int mini = 0;
            int maxi = 0;
            // TODO, binary search
            // search mini
            do {
                sprintf (filename, fmt.c_str(), mini);
                if (0 == file_exist (filename)) {
                    break;
                }
                mini ++;
            } while (1);
            maxi = mini;
            do {
                sprintf (filename, fmt.c_str(), mini);
                if (0 != file_exist (filename)) {
                    break;
                }
                maxi ++;
            } while (1);
            min = mini;
            idx = mini;
            this->frames = maxi - mini;
        }
        return this->frames;
    }
    void set_frame_position (size_t pos) { idx = pos; }

private:
    void load_image () {
        char filename[300];
        sprintf (filename, fmt.c_str(), idx);
        img = cvLoadImage(fmt.c_str(), CV_LOAD_IMAGE_ANYCOLOR);
    }
    IplImage * img;
    std::string fmt;
    size_t idx;
    size_t min;
    size_t frames;
};

class my_capture_file_t : public my_capture_t {
public:
    my_capture_file_t () : capture(NULL) { std::cerr << "[DBG] my_capture_file_t() construct" << std::endl; }
    my_capture_file_t (char *fnvid) { capture = cvCreateFileCapture(fnvid); std::cerr << "[DBG] my_capture_file_t(" << fnvid << ") construct" << std::endl;}
    virtual ~my_capture_file_t () { cvReleaseCapture (&capture); }
    IplImage * peek_next_frame () {
        return cvQueryFrame(this->capture);
    }
    size_t get_width () { return cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH); }
    size_t get_height () { return cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT); }
    size_t get_fps () { return cvGetCaptureProperty(capture, CV_CAP_PROP_FPS); }
    size_t get_frames () { return cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); }
    void set_frame_position (size_t pos) { cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, pos);}
protected:
    CvCapture *capture;
};

class my_capture_cam_t : public my_capture_file_t {
public:
    my_capture_cam_t (int id) { capture = cvCreateCameraCapture (id); std::cerr << "[DBG] my_capture_cam_t construct" << std::endl; }
    //IplImage * peek_next_frame ();
    //size_t get_width ();
    //size_t get_height ();
    //size_t get_fps ();
    //size_t get_frames ();
};

my_capture_t *
create_capture (char *url)
{
    my_capture_t * cap = NULL;
    if (NULL != strstr (url, "%d")) {
        cap = new my_capture_png_var_t (1, url);
    } else if (NULL != strstr (url, "%0")) {
        cap = new my_capture_png_var_t (1, url);
    }
    if (NULL != cap) {
        return cap;
    }
    if (NULL != strstr (url, "cam://")) {
        int id;
        sscanf (url, "cam://%d", &id);
        cap = new my_capture_cam_t (id);
    }
    if (NULL != cap) {
        return cap;
    }
    cap = new my_capture_file_t (url);
    if (NULL != cap) {
        return cap;
    }
    cap = new my_capture_onepic_t (url);
    return cap;
}

void process_video (char *src, char *fnvid, int start_idx, int orig_idx, int comp_idx, char flg_showimg, FILE *fp, target_resolution_t *trout, size_t trsize)
{
    IplImage *img_orig = NULL;
    IplImage *img_compared = NULL;

    IplImage *img_orig_process = NULL;
    IplImage *img_compared_process = NULL;

    IplImage *img_orig_temp = NULL;
    IplImage *img_compared_temp = NULL;

    my_capture_t *cap_compared = create_capture (fnvid);
    if (NULL == cap_compared) {
        std::cerr << "Not found file " << fnvid << std::endl;
        return;
    }
    my_capture_t *cap_orig = create_capture (src);
    if (NULL == cap_orig) {
        std::cerr << "Not found file " << fnvid << std::endl;
        return;
    }
    if (NULL == fp) {
        fp = stdout;
    }

    int numFrames = cap_compared->get_frames();
    target_resolution_t onetr;
    onetr.width = cap_compared->get_width();
    onetr.height = cap_compared->get_height();
    if (NULL == fp) {
        fp = stdout;
    }
    if ((NULL == trout) || (trsize < 1)) {
        trout = &onetr;
        trsize = 1;
    }
    if (orig_idx > 0) {
        cap_orig-> set_frame_position (orig_idx);
    }
    if (comp_idx > 0) {
        cap_compared-> set_frame_position (comp_idx);
    }

    int i = 0;
    int j = 0;
    double ssim = -1;

#if MYDEBUG
    std::cerr << "WARNING: debug mode, only process the first 100 frames!!!!" << std::endl;
    if (numFrames > 100) numFrames = 100; // debug
#endif

    int maxw = 0, maxh = 0;
    if (flg_showimg) {
        get_desktop_resolution (maxw, maxh);
        cvNamedWindow( "mainWin", CV_WINDOW_AUTOSIZE );
    }

    fprintf (fp, "# matric file generated by Media Matrics v" PROG_VERSION_CSTR " yhfudev@gmail.com\n");
    fprintf (fp, "# original file: %s\n", src);
    fprintf (fp, "# original file start frame id #: %d\n", orig_idx);
    fprintf (fp, "# compared file: %s\n", fnvid);
    fprintf (fp, "# compared file start frame id #: %d\n", comp_idx);
    for (j = 0; j < trsize; j ++) {
        fprintf (fp, "# width_video_origin=%"PRIuSZ",height_video_origin=%"PRIuSZ",width_video_compared=%"PRIuSZ",height_video_compared=%"PRIuSZ",width_video_scaled=%"PRIuSZ",height_video_scaled=%"PRIuSZ",fps=%"PRIuSZ",frames=%"PRIuSZ"\n"
            , cap_orig->get_width(), cap_orig->get_height()
            , cap_compared->get_width(), cap_compared->get_height()
            , trout[j].width, trout[j].height
            , cap_compared->get_fps(), cap_compared->get_frames());
    }
    fprintf (fp, "# sequence number, metric type, scale size, value[s] ...\n");

    char flg_change_size_warning = true;
    std::cerr.precision(2);
    // sometimes, the opencv can't get the frames from video file(it's set to 0)
    for (i = 0; (i < numFrames) || (0 == numFrames); i++) {
        std::cerr << "-- " << i << "/" << numFrames << " (" << std::setiosflags(std::ios::fixed) << ((float)i*100/numFrames)<< "%) --               \r";
        img_orig = cap_orig->peek_next_frame ();
        if (NULL == img_orig) {
            std::cerr << "Error: Not found cap_orig next frame [" << i << "] #" << orig_idx + i << std::endl;
            break;
        }
        img_compared = cap_compared->peek_next_frame ();
        if (NULL == img_orig) {
            std::cerr << "Error: Not found cap_compared next frame [" << i << "] #" << comp_idx + i << std::endl;
            break;
        }
        for (j = 0; j < trsize; j ++) {
            // the img_xxx_process are used in real calculation
            // img_xxx_process are the scale values of img_xxx
            img_orig_process = img_orig;
            img_compared_process = img_compared;
            if ((img_orig->width != trout[j].width) || (img_orig->height != trout[j].height)) {
                if (flg_change_size_warning) {
                    //flg_change_size_warning = false;
                    std::cerr << "Warning: convert size from "
                        << img_orig->width << "x" << img_orig->height
                        << " to "
                        << trout[j].width << "x" << trout[j].height
                        << std::endl;
                }
                if (NULL != img_orig_temp) {
                    cvReleaseImage (&img_orig_temp);
                }
                img_orig_temp = cvCreateImage(cvSize(trout[j].width, trout[j].height), img_orig->depth, img_orig->nChannels);
                cvResize(img_orig, img_orig_temp, CV_INTER_LINEAR);
                img_orig_process = img_orig_temp;
            }
            if ((img_compared->width != trout[j].width) || (img_compared->height != trout[j].height)) {
                if (flg_change_size_warning) {
                    std::cerr << "Warning: convert size from "
                        << img_compared->width << "x" << img_compared->height
                        << " to "
                        << trout[j].width << "x" << trout[j].height
                        << std::endl;
                }
                if (NULL != img_compared_temp) {
                    cvReleaseImage (&img_compared_temp);
                }
                img_compared_temp = cvCreateImage(cvSize(trout[j].width, trout[j].height), img_orig->depth, img_orig->nChannels);
                cvResize(img_compared, img_compared_temp, CV_INTER_LINEAR);
                img_compared_process = img_compared_temp;
            }
            if (j + 1 >= trsize) flg_change_size_warning = false;

            double dssim = 0;
            //cv::Mat img = cv::cvarrToMat(&ipl_img, false);
            // IplImage ipl_img = img;
            ssim = calculate_ssim_realval (img_orig_process, img_compared_process);
            dssim = 1.0 / ssim - 1;

            assert (NULL != fp);
            fprintf (fp, "%d\tSSIM\t%"PRIuSZ"x%"PRIuSZ"\t%f\t%f\n", start_idx + i, trout[j].width, trout[j].height, ssim, dssim);

            ssim = calculate_msssim_realval (img_orig_process, img_compared_process);
            fprintf (fp, "%d\tMSSSIM\t%"PRIuSZ"x%"PRIuSZ"\t%f\n", start_idx + i, trout[j].width, trout[j].height, ssim);

            double psnr = calculate_psnr_realval (img_orig_process, img_compared_process);
            fprintf (fp, "%d\tPSNR\t%"PRIuSZ"x%"PRIuSZ"\t%f\n", start_idx + i, trout[j].width, trout[j].height, psnr);

            //ssim = calculate_vifp_realval (img_orig_process, img_compared_process);
            //fprintf (fp, "%d\tVIFP\t%"PRIuSZ"x%"PRIuSZ"\t%f\n", i, trout[j].width, trout[j].height, ssim);

            ssim = calculate_mse_realval (img_orig_process, img_compared_process);
            fprintf (fp, "%d\tMSE\t%"PRIuSZ"x%"PRIuSZ"\t%f\n", start_idx + i, trout[j].width, trout[j].height, ssim);

        }

        if (flg_showimg) {
            //cvShowImage( "mainWin", img_orig );
            //cvShowManyImages ( "mainWin", maxw, maxh, 2, img_orig, img_compared );
            cvShowManyImages ( "mainWin", maxw, maxh, 2, img_orig, img_compared_process );
            //cvShowManyImages ( "mainWin", maxw, maxh, 2, img_orig_process, img_compared_process );
            char key = cvWaitKey(2);
        }
        for (j = 0; j < trsize; j ++) {
            fflush (fp);
        }
    }
    if (NULL != img_orig_temp) {
        cvReleaseImage (&img_orig_temp);
    }
    if (NULL != img_compared_temp) {
        cvReleaseImage (&img_compared_temp);
    }
    delete cap_compared;
    delete cap_orig;
    if (flg_showimg) {
        char key = cvWaitKey(3000);
        cvDestroyWindow("mainWin");
    }
}

void usage(char * progname)
{
    std::ostream & output = std::cerr;
    output << "Media Matrics v" PROG_VERSION_CSTR " yhfudev@gmail.com" << std::endl;
    output << "" << std::endl;
    output << "  generate the media metrics for meida file(s)," << std::endl;
    output << "  the supported metrics include MSE/PSNR/MS-SSIM etc." << std::endl;
    output << "" << std::endl;
    output << "Usage" << std::endl;
    output << "    " << progname << " [options] <source> <compared>" << std::endl;
    output << "" << std::endl;
    output << "Options:" << std::endl;
    output << "\t-g\tDisable GPU if available. default " << (IS_USEGPU()?"enabled":"disabled") << std::endl;
    output << "\t-m\tshow images" << std::endl;
    output << "\t-o <output>\toutput file name" << std::endl;
    output << "\t-r <res>\tresolutions" << std::endl;
    output << "\t-b <#>\tthe output frame # of the first frame" << std::endl;
    output << "\t-s <#>\tthe frame # of the original input video" << std::endl;
    output << "\t-d <#>\tthe frame # of the compared input video" << std::endl;
    output << "\t-h\tshow this message" << std::endl;
    output << "" << std::endl;
    output << "<source>    the png files, use format string" << std::endl;
    output << "<compared>  the video file name" << std::endl;
    output << "" << std::endl;
    output << "the source format examples:" << std::endl;
    output << " -> " << "media.xiph.org/BBB/BBB-1080-png/big_buck_bunny_%05d.png" << std::endl;
    output << " -> " << "media.xiph.org/tearsofsteel/tearsofsteel-1080-png/graded_edit_final_%05d.png" << std::endl;
    output << " -> " << "1080/sintel_trailer_2k_%04d.png" << std::endl;
    output << " -> " << "media.xiph.org/sintel/sintel-4k-png/%08d.png" << std::endl;
    output << " -> " << "ED-1080-png/%05d.png" << std::endl;
    output << "" << std::endl;
    output << "the resolutions examples:" << std::endl;
    output << " -> " << "1920x1080" << std::endl;
    output << " -> " << "1920*1080,320x180" << std::endl;
}

size_t
create_target_resolution (char * resolutions, char * prefix, target_resolution_t ** ptrout)
{
    size_t trsize;

    assert (NULL != ptrout);

    // get how many items
    int i;
    char *p = resolutions;
    for (i = 0; ; i ++) {
        if ((p = strchr (p+1, ',')) == NULL) {
            break;
        }
    }
    *ptrout = (target_resolution_t *) malloc ((i+1) * sizeof (target_resolution_t) );
    if (*ptrout == NULL) {
        return 0;
    }
    trsize = 0;
    p = resolutions;
    char *pn = p;
    char *pend = p + strlen(p);
    int w = 0, h = 0;
    for (trsize = 0; p < pend; ) {
        if (2 != sscanf (p, "%dx%d", &w, &h)) {
            if (2 != sscanf (p, "%d*%d", &w, &h)) {
                break;
            }
        }
        (*ptrout)[trsize].width = w;
        (*ptrout)[trsize].height = h;

        trsize ++;

        if ((pn = strchr (p, ',')) == NULL) {
            pn = p + strlen(p);
        }
        p = pn + 1;
    }

    if (0 == trsize) {
        free (*ptrout);
    }
    return trsize;
}

void
close_target_resolution (target_resolution_t * trout, size_t trsize)
{
    if (NULL == trout) {
        trsize = 0;
    }
    int i = 0;
    if (NULL != trout) {
        free (trout);
    }
}

char flg_use_gpu = 1;

int main(int argc, char *argv[])
{
    int start_idx = 0; // the start serial number
    int orig_idx = 0; // origin video file start frame idx
    int comp_idx = 0; // compared video file start frame idx
    target_resolution_t *trout = NULL;
    size_t trsize = 0;
    char * fn_prefix = (char *)"output";
    char * fn_output = NULL;
    char flg_showimg = 0;
    if (argc < 3) {
        usage (argv[0]);
        exit (1);
    }

    int c;
    struct option longopts[]  = {
        { "showimage",    0, 0, 'm' },
        { "usegpu",       0, 0, 'g' },
        { "output",       1, 0, 'o' },
        { "resolutions",  1, 0, 'r' },
        { "beginidx",     1, 0, 'b' },
        { "origidx",      1, 0, 's' },
        { "compidx",      1, 0, 'd' },

        { "help",         0, 0, 'h' },
        { "verbose",      0, 0, 'v' },
        { 0,              0, 0,  0  },
    };

    while ((c = getopt_long( argc, argv, "b:d:mo:r:s:vh", longopts, NULL )) != EOF) {
        switch (c) {
        case 'b':
            start_idx = atoi (optarg);
            break;
        case 's':
            orig_idx = atoi (optarg);
            break;
        case 'd':
            comp_idx = atoi (optarg);
            break;
        case 'o':
            fn_output = optarg;
            break;
        case 'r':
            trsize = create_target_resolution (optarg, fn_prefix, &trout);
            break;
        case 'v':
            break;
        case 'h':
            break;
        case 'g':
            flg_use_gpu = 1;
            break;
        case 'm':
            flg_showimg = 1;
            break;
        }
    }
    FILE *fp = NULL;
    if (NULL != fn_output) {
        fp = fopen (fn_output, "a");
    }
    if (NULL == fp) {
        fp = stdout;
    }
    process_video (argv[optind], argv[optind + 1], start_idx, orig_idx, comp_idx, flg_showimg, fp, trout, trsize);

    if (NULL != fn_output) {
        if (fp != stdout) {
            fclose (fp);
        }
    }
    close_target_resolution (trout, trsize);
    return 0;
}

#if _WIN32
#include "wtypes.h"
// Get the horizontal and vertical screen sizes in pixel
int
get_desktop_resolution (int &horizontal, int &vertical)
{
    RECT desktop;
    // Get a handle to the desktop window
    const HWND hDesktop = GetDesktopWindow();
    // Get the size of screen to the variable desktop
    GetWindowRect(hDesktop, &desktop);
    // The top left corner will have coordinates (0,0)
    // and the bottom right corner will have coordinates
    // (horizontal, vertical)
    horizontal = desktop.right;
    vertical = desktop.bottom;
    return 0;
}
#else
int
get_desktop_resolution (int &w, int &h)
{
    const char *command = "xrandr | grep '*' | awk '{print $1}'";
    FILE *fpipe = (FILE*)popen(command, "r");
    char line[256];
    while ( fgets( line, sizeof(line), fpipe)) {
        sscanf (line, "%dx%d", &w, &h);
        break;
    }
    pclose(fpipe);
    return 0;
}
#endif

/*Function///////////////////////////////////////////////////////////////

Name:       cvShowManyImages

Purpose:    This is a function illustrating how to display more than one
               image in a single window using Intel OpenCV

Parameters: char *title: Title of the window to be displayed
            int nArgs:   Number of images to be displayed
            ...:         IplImage*, which contains the images

Language:   C++

The method used is to set the ROIs of a Single Big image and then resizing
and copying the input images on to the Single Big Image.

This function does not stretch the image...
It resizes the image without modifying the width/height ratio..

This function can be called like this:

cvShowManyImages("Images", 2, img1, img2);
or
cvShowManyImages("Images", 5, img2, img2, img3, img4, img5);

This function can display upto 12 images in a single window.
It does not check whether the arguments are of type IplImage* or not.
The maximum window size is 700 by 660 pixels.
Does not display anything if the number of arguments is less than
    one or greater than 12.

If you pass a pointer that is not IplImage*, Error will occur.
Take care of the number of arguments you pass, and the type of arguments,
which should be of type IplImage* ONLY.

Idea was from [[BettySanchi]] of OpenCV Yahoo! Groups.

If you have trouble compiling and/or executing
this code, I would like to hear about it.

You could try posting on the OpenCV Yahoo! Groups
[url]http://groups.yahoo.com/group/OpenCV/messages/ [/url]

Parameswaran,
Chennai, India.

cegparamesh[at]gmail[dot]com

...
///////////////////////////////////////////////////////////////////////*/

void cvShowManyImages(const char* title, int maxw, int maxh, int nArgs, ...) {

    va_list args;
    // img - Used for getting the arguments
    IplImage *img;

    // [[DispImage]] - the image in which input images are to be copied
    IplImage *DispImage;

    //int size;
    int i;
    int m, n;
    int x, y;

    // w - Maximum number of images in a row
    // h - Maximum number of images in a column
    int w, h;

    // scale - How much we have to resize the image
    float scale, scale2;
    int max;

    // If the number of arguments is lesser than 0 or greater than 12
    // return without displaying
    if(nArgs <= 0) {
        printf("Number of arguments too small....\n");
        return;
    }
    else if(nArgs > 12) {
        printf("Number of arguments too large....\n");
        return;
    }
    //fprintf (stderr, "resolution: %dx%d\n", maxw, maxh);
    maxw = maxw * 2 / 3;
    maxh = maxh * 2 / 3;
    x = maxw;
    y = maxh;
    va_start(args, nArgs);
    for (i = 0; i < nArgs; i++) {
        img = va_arg(args, IplImage*);
        if (x > img->width) {
            x = img->width;
            y = img->height;
        }
        if (y > img->height) {
            x = img->width;
            y = img->height;
        }
    }
    va_end(args);
    //fprintf (stderr, "changed resolution 1: %dx%d\n", x, y);

    // Determine the size of the image,
    // and the number of rows/cols
    // from number of arguments
    if (nArgs == 1) {
        w = h = 1;
    }
    else if (nArgs == 2) {
        w = 2; h = 1;
        if (maxw > w * x) {
            maxw = x;
            maxh = y;
        } else {
            maxw = maxw / w;
            maxh = maxh / w;
        }
    }
    else if (nArgs == 3 || nArgs == 4) {
        w = 2; h = 2;
        if (maxw > w * x) {
            maxw = x;
            maxh = y;
        } else {
            maxw = maxw / w;
            maxh = maxh / w;
        }
   }
    else if (nArgs == 5 || nArgs == 6) {
        w = 3; h = 2;
        if (maxw > w * x) {
            maxw = x;
            maxh = y;
        } else {
            maxw = maxw / w;
            maxh = maxh / w;
        }
    }
    else if (nArgs == 7 || nArgs == 8) {
        w = 4; h = 2;
        if (maxw > w * x) {
            maxw = x;
            maxh = y;
        } else {
            maxw = maxw / w;
            maxh = maxh / w;
        }
    }
    else {
        w = 4; h = 3;
        if (maxw > w * x) {
            maxw = x;
            maxh = y;
        } else {
            maxw = maxw / w;
            maxh = maxh / w;
        }
    }
    //fprintf (stderr, "changed resolution 2: %dx%d\n", maxw, maxh);

    // Create a new 3 channel image
    DispImage = cvCreateImage( cvSize(100 + maxw*w, 60 + maxh*h), 8, 3 );
    cvSet(DispImage, CV_RGB(0, 0, 0));
    // cv::Mat a; a.setTo (cv::Scalar(redVal,greenVal,blueVal))

    // Used to get the arguments passed
    va_start(args, nArgs);

    // Loop for nArgs number of arguments
    for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + maxw)) {

        // Get the Pointer to the IplImage
        img = va_arg(args, IplImage*);

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if(img == 0) {
            printf("Invalid arguments");
            cvReleaseImage(&DispImage);
            return;
        }

        // Find the width and height of the image
        x = img->width;
        y = img->height;

        // Find whether height or width is greater in order to resize the image
        scale = (float) ( (float) x / maxw );
        scale2 = (float) ( (float) y / maxh );
        if (scale2 > scale) {
            scale = scale2;
        }

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n+= 20 + maxh;
        }

        // Set the image ROI to display the current image
        cvSetImageROI(DispImage, cvRect(m, n, (int)( x/scale ), (int)( y/scale )));

        // Resize the input image and copy the it to the Single Big Image
        cvResize(img, DispImage);

        // Reset the ROI in order to display the next image
        cvResetImageROI(DispImage);
    }

    // Create a new window, and show the Single Big Image
    //cvNamedWindow( title, 1 );
    cvShowImage( title, DispImage);

    //cvWaitKey();
    //cvDestroyWindow(title);

    // End the number of arguments
    va_end(args);

    // Release the Image Memory
    cvReleaseImage(&DispImage);
}

// PSNR : Peak Signal-to-Noise Ratio
// PSNR(A,B) = 10 log10 (MAX * MAX / MSE(A,B) )
// MSE(A,B) = SUM((A-B)^2) / N
double
calculate_psnr_1 (IplImage * img_orig, IplImage * img_compared)
{
    cv::Mat I1 = cv::cvarrToMat(img_orig, false);
    cv::Mat I2 = cv::cvarrToMat(img_compared, false);

    cv::Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

#if USEGPU

double
calculate_psnr_gpu_optimized (IplImage * img_orig, IplImage * img_compared, BufferPSNR& b)
{
    cv::Mat I1 = cv::cvarrToMat(img_orig, false);
    cv::Mat I2 = cv::cvarrToMat(img_compared, false);

    b.gI1.upload(I1);
    b.gI2.upload(I2);

    b.gI1.convertTo(b.t1, CV_32F);
    b.gI2.convertTo(b.t2, CV_32F);

    cv::gpu::absdiff(b.t1.reshape(1), b.t2.reshape(1), b.gs);
    cv::gpu::multiply(b.gs, b.gs, b.gs);

    double sse = cv::gpu::sum(b.gs, b.buf)[0];

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}
#endif // USEGPU

int
calculate_ssim_1 ( ssim_pameters_t * param, IplImage * img_orig, IplImage * img_compared, CvScalar * ret_mssim, CvScalar * ret_mean_cs)
{
    assert (NULL != param);
    assert (NULL != img_orig);
    assert (NULL != img_compared);
    assert (NULL != ret_mssim);
    assert (NULL != ret_mean_cs);
    cv::Mat i1 = cv::cvarrToMat(img_orig, false);
    cv::Mat i2 = cv::cvarrToMat(img_compared, false);

    double C1 = (param->K1 * param->L) * (param->K1 * param->L); // 6.5025
    double C2 = (param->K2 * param->L) * (param->K2 * param->L); // 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(param->gaussian_window, param->gaussian_window), param->gaussian_sigma);
    cv::GaussianBlur(I2, mu2, cv::Size(param->gaussian_window, param->gaussian_window), param->gaussian_sigma);

    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(param->gaussian_window, param->gaussian_window), param->gaussian_sigma);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(param->gaussian_window, param->gaussian_window), param->gaussian_sigma);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(param->gaussian_window, param->gaussian_window), param->gaussian_sigma);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3, t4;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t4 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t4);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
//t3=numerator=((2*mu1_mu2 + C1).*(2*sigma12 + C2))
//t1=denominator=((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

//t2=numerator2=(2*sigma12 + C2)
//t4=denominator2=(sigma1_sq + sigma2_sq + C2) >>>
    cv::Mat cs_map;
    divide(t2, t4, cs_map);      // cs_map =  t2./t4;
    cv::Scalar mean_cs = cv::mean( cs_map ); // mean_cs = average of cs map

    cv::Scalar mssim = cv::mean( ssim_map ); // mssim = average of ssim map

    *ret_mssim = mssim;
    *ret_mean_cs = mean_cs;
    return 0;
}

#if USEGPU

#ifdef CV_VERSION_EPOCH
#define CVVER (CV_VERSION_EPOCH * 10000 + CV_VERSION_MAJOR * 100 + CV_VERSION_MINOR)
#elif defined (CV_SUBMINOR_VERSION)
#define CVVER (CV_MAJOR_VERSION * 10000 + CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION)
#endif

#if (CVVER <= 20301) // 2.3.1
namespace cv { namespace gpu {

CV_EXPORTS void
GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, GpuMat& buf, double sigma1, double sigma2 = 0, int rowBorderType=BORDER_DEFAULT, int columnBorderType=-1, Stream& stream=Stream::Null() )
{
    cv::gpu::GaussianBlur(src, dst, ksize, sigma1, sigma2, rowBorderType, columnBorderType);
}

CV_EXPORTS void
add(const GpuMat& a, const GpuMat& b, GpuMat& c, const GpuMat& mask=GpuMat(), int dtype=-1, Stream& stream=Stream::Null() )
{
    cv::gpu::add (a,b,c,stream);
}

CV_EXPORTS void
add(const GpuMat& a, const Scalar& sc, GpuMat& c, const GpuMat& mask=GpuMat(), int dtype=-1, Stream& stream=Stream::Null() )
{
    cv::gpu::add (a,sc,c,stream);
}

CV_EXPORTS void
subtract(const GpuMat& a, const GpuMat& b, GpuMat& c, const GpuMat& mask=GpuMat(), int dtype=-1, Stream& stream=Stream::Null() )
{
    cv::gpu::subtract (a, b, c, stream);
}

CV_EXPORTS void
multiply(const GpuMat& a, const GpuMat& b, GpuMat& c, double scale=1, int dtype=-1, Stream& stream=Stream::Null() )
{
    cv::gpu::multiply (a,b,c,stream);
}
CV_EXPORTS void
multiply(const GpuMat& a, const Scalar& sc, GpuMat& c, double scale=1, int dtype=-1, Stream& stream = Stream::Null())
{
    cv::gpu::multiply (a,sc,c,stream);
}

CV_EXPORTS void
divide(const GpuMat& a, const GpuMat& b, GpuMat& c, double scale=1, int dtype=-1, Stream& stream=Stream::Null() )
{
    cv::gpu::divide (a,b,c,stream);
}

}}
#else // 2.4.5
#endif

int
calculate_ssim_gpu_optimized ( ssim_pameters_t * param, IplImage * img_orig, IplImage * img_compared, CvScalar * ret_mssim, CvScalar * ret_mean_cs, struct BufferMSSIM& b)
{
    assert (NULL != param);
    assert (NULL != img_orig);
    assert (NULL != img_compared);
    assert (NULL != ret_mssim);
    assert (NULL != ret_mean_cs);
    cv::Mat i1 = cv::cvarrToMat(img_orig, false);
    cv::Mat i2 = cv::cvarrToMat(img_compared, false);

    double C1 = (param->K1 * param->L) * (param->K1 * param->L); // 6.5025
    double C2 = (param->K2 * param->L) * (param->K2 * param->L); // 58.5225;
    /***************************** INITS **********************************/

    b.gI1.upload(i1);
    b.gI2.upload(i2);

    cv::gpu::Stream stream;

    stream.enqueueConvert(b.gI1, b.t1, CV_32F);
    stream.enqueueConvert(b.gI2, b.t2, CV_32F);

    cv::gpu::split(b.t1, b.vI1, stream);
    cv::gpu::split(b.t2, b.vI2, stream);
    cv::Scalar mssim;
    cv::Scalar mean_cs;

    cv::gpu::GpuMat buf;

    for( int i = 0; i < b.gI1.channels(); ++i )
    {
        cv::gpu::multiply(b.vI2[i], b.vI2[i], b.I2_2, stream);        // I2^2
        cv::gpu::multiply(b.vI1[i], b.vI1[i], b.I1_2, stream);        // I1^2
        cv::gpu::multiply(b.vI1[i], b.vI2[i], b.I1_I2, stream);       // I1 * I2

        cv::gpu::GaussianBlur(b.vI1[i], b.mu1, cv::Size(param->gaussian_window, param->gaussian_window), buf, param->gaussian_sigma, 0, cv::BORDER_DEFAULT, -1, stream);
        cv::gpu::GaussianBlur(b.vI2[i], b.mu2, cv::Size(param->gaussian_window, param->gaussian_window), buf, param->gaussian_sigma, 0, cv::BORDER_DEFAULT, -1, stream);

        cv::gpu::multiply(b.mu1, b.mu1, b.mu1_2, stream);
        cv::gpu::multiply(b.mu2, b.mu2, b.mu2_2, stream);
        cv::gpu::multiply(b.mu1, b.mu2, b.mu1_mu2, stream);

        cv::gpu::GaussianBlur(b.I1_2, b.sigma1_2, cv::Size(param->gaussian_window, param->gaussian_window), buf, param->gaussian_sigma, 0, cv::BORDER_DEFAULT, -1, stream);
        cv::gpu::subtract(b.sigma1_2, b.mu1_2, b.sigma1_2, cv::gpu::GpuMat(), -1, stream);
        //b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation

        cv::gpu::GaussianBlur(b.I2_2, b.sigma2_2, cv::Size(param->gaussian_window, param->gaussian_window), buf, param->gaussian_sigma, 0, cv::BORDER_DEFAULT, -1, stream);
        cv::gpu::subtract(b.sigma2_2, b.mu2_2, b.sigma2_2, cv::gpu::GpuMat(), -1, stream);
        //b.sigma2_2 -= b.mu2_2;

        cv::gpu::GaussianBlur(b.I1_I2, b.sigma12, cv::Size(param->gaussian_window, param->gaussian_window), buf, param->gaussian_sigma, 0, cv::BORDER_DEFAULT, -1, stream);
        cv::gpu::subtract(b.sigma12, b.mu1_mu2, b.sigma12, cv::gpu::GpuMat(), -1, stream);
        //b.sigma12 -= b.mu1_mu2;

        //here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
        cv::gpu::multiply(b.mu1_mu2, 2, b.t1, 1, -1, stream); //b.t1 = 2 * b.mu1_mu2 + C1;
        cv::gpu::add(b.t1, C1, b.t1, cv::gpu::GpuMat(), -1, stream);
        cv::gpu::multiply(b.sigma12, 2, b.t2, 1, -1, stream); //b.t2 = 2 * b.sigma12 + C2;
        cv::gpu::add(b.t2, C2, b.t2, cv::gpu::GpuMat(), -12, stream);

        cv::gpu::multiply(b.t1, b.t2, b.t3, 1, -1, stream);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

        cv::gpu::add(b.mu1_2, b.mu2_2, b.t1, cv::gpu::GpuMat(), -1, stream);
        cv::gpu::add(b.t1, C1, b.t1, cv::gpu::GpuMat(), -1, stream);

        cv::gpu::add(b.sigma1_2, b.sigma2_2, b.t4, cv::gpu::GpuMat(), -1, stream);
        cv::gpu::add(b.t4, C2, b.t4, cv::gpu::GpuMat(), -1, stream);


        cv::gpu::multiply(b.t1, b.t4, b.t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        cv::gpu::divide(b.t3, b.t1, b.ssim_map, 1, -1, stream);      // ssim_map =  t3./t1;

        cv::gpu::divide(b.t2, b.t4, b.cs_map, 1, -1, stream);      // cs_map =  t2./t4;

        stream.waitForCompletion();

        cv::Scalar s = cv::gpu::sum(b.ssim_map, b.buf);
        mssim.val[i] = s.val[0] / (b.ssim_map.rows * b.ssim_map.cols);

        s = cv::gpu::sum(b.cs_map, b.buf);
        mean_cs.val[i] = s.val[0] / (b.cs_map.rows * b.cs_map.cols);
    }
    *ret_mssim = mssim;
    *ret_mean_cs = mean_cs;
    return 0;
}
#endif // USEGPU

#if MYDEBUG
int
calculate_ssim_2 ( ssim_pameters_t * param, IplImage * img_orig, IplImage * img_compared, CvScalar * ret_mssim, CvScalar * ret_mean_cs)
{
    assert (NULL != param);
    assert (NULL != img_orig);
    assert (NULL != img_compared);
    assert (NULL != ret_mssim);
    assert (NULL != ret_mean_cs);
    cv::Mat i1 = cv::cvarrToMat(img_orig, false);
    cv::Mat i2 = cv::cvarrToMat(img_compared, false);

    double C1 = (param->K1 * param->L) * (param->K1 * param->L); // 6.5025
    double C2 = (param->K2 * param->L) * (param->K2 * param->L); // 58.5225;

    IplImage
        * img1 = NULL, *img2 = NULL, *img1_img2 = NULL,
        *img1_temp = NULL, *img2_temp = NULL,
        *img1_sq = NULL, *img2_sq = NULL,
        *mu1 = NULL, *mu2 = NULL,
        *mu1_sq = NULL, *mu2_sq = NULL, *mu1_mu2 = NULL,
        *sigma1_sq = NULL, *sigma2_sq = NULL, *sigma12 = NULL,
        *ssim_map = NULL, *cs_map = NULL, *temp1 = NULL, *temp2 = NULL, *temp3 = NULL, *temp4 = NULL;

    img1_temp = img_orig;
    img2_temp = img_compared;

    int x = img1_temp->width, y = img1_temp->height;
    int nChan = img1_temp->nChannels, d = IPL_DEPTH_32F;
    CvSize size = cvSize(x, y);

    img1 = cvCreateImage(size, d, nChan);
    img2 = cvCreateImage(size, d, nChan);
    cvConvert(img1_temp, img1);
    cvConvert(img2_temp, img2);

    img1_sq = cvCreateImage(size, d, nChan);
    img2_sq = cvCreateImage(size, d, nChan);
    img1_img2 = cvCreateImage(size, d, nChan);

    cvPow(img1, img1_sq, 2);
    cvPow(img2, img2_sq, 2);
    cvMul(img1, img2, img1_img2, 1);

    mu1 = cvCreateImage(size, d, nChan);
    mu2 = cvCreateImage(size, d, nChan);

    mu1_sq = cvCreateImage(size, d, nChan);
    mu2_sq = cvCreateImage(size, d, nChan);
    mu1_mu2 = cvCreateImage(size, d, nChan);

    sigma1_sq = cvCreateImage(size, d, nChan);
    sigma2_sq = cvCreateImage(size, d, nChan);
    sigma12 = cvCreateImage(size, d, nChan);

    temp1 = cvCreateImage(size, d, nChan);
    temp2 = cvCreateImage(size, d, nChan);
    temp3 = cvCreateImage(size, d, nChan);
    temp4 = cvCreateImage(size, d, nChan);

    ssim_map = cvCreateImage(size, d, nChan);
    cs_map = cvCreateImage(size, d, nChan);
    /*************************** END INITS **********************************/

    //////////////////////////////////////////////////////////////////////////
    // PRELIMINARY COMPUTING
    cvSmooth(img1, mu1, CV_GAUSSIAN, param->gaussian_window, param->gaussian_window, param->gaussian_sigma);
    cvSmooth(img2, mu2, CV_GAUSSIAN, param->gaussian_window, param->gaussian_window, param->gaussian_sigma);
    cvReleaseImage (&img1);
    cvReleaseImage (&img2);

    cvPow(mu1, mu1_sq, 2);
    cvPow(mu2, mu2_sq, 2);
    cvMul(mu1, mu2, mu1_mu2, 1);
    cvReleaseImage (&mu1);
    cvReleaseImage (&mu2);

    cvSmooth(img1_sq, sigma1_sq, CV_GAUSSIAN, param->gaussian_window, param->gaussian_window, param->gaussian_sigma);
    cvAddWeighted(sigma1_sq, 1, mu1_sq, -1, 0, sigma1_sq);

    cvSmooth(img2_sq, sigma2_sq, CV_GAUSSIAN, param->gaussian_window, param->gaussian_window, param->gaussian_sigma);
    cvAddWeighted(sigma2_sq, 1, mu2_sq, -1, 0, sigma2_sq);
    cvReleaseImage (&img1_sq);
    cvReleaseImage (&img2_sq);

    cvSmooth(img1_img2, sigma12, CV_GAUSSIAN, param->gaussian_window, param->gaussian_window, param->gaussian_sigma);
    cvAddWeighted(sigma12, 1, mu1_mu2, -1, 0, sigma12);
    cvReleaseImage (&img1_img2);

    //////////////////////////////////////////////////////////////////////////
    // FORMULA

    // (2*mu1_mu2 + C1)
    cvScale(mu1_mu2, temp1, 2);
    cvAddS(temp1, cvScalarAll(C1), temp1);
    cvReleaseImage (&mu1_mu2);

    // (2*sigma12 + C2)
    cvScale(sigma12, temp2, 2);
    cvAddS(temp2, cvScalarAll(C2), temp2);
    cvReleaseImage (&sigma12);

    // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    cvMul(temp1, temp2, temp3, 1);

    // (mu1_sq + mu2_sq + C1)
    cvAdd(mu1_sq, mu2_sq, temp1);
    cvAddS(temp1, cvScalarAll(C1), temp1);
    cvReleaseImage (&mu1_sq);
    cvReleaseImage (&mu2_sq);

    // (sigma1_sq + sigma2_sq + C2)
    cvAdd(sigma1_sq, sigma2_sq, temp4);
    cvAddS(temp4, cvScalarAll(C2), temp4);
    cvReleaseImage (&sigma1_sq);
    cvReleaseImage (&sigma2_sq);

    // ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
    cvMul(temp1, temp4, temp1, 1);

    // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
    cvDiv(temp3, temp1, ssim_map, 1);

    cvDiv(temp2, temp4, cs_map, 1);

    CvScalar mssim = cvAvg(ssim_map);
    CvScalar mean_cs = cvAvg(cs_map);

    // through observation, there is approximately
    // 1% error max with the original matlab program

    //std::cout << "(R, G & B SSIM index)" << std::endl;
    //std::cout << index_scalar.val[2] * 100 << "%" << std::endl;
    //std::cout << index_scalar.val[1] * 100 << "%" << std::endl;
    //std::cout << index_scalar.val[0] * 100 << "%" << std::endl;
    cvReleaseImage (&temp1);
    cvReleaseImage (&temp2);
    cvReleaseImage (&temp3);
    cvReleaseImage (&temp4);
    cvReleaseImage (&ssim_map);
    cvReleaseImage (&cs_map);

    *ret_mssim = mssim;
    *ret_mean_cs = mean_cs;
    return 0;
}
#endif // MYDEBUG

// MSE : Mean Squared Error
// MSE(A,B) = SUM((A-B)^2) / N
CvScalar
calculate_mse_scalar(IplImage * img_orig, IplImage * img_compared)
{
    IplImage *src1 = img_orig;
    IplImage *src2 = img_compared;

    int x = src1->width, y = src1->height;
    int nChan = src1->nChannels, d = IPL_DEPTH_32F;
    //size before down sampling
    CvSize size = cvSize(x, y);

    //creating diff and difference squares
    IplImage *img1 = cvCreateImage(size, d, nChan);
    IplImage *img2 = cvCreateImage(size, d, nChan);
    IplImage *diff = cvCreateImage(size, d, nChan);
    IplImage *diff_sq = cvCreateImage(size, d, nChan);

    cvConvert(src1, img1);
    cvConvert(src2, img2);
    cvAbsDiff(img1, img2, diff);
    //Squaring the images thus created
    cvPow(diff, diff_sq, 2);

    CvScalar mse = cvAvg(diff_sq);

    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&diff);
    cvReleaseImage(&diff_sq);
    //cvReleaseImage(&src1);
    //cvReleaseImage(&src2);

    return mse;
}

// MS-SSIM : Multi-Scale Structural SIMilarity
// level = 5
// double beta[] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
CvScalar
calculate_msssim_scalar(ssim_pameters_t * param, IplImage * source1, IplImage * source2, int level, double *beta)
{
    CvScalar ms_ssim_value;
    // image dimensions
    int x, y;
    int nChan = source1->nChannels;
    int d = source1->depth;

    x = source1->width, y = source1->height;


#ifdef DEBUG
    cout<<"\nAlpha = "<<alpha[0]<<" "<<alpha[1]<<" "<<alpha[2]<<" "<<alpha[3]<<"\n";
    cout<<"\nBeta = "<<beta[0]<<" "<<beta[1]<<" "<<beta[2]<<" "<<beta[3]<<"\n";
    cout<<"\nGamma = "<<gamma[0]<<" "<<gamma[1]<<" "<<gamma[2]<<" "<<gamma[3]<<"\n";
#endif

    IplImage *downsampleSrc1 = NULL;
    IplImage *downsampleSrc2 = NULL;
    for (int i=0; i < level; i++) {
        //Downsampling of the original images
        //Downsampling the images
        CvSize downs_size = cvSize(x, y);
        downsampleSrc1 = cvCreateImage(downs_size, d, nChan);
        downsampleSrc2 = cvCreateImage(downs_size, d, nChan);
        cvResize(source1, downsampleSrc1, CV_INTER_NN);
        cvResize(source2, downsampleSrc2, CV_INTER_NN);

        #ifdef DEBUG
        cout<<"Values at level="<<i<<" \n";
        #endif

        CvScalar mssim_t;
        CvScalar mcs_t;
        calculate_ssim_1 ( param, downsampleSrc1, downsampleSrc2, &mssim_t, &mcs_t);

        // calculating the withed average to find ms-ssim
        for (int j = 0; j < 4; j++) {
            if (i == 0) {
                ms_ssim_value.val[j] = pow((mcs_t.val[j]), (double)(beta[i]));
            } else if (i == level - 1) {
                ms_ssim_value.val[j] = (ms_ssim_value.val[j]) * pow((mssim_t.val[j]), (double)(beta[i]));
            } else {
                ms_ssim_value.val[j] = (ms_ssim_value.val[j]) * pow((mcs_t.val[j]), (double)(beta[i]));
            }
        }
        //Release images
        cvReleaseImage(&downsampleSrc1);
        cvReleaseImage(&downsampleSrc2);
        x >>= 1;
        y >>= 1;
    }
    return ms_ssim_value;
}

// VIFP: Visual Information Fidelity, pixel domain version (VIFp)
void
applyGaussianBlur(const cv::Mat& src, cv::Mat& dst, int ksize, double sigma)
{
    int invalid = (ksize-1)/2;
    cv::Mat tmp(src.rows, src.cols, CV_32F);
    cv::GaussianBlur(src, tmp, cv::Size(ksize,ksize), sigma);
    tmp(cv::Range(invalid, tmp.rows-invalid), cv::Range(invalid, tmp.cols-invalid)).copyTo(dst);
}

const float SIGMA_NSQ = 2.0f;
static const int NLEVS = 4;

static void
_calculate_vifp (const cv::Mat& ref, const cv::Mat& dist, int N, double& num, double& den)
{
    int w = ref.cols - (N-1);
    int h = ref.rows - (N-1);

    cv::Mat tmp(h,w,CV_32F);
    cv::Mat mu1(h,w,CV_32F), mu2(h,w,CV_32F), mu1_sq(h,w,CV_32F), mu2_sq(h,w,CV_32F), mu1_mu2(h,w,CV_32F), sigma1_sq(h,w,CV_32F), sigma2_sq(h,w,CV_32F), sigma12(h,w,CV_32F), g(h,w,CV_32F), sv_sq(h,w,CV_32F);
    cv::Mat sigma1_sq_th, sigma2_sq_th, g_th;

    // mu1 = filter2(win, ref, 'valid');
    applyGaussianBlur(ref, mu1, N, N/5.0);
    // mu2 = filter2(win, dist, 'valid');
    applyGaussianBlur(dist, mu2, N, N/5.0);

    const float EPSILON = 1e-10f;

    // mu1_sq = mu1.*mu1;
    cv::multiply(mu1, mu1, mu1_sq);
    // mu2_sq = mu2.*mu2;
    cv::multiply(mu2, mu2, mu2_sq);
    // mu1_mu2 = mu1.*mu2;
    cv::multiply(mu1, mu2, mu1_mu2);

    // sigma1_sq = filter2(win, ref.*ref, 'valid') - mu1_sq;
    cv::multiply(ref, ref, tmp);
    applyGaussianBlur(tmp, sigma1_sq, N, N/5.0);
    sigma1_sq -= mu1_sq;
    // sigma2_sq = filter2(win, dist.*dist, 'valid') - mu2_sq;
    cv::multiply(dist, dist, tmp);
    applyGaussianBlur(tmp, sigma2_sq, N, N/5.0);
    sigma2_sq -= mu2_sq;
    // sigma12 = filter2(win, ref.*dist, 'valid') - mu1_mu2;
    cv::multiply(ref, dist, tmp);
    applyGaussianBlur(tmp, sigma12, N, N/5.0);
    sigma12 -= mu1_mu2;

    // sigma1_sq(sigma1_sq<0)=0;
    cv::max(sigma1_sq, 0.0f, sigma1_sq);
    // sigma2_sq(sigma2_sq<0)=0;
    cv::max(sigma2_sq, 0.0f, sigma2_sq);

    // g=sigma12./(sigma1_sq+1e-10);
    tmp = sigma1_sq + EPSILON;
    cv::divide(sigma12, tmp, g);

    // sv_sq=sigma2_sq-g.*sigma12;
    cv::multiply(g, sigma12, tmp);
    sv_sq = sigma2_sq - tmp;

    cv::threshold(sigma1_sq, sigma1_sq_th, EPSILON, 1.0f, cv::THRESH_BINARY);

    // g(sigma1_sq<1e-10)=0;
    cv::multiply(g, sigma1_sq_th, g);

    // sv_sq(sigma1_sq<1e-10)=sigma2_sq(sigma1_sq<1e-10);
    cv::multiply(sv_sq, sigma1_sq_th, sv_sq);
    cv::multiply(sigma2_sq, 1.0f - sigma1_sq_th, tmp);
    sv_sq += tmp;

    // sigma1_sq(sigma1_sq<1e-10)=0;
    cv::threshold(sigma1_sq, sigma1_sq, EPSILON, 1.0f, cv::THRESH_TOZERO);

    cv::threshold(sigma2_sq, sigma2_sq_th, EPSILON, 1.0f, cv::THRESH_BINARY);

    // g(sigma2_sq<1e-10)=0;
    cv::multiply(g, sigma2_sq_th, g);

    // sv_sq(sigma2_sq<1e-10)=0;
    cv::multiply(sv_sq, sigma2_sq_th, sv_sq);

    cv::threshold(g, g_th, 0.0f, 1.0f, cv::THRESH_BINARY);

    // sv_sq(g<0)=sigma2_sq(g<0);
    cv::multiply(sv_sq, g_th, sv_sq);
    cv::multiply(sigma2_sq, 1.0f - g_th, tmp);
    cv::add(sv_sq, tmp, sv_sq);

    // g(g<0)=0;
    cv::max(g, 0.0f, g);

    // sv_sq(sv_sq<=1e-10)=1e-10;
    cv::max(sv_sq, EPSILON, sv_sq);

    // num=num+sum(sum(log10(1+g.^2.*sigma1_sq./(sv_sq+sigma_nsq))));
    sv_sq += SIGMA_NSQ;
    cv::multiply(g, g, g);
    cv::multiply(g, sigma1_sq, g);
    cv::divide(g, sv_sq, tmp);
    tmp += 1.0f;
    cv::log(tmp, tmp);
    num += cv::sum(tmp)[0] / log(10.0f);

    // den=den+sum(sum(log10(1+sigma1_sq./sigma_nsq)));
    tmp = 1.0f + sigma1_sq / SIGMA_NSQ;
    cv::log(tmp, tmp);
    den += cv::sum(tmp)[0] / log(10.0f);
}

double
calculate_vifp_1 (IplImage * img_orig, IplImage * img_compared)
{
    cv::Mat original = cv::cvarrToMat(img_orig, false);
    cv::Mat processed = cv::cvarrToMat(img_compared, false);

    double num = 0.0;
    double den = 0.0;

    cv::Mat ref[NLEVS];
    cv::Mat dist[NLEVS];
    cv::Mat tmp1, tmp2;

    int w = original.cols; //img_compared->width;
    int h = original.rows; //img_compared->height;

    // for scale=1:4
    for (int scale = 0; scale < NLEVS; scale ++) {
        // N=2^(4-scale+1)+1;
        int N = (2 << (NLEVS-scale-1)) + 1;

        if (scale == 0) {
            original.copyTo(ref[scale]);
            processed.copyTo(dist[scale]);
        }
        else {
            // ref=filter2(win,ref,'valid');
            applyGaussianBlur(ref[scale-1], tmp1, N, N/5.0);
            // dist=filter2(win,dist,'valid');
            applyGaussianBlur(dist[scale-1], tmp2, N, N/5.0);

            w = (w-(N-1)) / 2;
            h = (h-(N-1)) / 2;

            ref[scale] = cv::Mat(h,w,CV_32F);
            dist[scale] = cv::Mat(h,w,CV_32F);

            // ref=ref(1:2:end,1:2:end);
            cv::resize(tmp1, ref[scale], cv::Size(w,h), 0, 0, cv::INTER_NEAREST);
            // dist=dist(1:2:end,1:2:end);
            cv::resize(tmp2, dist[scale], cv::Size(w,h), 0, 0, cv::INTER_NEAREST);
        }

        _calculate_vifp(ref[scale], dist[scale], N, num, den);
    }

    return (num/den);
}

