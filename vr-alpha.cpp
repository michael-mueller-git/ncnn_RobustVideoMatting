#include "net.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

static int detect_rvm(ncnn::Net &net, const cv::Mat &bgr, cv::Mat &pha) {
const int target_width = 512;
    const int target_height = target_width;

    ncnn::Extractor ex = net.create_extractor();
    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
    ncnn::Mat ncnn_in1 = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_width, target_height);

    ncnn_in1.substract_mean_normalize(mean_vals, norm_vals);
    ex.input("in0", ncnn_in1);
    ncnn::Mat pha_;
    ex.extract("out1", pha_);

    cv::Mat cv_pha = cv::Mat(pha_.h, pha_.w, CV_32FC1, (float *)pha_.data);

    resize(cv_pha, cv_pha, cv::Size(bgr.cols, bgr.rows), cv::INTER_LINEAR);

    cv::Mat pha8U;
    cv_pha.convertTo(pha8U, CV_8UC1, 255.0, 0);

    pha8U.copyTo(pha);

    return 0;
}


int main(int argc, char **argv) {
    if(argc != 2) {
        std::cout << "Usage: " << argv[0] << " video_file" << std::endl;
        return -1;
    }
    std::string videopath = argv[1];
    cv::VideoCapture cap(videopath);

    if(!cap.isOpened()) {
        std::cout << "Cannot open cam" << std::endl;
        return -1;
    }

    const int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);

    ncnn::Net left_net;
    left_net.opt.use_vulkan_compute = true;
    left_net.load_param("rvm_ts.ncnn.param");
    left_net.load_model("rvm_ts.ncnn.bin");

    ncnn::Net right_net;
    right_net.opt.use_vulkan_compute = true;
    right_net.load_param("rvm_ts.ncnn.param");
    right_net.load_model("rvm_ts.ncnn.bin");

    double scale = 0.25;

    cv::Size frame_size(round(scale * cap.get(cv::CAP_PROP_FRAME_WIDTH)), round(scale * cap.get(cv::CAP_PROP_FRAME_HEIGHT))); 

    std::cout << frame_size << std::endl;

    cv::VideoWriter out(videopath + "-alpha.mp4", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cap.get(cv::CAP_PROP_FPS), frame_size, false);

    int i = 0;
    while(cap.isOpened()) {
        std::cout << i++ << "/" << frame_count << std::endl;
        cv::Mat frame, resized_frame;

        cap >> frame;

        if (frame.empty())
            break;

        cv::resize(frame, resized_frame, cv::Size(), scale, scale, cv::INTER_AREA);

        cv::Mat left_frame;
        resized_frame(cv::Rect(0, 0, resized_frame.cols/2, resized_frame.rows)).copyTo(left_frame);

        cv::Mat right_frame;
        resized_frame(cv::Rect(resized_frame.cols/2, 0, resized_frame.cols/2, resized_frame.rows)).copyTo(right_frame);

        // std::cout << resized_frame.size() << std::endl;

        cv::Mat left_fgr, right_fgr;

        detect_rvm(left_net, left_frame, left_fgr);
        detect_rvm(left_net, right_frame, right_fgr);

        std::vector<cv::Mat> alpha_stack = {left_fgr, right_fgr};

        cv::Mat alpha;

        cv::hconcat(alpha_stack, alpha);

        out.write(alpha);
    }
    out.release();
    cap.release();

    return 0;
}
