#include "net.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "argparse.hpp"
#include <vector>
#include <fstream>

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

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

int main(int argc, char **argv) {
    argparse::ArgumentParser program("vr-alpha");

      program.add_argument("video")
        .help("video path");

    program.add_argument("type")
      .choices("eq", "fisheye180", "fisheye190", "fisheye200");

    program.add_argument("--scale")
        .default_value(0.25)
        .scan<'g', double>();
    
    try {
        program.parse_args(argc, argv);
      }
      catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
      }

    auto inputvideo = program.get<std::string>("video");
    auto videotype = program.get<std::string>("type");
    double scale =  program.get<double>("--scale");

    if (!fileExists(inputvideo)) {
        std::cout << "Input video not found" << std::endl;
        return -1;
    }
        
    std::string dest = inputvideo;
    if (dest.size() >= 4) {
        dest.erase(dest.size() - 4);
    }

    std::string fisheyevideo = inputvideo;
    if (videotype == "eq") { 
        std::string cmd1 = "ffmpeg -i \"";
        cmd1 += inputvideo;
        cmd1 += R"(" -filter_complex "[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]" -map '[v]' -c:a copy -crf 15 ")";
        fisheyevideo = dest + "-fisheye.mp4";
        cmd1 += fisheyevideo;
        cmd1 += "\"";
            
        std::cout << "cmd1: " << cmd1 << std::endl;

        if (int ret = system(cmd1.c_str()); ret != 0)  {
            return ret;
        }
    } else if (videotype == "fisheye180") {
        // Nothing to do here
    } else{
        std::cout << "Not implemented" << std::endl;
        return  -1;
    }

    cv::VideoCapture cap(fisheyevideo);

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


    cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)); 

    std::cout << frame_size << std::endl;

    cv::VideoWriter out(dest + "-alpha.avi", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(cv::CAP_PROP_FPS), frame_size, false);

    int i = 0;
    while(cap.isOpened()) {
        std::cout << i++ << "/" << frame_count << std::endl;
        cv::Mat frame, resized_frame;
        cap >> frame;

        if (frame.empty())
            break;

        auto target_size = frame.size();

        cv::resize(frame, resized_frame, cv::Size(), scale, scale, cv::INTER_AREA);

        cv::Mat left_frame;
        resized_frame(cv::Rect(0, 0, resized_frame.cols/2, resized_frame.rows)).copyTo(left_frame);

        cv::Mat right_frame;
        resized_frame(cv::Rect(resized_frame.cols/2, 0, resized_frame.cols/2, resized_frame.rows)).copyTo(right_frame);

        cv::Mat left_fgr, right_fgr;

        detect_rvm(left_net, left_frame, left_fgr);
        detect_rvm(left_net, right_frame, right_fgr);

        std::vector<cv::Mat> alpha_stack = {left_fgr, right_fgr};

        cv::Mat alpha;

        cv::hconcat(alpha_stack, alpha);

        cv::Mat alpha_final;
        cv::resize(alpha, alpha_final, target_size);

        out.write(alpha_final);
    }
    out.release();
    cap.release();


    std::string cmd2 = R"(ffmpeg -i ")";
    cmd2 += fisheyevideo ;
    cmd2 += "\" -i \"";
    cmd2 += dest + "-alpha.avi";
    cmd2 += R"(" -i "mask.png" -i ")";
    cmd2 += inputvideo;
    cmd2 += R"(" -filter_complex "[1]scale=iw*0.4:-1[alpha];[2][alpha]scale2ref[mask][alpha];[alpha][mask]alphamerge,split=2[masked_alpha1][masked_alpha2]; [masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; [masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; [0][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt];[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; [out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt];[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; [out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt];[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h" -c:v libx265 -crf 17 -preset veryfast -map 3:a:? -c:a copy ")";
    cmd2 += dest + "-result.mp4";
    cmd2 += "\" -y";

    std::cout << "cmd2: " << cmd2 << std::endl;

    return system(cmd2.c_str());
}
