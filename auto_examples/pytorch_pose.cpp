// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <algorithm>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <typeinfo>
#include "json.hpp"

int SPPE_TENSOR_W = 256;
int SPPE_TENSOR_H = 320;

static int detect_posenet(ncnn::Net& posenet, const cv::Mat& bgr, std::vector<KeyPoint>& keypoints, char* inp_layer, char* out_layer)
{

//    std::vector<KP> target;
    cv::Mat img_tmp = bgr.clone();
    cv::Scalar grey_value(128, 128, 128);
    cv::Mat sppe_padded_img(SPPE_TENSOR_H, SPPE_TENSOR_W, CV_8UC3, grey_value);

    int original_w = bgr.cols, original_h = bgr.rows;
    double resize_ratio = 1;
    double resize_ratio_1 = (double)SPPE_TENSOR_W/original_w;
    double resize_ratio_2 = (double)SPPE_TENSOR_H/original_h;

    double resize_ratio_final = resize_ratio_1 < resize_ratio_2 ? resize_ratio_1 : resize_ratio_2;
    resize_ratio = resize_ratio_final;

    double new_w = (double)original_w * resize_ratio;
    double new_h = (double)original_h * resize_ratio;
    double padded_x ;
    double padded_y ;
    if(original_w > original_h)
    {
        padded_x = 0;
        padded_y = (0.5)*((double)320.0 - new_h);
        //std::cout << "w: " << img.cols << " h: " << img.rows << " resize_ratio: " << resize_ratio <<std::endl;
    }
    else
    {
        padded_x = (0.5)*((double)256.0 - new_w);
        padded_y = 0;
        //std::cout << "w: " << img.cols << " h: " << img.rows << " resize_ratio: " << resize_ratio <<std::endl;
    }
    cv::Size new_sz(new_w,new_h);
    cv::resize(img_tmp, img_tmp, new_sz);
    cv::imshow("resized", img_tmp);
    img_tmp.copyTo(sppe_padded_img(cv::Rect(padded_x, padded_y, new_w, new_h)));

    cv::imshow("padded", sppe_padded_img);
//    cv::waitKey(0);


    auto start = std::chrono::steady_clock::now();


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(sppe_padded_img.data, ncnn::Mat::PIXEL_BGR2RGB, SPPE_TENSOR_W, SPPE_TENSOR_H, 256, 320);

    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = posenet.create_extractor();

    ex.input(inp_layer, in);

    ncnn::Mat out;
    ex.extract(out_layer, out);

    int out_w = out.w, out_h = out.h;

    // resolve point from heatmap
    keypoints.clear();
    for (int p = 0; p < out.c; p++)
    {
        const ncnn::Mat m = out.channel(p);
        std::cout << typeid(m).name()<<std::endl;
        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < out.h; y++)
        {
            const float* ptr = m.row(y);
            for (int x = 0; x < out.w; x++)
            {
                float prob = ptr[x];
                if (prob > max_prob)
                {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }

        KeyPoint keypoint;
        float coord_x = (float) ((float) max_x / (float) out_w * (float) SPPE_TENSOR_W - (float )padded_x) / (float) resize_ratio ;
        float coord_y = (float) ((float) max_y / (float) out_h * (float) SPPE_TENSOR_H - (float )padded_y) / (float) resize_ratio ;
        keypoint.p = cv::Point2f(coord_x, coord_y);
//        keypoint.p = cv::Point2f(max_x * w / (float)out.w, max_y * h / (float)out.h);
        keypoint.prob = max_prob;

        keypoints.push_back(keypoint);
    }

    return 0;
}

static cv::Mat draw_pose(const cv::Mat& bgr, const std::vector<KeyPoint>& keypoints)
{
    cv::Mat image = bgr.clone();

    // draw bone
    static const int joint_pairs[12][2] = {
        {5, 3}, {3, 1}, {1, 2}, {2, 4}, {4, 6}, {1, 7}, {2, 8}, {7, 8}, {7, 9}, {9, 11}, {8, 10}, {10, 12},
    };

    for (int i = 0; i < 12; i++)
    {
        const KeyPoint& p1 = keypoints[joint_pairs[i][0]];
        const KeyPoint& p2 = keypoints[joint_pairs[i][1]];

        if (p1.prob < 0.04f || p2.prob < 0.04f)
            continue;

        cv::line(image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
    }

    // draw joint
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const KeyPoint& keypoint = keypoints[i];

        fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

        if (keypoint.prob < 0.2f)
            continue;

        cv::circle(image, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
    }

    //cv::imshow("image", image);
    //cv::waitKey(0);
    return image;

}

int main(int argc, char** argv)
{   
    const char* imagepath = argv[1];

    ncnn::Net posenet;

    posenet.opt.use_vulkan_compute = true;
//    posenet.load_param("model_pose/model.param");
//    posenet.load_model("model_pose/model.bin");
    posenet.load_param("../../build/auto_examples/model_pose/model.param");
    posenet.load_model("../../build/auto_examples/model_pose/model.bin");
    struct stat s;
    int camera = int(*imagepath) -'0';
    int id = 0;

    // open camera
    if (camera == 0)
    {
        cv::VideoCapture capture(0);
        while(true)
        {
            cv::Mat m;
            capture >> m;
            cv::resize(m,m,cv::Size(416,416));
            std::vector<KeyPoint> keypoints;
            detect_posenet(posenet, m, keypoints, argv[3], argv[4]);

            // write to json
            json_data::write_json("example.json", "camera", 1, keypoints);

            cv::Mat image = draw_pose(m, keypoints);
            cv::imshow("video", image);
            cv::waitKey(1);

        }
    }
    // test single image 
    else if (stat (imagepath, &s) == 0 and s.st_mode & S_IFREG)
    {
        cv::Mat m = cv::imread(imagepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }

        std::vector<KeyPoint> keypoints;
        detect_posenet(posenet, m, keypoints, argv[3], argv[4]);

        // write to json
        json_data::write_json("example.json", imagepath, id, keypoints);
        id++;

        cv::Mat image = draw_pose(m, keypoints);
        cv::imshow("image", image);
        cv::waitKey(0);
    }
    // test img folder
    else if (stat (imagepath, &s) == 0 and s.st_mode & S_IFDIR)
    {
        int img_num = 0;
        std::string save_folder = argv[2];
        std::vector<std::string> fn;
        cv::glob(imagepath,fn,true);
        for (int i=0; i<fn.size(); i++)
        {
            std::cout<<fn[i]<<std::endl;
        }
        std::string json_file;
        json_file = save_folder + "example.json"; 
        std::cout << json_file << "??????????????????????????????????????????????????????????????????????????????????????????????????????" << std::endl;
        for (int i=0; i<fn.size(); i++){
            cv::Mat m = cv::imread(fn[i], 1);

            std::vector<KeyPoint> keypoints;
            detect_posenet(posenet, m, keypoints, argv[3], argv[4]);

            // write to json
            json_data::write_json(json_file, fn[i], id, keypoints);
            id++;

            cv::Mat image = draw_pose(m, keypoints);

            cv::resize(image, image, cv::Size(640, 480));
            cv::imshow("result", image);

            cv::waitKey(0);

            std::ostringstream out; 

            std::string img_extension = ".jpg";
            out << save_folder << img_num << img_extension;
            cv::imwrite(out.str(),image);
            img_num++;
        }
    }
    

    return 0;
}
