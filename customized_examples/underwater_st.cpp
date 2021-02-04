// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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
#include "Person.h"
#include "KalmanTracker.h"
#include "RegionProcessor.h"
#include "Hungarian.h"
#include "HumanProcessor.h"
#include "utils.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if CV_MAJOR_VERSION >= 33
#include <opencv2/videoio/videoio.hpp>
#endif

#include <vector>

#include <stdio.h>

#define NCNN_PROFILING
#define YOLOV4_TINY //Using yolov4_tiny, if undef, using original yolov4

#ifdef NCNN_PROFILING
#include "benchmark.h"
#endif

#define SPPE_PARAM "underwater_model/alphapose.param"
#define SPPE_MODEL "underwater_model/alphapose.bin"

#define CNN_PARAM "underwater_model/resnet18.param"
#define CNN_MODEL "underwater_model/resnet18.bin"

int sz_boxes = 10;
int sz_skeletons = 30;
int sz_cnt = 10;
int sz_predictions = 5;



struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct KP
{
    cv::Point2f p;
    float prob;
};
static int init_yolov4(ncnn::Net* yolov4, int* target_size)
{
    /* --> Set the params you need for the ncnn inference <-- */

    yolov4->opt.num_threads = 4; //You need to compile with libgomp for multi thread support

    yolov4->opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support

    yolov4->opt.use_winograd_convolution = true;
    yolov4->opt.use_sgemm_convolution = true;
    yolov4->opt.use_fp16_packed = true;
    yolov4->opt.use_fp16_storage = true;
    yolov4->opt.use_fp16_arithmetic = true;
    yolov4->opt.use_packing_layout = true;
    yolov4->opt.use_shader_pack8 = false;
    yolov4->opt.use_image_storage = false;

    /* --> End of setting params <-- */
    int ret = 0;

    // original pretrained model from https://github.com/AlexeyAB/darknet
    // the ncnn model https://drive.google.com/drive/folders/1YzILvh0SKQPS_lrb33dmGNq7aVTKPWS0?usp=sharing
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
#ifdef YOLOV4_TINY
    const char* yolov4_param = "underwater_model/yolov4-tiny-opt.param";
    const char* yolov4_model = "underwater_model/yolov4-tiny-opt.bin";
    *target_size = 416;
#else
    const char* yolov4_param = "underwater_model/yolov4-opt.param";
    const char* yolov4_model = "underwater_model/yolov4-opt.bin";
    *target_size = 608;
#endif

    ret = yolov4->load_param(yolov4_param);
    if (ret != 0)
    {
        return ret;
    }

    ret = yolov4->load_model(yolov4_model);
    if (ret != 0)
    {
        return ret;
    }

    return 0;
}

static int detect_yolov4(const cv::Mat& bgr, std::vector<Object>& objects, int target_size, ncnn::Net* yolov4)
{
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov4->create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static int draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, int is_streaming)
{
    static const char* class_names[] = {"background", "person", "bicycle",
                                        "car", "motorbike", "aeroplane", "bus", "train", "truck",
                                        "boat", "traffic light", "fire hydrant", "stop sign",
                                        "parking meter", "bench", "bird", "cat", "dog", "horse",
                                        "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase",
                                        "frisbee", "skis", "snowboard", "sports ball", "kite",
                                        "baseball bat", "baseball glove", "skateboard", "surfboard",
                                        "tennis racket", "bottle", "wine glass", "cup", "fork",
                                        "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                                        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                                        "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
                                        "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                                        "cell phone", "microwave", "oven", "toaster", "sink",
                                        "refrigerator", "book", "clock", "vase", "scissors",
                                        "teddy bear", "hair drier", "toothbrush"
                                       };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);

    if (is_streaming)
    {
        cv::waitKey(1);
    }
    else
    {
        cv::waitKey(0);
    }

    return 0;
}


void cropImageFrom(std::vector<cv::Mat> &target, const cv::Mat &src, const std::vector<Object> &obj)
{
    target.clear();
    printf("Crop Image...\n");

    //...
    for(auto itr = obj.begin(); itr != obj.end(); itr++) 
    {
        target.push_back(src(itr->rect).clone());
    }

    return;
}

std::vector<KP> sppeOne(const cv::Mat &src)
{
    std::vector<KP> target;
    printf("sppeOne...\n");
    static ncnn::Net sppeNet;
    static bool is_loaded_sppe = false;

    if(!is_loaded_sppe)
    {
        sppeNet.opt.use_vulkan_compute = true;

        sppeNet.load_param(SPPE_PARAM);
        sppeNet.load_model(SPPE_MODEL);
        is_loaded_sppe = true;
    }
    int w = src.cols, h = src.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 256, 320);

    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = sppeNet.create_extractor();

    ex.input("input.1", in);

    ncnn::Mat out;
    ex.extract("1123", out);

    for (int p = 0; p < out.c; p++)
    {
        const ncnn::Mat m = out.channel(p);

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

        KP keypoint;
        keypoint.p = cv::Point2f(max_x * w / (float)out.w, max_y * h / (float)out.h);
        keypoint.prob = max_prob;

        target.push_back(keypoint);
    }

    return target;
}

static void draw_pose(const cv::Mat& bgr, const std::vector<KP>& keypoints, int is_streaming)
{
    cv::Mat image = bgr.clone();

    // draw bone
    static const int joint_pairs[16][2] = {
        {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
    };

    for (int i = 0; i < 16; i++)
    {
        const KP& p1 = keypoints[joint_pairs[i][0]];
        const KP& p2 = keypoints[joint_pairs[i][1]];

        if (p1.prob < 0.04f || p2.prob < 0.04f)
            continue;

        cv::line(image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
    }

    // draw joint
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const KP& keypoint = keypoints[i];

        fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

        if (keypoint.prob < 0.2f)
            continue;

        cv::circle(image, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("pose", image);
    if(is_streaming)
    {
        cv::waitKey(1);
    }
    else
    {
        cv::waitKey(0);
    }
}

std::vector<float> cnn(const cv::Mat &src)
{
    printf("cnn...\n");
    std::vector<float> target;
    
    static ncnn::Net cnnNet;
    static bool is_loaded_cnn = false;
    if(!is_loaded_cnn)
    {
        cnnNet.opt.use_vulkan_compute = true;

        cnnNet.load_param(CNN_PARAM);
        cnnNet.load_model(CNN_MODEL);
        is_loaded_cnn = true;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_BGR, src.cols, src.rows, 224, 224);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = cnnNet.create_extractor();

    ex.input("input.1", in);

    ncnn::Mat out;
    ex.extract("191", out);

    {
        ncnn::Layer* softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        softmax->load_param(pd);

        softmax->forward_inplace(out, cnnNet.opt);

        delete softmax;
    }

    out = out.reshape(out.w * out.h * out.c);

    for(int i = 0; i < out.w; i++)
    {
        target.push_back(out[i]);
    }

    return target;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    cv::Mat frame;
    std::vector<Object> objects;

    cv::VideoCapture cap;

    ncnn::Net yolov4;

    const char* devicepath;

    int target_size = 0;
    int is_streaming = 0;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [v4l input device or image]\n", argv[0]);
        return -1;
    }

    devicepath = argv[1];

#ifdef NCNN_PROFILING
    double t_load_start = ncnn::get_current_time();
#endif

    int ret = init_yolov4(&yolov4, &target_size); //We load model and param first!
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load model or param, error %d", ret);
        return -1;
    }

#ifdef NCNN_PROFILING
    double t_load_end = ncnn::get_current_time();
    fprintf(stdout, "NCNN Init time %.02lfms\n", t_load_end - t_load_start);
#endif

    if (strstr(devicepath, "/dev/video") == NULL)
    {
        frame = cv::imread(argv[1], 1);
        if (frame.empty())
        {
            fprintf(stderr, "Failed to read image %s.\n", argv[1]);
            return -1;
        }
    }
    else
    {
        cap.open(devicepath);

        if (!cap.isOpened())
        {
            fprintf(stderr, "Failed to open %s", devicepath);
            return -1;
        }

        cap >> frame;

        if (frame.empty())
        {
            fprintf(stderr, "Failed to read from device %s.\n", devicepath);
            return -1;
        }

        is_streaming = 1;
    }

    std::vector<cv::Mat> imgs;
    std::vector<std::vector<KP>> skeletons;
    std::vector<std::vector<float>> predictions;

    List ls;

    while (1)
    {
        if (is_streaming)
        {
#ifdef NCNN_PROFILING
            double t_capture_start = ncnn::get_current_time();
#endif

            cap >> frame;

#ifdef NCNN_PROFILING
            double t_capture_end = ncnn::get_current_time();
            fprintf(stdout, "NCNN OpenCV capture time %.02lfms\n", t_capture_end - t_capture_start);
#endif
            if (frame.empty())
            {
                fprintf(stderr, "OpenCV Failed to Capture from device %s\n", devicepath);
                return -1;
            }
        }

// #ifdef NCNN_PROFILING
//         double t_detect_start = ncnn::get_current_time();
// #endif

        detect_yolov4(frame, objects, target_size, &yolov4); //Create an extractor and run detection

// #ifdef NCNN_PROFILING
//         double t_detect_end = ncnn::get_current_time();
//         fprintf(stdout, "NCNN detection time %.02lfms\n", t_detect_end - t_detect_start);
// #endif

// #ifdef NCNN_PROFILING
//         double t_draw_start = ncnn::get_current_time();
// #endif

        draw_objects(frame, objects, is_streaming); //Draw detection results on opencv image

// #ifdef NCNN_PROFILING
//         double t_draw_end = ncnn::get_current_time();
//         fprintf(stdout, "NCNN OpenCV draw result time %.02lfms\n", t_draw_end - t_draw_start);
// #endif
        //kalman filter...

        //update list...
        //ls.update(/*Result from kalman tracker*/tb_list);

        //do sppe if needed...

        //update sppe result...
        //convert std::vector<KP(KeyPoint)> to std::vector<skeleton> first
        //ls.updateAllSkeletons(/*sppe result*/kpts);
        //do cnn if needed...

        //update cnn result...
        //ls.updateAllPredictions(/*cnn result*/preds);
        skeletons.clear();
        predictions.clear();

        cropImageFrom(imgs, frame, objects);
        for(auto itr = imgs.begin(); itr != imgs.end(); itr++)
        {
            skeletons.push_back(sppeOne(*itr));
            predictions.push_back(cnn(*itr));
            draw_pose(*itr, skeletons[itr-imgs.begin()], is_streaming);
            print_topk(predictions[itr-imgs.begin()], 2);
        }


        if (!is_streaming)
        {   //If it is a still image, exit!
            return 0;
        }
    }

    return 0;
}
