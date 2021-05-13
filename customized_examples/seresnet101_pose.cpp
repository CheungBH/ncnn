#include "net.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <sys/stat.h>
#define YOLOV4_TINY 1 //0 or undef for yolov4

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_yolov4(ncnn::Net& yolov4, const cv::Mat& bgr, std::vector<Object>& objects)
{


    int img_w = bgr.cols;
    int img_h = bgr.rows;
    const int target_size = 416;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov4.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    //     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);
    //printf(values);

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

static cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
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
    cv::waitKey(1);
    return image;
}

int main(int argc, char** argv)
// {
//     if (argc != 2)
//     {
//         fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
//         return -1;
//     }

//     const char* imagepath = argv[1];

//     cv::Mat m = cv::imread(imagepath, 1);
//     if (m.empty())
//     {
//         fprintf(stderr, "cv::imread %s failed\n", imagepath);
//         return -1;
//     }

//     std::vector<Object> objects;
//     detect_yolov4(m, objects);

//     draw_objects(m, objects);

//     return 0;
// }
{
    const char* imagepath = argv[1];
    printf(imagepath ,"\n");
    ncnn::Net yolov4;
    struct stat s;
    yolov4.opt.use_vulkan_compute = true;

#if YOLOV4_TINY
    yolov4.load_param("det_model/yolo.param");
    yolov4.load_model("det_model/yolo.bin");
    const int target_size = 416;
#else
    yolov4.load_param("det_model/yolo.param");
    yolov4.load_model("det_model/yolo.bin");
    const int target_size = 416;
#endif

    // int camera = int(*imagepath) -'0';
    if (imagepath == "0")
    {
        // printf(camera);
        cv::VideoCapture capture(0);
        while(true)
        {
            cv::Mat m;
            capture >> m;
            cv::resize(m,m,cv::Size(416,416));

            std::vector<Object> objects;
            detect_yolov4(yolov4,m, objects);
            cv::Mat image = draw_objects(m, objects);
        }
    }

    else if (stat (imagepath, &s) == 0 and s.st_mode & S_IFREG)
    {
        printf("img: ", "\n");
        cv::Mat m = cv::imread(imagepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }

            std::vector<Object> objects;
            detect_yolov4(yolov4,m, objects);
            cv::Mat image = draw_objects(m, objects);
    }
    else if (stat (imagepath, &s) == 0 and s.st_mode & S_IFDIR)
    {
        int img_num = 0;
        std::string save_folder = argv[2];
        std::vector<std::string> fn;
//        cv::glob(imagepath,fn,true);
        for (int i=0; i<fn.size(); i++)
        {
            std::cout<<fn[i]<<std::endl;
        }
        for (int i=0; i<fn.size(); i++){
        cv::Mat m = cv::imread(fn[i], 1);
        std::vector<Object> objects;
        detect_yolov4(yolov4,m, objects);
        cv::Mat image = draw_objects(m, objects);
        std::ostringstream out; 

        std::string img_extension = ".jpg";
        out << save_folder << img_num << img_extension;
        cv::imwrite(out.str(),image);
        img_num++;

    }
}

    return 0;
}