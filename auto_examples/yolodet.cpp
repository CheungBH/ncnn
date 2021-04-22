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

int boundary(int n, int lower, int upper)
{
    return (n > upper ? upper : (n < lower ? lower : n));
}

static int detect_yolov4(ncnn::Net& yolov4, const cv::Mat& bgr, std::vector<Object>& objects)
{


    int img_w = bgr.cols;
    int img_h = bgr.rows;
    const int target_size = 416;

    cv::Mat tmp = bgr.clone();
    double resize_ratio;
    cv::Scalar grey_value(128 , 128, 128);
    cv::Mat gray_img(target_size, target_size, CV_8UC3, grey_value);

    if(img_w > img_h)
    {
        resize_ratio = (double)target_size/(double)img_w ;
    }
    else
    {
        resize_ratio = (double)target_size/(double)img_h ;
    }
    double padded_x, padded_y;
    double new_w  = img_w * resize_ratio;
    double new_h = img_h * resize_ratio;
    cv::Size new_sz(new_w,new_h);

    if(img_w > img_h)
    {
        padded_x = 0;
        padded_y = (0.5)*((double)416 - new_h);
    }
    else
    {
        padded_x = (0.5)*((double)416 - new_w);
        padded_y = 0;
    }
    cv::resize(tmp, tmp, new_sz);

    tmp.copyTo(gray_img(cv::Rect(padded_x, padded_y, new_w, new_h)));
    cv::imshow("test", gray_img);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(gray_img.data, ncnn::Mat::PIXEL_BGR, gray_img.cols, gray_img.rows, target_size, target_size);

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

        double xmin = ( values[2]* img_w + (-(double)0.5*((double)target_size - (resize_ratio * img_w) )) ) * ((double)img_w / (double)new_w);
        double ymin = ( values[3]* img_h + (-(double)0.5*((double)target_size - (resize_ratio * img_h) )) ) * ((double)img_h / (double)new_h);
        double xmax = ( values[4]* img_w + (-(double)0.5*((double)target_size - (resize_ratio * img_w) )) ) * ((double)img_w / (double)new_w);
        double ymax = ( values[5]* img_h + (-(double)0.5*((double)target_size - (resize_ratio * img_h) )) ) * ((double)img_h / (double)new_h);
//        double width = xmax - xmin;
//        double height = ymax - ymin;

        double temp[4] = {xmin, ymin, xmax, ymax};
        for (int j = 0; j < 4; j++)
        {
            temp[j] = boundary(temp[j], 0, (j % 2 != 1 ? img_w - 1 : img_h - 1));
        }

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = temp[0];
        object.rect.y = temp[1];
        object.rect.width = temp[2]-temp[0];
        object.rect.height = temp[3]-temp[1];
        if(object.rect.width != 0 && object.rect.height !=0)
        {
            objects.push_back(object);
        }
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
    yolov4.load_param("../../build/auto_examples/model_yolo/model.param");
    yolov4.load_model("../../build/auto_examples/model_yolo/model.bin");
//    yolov4.load_param("model_yolo/model.param");
//    yolov4.load_model("model_yolo/model.bin");
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
//            cv::resize(m,m,cv::Size(416,416));

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
        cv::imshow("res", image);
        cv::waitKey(0);
    }
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