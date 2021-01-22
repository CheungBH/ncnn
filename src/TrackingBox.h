#ifndef TRACKINGBOX_H
#define TRACKINGBOX_H
#include <opencv2/opencv.hpp>

struct TrackingBox {
	int frame;
	int id;
	cv::Rect_<float> box;
};

#endif