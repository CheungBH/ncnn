#ifndef SRBOX_H
#define SRBOX_H

#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

class SRBox {
  public:
      std::pair<double, double> location;
	  double height, width;

	  std::pair<double, double> center;
	  double ymin, ymax , xmin , xmax;
      int exist;
      int disappear;
      int disappear_max = 20;
      int exist_max = 150;
      int alarm_cnt = 100;  

	  void clear();
	  void update(int flag);
	  void update_exist(int flag);
      void update_disappear();

	  void visualize_srbox(cv::Mat image, cv::Scalar color);

	  bool if_warning();

	  void cnt_color();
      SRBox(std::pair<double, double> loc, double h, double w);

};
#endif