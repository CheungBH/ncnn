#ifndef REGIONPROCESSOR_H
#define REGIONPROCESSOR_H

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <math.h> 
#include "TrackingBox.h"
#include "SRBox.h"

class RegionProcessor {
public:
	double height, width;
	double height_num, width_num;
	int region_cnt;
	double h_interval, w_interval; // the h and w of each SRBox
	std::vector<std::pair<double, double>> region_idx;
	std::map<std::pair<double, double>, SRBox> REGIONS;
	bool if_write;
	//AreaVisualizer Visualize;

	//Storing the unit location of all the cases (x,y) = (0,0) to (10,10)
	std::vector< std::pair<double, double> > keep_case;
	std::vector< std::pair<double, double> > update_case;
	std::vector< std::pair<double, double> > empty_case;
	std::vector< std::pair<double, double> > alarm_case;

	void clear();
	//find the location of a point , point coord unit is in pixel
	std::pair<double, double> locate(std::pair<double, double> pt);
	//Input the top-left and bottom-right coords to return a vector of all the location that 
	//is covered by the bounding box
	std::vector< std::pair<double, double> > locate_cover(std::pair<double, double> pt_tl, std::pair<double, double> pt_br);
	cv::Mat draw_cnt_map(cv::Mat img);
	std::vector<std::vector<std::pair<double, double>>> box2region(std::vector<std::pair<double, double>> centers, std::vector<std::pair<double, double>> covers, std::vector<std::pair<double, double>> occupies);
	std::vector<std::vector<double>> rect2vec(std::vector<cv::Rect> boxes);
	std::vector<std::vector<std::pair<double, double>>> get_origin_operator(std::vector<cv::Rect> rect_boxes);
	//??
	std::vector< std::pair<double, double> > locate_occupy(std::pair<double, double> pt_tl, std::pair<double, double> pt_br);
	std::vector<std::pair<double, double>> trigger_alarm();


	//Input a vector of box array and get a vector of pair of 
	std::vector< std::pair<double, double> > center_region(std::vector<std::vector<double>> boxes);
	std::vector< std::pair<double, double> > cover_region(std::vector<std::vector<double>> boxes);

	std::vector<std::pair<double, double>> occupy_region(std::vector<std::vector<double>> boxes);

	void update_region(std::vector<std::vector<std::pair<double, double>>> region_ls);
	RegionProcessor(double w, double h, double w_num, double h_num);
	std::vector<int> get_alarming_id(std::vector<TrackingBox> tracked_boxes);
	std::vector<int> region_range(cv::Rect box);
};
#endif