#ifndef HUMANPROCESSOR_H
#define HUMANPROCESSOR_H

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <math.h> 
#include "Person.h"
#include "TrackingBox.h"


class HumanProcessor {
public:
	std::vector<int> stored_id;
	std::vector<int> curr_id;
	std::vector<int> RD_id;
	std::vector<int> warning_id;
	std::vector<int> untracked_id;
	std::map<int, Person> PEOPLE;

	void clear();

	void update(std::vector<TrackingBox> tracked_boxes);

	HumanProcessor();

};
#endif