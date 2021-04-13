#include "json.hpp"
#include <fstream>
#include <iomanip>

using json = nlohmann::json;

void json_data::write_json(const std::string& filename, const std::string& image_path, const int& id, const std::vector<KeyPoint>& keypoints) {
	std::string image_name = image_path.substr(image_path.find_last_of("/") + 1, image_path.size());

	// read a JSON file
	std::ifstream i(filename);
	json images;

	json image = {
		{"id", id},
		{"image_name", image_name},
		{"keypoints", {
			keypoints[0].p.x,
			keypoints[0].p.y,
			keypoints[5].p.x,
			keypoints[5].p.y,
			keypoints[3].p.x,
			keypoints[3].p.y,
			keypoints[1].p.x,
			keypoints[1].p.y,
			keypoints[2].p.x,
			keypoints[2].p.y,
			keypoints[4].p.x,
			keypoints[4].p.y,
			keypoints[6].p.x,
			keypoints[6].p.y,
			keypoints[7].p.x,
			keypoints[7].p.y,
			keypoints[8].p.x,
			keypoints[8].p.y,
			keypoints[9].p.x,
			keypoints[9].p.y,
			keypoints[11].p.x,
			keypoints[11].p.y,
			keypoints[10].p.x,
			keypoints[10].p.y,
			keypoints[12].p.x,
			keypoints[12].p.y,
		}}
	};

	// check file empty
	if (i.peek() == std::ifstream::traits_type::eof()){
		images = {
			{"images", {image}}
		};
	}
	else{
		i >> images;
		// set JSON format
		images["images"].push_back(image);
	}

	i.close();
	// write JSON to file
	std::ofstream o(filename, std::ofstream::trunc);
	o << std::setw(4) << images << std::endl;
	o.close();
}