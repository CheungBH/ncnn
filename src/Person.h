#ifndef PERSON_H
#define PERSON_H
#include <iostream>
#include <vector>
#include <map>
#include "TrackingBox.h"
/*
functions in Person class:
-update functions for updating person's states (boxes, skeletons, predictions)	* All update functions return void
	-update_boxes(trackingbox)
	-update_skeletons(float[34]), update_skeletons(skeleton)    *choose the one you like to use
	-update_predictions(char)

-get functions returning DEEP COPY of person's states (boxes, skeletons, predictions)
	-get_boxes()		  *return std::vector<cv::Rect>
	-get_skeletons()	  *return std::vector<skeleton>
	-get_predictions()	  *return std::vector<char>
	-get_current_box()    *return cv::Rect (The newest cv::Rect in boxes)
	-get_current_pos()	  *return point (A pair(x,y) representing x & y coordinates of the center of the newest cv::Rect)

-functions returning values(int) for tracking person's states (boxes, skeletons, predictions)
	-num_boxes()		*return number of boxes currently stored
	-num_skeletons()	*return number of skeletons currently stored
	-num_predictions()	*return number of predictions currently stored
	-age()				*return current value of cnt

-funtions returning values(bool) for tracking perons's states (boxes, skeletons, predictions)
	-should_warn()		*return true if 8 or more boxes have their height/width > 3 in boxes
	-enough_kpts()		*return true if number of skeletons stored reaches maximum

-functions being used for updating std::vector<Person> (Not object oriented, not recommended)
	-set_found(bool)	*set found = boolean value passed
	-get_found()		*return found

function in List class:
-update functions for updating list's states (list, warning_list, list_list)	* All update functions return void
	-updateList(std::vector<trackingbox>)
	-updateWarningList(std::vector<int>)
	-updateLSTMList()		*use this function after updateWarningList, otherwise the result is not the most updated (result become  t-1'th frame's result instead)

-print functions for showing list's states (list, warning_list, lstm_list)		* All print functions return void
	-printList()
	-printWarningList()
	-printLSTMList()

-itr functions for getting itr to access list's states (list, warning_list, lstm_list)
	-itrListBegin()				*return list.begin()
	-itrListEnd()				*return list.end()
	-itrWarningListBegin()		*return warning_list.begin()
	-itrWarningListEnd()		*return_warning_list.end()
	-getPersonItrById(int)		*return itr of person which id=value passed. Return list.end() if item is not found. Slower
	-getPersonItrByPos(int)		*return the i'th element in the list. Return list.end() if i > length of the list. Faster 

-get functions for getting DEEP COPY of list's states (list, warning_list)	*not recommended, use itr functions instead
	-getWarningList(bool)		*return std::vector<int> storing position of warned persons in the person list if true. Else simply return their ids.
	-getWarningList()			*return std::map<int,int> storing (id, pos in the list) pair for each item

-opencv functions
	-cropImg(cv::Mat)	*return 3D cv::Mat ({length of list, 256, 320}) which stored cropped image of every person from image in order.
	-cropImg(int,cv::Mat) *return cv::Mat which is the cropped image of i'th item from image
*/


#include <opencv2/opencv.hpp>

extern int sz_boxes;
extern int sz_skeletons;
extern int sz_cnt;
extern int sz_predictions;


struct trackingbox {
	int frame;
	int id;
	cv::Rect bbox;
};
struct point {
	float x, y;
	point(float x, float y) : x(x), y(y) {}
};
struct skeleton {
	float pts[34];
	skeleton(float pts[34]) {
		for (int i = 0; i < 34; i++)
		{
			this->pts[i] = pts[i];
		}
	}
};

class Person {
public:
	Person(const TrackingBox& tb) : id(tb.id) //constrcutor
	{
		if (sz_skeletons <= 0 || sz_boxes <= 0)
			throw "Exception: sz_skeletons / sz_boxes should be greater than 0!";
		this->cnt = 0;
		this->found = true;
		this->bboxes.push_back(tb.box);
	}
	int get_id() const { return this->id; }

	void update_boxes(const TrackingBox& tb) // update current position
	{
		if (this->id != tb.id)
			throw "Exception: Not the same person!";
		this->bboxes.push_back(tb.box);
		this->found = true;
#ifdef LOG
		//std::cout << "Person " << this->id << ": Box stored.\nNumber of boxes stored: " << this->bboxes.size() << std::endl;
#endif
		if ((int)bboxes.size() > sz_boxes) {
#ifdef LOG
			//std::cout << "Number of boxes stored exceeds" << sz_boxes << ": Remove the eldest." << std::endl;
#endif
			this->bboxes.erase(bboxes.begin());
	}
		this->cnt = 0;
}
	std::vector<cv::Rect> get_boxes() const { return this->bboxes; }
	cv::Rect get_current_box() const { return this->bboxes.back(); }
	int num_boxes() const { return this->bboxes.size(); }

	void update_skeletons(const skeleton& data) // pass in skeleton sample to store in Person::skeletons
	{
		//skeleton data(sample);
		this->skeletons.push_back(data);
#ifdef LOG
		//std::cout << "Person " << this->id << ": Skeleton stored.\nNumber of skeletons stored: " << this->skeletons.size() << std::endl;
#endif
		if ((int)this->skeletons.size() > sz_skeletons)
		{
#ifdef LOG
			//std::cout << "Number of skeletons stored exceeds " << sz_skeletons << ": Remove the eldest." << std::endl;
#endif
			this->skeletons.erase(skeletons.begin());
	}
	}
	void update_skeletons(float sample[34])
	{
		skeleton data(sample);
		update_skeletons(data);
	}
	std::vector<skeleton> get_skeletons() const { return skeletons; }
	int num_skeletons() const { return this->skeletons.size(); }

	void update_predictions(const char prediction)
	{
		this->predictions.push_back(prediction);
#ifdef LOG
		//std::cout << "Person " << this->id << ": Prediction stored.\nNumber of predictions stored: " << this->predictions.size() << std::endl;
#endif
		if ((int)predictions.size() > sz_predictions)
		{
#ifdef LOG
			//std::cout << "Number of predictions stored exceeds " << sz_predictions << ": Remove the eldest." << std::endl;
#endif
			this->predictions.erase(this->predictions.begin());
		}
	}
	std::vector<char> get_predictions() const { return this->predictions; }
	int num_predictions() const { return this->predictions.size(); }

	int age() const { return this->cnt; }

	bool should_warn() const
	{
		if (this->bboxes.empty())
			throw "Exception: Position data is empty!";
		int temp = 0;
		for (auto itr = this->bboxes.begin(); itr != this->bboxes.end(); itr++)
		{
			float woh = (float)itr->width / (float)itr->height;
			if (woh >= /*3*/0)
				temp++;
			if (temp == 8)
				return true;
		}
		return false;
	}
	point get_current_pos() const
	{
		float x = (float)this->bboxes.back().x + (float)this->bboxes.back().width / 2.0f;
		float y = (float)this->bboxes.back().y + (float)this->bboxes.back().height / 2.0f;
		return point(x, y);
	}
	bool enough_kpts() const
	{
		if ((int)skeletons.size() == sz_skeletons)
			return true;
		return false;
	}
	void set_found(bool flag) { this->found = flag; }
	bool is_found() const { return this->found; }
	void incr_cnt()
	{
		if (this->cnt < sz_cnt)
			cnt++;
	}
	bool cnt_reach_max() const { return this->cnt == sz_cnt; }
	~Person() //destructor
	{}
private:
	int id;
	std::vector<cv::Rect> bboxes;
	std::vector<skeleton> skeletons;
	std::vector<char> predictions;
	int cnt;
	bool found;
};

void tracking2person(std::vector<TrackingBox>& track, std::vector<Person>& list)
{
	if (track.empty())
		return;
	bool found = false;
	for (auto itr = list.begin(); itr != list.end(); itr++)
	{
		if (track[0].id == itr->get_id())
		{
#ifdef LOG
			//std::cout << "Person found. Update..." << std::endl;
#endif // LOG

			itr->update_boxes(track[0]);
			found = true;
			break;
		}
	}
	if (!found)
	{
#ifdef LOG
		//std::cout << "Person not found. Add to the list..." << std::endl;
#endif // LOG
		list.push_back(Person(track[0]));
	}
	track.erase(track.begin());
	tracking2person(track, list);
}
void remove_ppl(std::vector<Person>& list)
{
	auto itr = list.begin();
	while (itr != list.end())
	{
		if (itr->cnt_reach_max())
		{
			auto temp = itr + 1;
			list.erase(itr);
			itr = temp;
		}
		else {
			itr++;
		}
	}
}
void update(std::vector<TrackingBox>& track, std::vector<Person>& list)
{
	//Step 1. Set flags of all people to false.
	for (auto itr = list.begin(); itr != list.end(); itr++)
	{
		itr->set_found(false);
	}
	//Step 2. Update the list of people according to tracking boxes provided.
	tracking2person(track, list);
	//All updated/new people's flag is true.
	//Step 3. Increment the cnt of whose flag remains false
	for (auto itr = list.begin(); itr != list.end(); itr++)
	{
		if (!itr->is_found())
		{
			itr->incr_cnt();
		}
	}
	//Step 4. Remove those who cnt reaches maximum.
	remove_ppl(list);
}
std::vector<int> get_warning_ids(std::vector<Person>& list)
{
	std::vector<int> temp;
	for (auto itr = list.begin(); itr != list.end(); itr++)
	{
		if (itr->should_warn())
			temp.push_back(itr->get_id());
	}
	return temp;
}
std::vector<Person>::iterator get_person_itr(std::vector<Person>& list, const int& id)// return person itr if found; null else
{
	for (auto itr = list.begin(); itr != list.end(); itr++)
	{
		if (itr->get_id() == id)
			return itr;
	}
	return list.end();
}
int n_space(int field, int value)
{
	while (value >= 10)
	{
		value /= 10;
		field--;
	}
	return field - 1;
}
std::string spacing(int n_space)
{
	std::string space = "";
	for (int i = 0; i < n_space; i++)
	{
		space += " ";
	}
	return space;
}
class List {
public:
	List() {
		list = {};
	}
	std::vector<Person>::iterator getPersonItrById(const int& id) // return person iterator if found, else null iterator is returned
	{
		if (list.empty())
			return list.end();
		auto itr = list.begin();
		while (itr != list.end())
		{
			if (itr->get_id() == id)
				return itr;
			itr++;
		}
		return itr;
	}
	std::vector<Person>::iterator getPersonItrByPos(const int& pos) 
	{
		if (pos >= list.end() - list.begin())
			return list.end();
		return list.begin() + pos;
	}
	std::vector<Person>::iterator itrListBegin() { return list.begin(); }
	std::vector<Person>::iterator itrListEnd() { return list.end(); }
	std::map<int, int>::iterator itrWarningListBegin() { return warning_list.begin(); }
	std::map<int, int>::iterator itrWarningListEnd() { return warning_list.end(); }
	std::map<int, int>::iterator itrLSTMListBegin() { return lstm_list.begin(); }
	std::map<int, int>::iterator itrLSTMListEnd() { return lstm_list.end(); }
	void updateList(std::vector<TrackingBox>& tb_list)
	{
		if (tb_list.empty())
			return;
		auto itr = list.begin();
		while (itr != list.end())
		{
			auto tbitr = tb_list.begin();
			while (tbitr != tb_list.end())
			{
				if (tbitr->id == itr->get_id())
					break;
				tbitr++;
			}

			if (tbitr == tb_list.end())
			{
				itr->incr_cnt();
				if (itr->cnt_reach_max())
				{
					int pos = itr - list.begin();
					list.erase(itr);
					itr = list.begin() + pos;
					continue;
				}
			}
			else
			{
				TrackingBox tb = *tbitr;
				itr->update_boxes(tb);
				tb_list.erase(tbitr);
			}
			itr++;
		}
		for (auto itr2 = tb_list.begin(); itr2 != tb_list.end(); itr2++)
		{
			TrackingBox tb = *itr2;
			list.push_back(Person(tb));
		}
		tb_list.clear();
	}
	void printList() const
	{
		std::cout << "Person list\n"
			<< "Number of items: " << list.size()
			<< "\nID  box    ske    pre    cnt\n";
		for (auto itr = list.begin(); itr != list.end(); itr++)
		{
			
			std::cout << itr->get_id() << spacing(n_space(4, itr->get_id()))
					  << itr->num_boxes() << "/" << sz_boxes << spacing(n_space(3, itr->num_boxes()) + n_space(3, sz_boxes))
					  << itr->num_skeletons() << "/" << sz_skeletons << spacing(n_space(3, itr->num_skeletons()) + n_space(3, sz_skeletons))
					  << itr->num_predictions() << "/" << sz_predictions << spacing(n_space(3, itr->num_predictions()) + n_space(3, sz_predictions))
					  << itr->age() << "/" << sz_cnt << std::endl;
					 
		}
		std::cout << "------------------------" << std::endl;
	}
	void updateWarningList(const std::vector<int>& ids)
	{
		warning_list.clear();
		for (auto itr = ids.begin(); itr != ids.end(); itr++)
		{
			auto itr_person = getPersonItrById(*itr);
			if (itr_person != list.end())
			{
				if (itr_person->should_warn())
				{
					warning_list[itr_person->get_id()] = itr_person - list.begin();
				}
			}
		}
	}
	void printWarningList() const
	{
		std::cout << "Warning list:\n"
			<< "Number of items: " << warning_list.size()
			<< "\nID      Position\n";
		for (auto itr = warning_list.begin(); itr != warning_list.end(); itr++)
		{
			std::cout << itr->first << spacing(n_space(8,itr->first)) << itr->second << std::endl;
		}
		std::cout << "------------------------" << std::endl;
	}
	std::map<int, int> getWarningList() const { return warning_list; }

	std::vector<int> getWarningList(bool usePosition) const
	{
		std::vector<int> temp = {};
		for (auto itr = warning_list.begin(); itr != warning_list.end(); itr++)
			temp.push_back((usePosition ? itr->second : itr->first));
		return temp;
	}
	void updateLSTMList()
	{
		lstm_list.clear();
		for (auto itr = warning_list.begin(); itr != warning_list.end(); itr++)
		{
			auto itr_person = getPersonItrByPos(itr->second);
			if (itr_person != list.end())
			{
				if (itr_person->enough_kpts())
					lstm_list[itr->first] = itr->second;
			}
		}
	}
	int numWarnedPerson()const { return warning_list.size(); }
	void printLSTMList() const
	{
		//std::cout << "LSTM list:\n"
			//<< "Number of items: " << lstm_list.size()
			//<< "\nID      Position\n";
		for (auto itr = lstm_list.begin(); itr != lstm_list.end(); itr++)
		{
			//std::cout << itr->first << spacing(n_space(8, itr->first)) << itr->second << std::endl;
		}
		//std::cout << "------------------------" << std::endl;
	}
	void cropImg(std::vector<cv::Mat> &target, cv::Mat& img) const
	{
		target.clear();
		for (auto itr = warning_list.begin(); itr != warning_list.end(); itr++)
		{
			cv::Mat temp = img(list[itr->second].get_current_box()).clone();
			cv::resize(temp, temp, cv::Size(256,320));
			target.push_back(temp);
		}
	}
	cv::Mat &cropImg(int pos, cv::Mat& img) const
	{
		int i = 0;
		int index = -1;
		for (auto itr = warning_list.begin(); itr != warning_list.end(); itr++)
		{
			if (i == pos)
			{
				index = itr->second;
				break;
			}
			i++;
		}
		img = img(list[index].get_current_box());
		cv::resize(img, img, cv::Size(256, 320));
		return img;
	}
	void updateSkeletons(int pos, skeleton kpt)
	{
		if (pos >= list.end() - list.begin())
			throw "Exception:Postion Out of Range!";
		list[pos].update_skeletons(kpt);
	}
	void updatePredictions(int pos, char prediction)
	{
		if (pos >= list.end() - list.begin())
			throw "Exception:Postion Out of Range!";
			list[pos].update_predictions(prediction);
	}
	void updateAllSkeletons(const std::vector<skeleton> &kpts)
	{
		if (kpts.size() != warning_list.size())
			throw "Exception: Kpts length mismatched!";
		int index = 0;
		for (auto itr = warning_list.begin(); itr != warning_list.end(); itr++)
		{
			list[itr->second].update_skeletons(kpts[index++]);
		}
	}
	void updateAllPredictions(const std::vector<char> &predicts)
	{
		if (predicts.size() != lstm_list.size())
			throw "Exception: Kpts length mismatched!";
		int index = 0;
		for (auto itr = lstm_list.begin(); itr != lstm_list.end(); itr++)
		{
			list[itr->second].update_predictions(predicts[index++]);
		}
	}
private:
	std::vector<Person> list;
	std::map<int, int> warning_list; // first: id; second: position relative to list.begin(); (for acessing person in faster way)
	std::map<int, int> lstm_list;	// first: id; second: position relative to list.begin(); (for acessing person in faster way)
};
#endif

