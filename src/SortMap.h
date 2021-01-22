#include <bits/stdc++.h> 
#include <opencv2/opencv.hpp>

using namespace std; 
  
// Comparator function to sort pairs 
// according to second value 


bool compare_by_key(std::pair<int, cv::Mat>& a, std::pair<int, cv::Mat>& b)
{
    return a.first < b.first;
}

// Function to sort the map according 
// to value in a (key-value) pairs 

void sort_by_key(std::map<int, cv::Mat>& M) 
{ 
  
    // Declare vector of pairs 
    std::vector<std::pair<int, cv::Mat> > A; 
  
    // Copy key-value pair from Map 
    // to vector of pairs 
    for (auto& it : M) { 
        A.push_back(it); 
    } 
  
    // Sort using comparator function 
    std::sort(A.begin(), A.end(), compare_by_key); 
  
    // Print the sorted value 
    // for (auto& it : A) { 
  
    //     cout << it.first << ' '
    //          << it.second << endl; 
    // } 
} 