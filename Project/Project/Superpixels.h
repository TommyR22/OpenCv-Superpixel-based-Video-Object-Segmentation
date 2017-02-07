//
//  FgModeling.h
//  Project
//
//  Created by Tommaso Ruscica on 15/09/15.
//  Copyright (c) 2015 Tommaso Ruscica. All rights reserved.
//

#ifndef __Project__FgModeling__
#define __Project__FgModeling__

#include <stdio.h>
#include <opencv2/opencv.hpp>
extern "C" {
#include "/Users/TommyR22/Desktop/vlfeat/vl/generic.h"
#include "/Users/TommyR22/Desktop/vlfeat/vl/slic.h"
}

#endif /* defined(__Project__FgModeling__) */


cv::Mat segmentVLFeat (cv::Mat image);    //create superpixels
std::vector<int> getMotionSuperpixels(cv::Mat image_superpixels, cv::Mat image_superpixels2, std::vector<std::pair<int,int>> vect_pair, double number_superpixels);
//double getNumberSuperpixel (cv::Mat image_superpixels); //get number of superpixels
std::vector<std::pair<int,int>> overlappingSuperpixel (cv::Mat image_superpixels,cv::Mat image_superpixels2); //vector of overlap superpixels between frame t and t+1
cv::Mat masking (cv::Mat image, int label);  //mask image for each superpixel
cv::Mat masking_motionSuperpixels(cv::Mat image_superpixels,std::vector<int> motionSuperpixels);
cv::Mat segment( cv::Mat image, int &numberOfLabels, cv::Mat &imageSegmented);
