//
//  Utility.hpp
//  Project
//
//  Created by Tommaso Ruscica on 21/01/16.
//  Copyright © 2016 Tommaso Ruscica. All rights reserved.
//

#ifndef Utility_h
#define Utility_h

#include <stdio.h>
#include <opencv2/opencv.hpp>

#endif /* Utility_h */

cv::Ptr<cv::ml::EM> ExpectationMaximization (cv::Ptr<cv::ml::EM> em, cv::Mat trainData , int nClusters, int currentCluster,cv::String type);
cv::Ptr<cv::ml::EM> ExpectationMaximization_update (cv::Ptr<cv::ml::EM> em, cv::Mat trainData , std::vector<cv::Mat> nClusters, int currentCluster,cv::String type);
std::vector<std::vector<float>> DiscretizePdf(cv::Mat sample, cv::Mat probs, cv::Ptr<cv::ml::EM> em, std::vector<float> discrete_pdf,std::vector<std::vector<float>> vector_discrete, int i);
std::vector<int> filteredMovingSuperpixelsMap(cv::Mat currSuperpixelsMap, std::vector<int> motionSuperpixels);
std::vector<std::vector<cv::Point>> filterBackground(cv::Mat image, std::vector<std::vector<cv::Point>> contours_mask,std::vector<cv::Ptr<cv::ml::EM>> BGmodels,std::vector<cv::Ptr<cv::ml::EM>> FGmodels, int p, int nClusters);
std::vector<std::vector<cv::Vec3b>> getPixelsFromMask (cv::Mat image, std::vector<std::vector<cv::Point>> contours_mask);
double MinBG(std::vector<float> vector_discrete,std::vector<cv::Ptr<cv::ml::EM>> BGmodels);
double MinFG(std::vector<float> vector_discrete,std::vector<cv::Ptr<cv::ml::EM>> FGmodels);
std::vector<cv::Mat> createWindows(cv::Mat image, std::vector<std::vector<cv::Point>> const& contours_mask, int winInc);
std::vector<std::pair<int,int>> fixSuperpixelsMap(std::vector<std::pair<int,int>> image_superpixels); //remove superpixels having only one neighbour
double HellingerDistance(std::vector<std::vector<float>> vector_discrete,  std::vector<cv::Ptr<cv::ml::EM>> vector_EM_BG, std::vector<cv::Ptr<cv::ml::EM>> vector_EM_FG ,int nClusters);
std::map <int, std::vector<cv::Vec3b>> hashSuperpixels(int number_superpixel, cv::Mat currSuperpixelsMap, cv::Mat currFrame);
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);
std::vector<cv::Mat> getImages(std::string directory);
