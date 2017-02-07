//
//  PerceptualOrganizationEnergy.h
//  Project
//
//  Created by Tommaso Ruscica on 05/11/15.
//  Copyright © 2015 Tommaso Ruscica. All rights reserved.
//

#ifndef PerceptualOrganizationEnergy_h
#define PerceptualOrganizationEnergy_h

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "cvblob.h"

#endif /* PerceptualOrganizationEnergy_h */


float boundaryComplexity(const cv::Mat& patch_mask, const std::vector<cv::Point>& contour, int s, int k, int sn);
double calculateSimmetry(cv::Mat image_superpixels1, cv::Mat image_superpixels2);

cv::Point centroid(const std::vector<cv::Point>& points);

std::vector<cv::Point> fixContour(const std::vector<cv::Point>& contour);
bool inNeighbourhood(cv::Point point_1, cv::Point point_2);
std::vector<cv::Point> getPointsBetween(const cv::Point& p1, const cv::Point& p2);

void drawContour(const std::vector<cv::Point>& points, cv::Mat& frame, cv::Scalar color, int thickness_or_filled);

// Compute distance between points
inline float pointDistance(const cv::Point& p1, const cv::Point& p2)
{
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

// Compute sum between numbers with wrapping to 0 after maximum value (excluded)
inline int wrap(int a, int m)
{
    // Apply modified mod function
    return ((a % m) + m) % m;
}


cv::Mat commonBoundary_1(const cvb::CvBlob& blob1, const cvb::CvBlob& blob2);
std::vector<cv::Point> commonBoundary_2(const cvb::CvBlob& blob1, const cvb::CvBlob& blob2);

float cohesivenessStrength(const cvb::CvBlob& patch_1, const cvb::CvBlob& patch_2);
cv::Mat blobToMat(const cvb::CvBlob& blob, int position = 1, bool filled = true, int output_width = -1, int output_height = -1);
int binaryArea(const cv::Mat& img);
std::vector<cv::Point> blobToPointVector(const cvb::CvBlob& blob);
//bool isBlobValid(const cvb::CvBlob& blob, unsigned int max_width, unsigned int max_height);
cvb::CvBlob* createBlob(const std::vector<cv::Point>& contour);
std::vector<cv::Point> boundingBoxFromContour(const std::vector<cv::Point>& contour);
cvb::CvBlob* createBlob(const std::vector<cv::Point>& bounding_box, const std::vector<cv::Point>& contour);
unsigned char pointDifferenceToChainCode(cv::Point last, cv::Point next);
void showBlob(const std::string&, const cvb::CvBlob& blob, int position = 0, bool filled = true, int output_width = -1, int output_height = -1);


int calculateContinuity(cv::Point P1,cv::Point P2, cv::Mat mask);
float calculatePOM(const cvb::CvBlob& patch_1, const cvb::CvBlob& patch_2, float complexity , float simmetry, int continuity, float proximity);
float POM(std::vector<std::pair<int,int>> neighboursPair, std::vector<std::vector<cv::Point>> &contours, cv::Mat currSuperpixelsMap, cv::Mat mask1, cv::Mat mask2, int s,int k,int ns);




