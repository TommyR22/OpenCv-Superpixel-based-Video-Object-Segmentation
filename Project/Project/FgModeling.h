//
//  FgModeling.hp
//  Project
//
//  Created by Tommaso Ruscica on 06/10/15.
//  Copyright © 2015 Tommaso Ruscica. All rights reserved.
//

#ifndef FgModeling_hp
#define FgModeling_hp

#include <stdio.h>
#include <opencv2/opencv.hpp>


#endif /* defined(__Project__FgModeling__) */

class FgModeling {
    
public:
    FgModeling ();
    void init (cv::Mat image,std::vector<std::vector<cv::Point>> contours_mask, int nClusters);
    void update (cv::Mat image, cv::Mat mask_fg, int nClusters);
    std::vector<std::vector<cv::Vec3b>> getPixelsFromMask (cv::Mat image, std::vector<std::vector<cv::Point>> contours_mask);
    //cv::Mat map (cv::Mat image);
    std::vector<cv::Ptr<cv::ml::EM>> getVector_EM(){return vector_EM;}
    
private:
    std::vector<std::vector<cv::Vec3b>> vector_prec_object;  //cluster's vector with pixels(MotionSuperpixels) of precedent frame
    std::vector<std::vector<cv::Vec3b>> vector_prec_object_temp;  //cluster's vector with pixels(MotionSuperpixels) of precedent frame
    
    std::vector<cv::Ptr<cv::ml::EM>> vector_EM;  //EM's vector(models)
    
    std::vector<std::vector<float>> vector_discrete_pdf_precedent;
    std::vector<std::vector<float>> vector_discrete_pdf_current;
    std::vector<int> counter;
    double Tfg = 1;   //threshold KL divergence
    int Tf = 10;    //threshold to remove model from model set

};