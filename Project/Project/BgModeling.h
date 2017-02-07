//
//  BgModeling.h
//  Project
//
//  Created by Tommaso Ruscica on 03/09/15.
//  Copyright (c) 2015 Tommaso Ruscica. All rights reserved.
//

#ifndef __Project__BgModeling__
#define __Project__BgModeling__

#include <stdio.h>
#include <opencv2/opencv.hpp>

#endif /* defined(__Project__BgModeling__) */

class BgModeling {
    
public:
    BgModeling ();
    void init (cv::Mat image, cv::Mat label, cv::Mat mask_fg, int nClusters);
    void update (cv::Mat image, cv::Mat mask_fg, int nClusters);
    cv::Mat map (cv::Mat image);
    std::vector<cv::Ptr<cv::ml::EM>> getVector_EM(){return vector_EM;}
    std::vector<float> getVector_prior(){return vector_prior;}

    
private:
    std::vector<std::vector<cv::Vec3b>> vector_prec_cluster;  //cluster's vector with pixels of precedent frame
    std::vector<std::vector<cv::Vec3b>> vector_prec_cluster_temp;  //cluster's vector with pixels of precedent frame

    std::vector<cv::Ptr<cv::ml::EM>> vector_EM;  //EM's vector(models)
    std::vector<float> vector_prior;

};