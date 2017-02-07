//
//  Adaptive_Kmeans.h
//  Project
//
//  Created by Tommaso Ruscica on 03/09/15.
//  Copyright (c) 2015 Tommaso Ruscica. All rights reserved.
//

#ifndef __Project__Adaptive_Kmeans__
#define __Project__Adaptive_Kmeans__

#include <stdio.h>
#include <iostream>
#include "Clustering.h"
#include <opencv2/opencv.hpp>
#include <vector>


#endif /* defined(__Project__Adaptive_Kmeans__) */


class Adaptive_Kmeans: public Clustering
{
public:
        
    virtual cv::Mat cluster (cv::Mat image);
    
};