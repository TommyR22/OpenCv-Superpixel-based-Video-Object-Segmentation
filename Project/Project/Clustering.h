//
//  Clustering.h
//  Project
//
//  Created by Tommaso Ruscica on 03/09/15.
//  Copyright (c) 2015 Tommaso Ruscica. All rights reserved.
//

#ifndef __Project__Clustering__
#define __Project__Clustering__

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#endif /* defined(__Project__Clustering__) */

class Clustering {
    
public:
    // pure virtual function
    virtual cv::Mat cluster (cv::Mat image) = 0;
};