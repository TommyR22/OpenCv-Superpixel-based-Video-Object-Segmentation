//
//  GraphCut.hp
//  Project
//
//  Created by Tommaso Ruscica on 05/11/15.
//  Copyright © 2015 Tommaso Ruscica. All rights reserved.
//

#ifndef GraphCut_hp
#define GraphCut_hp

#include <stdio.h>
#include <opencv2/opencv.hpp>

#endif /* GraphCut_hp */

//function to obtain matrix of links between node superpixels
std::vector<std::pair<int,int>> NeighboursSuperpixelMap(cv::Mat image_superpixels);
