//
//  GraphCut.cpp
//  Project
//
//  Created by Tommaso Ruscica on 05/11/15.
//  Copyright © 2015 Tommaso Ruscica. All rights reserved.
//

#include "GraphCut.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "graph.h"


using namespace std;
using namespace cv;

enum Direction{
    ShiftUp=2, ShiftRight=1, ShiftDown=3, ShiftLeft=0
};


//-----------------------------------------------------//
//                  NODES GRAPHCUT                      //
//-----------------------------------------------------//
vector<pair<int,int>> NeighboursSuperpixelMap(Mat image_superpixels){
    //function to obtain matrix of links between superpixels
    //shift in all direction all pixels of image to know the neighbors of each superpixels (only 1 shift position)
    vector<pair<int,int>> vect_pair;
    Direction direction_shift;
    cv::Mat temp;
    bool running = true;
    
    direction_shift = ShiftLeft;  //init
    
    while(running){
        switch (direction_shift)
        {
        case(ShiftLeft) :
            for(int i=1; i< 2; i++){
                temp = cv::Mat::zeros(image_superpixels.size(), image_superpixels.type());
                image_superpixels(cv::Rect(i, 0, image_superpixels.cols - i, image_superpixels.rows)).copyTo(temp(cv::Rect(0, 0, image_superpixels.cols - i, image_superpixels.rows)));

                //loop for all pixels in image_superpixels
                for(int j=0; j< image_superpixels.rows; j++){
                    const int* p_image = image_superpixels.ptr<int>(j);
                    const int* p_temp = temp.ptr<int>(j);
    
                    for(int x=0; x< image_superpixels.cols; x++){
                        pair<int,int> aPair;
                        aPair.first = p_image[x];   //pixel frame before shift(same position)
                        aPair.second = p_temp[x];   //pixel frame after shift(same position)
                        
                        if(aPair.first == 0 | aPair.second == 0){
                            //no insert pair
                        }
                        else if(aPair.first != aPair.second){
                            bool exist = false;
                            for(auto & element : vect_pair){
                                if(element.first == aPair.first && element.second == aPair.second){ //vector contains pair(not insert)
                                    exist = true;
                                    break;
                                }else if(element.first == aPair.second && element.second == aPair.first){   //not insert reflected pair es. (2-1 and 1-2)(delete only one)
                                    exist = true;
                                    break;
                                }else{
                                    //not exist
                                }
                            }
                            if(!exist){
                                vect_pair.push_back(aPair);
                            }
                        }
                    }
                }
            }
            direction_shift = ShiftRight;
            break;
                
        case(ShiftRight) :
            for(int i=1; i< 2; i++){
                temp = cv::Mat::zeros(image_superpixels.size(), image_superpixels.type());
                image_superpixels(cv::Rect(0, 0, image_superpixels.cols - i, image_superpixels.rows)).copyTo(temp(cv::Rect(i, 0, image_superpixels.cols - i, image_superpixels.rows)));
    
                //loop for all pixels in image_superpixels
                for(int j=0; j< image_superpixels.rows; j++){
                    const int* p_image = image_superpixels.ptr<int>(j);
                    const int* p_temp = temp.ptr<int>(j);
                    
                    for(int x=0; x< image_superpixels.cols; x++){
                        pair<int,int> aPair;
                        aPair.first = p_image[x];   //pixel frame before shift(same position)
                        aPair.second = p_temp[x];   //pixel frame after shift(same position)
                        
                        if(aPair.first == 0 | aPair.second == 0){
                            //no insert pair
                        }
                        else if(aPair.first != aPair.second){
                            bool exist = false;
                            for(auto & element : vect_pair){
                                if(element.first == aPair.first && element.second == aPair.second){
                                    exist = true;
                                    break;
                                }else if(element.first == aPair.second && element.second == aPair.first){
                                    exist = true;
                                    break;
                                }else{
                                    //not exist
                                }
                            }
                            if(!exist){
                                vect_pair.push_back(aPair);
                            }
                        }
                    }
                }
            }            direction_shift = ShiftUp;
            break;
    
        case(ShiftUp) :
                for(int i=1; i< 2; i++){
                    temp = cv::Mat::zeros(image_superpixels.size(), image_superpixels.type());
                    image_superpixels(cv::Rect(0, i, image_superpixels.cols, image_superpixels.rows - i)).copyTo(temp(cv::Rect(0, 0, image_superpixels.cols, image_superpixels.rows - i)));
    
                    //loop for all pixels in image_superpixels
                    for(int j=0; j< image_superpixels.rows; j++){
                        const int* p_image = image_superpixels.ptr<int>(j);
                        const int* p_temp = temp.ptr<int>(j);
                        
                        for(int x=0; x< image_superpixels.cols; x++){
                            pair<int,int> aPair;
                            aPair.first = p_image[x];   //pixel frame before shift(same position)
                            aPair.second = p_temp[x];   //pixel frame after shift(same position)
                            
                            if(aPair.first == 0 | aPair.second == 0){
                                //no insert pair
                            }
                            else if(aPair.first != aPair.second){
                                bool exist = false;
                                for(auto & element : vect_pair){
                                    if(element.first == aPair.first && element.second == aPair.second){
                                        exist = true;
                                        break;
                                    }else if(element.first == aPair.second && element.second == aPair.first){
                                        exist = true;
                                        break;
                                    }else{
                                        //not exist
                                    }
                                }
                                if(!exist){
                                    vect_pair.push_back(aPair);
                                }
                            }
                        }
                    }
                }                direction_shift = ShiftDown;
                break;
    
        case(ShiftDown) :
                for(int i=1; i< 2; i++){
                    temp = cv::Mat::zeros(image_superpixels.size(), image_superpixels.type());
                    image_superpixels(cv::Rect(0, 0, image_superpixels.cols, image_superpixels.rows - i)).copyTo(temp(cv::Rect(0, i, image_superpixels.cols, image_superpixels.rows - i)));
    
                    //loop for all pixels in image_superpixels
                    for(int j=0; j< image_superpixels.rows; j++){
                        const int* p_image = image_superpixels.ptr<int>(j);
                        const int* p_temp = temp.ptr<int>(j);
                        
                        for(int x=0; x< image_superpixels.cols; x++){
                            pair<int,int> aPair;
                            aPair.first = p_image[x];   //pixel frame before shift(same position)
                            aPair.second = p_temp[x];   //pixel frame after shift(same position)
                            
                            if(aPair.first == 0 | aPair.second == 0){
                                //no insert pair
                            }
                            else if(aPair.first != aPair.second){
                                bool exist = false;
                                for(auto & element : vect_pair){
                                    if(element.first == aPair.first && element.second == aPair.second){ //vector contains pair(not insert)
                                        exist = true;
                                        break;
                                    }else if(element.first == aPair.second && element.second == aPair.first){   //not insert reflected pair es. (2-1 and 1-2)(delete only one)
                                        exist = true;
                                        break;
                                    }else{
                                        //not exist
                                    }
                                }
                                if(!exist){
                                    vect_pair.push_back(aPair);
                                }
                            }
                        }
                    }
                }
                direction_shift = ShiftLeft;
                running = false;
                break;
        } //end switch
    } //end while
    
    cout << "Size neighbourPair: "<< vect_pair.size() << endl;
//    for(int i=0; i<vect_pair.size(); i++){
//        cout<< vect_pair[i].first << " - " << vect_pair[i].second <<endl;
//    }
    
    return vect_pair;
}




