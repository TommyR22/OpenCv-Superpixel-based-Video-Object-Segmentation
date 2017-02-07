//
//  BgModeling.cpp
//  Project
//
//  Created by Tommaso Ruscica on 03/09/15.
//  Copyright (c) 2015 Tommaso Ruscica. All rights reserved.
//

#include "BgModeling.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <numeric>
#include "Utility.h"

using namespace std;
using namespace cv;


//dafault constructor
BgModeling::BgModeling () {
}

//-----------------------------------------------------//
//                  INIT FUNCTION                      //
//-----------------------------------------------------//
void BgModeling::init(Mat image, Mat label, Mat mask_fg, int nClusters){
    cout<<endl;
    cout<<"// BG MODELS(init) //"<<endl;
    int image_rows = image.rows;
    int image_cols = image.cols;
    double time;

    //Looking for max and min value in label
    double minVal,maxVal;
    Point minLoc,maxLoc;
    minMaxLoc( label, &minVal, &maxVal, &minLoc, &maxLoc );
    cout << "LABELS -> min val: " << minVal << " || "; cout << "max val: " << maxVal << endl;
    
    time = (double) getTickCount();
    //Create vector of clusters created by label contains pixel's value
    vector<vector<Vec3b>> vector_cluster;
    for(int i=0; i<maxVal; i++){
        vector <Vec3b> cluster;
        vector_cluster.push_back(cluster);
    }
    
    //al singolo elemento di label corrisponde un pixel dell'img che però è di tipo vec3b.
    //loop to getting pixel's value from value of label and put it in vector_cluster
    for(int i = 0; i < label.rows; i++)
    {
        const int* p_label = label.ptr<int>(i);
        const Vec3b* p_image = image.ptr<Vec3b>(i);
        const int* p_mask = mask_fg.ptr<int>(i);

        for(int j = 0; j < label.cols; j++){
            if(p_mask[j] == 0){ //if pixel's value in mask_fg is equals 0 its background(save it in vector_cluster) else foreground(discard)
                vector_cluster[p_label[j]-1].push_back(p_image[j]); //"-1" because vector begin from 0 and minVal=1;
            }
        }
    }
    //saving current pixel's clusters for update
    this->vector_prec_cluster = vector_cluster;
    //init vector and clear data in it.Used for update function
    this->vector_prec_cluster_temp = vector_cluster;
    for(int g=0; g<this->vector_prec_cluster_temp.size(); g++){
        this->vector_prec_cluster_temp[g].clear();
    }
    
    //cout<<"size vector_cluster: "<<vector_cluster[0].size()<<endl;
    //cout<<"size vector_cluster: "<<vector_cluster[1].size()<<endl;
    //cout<<"size vector_cluster: "<<vector_cluster[2].size()<<endl;
    //cout<<"size vector_cluster: "<<vector_cluster[3].size()<<endl;
    //cout<<"size vector_cluster: "<<vector_cluster[4].size()<<endl;
    //cout<<"Total elements in clusters: "<<vector_cluster[0].size()+vector_cluster[1].size()+vector_cluster[2].size()+vector_cluster[3].size()+vector_cluster[4].size()<<endl;
    //cout<<"1 element vector_cluster: "<<vector_cluster[0][0]<<endl;
    time = 1000*((double)getTickCount() - time)/getTickFrequency();
    cout << "Time Elapsed loop: " << time << " milliseconds." << endl;
    
    //-------------------------- Expectation Maximization(init) ----------------------------
    //--------------------------------------------------------------------------------------
    cout<<"-> Expectation Maximization(init)"<<endl;
    cout<<"Model's number: "<<vector_cluster.size()<<endl;

    for(int i=0; i<vector_cluster.size(); i++){
        //Create train data for EM algorithm from vector_cluster.Convert each cluster to Mat
        Mat trainData(vector_cluster[i].size(), 3, CV_64F);
        for(int k=0; k<trainData.rows; ++k){
            for(int j=0; j<trainData.cols; ++j){
                trainData.at<double>(k, j) = vector_cluster[i][k][j];
            }
        }
        
        //create model, one for each cluster
        Ptr<ml::EM> em = ml::EM::create();
        
        //train model
        em = ExpectationMaximization(em, trainData, nClusters, i, "cluster");

        //saving current model
        this->vector_EM.push_back(em);

        vector<Mat> covs;
        //Calculating prior model
        float prior=((double)vector_cluster[i].size())/(image_rows*image_cols);
        this->vector_prior.push_back(prior);
        cout<<endl;
        em->getCovs(covs);
        cout<<"BG MODEL: "<<i<<" ||";
        cout<<" Prior: "<< prior;
        cout<<" Weights: "<< em->getWeights()<<endl;
        cout<<" Means model: "<< em->getMeans()<<endl;
        //for(int i=0;i<covs.size();++i){
        //    cout<<covs[i]<<endl;
        //}
    }
    cout<<endl;
}

//-----------------------------------------------------//
//                  UPDATE FUNCTION                    //
//-----------------------------------------------------//
void BgModeling::update(Mat image, Mat mask_fg, int nClusters){
    cout<<endl;
    cout<<"// BG MODELS(update) //"<<endl;
    cout<<"-> Expectation Maximization(update)"<<endl;
    cout<<"Model's number: "<< this->vector_prec_cluster.size()<<endl;
    //Update have as input models from vector_prec_clusters

    int image_rows = image.rows;
    int image_cols = image.cols;
    double time = 0.0;
    
    //loop for each pixel in current frame
    //for each pixel get P(model|pixel) and put pixel in cluster where P is max.
    for(int i=0; i<image.rows; ++i){
        
        const int* p_mask = mask_fg.ptr<int>(i);
        
        for(int j=0; j<image.cols; ++j){
            if(p_mask[j] == 0){ //check from mask if pixel is labelled as background (0) or foreground (1)
                //get pixel
                Vec3b pixel = image.at<Vec3b>(Point(i,j));
                //pixel to Mat
                Mat mat_pixel(1, 3, CV_64FC1);
                for(int x=0; x<3; x++){
                    mat_pixel.at<double>(0,x) = pixel[x];
                }
            
                int model_with_max_prob = 0;
                float current_value = 0;
                float last_max_value = 0;
                
                //loop for Models
                for(int k=0; k< this->vector_EM.size(); k++){
                    //get model
                    ml::EM* em = this->vector_EM[k];
                    //predict for current pixel
                    Mat probs(1, nClusters, CV_64FC1);
                    Vec2d v = em->predict2(mat_pixel ,probs);
                    //get max P(model|p)
                    current_value = this->vector_prior[k]*exp(v[0]);
                    //check for max P(model|p) for current pixel in models and get the model.
                    if(current_value > last_max_value){
                        model_with_max_prob = k;
                        last_max_value = current_value; //precedente_value = current max value
                    }
                }
            //put pixel into model where P(model|p) is max
            this->vector_prec_cluster[model_with_max_prob].push_back(pixel);
            //for re-update
            this->vector_prec_cluster_temp[model_with_max_prob].push_back(pixel);
            }
        }
    }

    //cout<<"Total elements in clusters(update): "<<this->vector_prec_cluster[0].size()+this->vector_prec_cluster[1].size()+this->vector_prec_cluster[2].size()+this->vector_prec_cluster[3].size()+this->vector_prec_cluster[4].size()<<endl;
    //cout<<"Total elements in clusters TEMP(update): "<<this->vector_prec_cluster_temp[0].size()+this->vector_prec_cluster_temp[1].size()+this->vector_prec_cluster_temp[2].size()+this->vector_prec_cluster_temp[3].size()+this->vector_prec_cluster_temp[4].size()<<endl;
    
    //-------------------------- Update Models(re-train) ---------------------------------------
    //------------------------------------------------------------------------------------------
    //from vector_prec_cluster to Mat for EM
    for(int i=0; i< this->vector_prec_cluster.size(); ++i){
        //Create train data for EM algorithm from vector_prec_cluster(update)
        Mat trainData(this->vector_prec_cluster[i].size(), 3, CV_64F);
        for(int k=0; k<trainData.rows; ++k){
            for(int j=0; j<trainData.cols; ++j){
                trainData.at<double>(k, j) = this->vector_prec_cluster[i][k][j];
            }
        }
        //get precedent model
        Ptr<ml::EM> em = this->vector_EM[i];
        
        //re-train it with new pixels starting with means,cov and weights from precedent model
        vector<Mat> covs;
        
        em = ExpectationMaximization_update(em, trainData, covs, i, "cluster");

        //update current model(overwrite precedent)
        this->vector_EM[i] = em;

        //update prior(overwrite precedent)
        float prior = ((double)this->vector_prec_cluster[i].size())/(image_rows*image_cols);
        this->vector_prior[i] = prior;
        cout<< "Prior model " << i << " --> " << prior <<endl;
        
        //clear clusters, insert current pixel's image in relative cluster for re-update(deleting old pixel from clusters) and clear temp cluster
        this->vector_prec_cluster[i].clear();
        this->vector_prec_cluster[i] = this->vector_prec_cluster_temp[i];
        this->vector_prec_cluster_temp[i].clear();
    }
    

}

//-----------------------------------------------------//
//                  MAP FUNCTION                       //
//-----------------------------------------------------//
Mat BgModeling::map(Mat image){
    int nClusters=2;
    
    //image with probs
    Mat image_out(image.rows, image.cols, CV_64FC1);

    //loop for each pixel in image
    for(int i=0; i<image.rows; ++i){
        for(int j=0; j<image.cols; ++j){
            //get one pixel
            Vec3b pixel = image.at<Vec3b>(Point(i,j));
            Mat mat_pixel(1, 3, CV_64FC1);
            //pixel to Mat
            for(int x=0;x<3;x++){
                mat_pixel.at<double>(0,x) = pixel[x];
            }
            float current_value = 0;
            float last_max_value = 0;
            
            //loop for models
            for(int k=0;k < this->vector_EM.size();k++){
                //get model
                Ptr<ml::EM> em = this->vector_EM[k];
                //predict P(model|p) for current pixel in this model
                Mat probs(1, nClusters, CV_64FC1);
                em->predict2(mat_pixel ,probs);
                
                //check for max P(model|p) for current pixel in models and get the mx value of P
                current_value = this->vector_prior[k]*probs.at<float>(0,0);
                if(current_value > last_max_value){
                    last_max_value = current_value; //precedente_value = current max value
                }
            }
            //save max P in image_out
            image_out.at<double>(i,j) = last_max_value;
            
        }
    }
    return image_out;
}
