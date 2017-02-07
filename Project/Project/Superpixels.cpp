//
//  FgModeling.cpp
//  Project
//
//  Created by Tommaso Ruscica on 15/09/15.
//  Copyright (c) 2015 Tommaso Ruscica. All rights reserved.
//

#include "Superpixels.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <numeric>
#include "SLIC.h"

extern "C" {
#include "/Users/TommyR22/Desktop/vlfeat/vl/generic.h"
#include "/Users/TommyR22/Desktop/vlfeat/vl/slic.h"
}

//NOTE:insert library libvl.dylib(vlfeat) in usr/local/lib (MAC OS)

using namespace std;
using namespace cv;

Mat segmentVLFeat(Mat segment);
vector<int> getMotionSuperpixels(cv::Mat image_superpixels, cv::Mat image_superpixels2, vector<pair<int,int>> vect_pair, int number_superpixels);
double getNumberSuperpixel(Mat image_superpixels);
vector<pair<int,int>> overlappingSuperpixel (Mat image_superpixels,Mat image_superpixels2);
Mat masking(Mat image, int label);
Mat masking_motionSuperpixels(Mat image_superpixels,vector<int> motionSuperpixels);
Mat segment(Mat segment);

//--------------------------------------------------------//
//                   SUPERPIXELS(VLFEAT)                  //
//--------------------------------------------------------//
Mat segmentVLFeat(Mat image){
    cout<<"-> Superpixels(segmentation)"<<endl;
    // Convert image to one-dimensional array.
    float* image_vector = new float[image.rows*image.cols*image.channels()];
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            // Assuming three channels ...
            image_vector[j + image.cols*i + image.cols*image.rows*0] = image.at<cv::Vec3b>(i, j)[0];
            image_vector[j + image.cols*i + image.cols*image.rows*1] = image.at<cv::Vec3b>(i, j)[1];
            image_vector[j + image.cols*i + image.cols*image.rows*2] = image.at<cv::Vec3b>(i, j)[2];
        }
    }
    
    // The algorithm will store the final segmentation in a one-dimensional array.
    vl_uint32* segmentation = new vl_uint32[image.rows*image.cols];
    vl_size height = image.rows;
    vl_size width = image.cols;
    vl_size channels = image.channels();
    
    // The region size defines the number of superpixels obtained.
    // Regularization describes a trade-off between the color term and the
    // spatial term.
    vl_size region = 7;
    float regularization = 2000; //m^2
    vl_size minRegion = 10;
    //segmentation function using VLFEAT
    vl_slic_segment(segmentation,image_vector , width, height, channels, region, regularization, minRegion);
    
    // Convert segmentation.
    int** slic_labels = new int*[image.rows];
    for (int i = 0; i < image.rows; ++i) {
        slic_labels[i] = new int[image.cols];
        for (int j = 0; j < image.cols; ++j) {
            slic_labels[i][j] = (int) segmentation[j + image.cols*i];
        }
    }
    
    //draw the segments in image
    int slic_label = 0;
    int slic_labelTop = -1;
    int slic_labelBottom = -1;
    int slic_labelLeft = -1;
    int slic_labelRight = -1;
    
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            
            slic_label = slic_labels[i][j];
            
            slic_labelTop = slic_label;
            if (i > 0) {
                slic_labelTop = slic_labels[i - 1][j];
            }
            
            slic_labelBottom = slic_label;
            if (i < image.rows - 1) {
                slic_labelBottom = slic_labels[i + 1][j];
            }
            
            slic_labelLeft = slic_label;
            if (j > 0) {
                slic_labelLeft = slic_labels[i][j - 1];
            }
            
            slic_labelRight = slic_label;
            if (j < image.cols - 1) {
                slic_labelRight = slic_labels[i][j + 1];
            }
            
            if (slic_label != slic_labelTop || slic_label != slic_labelBottom || slic_label!= slic_labelLeft || slic_label != slic_labelRight) {
                image.at<cv::Vec3b>(i, j)[0] = 0;
                image.at<cv::Vec3b>(i, j)[1] = 0;
                image.at<cv::Vec3b>(i, j)[2] = 255;
            }
        }
    }
    
    //convert label vector in Mat
    Mat image_superpixels(image.rows, image.cols, CV_32S);
    for(int i=0;i< image.rows; ++i){
        signed int *row_image_superpixels = image_superpixels.ptr<signed int>(i);
        for(int j=0; j< image.cols; ++j){
            row_image_superpixels[j] = slic_labels[i][j]+1; //////////////start numbering superpixel from 1
        }
    }
    //cout<<image_superpixels<<endl;
    imshow(to_string(rand() % 100), image);
    waitKey(0);
    
    return image_superpixels;
}

////-----------------------------------------------------//
////              GET NUMBER SUPERPIXELS                 //
////-----------------------------------------------------//
//double getNumberSuperpixel(Mat image_superpixels){
//    double min, max;
//    
//    cv::minMaxLoc(image_superpixels, &min, &max);
//    cout<<"number of superpixel: "<< max+1 <<endl;
//    
//    return max+1;
//}


//-----------------------------------------------------------------------------------//
//              CREATE VECTOR RAPRESENT OVERLAP SUPERPIXELS OF TWO FRAME             //
//-----------------------------------------------------------------------------------//
vector<pair<int,int>> overlappingSuperpixel(cv::Mat image_superpixels, cv::Mat image_superpixels2){
    //cout<<"Superpixel(overlapping)"<<endl;
    //vector pair to store overlap superpixel between the two frame
    vector<pair<int,int>> vect_pair;
    
    int init=0;
    //create vector overlapping between two frame
    for(int i=0;i< image_superpixels.rows;i++){
        const int* p_image = image_superpixels.ptr<int>(i);
        const int* p_image2 = image_superpixels2.ptr<int>(i);
        for(int j=0; j< image_superpixels.cols; ++j){
            //for init vector_pair.size().The first Pair discovered is add to vector_pair
            if(init == 0){
                //cout<<"init"<<endl;
                pair<int,int> aPair;
                aPair.first = p_image[j];
                aPair.second = p_image2[j];
                vect_pair.push_back(aPair);
                init = 1;
                break;
            }
            int exist = 0;
            //search in vector pair for no pair duplicate
            for(int k=0; k< vect_pair.size(); ++k){
                if(vect_pair[k].first == p_image[j] && vect_pair[k].second == p_image2[j]){
                    //nothing, pair already exist in vector_pair
                    //cout<<"esiste"<<endl;
                    exist = 1;
                    break;
                }
            }
            //if pair not exist in vector
            if(exist == 0){
                //add pair to vector_pair
                pair<int,int> aPair;
                aPair.first = p_image[j];
                aPair.second = p_image2[j];
                vect_pair.push_back(aPair);
            }
        }
    }
    
//    for(int i=0;i<vect_pair.size();i++){
//        cout<< vect_pair[i].first <<" - "<< vect_pair[i].second <<endl;
//    }
    
    return vect_pair;
}


//---------------------------------------------------------------------------//
//              CALCULATE JACCARD DISTANCE AND MOTION SUPERPIXELS            //
//---------------------------------------------------------------------------//
vector<int> getMotionSuperpixels(cv::Mat image_superpixels, cv::Mat image_superpixels2, vector<pair<int,int>> vect_pair, double number_superpixels){
    cout<<"Getting Motion Superpixel"<<endl;
    //NOTE:number_superpixels is the numbers of total superpixels in frame using vlfeat.Order label by cols starting from first row.
    //for threshold T(store all maximum pascal score between all superpixels in frame t+1)
    vector<double> max_distance;
    //vector containing motion Superpixels
    vector<int> motionSuperpixels;

    //loop for each superpixel in frame
    for(int k=1; k< number_superpixels; ++k){
        //save all distance of current superpixel in a temp vector(pascal_score)
        vector<double> pascal_score;
        //get the current superpixel ,find it vector_pair.second(right side)(frame t+1) and calculate jaccard distance with superpixel in left side of vect_pair(frame t)
        for(int x=0; x< vect_pair.size(); ++x){
            if(vect_pair[x].second == k){
                int y = vect_pair[x].first;
                
                //create binary masks of two images where pixels are 1 if equals current superpixel, 0 otherwise.
                Mat mask1 = masking(image_superpixels, y);
                Mat mask2 = masking(image_superpixels2, k);
                
                //calculate pascal score
                Mat AND,OR;
                bitwise_and(mask1,mask2,AND);
                bitwise_or(mask1,mask2,OR);
                
                //number pixel 1 in AND and OR mask
                double and_pixel = countNonZero(AND);
                double or_pixel = countNonZero(OR);

                //pascal score
                double pascalScore = (and_pixel/or_pixel);
                //cout<<"Distance: "<< pascalScore <<endl;
                
                //save it in pascal_score
                pascal_score.push_back(pascalScore);
                
            }
        }
        //check if current superpixels not exist(because union of two superpixels)
        if(pascal_score.size()==0){
            //do nothing
        }else{
            //calculate maximum and if above a T set it as motion superpixel
            double max = *max_element(pascal_score.begin(), pascal_score.end());
            //cout<< max <<endl;
            
            //for threshold T(store all minimum jaccard distance between all superpixels in frame t+1)
            max_distance.push_back(max);        //size equal number_superpixel
        }

        //at least clear pascal_score for next superpixel
        pascal_score.clear();
        
    }
    //cout<<"SIZE max_distance:"<<max_distance.size()<<endl;
    
    //calculate threshold T (as average of the maximum pascal score between all superpixels in frame t+1
    double T;
    //mean
    double sum = std::accumulate(max_distance.begin(), max_distance.end(), 0.0);
    //cout<<sum<<endl;
    double mean = sum / max_distance.size();
    cout<< "Mean: "<< mean <<" | ";
    //standard deviation
    double sq_sum = std::inner_product(max_distance.begin(), max_distance.end(), max_distance.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / max_distance.size() - mean * mean);
    cout<< "Standard deviation: "<< stdev <<" | ";

    double eps = std::numeric_limits<double>::epsilon();

    if(mean > 0.5){
        T = (mean) - 1.3*stdev - eps;
    }else{
        T =(mean) - 1.1*stdev - eps;
    }
    //T = mean;
    cout<<"Threshold T: "<< T << endl;
    
    //compare max_distance with T, if max_distance[x] < T ,it's Motion Superpixel!
    //loop for min_distance
    for(int i=0 ; i < max_distance.size(); ++i){
        if(max_distance[i] < T){  //if true, i = Motion superpixel!   //i+1 è superpixel poichè la dimensione di max_distance è = number_superpixel
            motionSuperpixels.push_back(i+1);   //+1 poichè le label dei superpixels partono da 1 e non da 0.
        }
    }
//    for(int x=0; x < motionSuperpixels.size(); x++){
//        cout<<"Motion Superpixel: "<< motionSuperpixels[x] <<endl;
//    }
    return motionSuperpixels;
}


//--------------------------------------------------------------------------------------//
//              CREATE MASK EACH SUPERPIXELS (1 FOR SUPERPIXELS,0 OTHERWISE )           //
//--------------------------------------------------------------------------------------//
Mat masking(Mat image_superpixels, int label){
    //binary mask for each superpixel, 1 if pixel is contained in superpixel, 0 otherwise.
    Mat mask = Mat(image_superpixels.rows,image_superpixels.cols, CV_8UC1, uchar(0));   //32S // 1 = 255
    for(int i=0; i< image_superpixels.rows; ++i){
        uchar *row_mask = mask.ptr<uchar>(i);
        int *row_image_superpixels = image_superpixels.ptr<int>(i);

        for(int j=0; j< image_superpixels.cols; ++j){
            //if pixel label == label of actual superpixel, substitute 1 to 0 in mask1 for precedent frame
            if(row_image_superpixels[j] == label){
                row_mask[j] = 255;
            }
        }
    }
    
//    if(label==2565){
//        imshow("prova", mask);
//        waitKey(0);
//    }
    
    int count = countNonZero(mask);
    if(count == 0){
        cout<<"Mask with all pixel 0";
    }
    //cout<<mask<<endl;
    return mask;
}

//----------------------------------------------------------------------------------------------------------------//
//              CREATE MASK FOR DETECT OBJECT MOTION SUPERPIXELS (1 MOTION SUPERPIXELS,0 OTHERWISE)               //
//----------------------------------------------------------------------------------------------------------------//
Mat masking_motionSuperpixels(Mat image_superpixels,vector<int> motionSuperpixels){
    //binary mask for each superpixel, 1 if pixel is contained in superpixel, 0 otherwise.
    Mat mask = Mat(image_superpixels.rows,image_superpixels.cols, CV_8UC1, char(0)); //CV_8UC1 for findContours
    
    for(int m=0; m< motionSuperpixels.size(); ++m){
        int motion_superpixel = motionSuperpixels[m];
        
        for(int i=0; i< image_superpixels.rows; ++i){
            unsigned char *row_mask = mask.ptr<unsigned char>(i);
            signed int *row_image_superpixels = image_superpixels.ptr<signed int>(i);

            for(int j=0; j< image_superpixels.cols; ++j){
                if(row_image_superpixels[j] == motion_superpixel){
                    row_mask[j] = 1;
                }
            }
        }
    }
    //cout<<mask<<endl;
    return mask;
}


//--------------------------------------------------------//
//                       SUPERPIXELS                      //
//--------------------------------------------------------//
Mat segment( Mat mat, int &numberOfLabels, Mat &imageSegmented)
{
    typedef unsigned int UINT;

    //superpixels params
    int spNum = 0;
    int spNumMax = (mat.rows*mat.cols)/49;;
    double compactness = 10.0;

    int width = mat.cols;
    int height = mat.rows;
    int sz = width*height;
    
    // Convert matrix to unsigned int array.
    unsigned int* image = new unsigned int[mat.rows*mat.cols];
    unsigned int value = 0x0000;
    
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            
            int b = mat.at<cv::Vec3b>(i,j)[0];
            int g = mat.at<cv::Vec3b>(i,j)[1];
            int r = mat.at<cv::Vec3b>(i,j)[2];
            
            value = 0x0000;
            value |= (0x00FF0000 & (r << 16));
            value |= (0x0000FF00 & (b << 8));
            value |= (0x000000FF & g);
            
            image[j + mat.cols*i] = value;
        }
    }
//    unsigned int* image = new unsigned int[mat.rows*mat.cols*mat.channels()];
//    for (int i = 0; i < mat.rows; ++i) {
//        for (int j = 0; j < mat.cols; ++j) {
//            // Assuming three channels ...
//            image[j + mat.cols*i + mat.cols*mat.rows*0] = mat.at<cv::Vec3b>(i, j)[0];
//            image[j + mat.cols*i + mat.cols*mat.rows*1] = mat.at<cv::Vec3b>(i, j)[1];
//            image[j + mat.cols*i + mat.cols*mat.rows*2] = mat.at<cv::Vec3b>(i, j)[2];
//        }
//    }
    
    int* label = new int[sz];
    SLIC slic;
    int* segmentation = new int[mat.rows*mat.cols];
    //int numberOfLabels = 0;

    slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(image, mat.cols, mat.rows, segmentation, numberOfLabels, spNumMax, compactness);
    
    cout<<"numberOfLabels: "<< numberOfLabels <<endl;

    Mat superpixels(mat.size(), CV_16U);
    for (int i = 0; i<superpixels.rows; i++){
        for (int j = 0; j<superpixels.cols; j++){
            superpixels.at<ushort>(i, j) = label[i + j*superpixels.rows];
        }
    }
    delete image, label;
    
    // Convert labels.
    int** slic_labels = new int*[mat.rows];
    for (int i = 0; i < mat.rows; ++i) {
        slic_labels[i] = new int[mat.cols];
        
        for (int j = 0; j < mat.cols; ++j) {
            slic_labels[i][j] = segmentation[j + i*mat.cols];
        }
    }
    
    Mat temp = mat.clone();

    //draw the segments in image
    int slic_label = 0;
    int slic_labelTop = -1;
    int slic_labelBottom = -1;
    int slic_labelLeft = -1;
    int slic_labelRight = -1;
    
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            
            slic_label = slic_labels[i][j];
            
            slic_labelTop = slic_label;
            if (i > 0) {
                slic_labelTop = slic_labels[i - 1][j];
            }
            
            slic_labelBottom = slic_label;
            if (i < mat.rows - 1) {
                slic_labelBottom = slic_labels[i + 1][j];
            }
            
            slic_labelLeft = slic_label;
            if (j > 0) {
                slic_labelLeft = slic_labels[i][j - 1];
            }
            
            slic_labelRight = slic_label;
            if (j < mat.cols - 1) {
                slic_labelRight = slic_labels[i][j + 1];
            }
            
            if (slic_label != slic_labelTop || slic_label != slic_labelBottom || slic_label!= slic_labelLeft || slic_label != slic_labelRight) {
                temp.at<cv::Vec3b>(i, j)[0] = 0;
                temp.at<cv::Vec3b>(i, j)[1] = 0;
                temp.at<cv::Vec3b>(i, j)[2] = 255;
            }
        }
    }
    
    //convert label vector in Mat
    Mat image_superpixels(mat.rows, mat.cols, CV_32S);
    for(int i=0;i< mat.rows; ++i){
        signed int *row_image_superpixels = image_superpixels.ptr<signed int>(i);
        for(int j=0; j< mat.cols; ++j){
            row_image_superpixels[j] = slic_labels[i][j]+1; //////////////start numbering superpixel from 1
        }
    }
    //cout<<image_superpixels<<endl;
    
    //imshow(to_string(rand() % 100), temp);
    //waitKey(0);
    
    imageSegmented = temp.clone();
    
    return image_superpixels;
}






