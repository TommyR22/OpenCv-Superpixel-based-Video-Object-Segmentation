//
//  Utility.cpp
//  Project
//
//  Created by Tommaso Ruscica on 21/01/16.
//  Copyright © 2016 Tommaso Ruscica. All rights reserved.
//

#include "Utility.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <numeric>
#include "Superpixels.h"
#include "PerceptualOrganizationEnergy.h"

using namespace std;
using namespace cv;



//-------------------------------------------------------------//
//-------------------------- Get Images -----------------------//
//-------------------------------------------------------------//

vector<Mat> getImages(string directory){

    //path's vector of images in directoryName
    std::vector <string> path_images;
    //Mat's vector of images in directoryName
    std::vector <Mat> images;

    DIR *dir;
    dir = opendir(directory.c_str());
    string imgName;
    struct dirent *ent;

    //loop for read all images in the directoryName
    if (dir != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            imgName= ent->d_name;
            //read file images except OS's files like "." , ".." , "DS_Store"
            if(imgName.compare(".")!= 0 && imgName.compare("..")!= 0 && imgName.compare(".DS_Store")!= 0)
            {
                string path;
                path.append(directory);
                path.append(imgName);
                Mat image= imread(path);
                //image.convertTo(image, CV_8UC3);
                //cv::cvtColor(image, image, CV_BGR2Lab);
                images.push_back(image);
                path_images.push_back(path);
                //cout << path << endl;
            }
        }
        closedir (dir);
        
    }else {
        cout<<"Directory not present"<<endl;
    }
    cout << "N.images in directory: "<<directory<< " -> " + to_string(images.size()) << endl;
    
    
    return images;
}

//-----------------------------------------------------//
//             EXPECTATION MAXIMIZATION                //
//-----------------------------------------------------//
Ptr<ml::EM> ExpectationMaximization(Ptr<ml::EM> em, Mat trainData , int nClusters, int currentCluster, String type){
    
    //set Params model
    em->setTermCriteria( TermCriteria( TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, FLT_EPSILON ));///////////////////
    em->setCovarianceMatrixType( ml::EM::COV_MAT_DIAGONAL);
    em->setClustersNumber(nClusters);
    
    double time = (double) getTickCount();
    //cout<<"Start Training "<< type <<" : "<<currentCluster<<endl;
    //Train cluster with default weights,means,covs.
    em->trainEM(trainData);
    //cout<<"Finish Training"<<endl;
    ////cout<<"Means model after training: "<<em->getMeans()<<endl;
    time = 1000*((double)getTickCount() - time)/getTickFrequency();
    //cout << "Time Elapsed Training: " << time << " milliseconds." << endl;
    
    return em;
}


//-----------------------------------------------------//
//         EXPECTATION MAXIMIZATION UPDATE             //
//-----------------------------------------------------//
Ptr<ml::EM> ExpectationMaximization_update(Ptr<ml::EM> em,Mat trainData ,vector<Mat> covs , int currentCluster,String type){
    
    em->getCovs(covs);
    em->setTermCriteria( TermCriteria( TermCriteria::MAX_ITER, 10, FLT_EPSILON ));///////////////////////
    
    double time = (double) getTickCount();
    cout << endl;
    cout << "Start Re-Training "<< type << " : " << currentCluster << endl;
    //train start with means,covs and weights from precedent model
    em->trainE(trainData, em->getMeans(), covs, em->getWeights());
    cout << "Finish Re-Training" << endl;
    //cout<<"Means model after training(update): "<<em->getMeans()<<endl;
    time = 1000*((double)getTickCount() - time)/getTickFrequency();
    cout << "Time Elapsed Re-Training: " << time << " milliseconds." << endl;
    
    return em;
}



//-----------------------------------------------------//
//                   DISCRETIZE PDF                    //
//-----------------------------------------------------//
vector<vector<float>> DiscretizePdf(Mat sample, Mat probs, Ptr<cv::ml::EM> em, vector<float> discrete_pdf, vector<vector<float>> vector_discrete, int i){
    double sum = 0;
    for(int red=0 ; red<=255 ; red+=25){
        for(int green=0 ; green<=255 ; green+=25){
            for(int blue=0 ; blue<=255 ; blue+=25){
                sample.at<double>(0,0) = red;
                sample.at<double>(0,1) = green;
                sample.at<double>(0,2) = blue;
                
                //cout<<sample<<endl;
                Vec2d result = em->predict2(sample, probs);
                //cout<<result<<endl;
                discrete_pdf.push_back(exp(result[0]));
                sum += exp(result[0]);
            }
        }
    }
    
    //normalize results of discretization
    double sum2=0;
    for(int j=0 ; j< discrete_pdf.size() ; j++){
        double epsd = std::numeric_limits<double>::epsilon();////////////////////////////////AGGIUNTA PER NON AVERE INF IN KL DISTANCE
        double temp = (discrete_pdf[j]+epsd)/sum;
        discrete_pdf[j] = temp;
        vector_discrete[i].push_back(temp); //save final result discretization in a vector //////////////
        sum2 += temp;
    }
    //cout<<"Totale probs: "<<sum2<<endl;
    
    return vector_discrete;
}


//--------------------------------------------------------------------------//
//                   FILTER MOVING SUPERPIXEL BY RADIUS                     //
//--------------------------------------------------------------------------//
vector<int> filteredMovingSuperpixelsMap(Mat currSuperpixelsMap, vector<int> motionSuperpixels){
    int radius = 15;
    //vector of type Pair.first element is movingSuperpixel, second element is the number of movingSupepixels neighbour
    vector<pair<int,int>> vect_pair;
    
    for(int i = 0; i<motionSuperpixels.size(); i++){
        int contatore = 0;
        Mat AND_result = Mat(currSuperpixelsMap.size(), CV_32S);
        Mat mask_superpixel = Mat(currSuperpixelsMap.size(), CV_8UC1, uchar(0));
        Mat mask_superpixel2 = Mat(currSuperpixelsMap.size(), CV_8UC1, uchar(0));
        //get current motionSuperpixel and calculate centroid
        int label = motionSuperpixels[i];
        mask_superpixel = masking(currSuperpixelsMap, label);
        vector<vector<Point>> contours_mask1;
        findContours(mask_superpixel ,contours_mask1 , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
        Point center1 = centroid(contours_mask1[0]);
        Mat centroid_mask = Mat::zeros( mask_superpixel.size(), CV_8UC1 );
        for( int ii = 0; ii< contours_mask1.size(); ii++ ){
            Scalar color( 255, 255, 255 );
            drawContours( centroid_mask, contours_mask1, ii, color, CV_FILLED);
        }
        Scalar color( 255, 255, 255 );
        circle(centroid_mask, center1, radius, color, CV_FILLED);
        //imshow("circle", centroid_mask);
        //waitKey(0);
        centroid_mask.convertTo(centroid_mask, CV_32S);
        //AND operation to retrieve label superpixels inside circle
        bitwise_and(currSuperpixelsMap, centroid_mask, AND_result);
        vector<int> temp;
        for(int i=0; i< AND_result.rows; ++i){
            for(int j=0; j< AND_result.cols; ++j){
                signed int element = AND_result.at<signed int>(i,j);
                if(element != 0){
                    signed int label_neighbour = currSuperpixelsMap.at<signed int>(i,j);
                    //check if label superpixels is a movingSuperpixel
                    bool is_movingSuperpixel = std::find(motionSuperpixels.begin(), motionSuperpixels.end(),label_neighbour)!= motionSuperpixels.end();
                    if(is_movingSuperpixel == 1){
                        //cout<<label_neighbour<<endl;
                        if(std::find(temp.begin(), temp.end(), label_neighbour) != temp.end()){
                            //present
                        }else{
                            //not present
                        temp.push_back(label_neighbour);
                        }
                    }
                }
            }
        }
        pair<int,int> aPair;
        aPair.first = label;
        aPair.second = temp.size();
        vect_pair.push_back(aPair);
    }
    //delete moving superpixels that have < 4 neighbours.(4+1 because superpixels count itself as neighbours)
    vect_pair.erase( std::remove_if( vect_pair.begin(), vect_pair.end(), [](pair<int,int> i){ return i.second < 5;} ), vect_pair.end() );
    motionSuperpixels.clear();
    //re-create motionSuperpixels vector without movingSuperpixels filtered
    for(int x =0; x< vect_pair.size();x++){
        motionSuperpixels.push_back(vect_pair[x].first);
    }

    return motionSuperpixels;
}



//---------------------------------------------------------//
//                   FILTER BACKGROUND                     //
//---------------------------------------------------------//
vector<vector<Point>> filterBackground(Mat image, vector<vector<Point>> contours_mask, vector<Ptr<ml::EM>> BGmodels, vector<Ptr<ml::EM>> FGmodels, int p, int nClusters){
    
    //get number of objects frame and save pixels in vector_object
    vector<vector<Vec3b>> vector_object = getPixelsFromMask(image , contours_mask);
    
    vector<vector<float>> vector_discrete;
    for(int i=0; i< vector_object.size(); i++){
        vector <float> discrete;
        vector_discrete.push_back(discrete);
    }

    cout<<"MOG and KL divergence"<<endl;
    cout<<"Model's number currFrame: "<< vector_object.size()<<endl;
    
    for(int i=0; i< vector_object.size(); i++){
        
        double minBG;
        double minFG;
        
        vector<float> discrete_pdf;
        Mat probs(1, nClusters, CV_64FC1);
        Mat sample(1, 3, CV_64FC1);
        
        //Create train data for EM algorithm from vector_object.Convert each object to Mat
        Mat trainData(vector_object[i].size(), 3, CV_64F);
        for(int k=0; k<trainData.rows; ++k){
            for(int j=0; j<trainData.cols; ++j){
                trainData.at<double>(k, j) = vector_object[i][k][j];
            }
        }
        
        //create model, one for each cluster
        Ptr<ml::EM> em = ml::EM::create();
        
        em = ExpectationMaximization(em, trainData, nClusters, i, "object");
        
        //discretize pdf
        vector_discrete = DiscretizePdf(sample, probs, em, discrete_pdf,vector_discrete, i);
        
        //vector_discrete[i] current pdf object
        minBG = MinBG(vector_discrete[i],BGmodels);
        minFG = MinFG(vector_discrete[i],FGmodels);

        if(p*minBG < minFG){
            //contours_mask.erase(contours_mask.begin()+i);       //////////////ELIMINO MALE I SUPERPIXELS!! ERRORE
            //TODO prendere maskera connected component da eliminare, prendere suoi movingsuperpixel ed eliminarli dal vettore movingSuperpixel
        }

    }
    
    return contours_mask;
}
    
//-------------------------------------------------------//
//              GET PIXELS FROM MASK FUNCTION            //
//-------------------------------------------------------//
vector<vector<Vec3b>> getPixelsFromMask(Mat image,vector<vector<Point>> contours_mask){
    //create a mask for each motionsuperpixels in frame image
    //vector<vector<Point>> contours_mask;
    //findContours(mask,contours_mask, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    int nObjects = contours_mask.size();
    //cout<<"Number Objects: "<< nObjects <<endl;
        
    //Create vector of Object created by numbers of motion superpixels contains pixel's value
    vector<vector<Vec3b>> vector_object;
        
    //drawing objects
    for( int n = 0; n< nObjects; n++ ){
        vector <Vec3b> object;
        vector_object.push_back(object);
        
        Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
        Scalar color( 255, 255, 255 );
        drawContours( drawing, contours_mask, n, color, CV_FILLED);
        Mat drawing_gray;
        cvtColor( drawing, drawing_gray, CV_BGR2GRAY );
            
        for(int i = 0; i < image.rows; i++)
        {
            const Vec3b* p_image = image.ptr<Vec3b>(i);
            const uchar* p_drawing = drawing_gray.ptr<uchar>(i);
                
            for(int j = 0; j < image.cols; j++){
                if(p_drawing[j] != 0){ //if pixel's value in mask is equals 0 its background else foreground(Object)(save it in object)
                    vector_object[n].push_back(p_image[j]);
                }
            }
        }
        //cout<<vector_object[n].size()<<endl;
        //imshow(to_string(rand() % 100), drawing);
        //waitKey(0);
    }
    
    return vector_object;   //vector with vector of foreground pixels
}


//-------------------------------------------------------//
//                          MIN BG                       //
//-------------------------------------------------------//
double MinBG(vector<float> vector_discrete,std::vector<cv::Ptr<cv::ml::EM>> BGmodels){
    
    vector<double> vect_minBG;
    
    double minBG;
    
    vector<vector<float>> vector_discreteBG;
        for(int i=0; i< BGmodels.size(); i++){
        vector <float> discrete;
        vector_discreteBG.push_back(discrete);
    }
    
    for(int i=0; i< BGmodels.size(); i++){
        int nClusters = 2;
        vector<float> discrete_pdf;
        Mat probs(1, nClusters, CV_64FC1);
        Mat sample(1, 3, CV_64FC1);
        vector_discreteBG = DiscretizePdf(sample, probs, BGmodels[i], discrete_pdf,vector_discreteBG, i);
    }
    
    double dist = 0;

    for(int k=0; k< vector_discreteBG.size(); k++){
        for(int i=0; i< vector_discrete.size(); i++){
            double ratio = vector_discrete[i] / vector_discreteBG[k][i] ;   //+ esp PER ELIMINARE inf IN CALCOLO DISTANZA
            if (ratio > 0) {
                dist += vector_discrete[i] * log(ratio);
            }
        }
        cout<<"KL DISTANCE | precedent obj: "<<k<<" | distance: "<<dist<<endl;
        vect_minBG.push_back(dist);
    }
    auto min = min_element(vect_minBG.begin(), vect_minBG.end());
    minBG = *min;
    
    return minBG;

}

//-------------------------------------------------------//
//                          MIN FG                       //
//-------------------------------------------------------//
double MinFG(vector<float> vector_discrete,std::vector<cv::Ptr<cv::ml::EM>> FGmodels){
    
    vector<double> vect_minFG;
    
    double minFG;
    
    vector<vector<float>> vector_discreteFG;
    for(int i=0; i< FGmodels.size(); i++){
        vector <float> discrete;
        vector_discreteFG.push_back(discrete);
    }
    
    for(int i=0; i< FGmodels.size(); i++){
        int nClusters = 2;
        vector<float> discrete_pdf;
        Mat probs(1, nClusters, CV_64FC1);
        Mat sample(1, 3, CV_64FC1);
        vector_discreteFG = DiscretizePdf(sample, probs, FGmodels[i], discrete_pdf,vector_discreteFG, i);
    }
    
    double dist = 0;
    
    for(int k=0; k< vector_discreteFG.size(); k++){
        for(int i=0; i< vector_discrete.size(); i++){
            double ratio = vector_discrete[i] / vector_discreteFG[k][i] ;   //+ esp PER ELIMINARE inf IN CALCOLO DISTANZA
            if (ratio > 0) {
                dist += vector_discrete[i] * log(ratio);
            }
        }
        cout<<"KL DISTANCE | precedent obj: "<<k<<" | distance: "<<dist<<endl;
        vect_minFG.push_back(dist);
    }
    auto min = min_element(vect_minFG.begin(), vect_minFG.end());
    minFG = *min;
    
    return minFG;
    
}


vector<Mat> createWindows(Mat image, std::vector<std::vector<cv::Point> > const& contours_mask, int winInc)
{
    //vector with all windows
    vector<Mat> vect_windows;

    //connected component not processed in a window yet
    vector<bool> NotProcessed;
    for (int i =0; i<contours_mask.size(); i++){
        NotProcessed.push_back(true);
    }

    for( int i = 0; i< contours_mask.size(); i++ ){
        if(NotProcessed[i]){
            //drawing rectangle current contour
            int maxX = 0, minX = image.cols, maxY=0, minY = image.rows;
                for(int cc=0; cc<contours_mask[i].size(); cc++)
                {
                    Point p = contours_mask[i][cc];
                    maxX = max(maxX, p.x);
                    minX = min(minX, p.x);
                    maxY = max(maxY, p.y);
                    minY = min(minY, p.y);
                }
            Scalar color( 255, 255, 255 );
            Mat boundingBox = Mat::zeros( image.size(), CV_8UC1 );
            //cv::rectangle(boundingBox, cv::boundingRect(contours_mask[i]), cv::Scalar(255,255,255), CV_FILLED);
            rectangle( boundingBox, Point(minX,minY), Point(maxX, maxY), cv::Scalar(255,255,255) ,CV_FILLED);

            NotProcessed[i] = false;
            //imshow( "BB1", boundingBox );
            //waitKey(0);
        
            //loop for all contour mask not processed with new mask yet
            loop_windows:
            for( int j = 0 ; j<contours_mask.size();j++){
                if(NotProcessed[j]){
                    Scalar color( 255, 255, 255 );
                    Mat dst = Mat::zeros( boundingBox.size(), CV_8UC1 );
                    Mat boundingBox2 = Mat::zeros( boundingBox.size(), CV_8UC1 );
                    cv::rectangle(boundingBox2, cv::boundingRect(contours_mask[j]), cv::Scalar(255,255,255), CV_FILLED);
                    bitwise_and(boundingBox, boundingBox2, dst);
                    if(countNonZero(dst) != 0){
                        NotProcessed[j] = false;
                        
                        int maxX = 0, minX = image.cols, maxY=0, minY = image.rows;
                        for(int cc=0; cc<contours_mask[j].size(); cc++)
                        {
                            Point p = contours_mask[j][cc];
                            maxX = max(maxX, p.x);
                            minX = min(minX, p.x);
                            maxY = max(maxY, p.y);
                            minY = min(minY, p.y);
                        }
                        rectangle( boundingBox, Point(minX,minY), Point(maxX, maxY), cv::Scalar(255,255,255) ,CV_FILLED);
                        //cv::rectangle(boundingBox, cv::boundingRect(contours_mask[j]), cv::Scalar(255,255,255), CV_FILLED);
                        //imshow( "BB", boundingBox );
                        //waitKey(0);
                        goto loop_windows;
                    }
                }
            }
            //enlarge current window
            vector<vector<Point>> contours_mask_final;
            findContours(boundingBox,contours_mask_final, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            maxX = 0, minX = image.cols, maxY=0, minY = image.rows;
            for(int cc=0; cc<contours_mask_final[0].size(); cc++)
            {
                Point p = contours_mask_final[0][cc];
                maxX = max(maxX, p.x);
                minX = min(minX, p.x);
                maxY = max(maxY, p.y);
                minY = min(minY, p.y);
            }
            Mat boundingBox_final = Mat::zeros( image.size(), CV_8UC1 );
            //cv::rectangle(boundingBox, cv::boundingRect(contours_mask[i]), cv::Scalar(255,255,255), CV_FILLED);
            rectangle( boundingBox_final, Point(minX - winInc,minY - winInc), Point(maxX + winInc, maxY + winInc), cv::Scalar(255,255,255) ,CV_FILLED);
            //add current window in a vector
            vect_windows.push_back(boundingBox_final);
            //imshow( "BB FINALE", boundingBox_final );
            //waitKey(0);
        }
    }
    //cout<<"SIZE bb "<< boundingBoxMats.size() <<endl;
    return vect_windows;
}



//----------------------------------------------------------------//
//                       FIX SUPERPIXELS MAP                      //
//----------------------------------------------------------------//
//remove superpixels having only one neighbour
vector<pair<int,int>> fixSuperpixelsMap(vector<pair<int,int>> neighboursPair){
    for(int i=0; i<=neighboursPair.size(); i++){
        
        int j=0;
        int k=0;
        int superpixel1 = neighboursPair[i].first;
        int superpixel2 = neighboursPair[i].second;
        
        for(int x=0; x<=neighboursPair.size(); x++){
            if(neighboursPair[x].first == superpixel1 || neighboursPair[x].second == superpixel1){
                j+=1;
            }
            if(neighboursPair[x].first == superpixel2 || neighboursPair[x].second == superpixel2){
                k+=1;
            }
        }
        
        if(j==1){
            for(int x=0; x<=neighboursPair.size(); x++){
                if(neighboursPair[x].first == superpixel1){
                    neighboursPair[x].second = superpixel1;
                }
            }
        }
        if(k==1){
            for(int x=0; x<=neighboursPair.size(); x++){
                if(neighboursPair[x].first == superpixel2){
                    neighboursPair[x].second = superpixel2;
                }
            }
        }
    }
    
    return neighboursPair;
}


//----------------------------------------------------//
//                HELLINGER DISTANCE                  //
//----------------------------------------------------//
//hellinger distance for edge(graphcut) between two superpixels neighbours
double HellingerDistance(vector<vector<float>> vector_discrete,  vector<Ptr<cv::ml::EM>> vector_EM_BG, vector<Ptr<cv::ml::EM>> vector_EM_FG , int nClusters){
    
    vector<double> vector_distances;
    
    for(int i=0; i<vector_EM_BG.size(); ++i){
        Ptr<ml::EM> em = vector_EM_BG[i];
        vector<float> discrete_pdf;
        Mat probs(1, nClusters, CV_64FC1);
        Mat sample(1, 3, CV_64FC1);
        
        vector<vector<float>> vector_discrete_model;
        vector <float> discrete;
        vector_discrete_model.push_back(discrete);
        
        //sampling pdf model with step of 25 to discretize it.
        vector_discrete_model = DiscretizePdf(sample, probs, em, discrete_pdf,vector_discrete_model, 0);
                
        double sommatoria = 0;
        double dist = 0;

        for(int j=0; j<vector_discrete[0].size(); j++){
            sommatoria += vector_discrete[0][j] * vector_discrete_model[0][j];
        }
        dist = sqrt(1 - sqrt(sommatoria));
        vector_distances.push_back(dist);
    }
    for(int i=0; i<vector_EM_FG.size(); ++i){
        Ptr<ml::EM> em = vector_EM_FG[i];
        vector<float> discrete_pdf;
        Mat probs(1, nClusters, CV_64FC1);
        Mat sample(1, 3, CV_64FC1);
        
        
        vector<vector<float>> vector_discrete_model;
        vector <float> discrete;
        vector_discrete_model.push_back(discrete);
        
        //sampling pdf model with step of 25 to discretize it.
        vector_discrete_model = DiscretizePdf(sample, probs, em, discrete_pdf,vector_discrete_model, 0);

        double sommatoria = 0;
        double dist = 0;
        
        for(int j=0; j<vector_discrete[0].size(); j++){
            sommatoria += vector_discrete[0][j] * vector_discrete_model[0][j];
        }
        dist = sqrt(1 - sqrt(sommatoria));
        vector_distances.push_back(dist);
    }
    double distance = 1 - *min_element(vector_distances.begin(), vector_distances.end());
    
    if(isnan(distance)){
        double epsd = std::numeric_limits<double>::epsilon();//aggiunta per non avere nan
        distance = epsd;
    }
    
    return distance;
}


//----------------------------------------------------//
//                 HASH SUPERPIXELS                   //
//----------------------------------------------------//
//hash(superpixel, pixels of superpixel)
map <int, vector<Vec3b>> hashSuperpixels(int number_superpixel, Mat currSuperpixelsMap, Mat currFrame){
    map <int, vector<Vec3b>> hash_pixels;

    for(int nSuperpixels=0; nSuperpixels < number_superpixel; ++nSuperpixels){
        int superpixel = nSuperpixels;
        vector<Vec3b> pixels;
        for(int i = 0; i < currSuperpixelsMap.rows; ++i){
            const int* p_label = currSuperpixelsMap.ptr<int>(i);
            const Vec3b* p_image = currFrame.ptr<Vec3b>(i);
            for(int j = 0; j < currSuperpixelsMap.cols; ++j){
                if(p_label[j] == superpixel){
                    pixels.push_back(p_image[j]);
                }
            }
        }
        hash_pixels[superpixel] = pixels;
    }
    
    return hash_pixels;
}



//----------------------------------------------------//
//                   MAT IS EQUAL                     //
//----------------------------------------------------//
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2){
    // treat two empty mat as identical as well
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    // if dimensionality of two mat is not identical, these two mat is not identical
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
        return false;
    }
    cv::Mat diff;
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
    int nz = cv::countNonZero(diff);
    return nz==0;
}

