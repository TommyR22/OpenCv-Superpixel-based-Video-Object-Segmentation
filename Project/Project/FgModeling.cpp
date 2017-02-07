//
//  FgModeling.cpp
//  Project
//
//  Created by Tommaso Ruscica on 06/10/15.
//  Copyright © 2015 Tommaso Ruscica. All rights reserved.
//

#include "FgModeling.h"
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

//TODO CONTROLLARE COSA SUCCEDE SE AUMENTA IL NUMERO DI OGGETTI IN FRAME SUCCESSIVE

vector<vector<Vec3b>> getPixelsFromMask(Mat image , Mat mask);

//dafault constructor
FgModeling::FgModeling () {
}


//-----------------------------------------------------//
//                  INIT FUNCTION                      //
//-----------------------------------------------------//
void FgModeling::init(Mat image, vector<vector<Point>> contours_mask, int nClusters){
    cout<<endl;
    cout<<"// FG MODELS(init) //"<<endl;
    double time;
    
    time = (double) getTickCount();
    
    //get number of objects frame and save pixels in vector_object
    vector<vector<Vec3b>> vector_object = getPixelsFromMask(image , contours_mask);
    
    vector<vector<float>> vector_discrete;
    for(int i=0; i< vector_object.size(); i++){
        vector <float> discrete;
        vector_discrete.push_back(discrete);
        this->counter.push_back(0);
    }

    //saving current pixel's object for update
    this->vector_prec_object = vector_object;
    //init vector and clear data in it.Used for update models
    this->vector_prec_object_temp = vector_object;
    for(int g=0; g<this->vector_prec_object_temp.size(); g++){
        this->vector_prec_object_temp[g].clear();
    }
    
    //cout<<"Total pixels in objects: "<< vector_object[0].size()+vector_object[1].size() <<endl;
    time = 1000*((double)getTickCount() - time)/getTickFrequency();
    cout << "Time Elapsed loop: " << time << " milliseconds." << endl;
    
    //----------------------- MOG and Kullback-Leibler divergence --------------------------
    //--------------------------------------------------------------------------------------
    cout<<"MOG and KL divergence(init)"<<endl;
    cout<<"Model's number: "<< vector_object.size()<<endl;
    
    for(int i=0; i< vector_object.size(); i++){
        
        vector<float> discrete_pdf;
        Mat probs(1, nClusters, CV_64FC1);
        Mat sample(1, 3, CV_64FC1);
        //double sum = 0;
        
        //Create train data for EM algorithm from vector_object.Convert each object to Mat
        Mat trainData(vector_object[i].size(), 3, CV_64F);
        for(int k=0; k<trainData.rows; ++k){
            double* Mi = trainData.ptr<double>(k);
            for(int j=0; j<trainData.cols; ++j){
                Mi[j] = vector_object[i][k][j];
            }
        }
        
        //create model, one for each cluster
        Ptr<ml::EM> em = ml::EM::create();
        
        em = ExpectationMaximization(em, trainData, nClusters, i, "object");
                
        //saving current model
        this->vector_EM.push_back(em);
        
        //sampling pdf model with step of 25 to discretize it.
        vector_discrete = DiscretizePdf(sample, probs, em, discrete_pdf,vector_discrete, i);
        
    }
    //save samples from pdf of current model's frame(for distance KL)
    this->vector_discrete_pdf_current = vector_discrete;
    this->vector_discrete_pdf_precedent = vector_discrete;
    
}



//-------------------------------------------------------//
//                  UPDATE FUNCTION                      //
//-------------------------------------------------------//
void FgModeling::update(Mat image, Mat mask_fg, int nClusters){
    cout<<endl;
    cout<<"// FG MODELS(update) //"<<endl;
    
    vector<vector<Point>> contours_mask;
    findContours(mask_fg,contours_mask, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    //double time = 0.0;
    
    vector<int> nModelsAdd;
    
    //get number of object and save pixels in vector_object
    vector<vector<Vec3b>> vector_object = getPixelsFromMask(image , contours_mask);  /// CONTROLLARE SE TIENE LO STESSO ORDINAMENTO (in posizione 1 pixels object 1...)
    
    vector<vector<float>> vector_discrete;
    for(int i=0; i< vector_object.size(); i++){
        vector <float> discrete;
        vector_discrete.push_back(discrete);
    }
    
    //----------------------- MOG and Kullback-Leibler divergence --------------------------//
    //--------------------------------------------------------------------------------------//
    //LOOP for each object in frame
    //drawing objects and identify the foreground model which best matches object using the Kullback-Leibler (KL) divergence.
    for( int n = 0; n< vector_object.size() ; n++ ){
        //Create train data for EM algorithm from vector_object.Convert each object to Mat
        Mat trainData(vector_object[n].size(), 3, CV_64F);
        for(int k=0; k< trainData.rows; ++k){
            for(int j=0; j<trainData.cols; ++j){
                trainData.at<double>(k, j) = vector_object[n][k][j];
            }
        }
        
        //create model, one for each cluster
        Ptr<ml::EM> em = ml::EM::create();
        
        em = ExpectationMaximization(em, trainData, nClusters, n, "object");
    
        //-------------------------- KL Divergence -----------------------------//
        //----------------------------------------------------------------------//
        //Kullback-Leibler Distance between current model and precedent models
        vector<float> discrete_pdf;
        Mat probs(1, nClusters, CV_64FC1);
        Mat sample(1, 3, CV_64FC1);
        
        //sampling pdf model with step of 25 to discretize it.
        vector_discrete = DiscretizePdf(sample, probs, em, discrete_pdf,vector_discrete, n);

        
    }
    //save discretization to calculate KL distance
    this->vector_discrete_pdf_current.clear();
    this->vector_discrete_pdf_current = vector_discrete;

    //vector[row][col] store kl distance between objects where row is current object, col is precedent object
    //vector<vector<double>> kl_distances;
    //INVECE DI CREARE IL VETTORE ,QUANDO CALCOLO LA DISTANZA POSSO VEDERE SE < T, in tal caso break e metto pixel in relativo modello.(vedere cout successivo)
    //Loop per numero oggetti in current frame!!
    for (int x=0; x< this->vector_discrete_pdf_precedent.size(); x++){  //init counter ( all model +=1 ), after update the model updated will have counter = 0
        this->counter[x] +=1;
    }
    //double dist=0;
    for(int j=0; j<this->vector_discrete_pdf_current.size(); j++){
        int add = 0;  //usato per capire se pixels modello corrente sono stati aggiunti a un modello esistente, in caso aggiungere nuovo modello
        for(int k=0; k< this->vector_discrete_pdf_precedent.size(); k++){
            double dist=0;
            double sommatoria=0;
            for(int i=0; i< this->vector_discrete_pdf_current[j].size(); i++){
//                double ratio = this->vector_discrete_pdf_current[j][i] / this->vector_discrete_pdf_precedent[k][i] ;   //KL DIVERGENCE
//                if (ratio > 0) {
//                    dist += this->vector_discrete_pdf_current[j][i] * log(ratio);
//                }
                sommatoria += vector_discrete_pdf_current[j][i] * vector_discrete_pdf_precedent[k][i];  //HELLINGER DISTANCE
                dist = sqrt(1 - sqrt(sommatoria));
            }
            cout<<"KL/HELLINGER DISTANCE |"<<" current obj: "<<j<<" precedent obj: "<<k<<" | distance: "<<dist<<endl;
            //if distance between current MOG j and precedent model k is smaller than a threshold, add pixels current object in model, else add new model
            if(dist < this->Tfg){
                add = 1;
                this->counter[k] = 0;
                for(int y=0; y<vector_object[j].size(); y++){
                    this->vector_prec_object[k].push_back(vector_object[j][y]);
                    this->vector_prec_object_temp[k].push_back(vector_object[j][y]);
                }
            }
            dist = 0;
            if(this->counter[k] == this->Tf){
                this->vector_prec_object.erase(this->vector_prec_object.begin() + k);
                this->vector_prec_object_temp.erase(this->vector_prec_object_temp.begin() + k);
                this->vector_EM.erase(this->vector_EM.begin() + k);
                this->vector_discrete_pdf_precedent.erase(this->vector_discrete_pdf_precedent.begin() + k);

            }
        }
        //crea nuovo modello e inserisci i pixels del corrente oggetto.
        if(add == 0){
            cout<<"Add new model"<<endl;
            this->vector_prec_object.push_back(vector_object[j]);
            this->vector_prec_object_temp.push_back(vector_object[j]);
            this->counter.push_back(0);
            //cout<<this->vector_prec_object.size()<<endl;
            nModelsAdd.push_back(this->vector_prec_object.size()-1);  //Add number of new model in a vector for train with default params(-1 because count start to 0)
        }
        //cout<<this->vector_prec_object.size()<<endl;
    }
    
    for (int i=0; i<this->vector_prec_object.size(); i++){
            cout<<"Modello finale: "<<i<<" | numero pixels: "<<this->vector_prec_object[i].size()<<endl;
    }
    
    for (int i=0; i<this->vector_prec_object_temp.size(); i++){
        cout<<"Modello temp finale: "<<i<<" | numero pixels: "<<this->vector_prec_object_temp[i].size()<<endl;
    }

    
    //-------------------------- Update Models(re-train) ----------------------------//
    //-------------------------------------------------------------------------------//
    //from vector_prec_object to Mat for EM
    for(int i=0;i< this->vector_prec_object.size();i++){
        //Create train data for EM algorithm from vector_prec_cluster(update)
        Mat trainData(this->vector_prec_object[i].size(), 3, CV_64F);
        for(int k=0; k<trainData.rows; ++k){
            for(int j=0; j<trainData.cols; ++j){
                trainData.at<double>(k, j) = this->vector_prec_object[i][k][j];
            }
        }
        
        //if new model , training with default params
        if ( std::find(nModelsAdd.begin(), nModelsAdd.end(), i) != nModelsAdd.end() ){
            //create model, one for each cluster
            Ptr<ml::EM> em = ml::EM::create();
            
            em = ExpectationMaximization(em, trainData, nClusters, i, "object");
            
            //saving current model
            this->vector_EM.push_back(em);
            
        }else{  //else training with params of precedent training
            //get precedent model
            Ptr<ml::EM> em = this->vector_EM[i];
            
            //re-train it with new pixels starting with means,cov and weights from precedent model
            vector<Mat> covs;
            
            ExpectationMaximization_update(em, trainData, covs, i, "object");
            
            //update current model(overwrite precedent)
            this->vector_EM[i] = em;
        }
        
        //clear clusters, insert current pixel's image in relative cluster for re-update(deleting old pixel from clusters) and clear temp cluster
        if(this->counter[i] == 0){
            this->vector_prec_object[i].clear();
            this->vector_prec_object[i] = this->vector_prec_object_temp[i];
            this->vector_prec_object_temp[i].clear();
        }else{
            this->vector_prec_object_temp[i].clear();
        }
        
    }
    
    for (int i=0; i<this->vector_prec_object.size(); i++){
        cout<<"Modello FINALE: "<<i<<" | numero pixels: "<<this->vector_prec_object[i].size()<<endl;
    }
//    for (int i=0; i<this->vector_prec_object_temp.size(); i++){
//        cout<<"Modello temp FINALE: "<<i<<" | numero pixels: "<<this->vector_prec_object_temp[i].size()<<endl;
//    }
    
    
    //dopo tutto devo impostare vector_discrete_pdf_precedent = vector_discrete_pdf_current!!
    this->vector_discrete_pdf_precedent = this->vector_discrete_pdf_current;
    this->vector_discrete_pdf_current.clear();
    
    //for (int i=0; i<this->counter.size(); i++){
    //    cout<< this->counter[i] <<endl;
    //}
}



//-------------------------------------------------------//
//              GET PIXELS FROM MASK FUNCTION            //
//-------------------------------------------------------//
vector<vector<Vec3b>> FgModeling::getPixelsFromMask(Mat image,vector<vector<Point>> contours_mask){
    //create a mask for each motionsuperpixels in frame image
    //vector<vector<Point>> contours_mask;
    //findContours(mask,contours_mask, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    int nObjects = contours_mask.size();
    cout<<"Number Objects: "<< nObjects <<endl;
    
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

