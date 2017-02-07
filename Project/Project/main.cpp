//
//  main.cpp
//  Project
//
//  Created by Tommaso Ruscica on 20/07/15.
//  Copyright (c) 2015 Tommaso Ruscica. All rights reserved.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include "adaptive_kmeans.h"
#include "BgModeling.h"
#include "Superpixels.h"
#include "FgModeling.h"
#include "GraphCut.h"
#include "PerceptualOrganizationEnergy.h"
#include "Utility.h"
#include "graph.h"
#include <map>
#include <numeric>


extern "C" {
#include "/Users/TommyR22/Desktop/vlfeat/vl/generic.h"
#include "/Users/TommyR22/Desktop/vlfeat/vl/slic.h"
}

using namespace std;
using namespace cv;


int main(int argc, const char * argv[]) {
    
    //path image's directory
    string directoryName = "/Users/TommyR22/Desktop/SegTrackv2/JPEGImages/birdfall/";
    string directoryName_groundTruth = "/Users/TommyR22/Desktop/SegTrackv2/GroundTruth/birdfall/";
    
    vector<Mat> images = getImages(directoryName);
    vector<Mat> images_groundTruth = getImages(directoryName_groundTruth);
    
    //size image, rows and columns
    cv::Size size = images[0].size();
    cout<<"Size images:"<< size << endl;
    
    //number of superpixels
    int numberOfLabels = 0;
    
    //display segmentation in image
    Mat imageSegmented;
    
    //first frame
    Mat currFrame = images[0];
    //imshow( "First Frame", currFrame );
    //waitKey(0);
    //first frame superpixel
    Mat temp_currFrame = currFrame.clone();
    Mat currSuperpixelsMap = segment(temp_currFrame, numberOfLabels, imageSegmented);
  
    //--------------//
    //*** PARAMS ***//
    //--------------//
    
    //Params for Filter
    //size filter parameters: remove group of moving superpixels that have size lower than
    //threshold sizeT = ceil((height+width)/sizeP)*superPixelDimension;
    int width = currFrame.cols;
    int height = currFrame.rows;
    int superPixelDimension = 49;
    int sizeP = 200;
    int sizeT = ((width*height)/sizeP)*superPixelDimension;
    int FilterSuperpixelsSize = 50;
    
    //Params for POM
    int s = 1;
    int k = 2;
    int ns = 2;
    
    //Params EM
    int nClusters = 3;
    
    //Param FilterBackground
    int p = 1;
    
    //Param windows
    int winInc = 3; //windows enlargment
    
    //Params graphcut
    int a1 = 1;
    int a2 = 1;
    
    //Param appearance
    int n = 3;  //if size1 >=n*size2
    int p_edge = 1; //parametro messo nel caso in cui nella normalizzazione di un edge, il max è = al min (e quindi denominatore a 0).
    
    //background/foreground models
    BgModeling *bg = new BgModeling();
    FgModeling *fg = new FgModeling();
    
    //for F-measure
    float total_tp;
    float total_fp;
    float total_fn;
    
    //--------------//
    

    //start from 2nd frame
    for(int frame = 1 ; frame < images.size()  ; ++frame){  /*curr_image <images.size()*/
        Mat imageSegmented;                             //display segmentation in image
        Mat currFrame = images[frame];                  //current frame
        Mat prevSuperpixelsMap = currSuperpixelsMap;    //previous frame
        Mat temp_currFrame = currFrame.clone();
        Mat currSuperpixelsMap = segment(temp_currFrame, numberOfLabels, imageSegmented);    //current frame superpixel
        

        //find neighbours of each superpixels
        //vector<pair<int,int>> nodes = NeighboursSuperpixelMap(currSuperpixelsMap);  //in GraphCut.cpp
        
        //get moving superpixels
        cout<<endl;
        cout<<"// MOVING SUPERPIXELS //"<<endl;
        vector<pair<int,int>> vect_pair = overlappingSuperpixel(prevSuperpixelsMap, currSuperpixelsMap);
        //double number_superpixel = getNumberSuperpixel(currSuperpixelsMap);
        vector<int> motionSuperpixels = getMotionSuperpixels(prevSuperpixelsMap, currSuperpixelsMap, vect_pair, numberOfLabels);
        Mat movingSuperpixelsMap = masking_motionSuperpixels(currSuperpixelsMap, motionSuperpixels);
        cout << "nMotionSuperpixels: " << motionSuperpixels.size() <<endl;
        
        //binary image superpixels
        vector<vector<Point>> contours_mask;
        findContours(movingSuperpixelsMap,contours_mask, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        Mat mask = Mat::zeros( movingSuperpixelsMap.size(), CV_8UC3 );
        for( int i = 0; i< contours_mask.size(); ++i ){
            Scalar color( 255, 255, 255 );
            drawContours( mask, contours_mask, i, color, CV_FILLED );
        }
        //imshow( "Result mask", mask );
        //waitKey(0);
        //show blending between current frame and mask superpixels
        double alpha = 0.6;
        double beta = ( 1.0 - alpha );
        Mat dst = Mat::zeros( movingSuperpixelsMap.size(), CV_8UC3 );
        addWeighted( imageSegmented, alpha, mask, beta, 0.0, dst);
        imshow( "Moving Superpixels", dst );
        waitKey(0);
        
        //Filter Moving Superpixels by Radius...
        motionSuperpixels = filteredMovingSuperpixelsMap(currSuperpixelsMap,motionSuperpixels);
        cout<<"nMotionSuperpixels after filter radius: "<< motionSuperpixels.size() <<endl;
        //typedef std::vector<cv::Vec4i> Hierarchy;
        //Hierarchy hierarchy;
        Mat movingSuperpixelsMap2 = masking_motionSuperpixels(currSuperpixelsMap, motionSuperpixels);
        //binary image superpixels
        vector<vector<Point>> contours_mask_filtered;
        findContours(movingSuperpixelsMap2,contours_mask_filtered, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        //...and by size
        for (vector<vector<Point> >::iterator it = contours_mask_filtered.begin(); it!=contours_mask_filtered.end();){
            if (it->size() < FilterSuperpixelsSize)
                it = contours_mask_filtered.erase(it);
            else
                ++it;
        }
        //draw
        Mat filteredSuperpixel = Mat::zeros( movingSuperpixelsMap2.size(), CV_8UC3 );
        for( int i = 0; i< contours_mask_filtered.size(); ++i ){
            Scalar color( 255, 255, 255 );
            drawContours( filteredSuperpixel, contours_mask_filtered, i, color, CV_FILLED);
        }
        addWeighted( imageSegmented, alpha, filteredSuperpixel, beta, 0.0, filteredSuperpixel);
        imshow( "Moving Superpixel filtered", filteredSuperpixel );
        waitKey(0);
        
        Mat mask_superpixel_fg = Mat::zeros( movingSuperpixelsMap2.size(), CV_8UC1 );
        for( int i = 0; i< contours_mask_filtered.size(); ++i ){
            Scalar color( 255, 255, 255 );
            drawContours( mask_superpixel_fg, contours_mask_filtered, i, color, CV_FILLED);
        }
        imshow( "mask_superpixel", mask_superpixel_fg );
        waitKey(0);
        
        
        //first processable frame
        if(frame == 1 ){
            //create appearance models
            //adaptive k-means
            Adaptive_Kmeans* kmeans = new Adaptive_Kmeans();
            Mat label = kmeans->cluster(currFrame);
            //Background modeling
            //Mat mask_fg = Mat(currFrame.rows,currFrame.cols, CV_64F, double(0)); //SI PUO' ELIMINARE IN INIT(?)
            bg->init( currFrame, label, mask_superpixel_fg, nClusters);
            //cout<<"VECTOR_EM: "<<bg->getVector_EM().size()<<endl; // anche getVector_prior();
            
            //Foreground modeling
            Mat groundTruth = images_groundTruth[frame];
            cvtColor(groundTruth, groundTruth, CV_RGB2GRAY);
            vector<vector<Point>> cc;
            findContours(groundTruth, cc, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            fg->init( currFrame, cc, nClusters);                                //////////////////////mask_superpixels_fg
        }
        
        //segmentation algorithm
        //at first iteration the windows are computed on the filtered moving superpixels

        //tengo traccia delle coppie di superpixels nodi graph e relativo contributo pom
        map <pair<int,int>,double> hash_edges_pom;
        //tengo traccia delle coppie di superpixels nodi graph e relativo contributo hellinger
        map <pair<int,int>,double> hash_edges_hellinger;
        //tengo traccia delle coppie di superpixels nodi graph e relativo contributo likelihood
        map <pair<int,int>,double> hash_edges_likelihood;
        
        //hash con chiave superpixel e come valore le capacità verso source e sink
        map <int,pair<double,double>> hash_likelihood;
        
        //hashmap(superpixel label , all pixel inside superpixel label)
        map <int, vector<Vec3b>> hash_pixels = hashSuperpixels( numberOfLabels, currSuperpixelsMap, currFrame);
        
        
        //Create windows connected components
        cout<<endl;
        cout<<"// WINDOWS //"<<endl;
        Mat boundingBox = Mat::zeros( movingSuperpixelsMap2.size(), CV_8UC3 );
        for( int i = 0; i< contours_mask_filtered.size(); i++ ){
            cv::rectangle(boundingBox, cv::boundingRect( contours_mask_filtered[i]), cv::Scalar(255));
        }
        //imshow( "boundingBox", boundingBox );
        //waitKey(0);
        
        int iteration = 0;
        
        //create Graph
        cout<<endl;
        cout<<"// GRAPHCUT //"<<endl;
        
        Mat tmp_mask = mask_superpixel_fg.clone();
        
        while(true){
            
            //maschera finale ad ogni ciclo while
            Mat final_mask = Mat::zeros( movingSuperpixelsMap2.size(), CV_8UC1 );

            //vector of windows (binary image Mat)
            vector<Mat> vect_windows;
            vect_windows = createWindows( movingSuperpixelsMap2, contours_mask_filtered, winInc);
            
            for(int window = 0 ; window < vect_windows.size(); ++window){
                cout<<"// current window: "<<window<<" //"<<endl;
                //sub image from contour mask current window
                
                //vettore per la normalizzazione
                vector<float> pom_normalization;
                vector<double> hellinger_normalization;
                vector<double> likelihood_normalization;
            
                vector<vector<Point>> contours_mask_graph;
                findContours(vect_windows[window],contours_mask_graph, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
               
                int maxX = 0, minX = images[0].cols, maxY=0, minY = images[0].rows;
                for(int cc = 0; cc<contours_mask_graph[0].size(); ++cc){
                    Point p = contours_mask_graph[0][cc];
                    maxX = max(maxX, p.x);
                    minX = min(minX, p.x);
                    maxY = max(maxY, p.y);
                    minY = min(minY, p.y);
                }
                Point p_min = *new Point(minX,minY);
                Point p_max = *new Point(maxX,maxY);
                Rect rect(p_min,p_max);
                //subImage from current pixel
                Mat subCurrSuperpixelsMap = currSuperpixelsMap(rect);
                
                //Mat prova = currFrame(rect);
                //imshow("subImage", prova);
                //waitKey(0);
                
                vector<pair<int,int>> neighboursPair = NeighboursSuperpixelMap(subCurrSuperpixelsMap);
            
                //remove superpixels having only one neighbour
                //neighboursPair = fixSuperpixelsMap(neighboursPair);
            
                //per ogni pixel del corrente superpixel :calcolo la likelihood per ogni modello di Fg e Bg, faccio la media per ogni modello e prendo solo il minimo dei modelli di Bg e il minimo tra quelli di Fg.
                vector<vector<float>> superpixel_likelihoodsBG; //contiene le likelihood per ogni modello
                vector<vector<float>> superpixel_likelihoodsFG;

                //initialize  likelihood vectors
                vector<float> likehoods_model;
                for(int i=0; i<bg->getVector_EM().size(); ++i){
                    superpixel_likelihoodsBG.push_back(likehoods_model);
                    superpixel_likelihoodsBG[i].clear();
                }
                for(int i=0; i<fg->getVector_EM().size(); ++i){
                    superpixel_likelihoodsFG.push_back(likehoods_model);
                    superpixel_likelihoodsFG[i].clear();
                }
                //superpixel_likelihoodsBG.clear();
                //superpixel_likelihoodsFG.clear();

                vector<double> vector_means;

                int id_node = 0; //used for label node in graphcut
            
                //usato per avere corrispondenza tra superpixel e relativo nodo graphcut
                map<int,int> pair_superpixelsNode;
            
                //Graphcut
                typedef Graph<double,double,double> GraphType;
                int numNodes = numberOfLabels;
                int numEdge = neighboursPair.size();
                GraphType *graph = new GraphType(/*estimated # of nodes*/ numNodes, /*estimated # of edges*/ numEdge );
            
                Mat mask_fg = Mat(currSuperpixelsMap.rows,currSuperpixelsMap.cols, CV_8UC1, char(0)); //CV_8UC1 for findContours
                Mat mask_bg = Mat(currSuperpixelsMap.rows,currSuperpixelsMap.cols, CV_8UC1, char(0)); //CV_8UC1 for findContours
            
                //calcolo likelihood bg e fg di ogni pixel di un superpixel per graphcut
                for ( int sp=0; sp<neighboursPair.size(); ++sp ){   //poichè in neighboursPair avrò tutti i pixel della corrente finestra //superpixel
                    cout << ".";
                    for( int nsp=0 ; nsp<2; ++nsp ){     //number superpixel, ciclo per prendere i due superpixel nella coppia
                    
                        for(int i=0; i<superpixel_likelihoodsBG.size(); ++i){
                            superpixel_likelihoodsBG[i].clear();
                        }
                        for(int i=0; i<superpixel_likelihoodsFG.size(); ++i){
                            superpixel_likelihoodsFG[i].clear();
                        }
                    
                        int superpixel;
                        if( nsp==0 ){
                            superpixel = neighboursPair[sp].first;
                        }else{
                            superpixel = neighboursPair[sp].second;
                        }
                        
                        
                      if( pair_superpixelsNode.find(superpixel) == pair_superpixelsNode.end() ){  //check if superpixel was computed
                          if( hash_likelihood.find(superpixel) == hash_likelihood.end() ){  //check if superpixel was computed
                              //cout<<"non esiste: "<<superpixel<<endl;
                            //not found
                            //int cc=0;
                            vector<Vec3b> pixels = hash_pixels[superpixel]; //prendo tutti i pixels del relativo superpixel
                            for( int pixel=0; pixel<pixels.size(); ++pixel ){
                                Mat mat_pixel(1, 3, CV_64FC1);
                                //pixel to Mat
                                for(int x=0;x<3;x++){
                                    mat_pixel.at<double>(0,x) = pixels[pixel][x];
                                }
                                //cout<<mat_pixel<<endl;
                                //if(cc==0)cout<<mat_pixel<<endl;
                                //cc++;
                                //loop for BGmodels
                                for( int bgmodels=0; bgmodels<bg->getVector_EM().size(); ++bgmodels ){
                                    //get model
                                    ml::EM* em = bg->getVector_EM()[bgmodels];
                                    //get likelihood
                                    Mat probs(1, nClusters, CV_64FC1);
                                    Vec2d v = em->predict2(mat_pixel ,probs);
                                    superpixel_likelihoodsBG[bgmodels].push_back(v[0]/**bg->getVector_prior()[bgmodels])*/); //PROVA CON POSTERIOR *bg->getVector_prior()[bgmodels]
                                }
                                //loop for FGmodels
                                for( int fgmodels=0; fgmodels<fg->getVector_EM().size(); ++fgmodels ){
                                    //get model
                                    ml::EM* em = fg->getVector_EM()[fgmodels];
                                    //get likelihood
                                    Mat probs(1, nClusters, CV_64FC1);
                                    Vec2d v = em->predict2(mat_pixel ,probs);
                                    superpixel_likelihoodsFG[fgmodels].push_back(v[0]/**(pixels.size()/(height*width))*/);//PROVA CON POSTERIOR *pixels.size()
                                }
                            }
                        
                            //alla fine del ciclo per tutti i pixel del superpixel calcolo le medie per ogni modello
                            for( int sizeModel=0 ; sizeModel<bg->getVector_EM().size(); ++sizeModel ){
                                double sum = std::accumulate(superpixel_likelihoodsBG[sizeModel].begin(), superpixel_likelihoodsBG[sizeModel].end(), 0.0);
                                double mean = sum / superpixel_likelihoodsBG[sizeModel].size();
                                vector_means.push_back(mean);
                            }
                            //get max mean of BgModels
                            double maxMeanBG = exp(*max_element(vector_means.begin(), vector_means.end()));
                            vector_means.clear();
                        
                            for( int sizeModel=0 ; sizeModel<fg->getVector_EM().size(); ++sizeModel ){
                                double sum = std::accumulate(superpixel_likelihoodsFG[sizeModel].begin(), superpixel_likelihoodsFG[sizeModel].end(), 0.0);
                                double mean = sum / superpixel_likelihoodsFG[sizeModel].size();
                                vector_means.push_back(mean);
                            }
                            //get max mean of FgModels
                            double maxMeanFG = exp(*max_element(vector_means.begin(), vector_means.end()));
                            vector_means.clear();
                        
                        
                            //usato per avere corrispondenza tra nodo graphcut e relativo superpixel
                            //pair<int,int>aPair2;
                            //aPair2.first=superpixel;//superpixel
                            //aPair2.second=id_node;//nodo graphcut
                            pair_superpixelsNode[superpixel] = id_node;
                        
                            double p_norm = maxMeanBG + maxMeanFG;
                              
                              //hash con chiave superpixel e valore le capacità verso source e sink
                              pair<double,double> aPair;//source-sink
                              aPair.first = (maxMeanBG/p_norm);//source
                              aPair.second = (maxMeanFG/p_norm);//sink
                              hash_likelihood[superpixel]= aPair;
                        
                            graph -> add_node(); //superpixels = i.first
                            graph -> add_tweights( /*label node*/ id_node,/* capacities */  -log(hash_likelihood[superpixel].first)/*source*/, -log(hash_likelihood[superpixel].second)/*sink*/);
                            //cout<<superpixel<<" | "<< maxMeanBG<<" - "<< maxMeanFG<<endl;
                    
                            //per visualizzare immagine binaria della stima per la finestra corrente
                            for(int i=0; i< currSuperpixelsMap.rows; ++i){
                                unsigned char *row_mask = mask_bg.ptr<unsigned char>(i);
                                unsigned char *row_mask2 = mask_fg.ptr<unsigned char>(i);
                                signed int *row_image_superpixels = currSuperpixelsMap.ptr<signed int>(i);
                            
                                for(int j=0; j< currSuperpixelsMap.cols; ++j){
                                    if(row_image_superpixels[j] == superpixel){
                                        row_mask[j] = 255*(hash_likelihood[superpixel].first);
                                        row_mask2[j] = 255*(hash_likelihood[superpixel].second);
                                    }
                                }
                            }
                            id_node+=1;
                              
                          }else{
                              //cout<<"non esiste ma già computato: "<< superpixel <<endl;

                              double p_norm = hash_likelihood[superpixel].first + hash_likelihood[superpixel].second;
                              
                              pair_superpixelsNode[superpixel] = id_node;
                              graph -> add_node(); //superpixels = i.first
                              graph -> add_tweights(/*label node*/ id_node,/* capacities */ -log(hash_likelihood[superpixel].first)/*source*/, -log(hash_likelihood[superpixel].second)/*sink*/);
                              
                              for(int i=0; i< currSuperpixelsMap.rows; ++i){
                                  unsigned char *row_mask = mask_bg.ptr<unsigned char>(i);
                                  unsigned char *row_mask2 = mask_fg.ptr<unsigned char>(i);
                                  signed int *row_image_superpixels = currSuperpixelsMap.ptr<signed int>(i);
                                  
                                  for(int j=0; j< currSuperpixelsMap.cols; ++j){
                                      if(row_image_superpixels[j] == superpixel){
                                          row_mask[j] = 255*(hash_likelihood[superpixel].first);
                                          row_mask2[j] = 255*(hash_likelihood[superpixel].second);
                                      }
                                  }
                              }
                              id_node+=1;
                          }
                      }else{
                          //cout<<"esiste e già computato: "<<superpixel<<endl;
                          //found
                          //do nothing
                          //just computed
                      }
                    }
                }
                cout <<"end computing likelihood source/sink graphcut"<< endl;
            
                //imshow( "Result mask_bg", mask_bg );
                //imshow( "Result mask_fg", mask_fg );
                //waitKey(0);
                
                //vettore con coppie di superpixels che usano la likelihood al posto dell'hellinger distance nell'appearance.
                vector<pair<int,int>> idx_likelihood;
            
                //calcolo POM e Hellinger distance/likelihood per graphcut
                for( int i=0; i< neighboursPair.size(); ++i ){
    
                    pair<int,int> aPair;
                    aPair.first = neighboursPair[i].first;
                    aPair.second = neighboursPair[i].second;
                    Mat mask1 = masking(currSuperpixelsMap,neighboursPair[i].first);
                    Mat mask2 = masking(currSuperpixelsMap,neighboursPair[i].second);
                
                    //POM
                    //check if POM edge was computed
                    if ( hash_edges_pom.find(aPair) == hash_edges_pom.end() ) {
                        // not found
                        //Calculate POM
                        vector<vector<Point>> contours ;        //contours union two superpixels
                        float pom = POM(neighboursPair, contours, currSuperpixelsMap, mask1, mask2, s, k, ns);
                        //cout << "POM: " << pom << endl;
                        pom_normalization.push_back(pom);
                        hash_edges_pom[aPair] = pom;
                    }else{
                        //found
                        pom_normalization.push_back(hash_edges_pom[aPair]);
                    }
                    
                
                    //APPEARANCE(LIKELIHOOD)
                    int size1 = countNonZero(mask1);
                    int size2 = countNonZero(mask2);

                    if(size1 >= n*size2 | size2 >= n*size1){
                        //cout<<"ENTRATO: "<<size1<<" - "<<size2<< endl;
                        //check if likelihood edge was computed
                        if( hash_edges_likelihood.find(aPair) == hash_edges_likelihood.end()){
                            
                            if(size1 >= n*size2){
                                vector<float> likelihoods;

                                //get pixels of superpixels 1 (the bigger)
                                vector<Vec3b> pixels1 = hash_pixels[neighboursPair[i].first];
                                Mat trainData(pixels1.size(), 3, CV_64F);
                                for(int l=0; l< trainData.rows; ++l){
                                    for(int j=0; j<trainData.cols; ++j){
                                        trainData.at<double>(l, j) = pixels1[l][j];
                                    }
                                }
                                //MOG of superpixel 1 (the bigger)
                                Ptr<ml::EM> em = ml::EM::create();
                                
                                //set Params model
                                em->setTermCriteria( TermCriteria( TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, FLT_EPSILON ));
                                em->setCovarianceMatrixType( ml::EM::COV_MAT_DIAGONAL);
                                em->setClustersNumber(nClusters);
                                //Train cluster with default weights,means,covs.
                                em->trainEM(trainData);

                                //get pixels of superpixels 2 (the smaller)
                                vector<Vec3b> pixels2 = hash_pixels[neighboursPair[i].second];
                                
                                for( int pixel=0; pixel<pixels2.size(); ++pixel ){
                                    Mat mat_pixel(1, 3, CV_64FC1);
                                    //pixel to Mat
                                    for(int x=0;x<3;x++){
                                        mat_pixel.at<double>(0,x) = pixels2[pixel][x];
                                    }
                                    //get likelihood
                                    Mat probs(1, nClusters, CV_64FC1);
                                    Vec2d v = em->predict2(mat_pixel ,probs);
                                    likelihoods.push_back((v[0]));
                                }
                                double sum = std::accumulate(likelihoods.begin(), likelihoods.end(), 0.0);
                                double mean = sum / likelihoods.size();
                                likelihood_normalization.push_back(mean);
                                hash_edges_likelihood[aPair] = mean;
                                idx_likelihood.push_back(aPair);
                                //cout<<"add_1: "<<aPair.first<<"-"<<aPair.second<<" | "<<mean<<endl;
                                
                            }else{ //size2 >=n*size1
                                vector<float> likelihoods;
                                //get pixels of superpixels 2 (the bigger)
                                vector<Vec3b> pixels2 = hash_pixels[neighboursPair[i].second];
                                Mat trainData(pixels2.size(), 3, CV_64F);
                                for(int l=0; l< trainData.rows; ++l){
                                    for(int j=0; j<trainData.cols; ++j){
                                        trainData.at<double>(l, j) = pixels2[l][j];
                                    }
                                }
                                //MOG of superpixel 2 (the bigger)
                                Ptr<ml::EM> em = ml::EM::create();
                                
                                //set Params model
                                em->setTermCriteria( TermCriteria( TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, FLT_EPSILON ));
                                em->setCovarianceMatrixType( ml::EM::COV_MAT_DIAGONAL);
                                em->setClustersNumber(nClusters);
                                //Train cluster with default weights,means,covs.
                                em->trainEM(trainData);
                            
                                //get pixels of superpixels 1 (the smaller)
                                vector<Vec3b> pixels1 = hash_pixels[neighboursPair[i].first];
                            
                                for( int pixel=0; pixel<pixels1.size(); ++pixel ){
                                    Mat mat_pixel(1, 3, CV_64FC1);
                                    //pixel to Mat
                                    for(int x=0;x<3;x++){
                                        mat_pixel.at<double>(0,x) = pixels1[pixel][x];
                                    }
                                    //get likelihood
                                    Mat probs(1, nClusters, CV_64FC1);
                                    Vec2d v = em->predict2(mat_pixel ,probs);
                                    likelihoods.push_back((v[0]));
                                }
                                double sum = std::accumulate(likelihoods.begin(), likelihoods.end(), 0.0);
                                double mean = sum / likelihoods.size();
                                likelihood_normalization.push_back(mean);
                                hash_edges_likelihood[aPair] = mean;
                                idx_likelihood.push_back(aPair);
                                //cout<<"add_2: "<<aPair.first<<"-"<<aPair.second<<" | "<<mean<<endl;

                            }
                        }else{
                            //found
                            idx_likelihood.push_back(aPair);
                            likelihood_normalization.push_back(hash_edges_likelihood[aPair]);
                            //cout<<"add_found: "<<aPair.first<<"-"<<aPair.second<<" | "<<endl;

                        }

                    } else {
                        //HELLINGER
                        if(hash_edges_hellinger.find(aPair) == hash_edges_hellinger.end()){
                            //not found
                            //PROVARE A PRENDERE PIXEL DIRETTAMENTE DA HASHMAP HASH_PIXEL
                            //get pixels from two superpixels
                            vector<vector<Point>> contours ;        //contours union two superpixels
                            Mat mask_union = Mat(currSuperpixelsMap.rows,currSuperpixelsMap.cols, CV_8UC1, uchar(0));
                            bitwise_or(mask1, mask2, mask_union); //mask union superpixels
                            findContours(mask_union ,contours , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
                            
                            vector<vector<Vec3b>> vector_object = getPixelsFromMask(currFrame , contours);
                            Mat trainData(vector_object[0].size(), 3, CV_64F);
                            for(int l=0; l< trainData.rows; ++l){
                                for(int j=0; j<trainData.cols; ++j){
                                    trainData.at<double>(l, j) = vector_object[0][l][j];
                                }
                            }
                            Ptr<ml::EM> em = ml::EM::create();
                            em = ExpectationMaximization(em, trainData, nClusters, 0, "Pair Superpixels");
                            
                            vector<float> discrete_pdf;
                            Mat probs(1, nClusters, CV_64FC1);
                            Mat sample(1, 3, CV_64FC1);
                            
                            vector<vector<float>> vector_discrete;
                            vector <float> discrete;
                            vector_discrete.push_back(discrete);
                            
                            //sampling pdf model with step of 25 to discretize it.
                            vector_discrete = DiscretizePdf(sample, probs, em, discrete_pdf,vector_discrete, 0);
                            
                            //calculate min Hellinger distance between BG and FG models
                            double hellinger_distance = HellingerDistance(vector_discrete, bg->getVector_EM(), fg->getVector_EM(), nClusters);
                            //cout<<"HL distance: "<< hellinger_distance <<endl;
                            hellinger_normalization.push_back(hellinger_distance);
                            hash_edges_hellinger[aPair] = hellinger_distance;
                            //cout<<"add_1_hell: "<<aPair.first<<"-"<<aPair.second<<" | "<<hellinger_distance<<endl;

                        }else{
                            //found
                            hellinger_normalization.push_back(hash_edges_hellinger[aPair]);
                            //cout<<"add_found_hell: "<<aPair.first<<"-"<<aPair.second<<" | "<<endl;

                        }
                    }
                    
                    cout<<".";
                }//end pom and hellinger/likelihood
            
                //normalizzo pom e hellinger distance e likelihood
                float min_pom = *min_element(pom_normalization.begin(), pom_normalization.end());
                float max_pom = *max_element(pom_normalization.begin(), pom_normalization.end());
                
                double min_likelihood=0;
                double max_likelihood=0;
                double den_likelihood=0;
                double min_hellinger=0;
                double max_hellinger=0;
                double den_hellinger=0;
                
                if(hellinger_normalization.empty()){
                    
                }else{
                    min_hellinger = *min_element(hellinger_normalization.begin(), hellinger_normalization.end());
                    max_hellinger = *max_element(hellinger_normalization.begin(), hellinger_normalization.end());
                    den_hellinger = max_hellinger - min_hellinger;
                }
                
                if(likelihood_normalization.empty()){
                    
                }else{
                    min_likelihood = *min_element(likelihood_normalization.begin(), likelihood_normalization.end());
                    max_likelihood = *max_element(likelihood_normalization.begin(), likelihood_normalization.end());
                    den_likelihood = max_likelihood - min_likelihood;
                }
                
            
                //normalization pom and hellinger:  norm = norm/den
                float den_pom = max_pom - min_pom;
                
                //save edge between two superpixels   ////////////////////////
                //hash_edges[aPair] =

                //ciclo per assegnare il valore finale dell'edge alle coppie di superpixels.
                for( int i=0; i< neighboursPair.size(); ++i ){
                    float pom_norm = (hash_edges_pom[neighboursPair[i]] - min_pom)/den_pom;

                    if(std::find(idx_likelihood.begin(), idx_likelihood.end(), neighboursPair[i]) != idx_likelihood.end()){
                        if(hash_edges_likelihood.find(neighboursPair[i]) == hash_edges_likelihood.end()){
                            //not found
                        }else{
                            float likelihood_norm = 0;
                            if(den_likelihood == 0)likelihood_norm = p_edge;
                            else likelihood_norm = (hash_edges_likelihood[neighboursPair[i]] - min_likelihood)/den_likelihood;
                            
                            graph -> add_edge( pair_superpixelsNode[neighboursPair[i].first], pair_superpixelsNode[neighboursPair[i].second],/* capacities */ (a1*likelihood_norm+a2*pom_norm)/2, (a1*likelihood_norm+a2*pom_norm)/2 );
                            //exp(-(1-(a1*likelihood_norm+a2*pom_norm)/2)), exp(-(1-(a1*likelihood_norm+a2*pom_norm)/2 )))
                        }
                    }else{
                        float hellinger_norm = (hash_edges_hellinger[neighboursPair[i]] - min_hellinger)/den_hellinger;

                        graph -> add_edge( pair_superpixelsNode[neighboursPair[i].first], pair_superpixelsNode[neighboursPair[i].second],/* capacities */ ((a1*hellinger_norm+a2*pom_norm)/2), ((a1*hellinger_norm+a2*pom_norm)/2) );
                        //exp(-(1-(a1*hellinger_norm+a2*pom_norm)/2)), exp(-(1-(a1*hellinger_norm+a2*pom_norm)/2 )))

                    }
                }
                
                cout << "end computing edges graphcut" << endl;
            
                //calculate maxflow graphcut
                int flow = graph -> maxflow();
                printf("Flow GraphCut = %d\n", flow);
            
                Mat tmp_mask_iteration = Mat(currSuperpixelsMap.rows,currSuperpixelsMap.cols, CV_8UC1, char(0)); //CV_8UC1 for findContours
                for (auto k : pair_superpixelsNode){
                    for(int i=0; i< currSuperpixelsMap.rows; ++i){
                        unsigned char *row_mask = tmp_mask_iteration.ptr<unsigned char>(i);
                        signed int *row_image_superpixels = currSuperpixelsMap.ptr<signed int>(i);
                
                        for(int j=0; j< currSuperpixelsMap.cols; ++j){
                            if(row_image_superpixels[j] == k.first){
                            
                                if (graph->what_segment(k.second) == GraphType::SOURCE){
                                    row_mask[j] = 255;
                                }
                                else{
                                    row_mask[j] = 0;
                                }
                            }
                        }
                    }
                }
                
                //prende contorno finestra corrente e la disegna su final_mask
                vector<vector<Point>> tmp_contours;
                findContours(tmp_mask_iteration,tmp_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
                for( int i = 0; i< tmp_contours.size(); ++i ){
                    Scalar color( 255, 255, 255 );
                    drawContours( final_mask, tmp_contours, i, color, CV_FILLED);
                }
            
            }//end current window
            
            //show binary image current iteraction
            //imshow( to_string(iteration), final_mask );
            //waitKey(0);

            if(matIsEqual(tmp_mask, final_mask)){
                cout<<"end while"<<endl;
                break;
            }else{
                cout<<"loop while iteration: "<< iteration <<endl;
                //creo maschera temporanea e contorno per ciclo while
                tmp_mask = final_mask.clone();
                contours_mask_filtered.clear();
                findContours(final_mask,contours_mask_filtered, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            }
            
            
            int new_superpixel = numberOfLabels + 1;
            
            //etichetto i pixel 1 della maschera in uscita come un nuovo pixel per ogni connected component
            for(int c=0; c< contours_mask_filtered.size(); ++c){
                
                Mat a = Mat::zeros( movingSuperpixelsMap2.size(), CV_8UC1 );
                drawContours( a, contours_mask_filtered, c, Scalar(255), CV_FILLED);
                
                vector<Vec3b> pixels;

                //loop per settare sia la nuova label dei superpixels nella mappa dei superpixels e sia per prendere i pixels relativi al nuovo superpixel.
                for(int i=0; i< currSuperpixelsMap.rows; ++i){
                    unsigned char *row_mask = a.ptr<unsigned char>(i);
                    signed int *row_image_superpixels = currSuperpixelsMap.ptr<signed int>(i);
                    const Vec3b* p_image = currFrame.ptr<Vec3b>(i);

                    for(int j=0; j< currSuperpixelsMap.cols; ++j){
                        if(row_mask[j] > 0){
                            pixels.push_back(p_image[j]);
                            row_image_superpixels[j] = new_superpixel;
                        }
                    }
                }
                //aggiungo i pixels del nuovo superpixels nell'hashmap
                hash_pixels[new_superpixel] = pixels;
                //cout<<hash_pixels[new_superpixel].size()<<endl;

                //setto gli edge verso source e sink del nuovo superpixel
                pair<double,double> aPair;//source-sink
                aPair.first = 0;//source
                aPair.second = 1;//sink
                hash_likelihood[new_superpixel]= aPair;
                
                numberOfLabels+=1;

            }
            //Mat b = masking(currSuperpixelsMap, 2565);
            
            iteration += 1;
            
        }//end while
        
        Mat a = images_groundTruth[frame];
        a = a > 127; //threshold to convert image grounTruth in a binary image.
        cvtColor(a, a, CV_RGB2GRAY);
        vector<vector<Point>> cc;
        findContours(a,cc, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        Mat binary_groundTruth = Mat::zeros( currSuperpixelsMap.size(), CV_8UC1 );
        for(int i=0; i< cc.size();++i){
            drawContours( binary_groundTruth, cc, i, Scalar(255), CV_FILLED);
        }

        Mat mat_true_positive = Mat::zeros( currSuperpixelsMap.size(), CV_8UC1 );
        Mat mat_false_positive = Mat::zeros( currSuperpixelsMap.size(), CV_8UC1 );
        Mat mat_false_negative = Mat::zeros( currSuperpixelsMap.size(), CV_8UC1 );

        //TruePositive
        bitwise_and(tmp_mask, binary_groundTruth, mat_true_positive);
        
        //FalsePositive
        Mat not_groundThuth = Mat::zeros( currSuperpixelsMap.size(), CV_8UC1 );
        bitwise_not(binary_groundTruth, not_groundThuth);
        bitwise_and(not_groundThuth, tmp_mask, mat_false_positive);
        
        //FalseNegative
        Mat not_mask = Mat::zeros( currSuperpixelsMap.size(), CV_8UC1 );
        bitwise_not(tmp_mask, not_mask);
        bitwise_and(binary_groundTruth, not_mask, mat_false_negative);

        float true_positive = countNonZero(mat_true_positive);
        float false_positive = countNonZero(mat_false_positive);
        float false_negative = countNonZero(mat_false_negative);
        
        //cout<<"TP:"<<true_positive<<endl;
        //cout<<"FP:"<<false_positive<<endl;
        //cout<<"FN:"<<false_negative<<endl;

        total_tp+=true_positive;
        total_fp+=false_positive;
        total_fn+=false_negative;
        
        Mat final = Mat::zeros( currFrame.size(), CV_8UC3 );
        Mat tmp = tmp_mask.clone();
        cvtColor(tmp, tmp, CV_GRAY2BGR);
        addWeighted( currFrame, alpha, tmp, beta, 0.0, final);
        //imshow( "TP", mat_true_positive );
        //imshow( "FP", mat_false_positive );
        //imshow( "FN", mat_false_negative );
        
        //update background/foreground models
        //bg->update(currFrame, tmp_mask, nClusters);//CONTROLLARE POICHE' IN FASE DI TEST E' CV_64F e non CV_8UC3
        //fg->update(currFrame, tmp_mask, nClusters);
        
        //F-measure
        float precision = total_tp/(total_tp+total_fp);
        float recall = total_tp/(total_tp+total_fn);
        float num = 2 * precision * recall;
        float den = precision + recall;
        float Fmeasure = num/den;
        cout<<"Precision: "<< precision<<endl;
        cout<<"Recall:"<<recall<<endl;
        cout<< "F-measure: " << Fmeasure <<endl;
        
        imshow( "Final mask", final );
        waitKey(0);
        
    }
    
    
    return 0;
}