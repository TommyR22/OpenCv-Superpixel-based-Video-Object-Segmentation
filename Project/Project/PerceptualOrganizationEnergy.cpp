//
//  PerceptualOrganizationEnergy.cpp
//  Project
//
//  Created by Tommaso Ruscica on 05/11/15.
//  Copyright © 2015 Tommaso Ruscica. All rights reserved.
//

#include "PerceptualOrganizationEnergy.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <numeric>
#include <complex>      // std::complex, std::norm
#include "cvblob.h"
#include "Superpixels.h"

#define PI 3.14159265;

using namespace std;
using namespace cv;

//--------------------------------------------------------//
//                      COMPLEXITY                        //
//--------------------------------------------------------//
// Compute boundary complexity
float boundaryComplexity(const Mat& patch_mask, const vector<Point>& contour, int s, int k, int sn)
{
    // Remove last point if it's the same as the first one
    vector<Point> c = contour;
    if(c[0] == c[c.size()-1])
    {
        c.erase(c.end()-1);
    }
    // Initialize result
    float B = 0.0f;
    // Get number of points
    int N = c.size();
    //cout << "N: " << N << endl;
    // Compute notch map
    bool* notch_map = new bool[N];
    for(int i=0; i<N; i++)	notch_map[i] = 0;
    // Process each point and look for notches
    for(int j=0; j<N; j += sn)
    {
        // Get current, previous and next points
        Point curr = c[j];
        Point prev = c[wrap(j-sn, N)];
        Point next = c[wrap(j+sn, N)];
        // Compute vectors
        Point v1 = prev - curr;
        Point v2 = next - curr;
        // Compute angle
        float cross = abs(v1.x*v2.y - v1.y*v2.x);
        float dot = v1.x*v2.x + v1.y*v2.y;
        float angle = abs(fastAtan2(cross, dot));
        if(angle > 180.0f) angle -= 180.0f;
        // Check is points are more or less aligned
        if(angle < 30.0f || angle > 150.0f)
        {
            // Skip
            continue;
        }
        // Get average point between the two extemities
        Point2f mid((prev.x+next.x)/2.0f, (prev.y+next.y)/2.0f);
        int mid_xs[] = {(int)floor(mid.x), (int)ceil(mid.x)};
        int mid_ys[] = {(int)floor(mid.y), (int)ceil(mid.y)};
        Point cand_1(mid_xs[0], mid_ys[0]);
        Point cand_2(mid_xs[0], mid_ys[1]);
        Point cand_3(mid_xs[1], mid_ys[0]);
        Point cand_4(mid_xs[1], mid_ys[1]);
        if     (cand_1 != prev && cand_1 != curr && cand_1 != next)	mid = cand_1;
        else if(cand_2 != prev && cand_2 != curr && cand_2 != next)	mid = cand_2;
        else if(cand_3 != prev && cand_3 != curr && cand_3 != next)	mid = cand_3;
        else if(cand_4 != prev && cand_4 != curr && cand_4 != next)	mid = cand_4;
        else continue; // Could not decide, skip point
        // Check if mid is inside or outside the object
        if(patch_mask.at<uchar>(mid.y, mid.x) == 0)
        {
            // Notch found
            notch_map[j] = 1;
        }
    }
    // Process each point and look for notches
    for(int d=0; d<N; d++)
    {
        // Compute A_sk
        //cout << "p_d: " << c[d].x << "," << c[d].y << endl;
        //cout << "p_d_ks: " << c[wrap(d+k*s,N)].x << "," << c[wrap(d+k*s,N)].y << endl;
        float A_num = pointDistance(c[d], c[wrap(d+s*k,N)]);
        float A_den = 0.0f;
        for(int j=1; j<=k; ++j)
        {
            float distance_to_next = pointDistance(c[wrap(d+j*s,N)], c[wrap(d+(j-1)*s,N)]);
            A_den += distance_to_next;
        }
        float A = 1.0f - A_num/A_den;   //--> A_sk
        //cout << "A: " << A << endl;
        // Count number of notches in the window
        float n = 0;    /////////////////////////////////0-1-2 come valori, quindi F_sk viene 0.
        // Compute notch window interval
        int n_start = d;
        int n_end = d+k*s;
        // Compute number of notches
        for(int i=n_start; i<=n_end; i++)
        {
            // Wrap i
            int wi = wrap(i,N);
            // Check notch
            if(notch_map[wi])
            {
                // Add notch
                n++;
            }
        }
        //cout<<"n: "<<n<<endl;
        // Compute F_sk
        float F = 1.0f - 2.0f*abs(0.5 - n/(N-3));   //--> F_sk
        // Add to B
        B += A*F;   //--> B
    }
    // Free notch map
    delete [] notch_map;
    // Compute final value for B
    B /= N;
    // Return result
    return B;
    
}


//--------------------------------------------------------//
//                        SIMMETRY                        //
//--------------------------------------------------------//
double calculateSimmetry(Mat image_superpixels1 , Mat image_superpixels2){
    double simmetry;
    
    vector<vector<Point>> contours_mask1 ;
    vector<vector<Point>> contours_mask2 ;

    findContours(image_superpixels1 ,contours_mask1 , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
    findContours(image_superpixels2 ,contours_mask2 , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);

    Point center1 = centroid(contours_mask1[0]);
    Point center2 = centroid(contours_mask2[0]);
    
//    vector<Point> prova;
//    prova.push_back(center1);
//    prova.push_back(center2);
    
//    cout<<center1<<endl;
//    cout<<center2<<endl;

    //compute simmetry
    simmetry = min(abs(center1.x - center2.x),1) * min(abs(center1.y - center2.y),1);

    
//    Point centro1 = centroid(contours_mask1[0]);
//    Point centro2 = centroid(contours_mask2[0]);
//
//    Mat drawing1 = Mat::zeros( image_superpixels1.size(), CV_8UC3 );
//
//    for( int i = 0; i< contours_mask1.size(); i++ ){
//        Scalar color( 255, 255, 255 );
//        drawContours( drawing1, contours_mask1, i, color, CV_FILLED);
//    }
//    for( int i = 0; i< contours_mask2.size(); i++ ){
//        Scalar color( 255, 255, 0 );
//        drawContours( drawing1, contours_mask2, i, color, CV_FILLED);
//    }
//    
//    Scalar color( 0, 0, 255 );
//    circle(drawing1, centro1, 1, color,CV_FILLED);
//    circle(drawing1, centro2, 1, color,CV_FILLED);
//
//    imshow( "Result draw", drawing1 );
//    waitKey(0);
    
    //PROVA SE ALLINEATI I CENTROIDI
//    Vec4f lines;
//    fitLine (prova,lines,2,0,0.01,0.01);
//    int lefty = (-lines[2]*lines[1]/lines[0])+lines[3];
//    int righty = ((image_superpixels1.cols-lines[2])*lines[1]/lines[0])+lines[3];
//    
//    line(drawing1,Point(image_superpixels1.cols-1,righty),Point(0,lefty),Scalar(255,255,255),2);
//    
//    imshow( "Result common", drawing1 );
//    waitKey(0);
    
    return simmetry;
}


//--------------------------------------------------------//
//               CONTINUITY & PROXIMITY                   //
//--------------------------------------------------------//
int CV_POSITION_ORIGINAL = 0;
int CV_POSITION_TOP_LEFT = 1;

int calculateContinuity(Point P1, Point P2 , Mat mask){
    
    int continuity = 0;
    
    int x1 = P1.x;
    int y1 = P1.y;
    int x2 = P2.x;
    int y2 = P2.y;
    int minx = min(x1, x2);
    int maxx = max(x1, x2);
    int miny = min(y1, y2);
    int maxy = max(y1, y2);
    int a = y2 - y1;
    int b = x2 - x1;
    
    vector<Point> points_out;
    
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if(mask.at<uchar>(i, j) == 255 ){
                Point p(i,j);
                if(p.y >= miny && p.y <= maxy){
                    //no calculation for pixel between two points.(y)
                }else if(p.x >= minx && p.x <= maxx){
                    //no calculation for pixel between two points.(x)
                }else{
                    points_out.push_back(p);
                }
            }
        }
    }
    //cout<<"SIZE:"<<points_out.size()<<endl;
    for( int y=0; y<points_out.size(); y++){
        if(b != 0){
            float m = a / b;
            float den = sqrt(1 + pow(m,2));
            float q = -x1 * m + y1;
            float d = abs(points_out[y].y - m * points_out[y].x - q)/sqrt(den) ;
            //cout<<"d: "<<d<<endl;
            if(d > 0.5){
                continuity = 0;
            }else{
                continuity = 1;
                break;
            }
        }else{
            //cout<<"b=0"<<endl;
            if(abs(points_out[y].x - x1) > 1){
                continuity = 0;
            }else{
                continuity = 1;
                break;
            }
        }
    }
    return continuity;
}

//Compute common boundary
vector<Point> commonBoundary_2(const cvb::CvBlob& blob1, const cvb::CvBlob& blob2)
{
    int init = 0;
    Point init_point;
    Point last_point;

    // Render blob2, without filling
    int width = (blob1.maxx > blob2.maxx ? blob1.maxx : blob2.maxx) + 3;
    int height = (blob1.maxy > blob2.maxy ? blob1.maxy : blob2.maxy) + 3;
    Mat mask2 = blobToMat(blob2, CV_POSITION_ORIGINAL, false, width, height);
    // Convert mask1 to point vector
    vector<Point> contour1 = blobToPointVector(blob1);
    // Initialize common boundary image
    Mat common = Mat::zeros(height, width, CV_8UC1);
    // Each contour point
    vector<Point> p_vect;   //vector with points of neighbouring contour
    for(vector<Point>::iterator it = contour1.begin(); it != contour1.end(); ++it)
    {
        // Get point
        Point p = *it;
        // Check if we have a neighbouring contour point in mask2
        int n_x = (p.x > 0 ? p.x-1 : p.x);
        int n_y = (p.y > 0 ? p.y-1 : p.y);
        int n_h = (n_y+2 < height ? 3 : height-n_y);
        int n_w = (n_x+2 < width ? 3 : width-n_x);
        Mat neighbourhood = mask2(Rect(n_x, n_y, n_w, n_h));
        if(binaryArea(neighbourhood) > 0)
        {
            p_vect.push_back(p);
//            if(init == 0){
//                init_point = p;
//                common.at<uchar>(p.y, p.x) = 255;
//                init++; //TOGLIERLO PER VEDERE TUTTA LA LINEA
//            }else{
//                last_point = p;
//            }
        }
    }
    
    common.at<uchar>(last_point.y, last_point.x) = 255;
    
    //get two point of common boundary (first and last)
    //vector<Point> p_vect;
    //p_vect.push_back(init_point);
    //p_vect.push_back(last_point);
    
    // TOGLIERLO PER VEDERE SOLI I DUE PUNTI
//    Vec4f lines;
//    fitLine (p_vect,lines,DIST_L1,0,0.01,0.01);
//    int lefty = (-lines[2]*lines[1]/lines[0])+lines[3];
//    int righty = ((common.cols-lines[2])*lines[1]/lines[0])+lines[3];
//    line(common,Point(common.cols-1,righty),Point(0,lefty),Scalar(255,255,255),1);
//    line(common,init_point,last_point,Scalar(255,255,255),1);

    // Return common boundary
    //imshow( "common2", common );
    //waitKey(0);
    
    return p_vect;
}


// Compute common boundary ( for cohesiveness strength )
Mat commonBoundary_1(const cvb::CvBlob& blob1, const cvb::CvBlob& blob2)
{
    // Render blob2, without filling
    int width = (blob1.maxx > blob2.maxx ? blob1.maxx : blob2.maxx) + 3;
    int height = (blob1.maxy > blob2.maxy ? blob1.maxy : blob2.maxy) + 3;
    Mat mask2 = blobToMat(blob2, CV_POSITION_ORIGINAL, false, width, height);
    // Convert mask1 to point vector
    vector<Point> contour1 = blobToPointVector(blob1);
    // Initialize common boundary image
    Mat common = Mat::zeros(height, width, CV_8UC1);
    // Each contour point
    for(vector<Point>::iterator it = contour1.begin(); it != contour1.end(); ++it)
    {
        // Get point
        Point p = *it;
        // Check if we have a neighbouring contour point in mask2
        int n_x = (p.x > 0 ? p.x-1 : p.x);
        int n_y = (p.y > 0 ? p.y-1 : p.y);
        int n_h = (n_y+2 < height ? 3 : height-n_y);
        int n_w = (n_x+2 < width ? 3 : width-n_x);
        Mat neighbourhood = mask2(Rect(n_x, n_y, n_w, n_h));
        if(binaryArea(neighbourhood) > 0)
        {
            common.at<uchar>(p.y, p.x) = 255;
        }
    }
    // Return common boundary
    //imshow( "common1", common );
    //waitKey(0);
    
    return common;
}


// Compute cohesiveness strength between two patches    // PROXIMITY //
float cohesivenessStrength(const cvb::CvBlob& patch_1, const cvb::CvBlob& patch_2)
{
    // Compute lambda
    float alpha = 20.0f;
    float beta = 3.0f;
    float lambda;
    // Compute boundary lengths
    int L_1 = patch_1.contour.chainCode.size();
    int L_2 = patch_2.contour.chainCode.size();
    Mat common_boundary = commonBoundary_1(patch_1, patch_2);
    int L_12 = binaryArea(common_boundary);
    // Compute lambda (proximity)
    //cout<<"L12 : "<<L_12<<endl;
    //check if L_1 is greater 3 times of L_2        ///SOSTITUIRE L_1 e L_2 SOLO NEI BRANCH CON IL NUM DI PIXELS DEI SUPERPIXELS
    if (L_1 < L_2*3) {
        lambda = beta*(exp(-alpha * L_12 / (L_1)*2));
    }else if (L_2 < L_1*3) {
        lambda = beta*(exp(-alpha * L_12 / (L_2)*2));
    }else{
        lambda = beta*exp(-alpha*L_12/(L_1+L_2));
    }
    

    return lambda;
}


//--------------------------------------------------------//
//                          POM                           //
//--------------------------------------------------------//
float calculatePOM(const cvb::CvBlob& patch_1, const cvb::CvBlob& patch_2, float complexity , float simmetry, int continuity, float proximity)
{

    float pom;
    int t1 = 18;
    float t2 = 3.5;
    
    // Compute boundary lengths
    int L_1 = patch_1.contour.chainCode.size();     
    int L_2 = patch_2.contour.chainCode.size();
    Mat common_boundary = commonBoundary_1(patch_1, patch_2);
    int L_12 = binaryArea(common_boundary);

    //check if L_1 is greater 3 times of L_2
    if (L_1 > L_2*3) {
        pom = exp(-(t1 * complexity + t2* proximity));
    }else if (L_2 > L_1*3) {
        pom = exp(-(t1 * complexity + t2* proximity));
    }else{
        pom = exp(-(t1 * complexity + t2* (simmetry + continuity) * proximity));
    }
    
    return pom;
}


Mat blobToMat(const cvb::CvBlob& blob, int position, bool filled, int output_width, int output_height)
{
//    // Check blob is valid
//    if(!isBlobValid(blob))
//    {
//        //throw MyException("Can't convert CvBlob to Mat, invalid blob");
//    }
    // Compute height and width of the output image size (add 1 padding pixel, otherwise drawContours won't work)
    if(position == CV_POSITION_ORIGINAL)
    {
        if((output_width > 0 && (unsigned int) output_width <= blob.maxx) || (output_height > 0 && (unsigned int) output_height <= blob.maxy))
        {
            stringstream error;
            error << "Output width (" << output_width << ") or height (" << output_height << ") too small for blob ([" << blob.x << "-" << blob.maxx << ", " << blob.y << "-" << blob.maxy << "]).";
            //throw MyException(error.str());
        }
    }
    else if(position == CV_POSITION_TOP_LEFT)
    {
        if((output_width > 0 && (unsigned int) output_width < blob.width()) || (output_height > 0 && (unsigned int) output_height < blob.height()))
        {
            stringstream error;
            error << "Output width (" << output_width << ") or height (" << output_height << ") too small for blob (" << blob.width() << "x" << blob.height() << ").";
            //throw MyException(error.str());
        }
    }
    else
    {
        //throw MyException("Invalid blob position value.");
    }
    int width = (output_width > 0 ? output_width : (position == CV_POSITION_ORIGINAL ? blob.maxx + 1 : blob.width())) + 2;
    int height = (output_height > 0 ? output_height : (position == CV_POSITION_ORIGINAL ? blob.maxy + 1 : blob.height())) + 2;
    // Create output image
    Mat drawn = Mat::zeros(height, width, CV_8UC3);
    // Compute offsets from top-left corner (with padding pixel)
    int x_offset = (position == CV_POSITION_ORIGINAL ? 0 : blob.minx) - 1;
    int y_offset = (position == CV_POSITION_ORIGINAL ? 0 : blob.miny) - 1;
    // Draw blob contour
    // Get starting point
    Point current = blob.contour.startingPoint;
    drawn.at<Vec3b>(current.y - y_offset, current.x - x_offset) = Vec3b(255,255,255);
    // Get chain code contour
    const cvb::CvChainCodes& chain_code = blob.contour.chainCode;
    for(cvb::CvChainCodes::const_iterator it=chain_code.begin(); it != chain_code.end(); it++)
    {
        current.x += cvb::cvChainCodeMoves[*it][0];
        current.y += cvb::cvChainCodeMoves[*it][1];
        drawn.at<Vec3b>(current.y - y_offset, current.x - x_offset) = Vec3b(255,255,255);
    }
    if(filled)
    {
        // Fill contour
        vector<vector<Point> > contours;
        Mat drawn_gs;
        cvtColor(drawn, drawn_gs, CV_BGR2GRAY);
        findContours(drawn_gs, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        for(unsigned int i=0; i<contours.size(); i++)
        {
            drawContours(drawn, contours, i, CV_RGB(255,255,255), -1);//CV_FILLED, 8);
        }
        
    }
    // Convert to 8-bit
    Mat final;
    cvtColor(drawn, final, CV_BGR2GRAY);
    // Remove padding pixel
    Rect selection(1, 1, width-2, height-2);
    Mat selected = final(selection);
    // Return result
    return selected;
}



int binaryArea(const Mat& img)
{
    // Check channel number
    if(img.channels() > 1)
    {
        //throw MyException("binaryArea(): input must be single-channel image.");
    }
    // Compute area, i.e. number of elements greater than 0
    int area = 0;
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            if(img.at<uchar>(i,j) > 0)
                //if(*(img.data + img.step[0]*i + img.step[1]*j) > 0)
            {
                area++;
            }
        }
    }
    // Return result
    return area;
}

vector<Point> blobToPointVector(const cvb::CvBlob& blob)
{
    // Define output vector
    vector<Point> contour;
    // Get starting point
    Point current = blob.contour.startingPoint;
    // Get chain code contour
    const cvb::CvChainCodes& chain_code = blob.contour.chainCode;
    // Add points
    contour.push_back(current);
    //int num_points = chain_code.size();
    for(cvb::CvChainCodes::const_iterator it=chain_code.begin(); it != chain_code.end(); it++)
    {
        current.x += cvb::cvChainCodeMoves[*it][0];
        current.y += cvb::cvChainCodeMoves[*it][1];
        contour.push_back(current);
    }
    // Return result
    return contour;
}

// Create blob from point vector
cvb::CvBlob* createBlob(const vector<Point>& contour)
{
    // Compute bounding box
    vector<Point> bounding_box = boundingBoxFromContour(contour);
    // Call other function
    return createBlob(bounding_box, contour);
}

// Compute bounding box from contour
vector<Point> boundingBoxFromContour(const vector<Point>& contour)
{
    // Initialize minimum and maximum coordinates
    int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;
    // Go through each point
    for(vector<Point>::const_iterator it = contour.begin(); it != contour.end(); ++it)
    {
        if(it->x < min_x)	min_x = it->x;
        if(it->y < min_y)	min_y = it->y;
        if(it->x > max_x)	max_x = it->x;
        if(it->y > max_y)	max_y = it->y;
    }
    // Create and fill output vector
    vector<Point> bounding_box;
    bounding_box.push_back(Point(min_x, min_y));
    bounding_box.push_back(Point(max_x, min_y));
    bounding_box.push_back(Point(max_x, max_y));
    bounding_box.push_back(Point(min_x, max_y));
    bounding_box.push_back(Point(min_x, min_y));
    // Return result
    return bounding_box;
}

// Calls fixContour() on contour
// Both bounding_box and contour can have the last point equal to che first point
cvb::CvBlob* createBlob(const vector<Point>& bounding_box, const vector<Point>& contour)
{
    // Check both arrays are non-empty
    if(bounding_box.size() == 0 || contour.size() == 0)
    {
        //throw MyException("Bounding box/contour vectors can't be empty");
    }
    // Allocate blob
    cvb::CvBlob *blob = new cvb::CvBlob();
    // Compute bounding box borders
    int minx = INT_MAX, maxx = 0, miny = INT_MAX, maxy = 0;
    for(vector<Point>::const_iterator it = bounding_box.begin(); it != bounding_box.end(); it++)
    {
        //cout << it->x << " " << it->y << endl;
        if(it->x < minx)	minx = it->x;
        if(it->x > maxx)	maxx = it->x;
        if(it->y < miny)	miny = it->y;
        if(it->y > maxy)	maxy = it->y;
    }
    // Check bounding box coordinates
    if(minx < 0 || miny < 0 || maxx < 0 || maxy < 0)
    {
        //throw MyException("Bounding box has negative coordinates");
    }
    blob->minx = minx;
    blob->maxx = maxx;
    blob->miny = miny;
    blob->maxy = maxy;
    // Fix contour
    vector<Point> fixed_contour = fixContour(contour);
    // Compute chaincode contour
    cvb::CvContourChainCode cc_contour;
    cc_contour.startingPoint.x = contour[0].x;
    cc_contour.startingPoint.y = contour[0].y;
    Point last = contour[0];
    for(vector<Point>::iterator it = fixed_contour.begin()+1; it != fixed_contour.end(); it++)
    {
        //cout << "c point: " << it->x << " " << it->y << endl;
        // Check points are different
        if(*it != last)
        {
            // Get corresponding chain code
            uchar code = pointDifferenceToChainCode(last, *it);
            //cout << "prev: (" << last.x << ", " << last.y << "), next: (" << it->x << ", " << it->y << "), code: " << (int)code << endl;
            // Add to contour
            cc_contour.chainCode.push_back(code);
            // Save last point
            last = *it;
        }
    }
    // Set contour chain code
    blob->contour = cc_contour;
    // Compute blob properties
    blob->area = 0;
    blob->m10 = 0;
    blob->m01 = 0;
    blob->m11 = 0;
    blob->m20 = 0;
    blob->m02 = 0;
    Mat blob_mat = blobToMat(*blob);
    for(int i=0; i<blob_mat.rows; ++i)
    {
        for(int j=0; j<blob_mat.cols; ++j)
        {
            if(blob_mat.at<uchar>(i,j) > 0)
            {
                blob->area++;
                blob->m10 += j;
                blob->m01 += i;
                blob->m11 += i*j;
                blob->m20 += j*j;
                blob->m02 += i*i;
            }
        }
    }
    blob->centroid.x = blob->x + blob->m10/blob->area;
    blob->centroid.y = blob->y + blob->m01/blob->area;
    // TODO FIXME compute also other properties
    // Return blob
    return blob;
}

unsigned char pointDifferenceToChainCode(Point last, Point next)
{
    int dx = next.x - last.x;
    int dy = next.y - last.y;
    if(dx == 0 && dy == -1)		return CV_CHAINCODE_UP;
    if(dx == 1 && dy == -1)		return CV_CHAINCODE_UP_RIGHT;
    if(dx == 1 && dy == 0)		return CV_CHAINCODE_RIGHT;
    if(dx == 1 && dy == 1)		return CV_CHAINCODE_DOWN_RIGHT;
    if(dx == 0 && dy == 1)		return CV_CHAINCODE_DOWN;
    if(dx == -1 && dy == 1)		return CV_CHAINCODE_DOWN_LEFT;
    if(dx == -1 && dy == 0)		return CV_CHAINCODE_LEFT;
    if(dx == -1 && dy == -1)	return CV_CHAINCODE_UP_LEFT;
    stringstream error;
    
    error << "Trying to get chain-code of non adjacent points ([ " << last.x << " " << last.y << "], [" << next.x << " " << next.y << "]).";
    //throw Exception(error.str());
    return 10;  ////////
}

// Show blob
void showBlob(const string& window, const cvb::CvBlob& blob, int position, bool filled, int output_width, int output_height)
{
    // Draw blob
    Mat blob_mat = blobToMat(blob, position, filled, output_width, output_height);
    // Show image
    imshow(window, blob_mat);
}


// Get centroid point from point vector
Point centroid(const vector<Point>& points)
{
    // Fix contour
    vector<Point> contour = fixContour(points);
    // Initialize minimum and maximum coordinates
    int avg_x = 0;
    int avg_y = 0;
    // Go through all points
    for(vector<Point>::const_iterator p_it = contour.begin(); p_it != contour.end(); p_it++)
    {
        avg_x += p_it->x;
        avg_y += p_it->y;
    }
    // Return center
    return Point(avg_x/contour.size(), avg_y/contour.size());
}


// Draw contour
void drawContour(const vector<Point>& points, Mat& frame, Scalar color, int thickness_or_filled)
{
    // Check contour
    vector<Point> contour = fixContour(points);
    // Build contour vector
    vector<vector<Point> > contours;
    contours.push_back(contour);
    // Call OpenCv drawing function
    drawContours(frame, contours, 0, color, thickness_or_filled);//CV_FILLED, 8);
    
}

vector<Point> fixContour(const vector<Point>& contour)
{
    vector<Point> new_contour;
    Point start = contour[0];
    Point last = start;
    new_contour.push_back(last);
    unsigned int i = 1;
    while(i < contour.size())
    {
        Point next = contour[i];
        if(next == last)
        {
            i++;
            continue;
        }
        if(inNeighbourhood(last, next))
        {
            new_contour.push_back(next);
            last = next;
            i++;
        }
        else
        {
            //cout << "point 1: " << last.x << " " << last.y << endl;
            //cout << "point 2: " << next.x << " " << next.y << endl;
            // Get points between
            vector<Point> between = getPointsBetween(last, next);
            //for(vector<Point>::iterator it = between.begin(); it != between.end(); it++)
            //	cout << "between: " << it->x << " " << it->y << endl;
            new_contour.insert(new_contour.end(), between.begin(), between.end());
            //cout << "new_contour now:" << endl;
            //for(vector<Point>::iterator it = new_contour.begin(); it != new_contour.end(); it++)
            //	cout << it->x << " " << it->y << endl;
            new_contour.push_back(next);
            last = next;
            i++;
        }
        if(last == start)
        {
            return new_contour;
        }
    }
    if(last != start)
    {
        // Get points between
        vector<Point> between = getPointsBetween(last, start);
        new_contour.insert(new_contour.end(), between.begin(), between.end());
    }
    return new_contour;
}


bool inNeighbourhood(Point point_1, Point point_2)
{
    return abs(point_1.x-point_2.x) <= 1 && abs(point_1.y-point_2.y) <= 1;
}

vector<Point> getPointsBetween(const Point& p1, const Point& p2)
{
    // Check if they are in a 3x3 neighbourhood
    if(inNeighbourhood(p1, p2))
    {
        return vector<Point>();
    }
    // Get the point midway
    Point mid((p1.x + p2.x)/2, (p1.y + p2.y)/2);
    // Get the list of points between p1 and mid, and between mid and p2
    vector<Point> p1_mid_points = getPointsBetween(p1, mid);
    vector<Point> mid_p2_points = getPointsBetween(mid, p2);
    // Join all points
    p1_mid_points.push_back(mid);
    p1_mid_points.insert(p1_mid_points.end(), mid_p2_points.begin(), mid_p2_points.end());
    // Return points
    return p1_mid_points;
}


//-------------------//
//       POM         //
//-------------------//
float POM(vector<pair<int,int>> neighboursPair, vector<vector<Point>> &contours , Mat currSuperpixelsMap, Mat mask1, Mat mask2, int s,int k,int ns){
    vector<double> simmetries;
    vector<float> complexities;
    double simmetry;
    float complexity;
    int continuity;
    float pom = 0;
    //conversions for FindContours
    //mask1.convertTo(mask1, CV_8UC1);    //mask one superpixels
    //mask2.convertTo(mask2, CV_8UC1);    //mask two superpixels(neighbour)
    Mat mask_simmetry1 = Mat(currSuperpixelsMap.rows,currSuperpixelsMap.cols, CV_8UC1, uchar(0));
    Mat mask_simmetry2 = Mat(currSuperpixelsMap.rows,currSuperpixelsMap.cols, CV_8UC1, uchar(0));
    Mat mask = Mat(currSuperpixelsMap.rows,currSuperpixelsMap.cols, CV_8UC1, uchar(0));
        
    mask_simmetry1 = mask1;
    mask_simmetry2 = mask2;
    bitwise_or(mask_simmetry1, mask_simmetry2, mask); //mask union superpixels
        
    simmetry = calculateSimmetry(mask_simmetry1, mask_simmetry2);
    simmetries.push_back(simmetry);
    //cout<< "Simmetry: "<< simmetry <<endl;
        
    findContours(mask ,contours , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
    Scalar color( 255, 255, 255 );
    drawContours( mask, contours, 0, color, CV_FILLED);
    //imshow("unionSuperpixels", mask);
    //waitKey(0);
        
    complexity = boundaryComplexity(mask,contours[0],s,k,ns);///////////////MOLTI 0 poichè F_sk = 0;
    complexities.push_back(complexity);
    //cout<< "Complexity: "<< complexity <<endl;
        
    //**--> compute Continuity & Proximity
    vector<vector<Point>> contours_one,contours_two ;
        
    //prendo le singole maschere dei superpixel per creare il blob relativo
    findContours(mask1 ,contours_one , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
    findContours(mask2 ,contours_two , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
        
    cvb::CvBlob* blob1 = createBlob(contours_one[0]);
    cvb::CvBlob* blob2 = createBlob(contours_two[0]);
        
    //Mat common = Mat::zeros(image_superpixels.rows, image_superpixels.cols, CV_8UC1);
    vector<Point> points;
    points = commonBoundary_2(*blob1, *blob2);  //vector of neighbouring contour
    
    Mat neighbouring_contour_mat = Mat(currSuperpixelsMap.rows,currSuperpixelsMap.cols, CV_8UC1, uchar(0));
    for(int i=0; i<points.size();++i){
        neighbouring_contour_mat.at<uchar>(points[i]) = 255;    //draw neighbouring contour
    }
    //image moment to find orientation
    Moments mu;
    mu = moments( points , false );
    
    float x = mu.m10/mu.m00;
    float y = mu.m01/mu.m00;
    float u20 = (mu.m20/mu.m00) - pow(x, 2);
    float u02 = (mu.m02/mu.m00) - pow(y, 2);
    float u11 = (mu.m11/mu.m00) - x * y;
    
    float orientation = atan2((float)(-2)*u11,u20 - u02)/2  * 180/PI;
    //cout << "ORIENTATION: " << orientation << endl;
    vector<float> coordinate_x;
    vector<float> coordinate_y;
    for(int npoints=0; npoints<points.size(); ++npoints){
        coordinate_x.push_back(points[npoints].x);
        coordinate_y.push_back(points[npoints].y);
    }
    auto leftmost = min_element(coordinate_x.begin(), coordinate_x.end());  //min x
    auto rightmost = max_element(coordinate_x.begin(), coordinate_x.end()); //max x
    auto topmost = max_element(coordinate_y.begin(), coordinate_y.end());   //max y
    auto bottommost = min_element(coordinate_y.begin(), coordinate_y.end());//min y
    //[top-left top-right right-top right-bottom bottom-right bottom-left left-bottom left-top]
    float minX_topLeft = currSuperpixelsMap.cols;
    float maxX_topRight = 0;
    float minY_rightTop = currSuperpixelsMap.rows;
    float maxY_rightBottom = 0;
    float maxX_bottomRight = 0;
    float minX_bottomleft = currSuperpixelsMap.cols;
    float maxY_leftBottom = 0;
    float minY_leftTop = currSuperpixelsMap.rows;
    Point p_topLeft, p_topRight, p_rightTop, p_rightBottom, p_bottomRight, p_bottomLeft, p_leftBottom, p_leftTop;
    
    for(int npoints=0; npoints<points.size(); ++npoints){
        //top-left (min y, min x)
        if(points[npoints].y == *bottommost){
            if(points[npoints].x <= minX_topLeft){
                minX_topLeft = points[npoints].x;
                p_topLeft.x = points[npoints].x;
                p_topLeft.y = points[npoints].y;
            }
        }
        //top-right (min y, max x)
        if(points[npoints].y == *bottommost){
            if(points[npoints].x >= maxX_topRight){
                maxX_topRight = points[npoints].x;
                p_topRight.x = points[npoints].x;
                p_topRight.y = points[npoints].y;
            }
        }
        //right-top (max x, min y)
        if(points[npoints].x == *rightmost){
            if(points[npoints].y <= minY_rightTop){
                minY_rightTop = points[npoints].y;
                p_rightTop.x = points[npoints].x;
                p_rightTop.y = points[npoints].y;
            }
        }
        //right-bottom (max x, max y)
        if(points[npoints].x == *rightmost){
            if(points[npoints].y >= maxY_rightBottom){
                maxY_rightBottom = points[npoints].y;
                p_rightBottom.x = points[npoints].x;
                p_rightBottom.y = points[npoints].y;
            }
        }
        //bottom-right (max y, max y)
        if(points[npoints].y == *topmost){
            if(points[npoints].x >= maxX_bottomRight){
                maxX_bottomRight = points[npoints].y;
                p_bottomRight.x = points[npoints].x;
                p_bottomRight.y = points[npoints].y;
            }
        }
        //bottom-left (max y, min y)
        if(points[npoints].y == *topmost){
            if(points[npoints].x <= minX_bottomleft){
                minX_bottomleft = points[npoints].y;
                p_bottomLeft.x = points[npoints].x;
                p_bottomLeft.y = points[npoints].y;
            }
        }
        //left-bottom (min x, max y)
        if(points[npoints].x == *leftmost){
            if(points[npoints].y >= maxY_leftBottom){
                maxY_leftBottom = points[npoints].y;
                p_leftBottom.x = points[npoints].x;
                p_leftBottom.y = points[npoints].y;
            }
        }
        //left-top (min x, min y)
        if(points[npoints].x == *leftmost){
            if(points[npoints].y <= minY_leftTop){
                minY_leftTop = points[npoints].y;
                p_leftTop.x = points[npoints].x;
                p_leftTop.y = points[npoints].y;
            }
        }
    }
    Point P1,P2;
    vector<Point> point;
    if(orientation == 0){
        P1 = (p_topLeft + p_bottomLeft) / 2;
        P2 = (p_topRight + p_bottomRight) / 2;
    }else if(orientation == 90 || orientation == -90){
        P1 = (p_topLeft + p_topRight) / 2;
        P2 = (p_bottomRight + p_bottomLeft) / 2;
    }else if(orientation < 0){
        P1 = p_topLeft;
        P2 = p_bottomRight ;
    }else{
        P1 = p_bottomLeft;
        P2 = p_topRight;
    }

    //draw 2 points of common boundaries!
    //cout<<"P1: "<<P1.x<<" - "<<P1.y<<endl;
    //cout<<"P2: "<<P2.x<<" - "<<P2.y<<endl;
    //vector<Point> v;
    //v.push_back(P1);
    //v.push_back(P2);
    //cvtColor(neighbouring_contour_mat, neighbouring_contour_mat, CV_GRAY2BGR);
    //Scalar color2( 0, 0, 255 );
    //circle(neighbouring_contour_mat, P1, 1, Scalar(0,255,0),CV_FILLED);
    //circle(neighbouring_contour_mat, P2, 1, Scalar(0,255,0),CV_FILLED);

    continuity = calculateContinuity(P1,P2, mask);
    //cout<< "Continuity: "<< continuity <<endl;
        
    //**-->compute cohesivenessStrength
    //alias lambda
    float proximity;
    proximity = cohesivenessStrength(*blob1 , *blob2);
    //cout<< "Proximity: "<< proximity <<endl;
    
    //Size size(800,640);
    //resize(neighbouring_contour_mat,neighbouring_contour_mat,size);
    //imshow("unionSuperpixels", neighbouring_contour_mat);
    //waitKey(0);
    
    pom = calculatePOM(*blob1, *blob2, complexity, simmetry, continuity, proximity);
    
    return pom;
}






