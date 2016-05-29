//
//  main.cpp
//  Augmented Reality SS16
//
//  Created by Strelchenko Vadym on 5/4/16.
//  Copyright © 2016 Strelchenko Vadym. All rights reserved.
//


// -I opencv/include/ -L opencv/lib/ -lm -lcxcore -lcvaux -lcv -lhighgui

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv/cv.h> /// include most of core headers
#include <opencv/highgui.h> /// include GUI-related headers

using namespace cv;

static const std::string MARKER_EXAMPLE_1 = "/Users/strelchenkovadym/Documents/Study/TUM/ss16/Augmented Reality/exercises/MarkerExample_1_640.mp4";
static const std::string AGUMENTED_REALITY_EXERCISE_1 = "Augmented Reality Ex.1";
static const std::string AGUMENTED_REALITY_EXERCISE_2 = "Augmented Reality Ex.2";
//use adaptive thresholding
static const bool ADAPTIVE = true;

static int sampleArea = 7;
int thresh = 100;

//wheather to draw or not the stripe
bool firstStripe = true;
//wheather to execute my code or solution's one
bool myWay = false;

//Sample image for subpixel values at the point p (which is not necessarily fully corresponds to a certain pixel)
int subpixSampleSafe ( const cv::Mat &pSrc, const cv::Point2f &p )
{
    int x = int( floorf ( p.x ) );
    int y = int( floorf ( p.y ) );
    if(x<0||x>=pSrc.cols -1|| y < 0 || y >= pSrc.rows - 1 )
        return 127;
    int dx = int ( 256 * ( p.x - floorf ( p.x ) ) );
    int dy = int ( 256 * ( p.y - floorf ( p.y ) ) );
    unsigned char* i = ( unsigned char* ) ( ( pSrc.data + y * pSrc.step ) + x );
    int a = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
    i += pSrc.step;
    int b = i[ 0 ] + ( ( dx * ( i[ 1 ] - i[ 0 ] ) ) >> 8 );
    return a + ( ( dy * ( b - a) ) >> 8 );
}

//Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//DOESN'T WORK T__T
/*float findExtremumParabola(const Point2f & p1, const Point2f & p2, const Point2f & p3) {
 std::cout << "Points:" << std::endl;
 std::cout << p1.x << " " << p1.y << " " << p2.x << " " << p2.y << " " << p3.x << " " << p3.y << std::endl;
 Mat A = (Mat_<float>(3, 3) << p1.x*p1.x, p1.x, 1,
 p2.x*p2.x, p2.x, 1,
 p3.x*p3.x, p3.x, 1);
 std::vector<float> x;
 Mat b = (Mat_<float>(1, 3) << p1. y, p2.y, p3.y);
 
 solve(A.t(), b.t(), x);
 transpose(x, x);
 
 float shift = - x[1] / (2 * x[0]);
 std::cout << "Needed shift: " << shift << std::endl;
 std::cout<< x[0] << " " << x[1] << " " << x[2] << std::endl;
 
 return shift;
 }*/

int main(int argc, const char * argv[]) {
    VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        //cap.open(MARKER_EXAMPLE_1);
        return -1;
    }
    
    //contour's storage
    CvMemStorage* memStorage = cvCreateMemStorage();
    
    int keyPressed;
    
    namedWindow(AGUMENTED_REALITY_EXERCISE_2, WINDOW_NORMAL | WINDOW_KEEPRATIO);
    namedWindow("The first stripe", CV_WINDOW_NORMAL);
    namedWindow("The first stripe (small)", CV_WINDOW_NORMAL);
    if (!ADAPTIVE) {
        createTrackbar("Threshold", AGUMENTED_REALITY_EXERCISE_2, &thresh, 255);
    }
    while (true)
    {
        Mat frame;
        Mat imgGray;
        Mat edges;
        cap >> frame; // get a new frame from camera
        //GaussianBlur(frame, frame, Size(sampleArea, sampleArea), 1.5);
        cvtColor(frame, imgGray, CV_BGR2GRAY);
        
        
        if (!ADAPTIVE) {
            threshold(imgGray, edges, thresh, 255, THRESH_BINARY);
        } else {
            //std::cout << "gaussian" << std::endl;
            adaptiveThreshold(imgGray, edges, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, sampleArea, 2);
        }
        
        Mat hierarchy;
        //contours container
        std::vector<std::vector<Point>> contours;
        //find contours // human way
        //findContours(edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        
        //find contours // idiotic way
        CvMat img_mono_(edges);
        CvSeq* contours_;
        cvFindContours(&img_mono_, memStorage, &contours_, sizeof(CvContour),
                       CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE
                       );
        
        /*
         CvSeq* result = cvApproxPoly(
         contours, sizeof(CvContour), memStorage, CV_POLY_APPROX_DP,
         cvContourPerimeter(contours)*0.02, 0
         );
         */
        
        //for (auto contour = contours.begin(); contour != contours.end(); contour++)
        for (; contours_; contours_ = contours_->h_next)
        {
            CvSeq* result = cvApproxPoly(
                                         contours_, sizeof(CvContour), memStorage, CV_POLY_APPROX_DP,
                                         cvContourPerimeter(contours_)*0.02, 0);
            
            if (result->total!=4){
                continue;
            }
            
            //a polygon with 4 points in API 2.x format
            cv::Mat result_ = cv::cvarrToMat(result); /// API 1.X to 2.x
            cv::Rect r = cv::boundingRect(result_);
            if (r.height<20 || r.width<20 || r.width > edges.cols - 10 || r.height > edges.rows - 10 || cvContourArea(result) < 2000) {
                continue;
            }
            
            //a pointer to polygonal data as (as Point-s)
            const cv::Point *rect = (const cv::Point*) result_.data;
            //number of points (always 4?)
            int npts = result_.rows;
            
            // draw the polygon
            cv::polylines(frame, &rect, &npts, 1,
                          true,           // draw closed contour (i.e. joint end to start)
                          CV_RGB(255, 0, 0),// colour RGB ordering (here = green)
                          2,              // line thickness
                          CV_AA, 0);
            
            
            // Added in Exercise 4 - Start *****************************************************************
            //
            //lineParams is shared
            float lineParams[16];
            //4 lines 4 parameters each
            Mat lineParamsMat(Size(4, 4), CV_32F, lineParams);
            //new edge centers container
            Point2f points[6];
            //keep track of data in points array
            int pointsCount;
            //
            // Added in Exercise 4 - End *****************************************************************
            
            //for each edge (2 points)
            //draw equi-spread circles along the edges of the squares
            for (int j = 0; j < 4; j++) {
                // Added in Exercise 4 - Start *****************************************************************
                //
                pointsCount = 0;
                //
                // Added in Exercise 4 - End *****************************************************************
                
                Point2f p1 = *(rect + j % 4);
                Point2f p2 = *(rect + (j + 1) % 4);
                float numOfChunks = 7;
                //distances between the point to draw & the centers of the stripes
                float dx = (p2.x - p1.x) / numOfChunks;
                float dy = (p2.y - p1.y) / numOfChunks;
                
                // Added in Exercise 3 - Start *****************************************************************
                //Setup stuff to work on a stripe
                //
                int stripeLength = 0.8 * sqrt(dx*dx + dy*dy);
                if (stripeLength < 5) {
                    stripeLength = 5;
                }
                //make stripeLength odd (because of the shift in nStop)
                stripeLength |= 1;
                
                Size stripeSize;
                stripeSize.width = 3;
                stripeSize.height = stripeLength;
                //8-bit Unsigned value with Channel number - 1
                Mat stripe(stripeSize, CV_8UC1);
                
                //normalized vectors parallel and orthogonal to the current edge
                Point2f stripeParalVec;
                Point2f stripeOrthoVec;
                float vecLength = sqrt(dx*dx + dy*dy);
                
                stripeParalVec.x = dx / vecLength;
                stripeParalVec.y = dy / vecLength;
                stripeOrthoVec.x = -stripeParalVec.x;
                stripeOrthoVec.y = stripeParalVec.y;
                // Added in Exercise 3 - End *******************************************************************
                
                int stripeEnd = stripeLength >> 1;
                int stripeStart = - stripeEnd;
                circle(frame, p1, 3, Scalar(255, 0, 0), -1);
                for (int i = 1; i < numOfChunks; i++) {
                    //circle coords to draw
                    Point2f currEdgeCenter(p1.x + i * dx, p1.y + i * dy); // Point p in the solution
                    //draw a circle to indicate a chunk
                    circle(frame, Point(currEdgeCenter.x, currEdgeCenter.y), 2, Scalar(0, 255, 0), -1);
                    
                    // Added in Exercise 3 *******************************************************************
                    //
                    //subpixel colors for analyzed stripe
                    for (int m = stripeStart; m <= stripeEnd; m++) {
                        for (int n = -1; n <= 1; n++) {
                            if (myWay) {
                                stripe.at<uchar>(m + stripeEnd, n + 1) = subpixSampleSafe(imgGray, currEdgeCenter + n * stripeOrthoVec + m * stripeParalVec);
                            } else {
                                cv::Point2f subPixel;
                                
                                subPixel.x = (double)currEdgeCenter.x + ((double)m * stripeParalVec.x) + ((double)n * stripeOrthoVec.x);
                                subPixel.y = (double)currEdgeCenter.y + ((double)m * stripeParalVec.y) + ((double)n * stripeOrthoVec.y);
                                
                                int pixel = subpixSampleSafe (imgGray, subPixel);
                                
                                int w = m + 1; //add 1 to shift to 0..2
                                int h = n + ( stripeLength >> 1 ); //add stripelenght>>1 to shift to 0..stripeLength
                                
                                
                                stripe.at<uchar>(h,w) = (uchar)pixel;
                            }
                        }
                    }
                    
                    //Sobelize dat shit
                    //
                    //values of the center points in the stripe after applying the Sobel operator
                    int sobelValues[stripeLength - 2];
                    int maxValue = 0;
                    int maxIndex = 0;
                    for (int m = 1; m <= stripeLength - 1; m++) {
                        if (myWay) {
                            //MY WAY //not the cause
                            sobelValues[m - 1] = -stripe.at<uchar>(m - 1, 0) - stripe.at<uchar>(m, 0) * 2 - stripe.at<uchar>(m                                               + 1, 0)
                            + stripe.at<uchar>(m - 1, 2) + stripe.at<uchar>(m, 2) * 2 + stripe.at<uchar>(m + 1, 2);
                        } else {
                            //THEIR WAY
                            unsigned char* stripePtr = &( stripe.at<uchar>(m-1,0) );
                            double r1 = -stripePtr[ 0 ] - 2 * stripePtr[ 1 ] - stripePtr[ 2 ];
                            
                            stripePtr += 2*stripe.step;
                            double r3 =  stripePtr[ 0 ] + 2 * stripePtr[ 1 ] + stripePtr[ 2 ];
                            sobelValues[m-1] = r1+r3;
                        }
                        
                        if (abs(sobelValues[m - 1]) > maxValue) {
                            maxValue = sobelValues[m];
                            maxIndex = m;
                        }
                    }
                    
                    //draw the first edge stripe
                    if (firstStripe) {
                        std::cout << "stripe.rows: " << stripe.rows << std::endl;
                        std::cout << "stripe.cols: " << stripe.cols << std::endl;
                        
                        Mat stripeTmp;
                        std::cout << "stripe.size(): " << stripe.size().height << ", " << stripe.size().width << std::endl;
                        resize( stripe, stripeTmp, Size(), 50, 50);
                        std::cout << "stripe resized.size(): " << stripeTmp.size().height << ", " << stripeTmp.size().width << std::endl;
                        imshow("The first stripe (small)", stripe);
                        imshow("The first stripe", stripeTmp);
                        
                        for (int k = 0; k < stripeLength - 2; k++) {
                            std::cout << sobelValues[k] << " ";
                        }
                        std::cout << std::endl;
                        
                        firstStripe = false;
                    }
                    
                    //why 0? fuck you, that's why
                    //not, but really - there has to be smth, may just as well be a 0
                    //findExtremumParabola(Point2f(0, (maxIndex <= 0) ? 0 : afterSobel[maxIndex - 1]), Point2f(1, maxValue), Point2f(2, (maxIndex >= stripeLength - 2) ? 0 : afterSobel[maxIndex + 1]));
                    
                    //parabola y coordinates fetch
                    double y0,y1,y2; // y0 .. y1 .. y2
                    y0 = (maxIndex <= 0) ? 0 : sobelValues[maxIndex-1];
                    y1 = sobelValues[maxIndex];
                    y2 = (maxIndex >= stripeLength-3) ? 0 : sobelValues[maxIndex+1];
                    
                    //Apparently the following shit solves the parabola equation and finds and extremum from the first derivative
                    //
                    //formula for calculating the x-coordinate of the vertex of a parabola, given 3 points with equal distances
                    //(xv means the x value of the vertex, d the distance between the points):
                    //xv = x1 + (d / 2) * (y2 - y0)/(2*y1 - y0 - y2)
                    double pos = (y2 - y0) / (4*y1 - 2*y0 - 2*y2 ); //d = 1 because of the normalization and x1 will be added later
                    //std::cout << "pos: " << pos << std::endl;
                    if (isnan(pos) || isinf(pos)) {
                        //std::cout << "pos is Nan" << std::endl;
                        pos = 0.;
                    }
                    
                    Point2f newEdgeCenter; //exact point with subpixel accuracy
                    int maxIndexShift = maxIndex - (stripeLength>>1);
                    
                    //shift the original edgepoint accordingly
                    newEdgeCenter.x = (double)currEdgeCenter.x + (((double)maxIndexShift+pos) * stripeOrthoVec.x);
                    newEdgeCenter.y = (double)currEdgeCenter.y + (((double)maxIndexShift+pos) * stripeOrthoVec.y);
                    //print debug
                    /*std::cout << "old edge: " << currEdgeCenter.x << " " << currEdgeCenter.y << std::endl;
                     std::cout << "new edge: " << newEdgeCenter.x << " " << newEdgeCenter.y << std::endl;*/
                    
                    
                    // Added in Exercise 4 *****************************************************************
                    //
                    points[pointsCount] = newEdgeCenter;
                    pointsCount++;
                }
                
                //fit dat line
                Mat pointsMat(Size(1, 6), CV_32FC2, points);
                //without a temp var (row) it doesn't work =/
                Mat col = lineParamsMat.col(j);
                
                /*std::cout << std::endl << "pointsMat : " << std::endl;
                 for (int i = 0; i < 6; i++) {
                 std::cout << pointsMat.at<Point2f>(0, i) << " ";
                 }
                 std::cout << std::endl;*/
                
                fitLine(pointsMat, lineParamsMat.col(j), CV_DIST_L2, 0, .01, .01);
                
                std::cout << std::endl << "lineParamsMat: " << std::endl;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        std::cout << lineParamsMat.at<float>(i, j) << " ";
                    }
                    std::cout << std::endl;
                }
                
                /*std::cout << std::endl << "row : " << std::endl;
                 for (int i = 0; i < 4; i++) {
                 std::cout << "[" << row.at<float>(0, i) << " | ";
                 std::cout << lineParamsMat.at<float>(j, i) << "] ";
                 }
                 std::cout << std::endl;*/
            }
            
            for (int i = 0; i < 4; i++) {
                if (myWay) {
                    //MY WAY // not the cause
                    //
                    //from #fitLine doc
                    //(vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line
                    //(vx1 vy1)k + (x01 y01) = (vx2 vy2)k + (x02 y02) -> k =
                    double vx1, vy1, x01, y01, vx2, vy2, x02, y02;
                     x01 = lineParamsMat.at<float>(2, i);           y01 = lineParamsMat.at<float>(3, i);
                     x02 = lineParamsMat.at<float>(2, (i + 1) % 4); y02 = lineParamsMat.at<float>(3, (i + 1) % 4);
                    
                     vx1 = lineParamsMat.at<float>(0, i);           vy1 = lineParamsMat.at<float>(1, i);
                     vx2 = lineParamsMat.at<float>(0, (i + 1) % 4); vy2 = lineParamsMat.at<float>(1, (i + 1) % 4);
                    
                    
                    double newCornerX = x02 * vx1 * vy2 - y02 * vx1 * vx2 + y01 * vx1 * vx2;//(x02 - x01) / (vx1 - vx2); //(x02 - x01) / (vx1 - vx2)
                    double newCornerY = -x01 * vy1 * vy2 + y01 * vx1 * vy2 + x02 * vy1 * vy2 - y02 * vy1 * vx2;//(y02 - y01) / (vy1 - vy2); //(y02 - y01) / (vy1 - vy2)
                    double c = vy2*vx1 - vy1*vx2;
                    
                    if ( fabs(c) < 0.001 ) //lines parallel?
                    {
                        std::cout << "lines parallel" << std::endl;
                        continue;
                    }
                    
                    newCornerX /= c;
                    newCornerY /= c;
                    
                    circle(frame, Point(newCornerX, newCornerY), 5, CV_RGB(5, 193, 255), -1);
                } else {
                    
                    //THEIR WAY
                    //
                    double x0,x1,y0,y1,u0,u1,v0,v1;
                    x0 = lineParamsMat.at<float>(2, i); y0 = lineParamsMat.at<float>(3, i); //x01, y01
                    x1 = lineParamsMat.at<float>(2, (i + 1) % 4); y1 = lineParamsMat.at<float>(3, (i + 1) % 4); //x02, y02
                    
                    u0 = lineParamsMat.at<float>(0, i); v0 = lineParamsMat.at<float>(1, i); //vx1, vy1
                    u1 = lineParamsMat.at<float>(0, (i + 1) % 4); v1 = lineParamsMat.at<float>(1, (i + 1) % 4); //vx2, vy2
                    
                    double a =  x1*u0*v1 - y1*u0*u1 - x0*u1*v0 + y0*u0*u1;
                    double b = -x0*v0*v1 + y0*u0*v1 + x1*v0*v1 - y1*v0*u1;
                    double c =  v1*u0-v0*u1;
                    
                    if ( fabs(c) < 0.001 ) //lines parallel?
                    {
                        std::cout << "lines parallel" << std::endl;
                        continue;
                    }
                    
                    a /= c;
                    b /= c;
                    
                    circle(frame, Point(a, b), 5, CV_RGB(5, 193, 255)  );
                    //
                    //THEIR WAY
                }
            }
            //
            // Added in Exercise 4 *****************************************************************
        }
        
        cvClearMemStorage(memStorage);
        imshow(AGUMENTED_REALITY_EXERCISE_2, frame);
        
        keyPressed = waitKey(-1);
        //std::cout << "Key code: " << keyPressed << std::endl;
        if ( keyPressed == 27 ) { //esc
            cvReleaseMemStorage(&memStorage);
            break;
        } else if (keyPressed == 32) {
            //just continue
        } else if (keyPressed == 97) {
            std::cout << "'A' pressed! Changing sampling area.." << std::endl;
            switch (sampleArea) {
                case 3:
                    sampleArea = 5;
                    break;
                case 5:
                    sampleArea = 7;
                    break;
                case 7:
                    sampleArea = 3;
                    break;
            }
        }
    }
    
    return 0;
}