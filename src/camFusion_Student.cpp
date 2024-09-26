
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
  //////////////////////////////

    // Initialize the vector to hold keypoint matches within the bounding box
    std::vector<cv::DMatch> kptMatchesInROI;

    // Define the ROI for the bounding box
    cv::Rect roi(boundingBox.roi.x, boundingBox.roi.y, boundingBox.roi.width, boundingBox.roi.height);

    // Iterate over all keypoint matches
    for (const auto &match : kptMatches)
    {
        // Get the keypoint from the previous frame
        cv::KeyPoint kpPrev = kptsPrev[match.queryIdx];

        // Get the keypoint from the current frame
        cv::KeyPoint kpCurr = kptsCurr[match.trainIdx];

        // Check if the current keypoint is within the bounding box's ROI
        if (roi.contains(kpCurr.pt))
        {
            // If within ROI, add the match to the vector
            kptMatchesInROI.push_back(match);
        }
    }

    // Update the bounding box with the filtered matches
    boundingBox.kptMatches = kptMatchesInROI;


  //////////////////////////////
}


/////////////////////////
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp> // Include this for drawMatches
///////////////////////////

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
  ///////////////////////////////////////////

 
    // Vector to store distance ratios between matched keypoints
    std::vector<double> distRatios;
    std::vector<double> disparities; // Vector to store disparities
  
    std::vector<double> distances;   // Vector to store distances


    // Loop over all keypoint matches
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        cv::KeyPoint kpOuterCurr = kptsCurr[it1->trainIdx];
        cv::KeyPoint kpOuterPrev = kptsPrev[it1->queryIdx];

        for (auto it2 = it1 + 1; it2 != kptMatches.end(); ++it2)
        {
            cv::KeyPoint kpInnerCurr = kptsCurr[it2->trainIdx];
            cv::KeyPoint kpInnerPrev = kptsPrev[it2->queryIdx];

            // Calculate distances between keypoints
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
          
         

            // Calculate distance ratio and add to the list
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= 1.0) // Avoid small distances
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);

                // Calculate disparity
                double disparity = kpOuterPrev.pt.x - kpOuterCurr.pt.x;
                disparities.push_back(disparity);
                distances.push_back(distCurr);
            }
        }
    }

    // Proceed only if there are valid distance ratios
    if (distRatios.empty())
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // Sort distance ratios and compute the median ratio
    std::sort(distRatios.begin(), distRatios.end());
    size_t n = distRatios.size();
    double medianRatio = (n % 2 == 0) ? (distRatios[n / 2 - 1] + distRatios[n / 2]) / 2.0 : distRatios[n / 2];

    // Compute time-to-collision (TTC)
    double dT = 1.0 / frameRate;
    if (medianRatio != 1.0)
    {
        TTC = -dT / (1.0 - medianRatio);
    }
    else
    {
        TTC = std::numeric_limits<double>::infinity(); // Handle the case where medianRatio is 1.0
    }
////////////////////////////
  // Calculate min distances for output
    double minXPrev = (distances.empty()) ? std::numeric_limits<double>::quiet_NaN() : *std::min_element(distances.begin(), distances.end());
    double minXCurr = (distances.empty()) ? std::numeric_limits<double>::quiet_NaN() : *std::min_element(distances.begin(), distances.end());

    // Output debug information for camera TTC calculation
    std::cout << "Camera TTC Calculation:" << std::endl;
    std::cout << "Previous frame min distance: " << minXPrev << std::endl;
    std::cout << "Current frame min distance: " << minXCurr << std::endl;
    std::cout << "TTC: " << TTC << std::endl;
  ///////////////////////////
    // Optional visualization
    if (visImg != nullptr)
    {
        cv::Mat visImage;
        std::vector<cv::DMatch> goodMatches;

        // Filter matches for visualization
        for (const auto &match : kptMatches)
        {
            cv::Point2f prevPoint = kptsPrev[match.queryIdx].pt;
            cv::Point2f currPoint = kptsCurr[match.trainIdx].pt;
            double disparity = prevPoint.x - currPoint.x;

            // Check if this disparity is valid
            if (std::find_if(disparities.begin(), disparities.end(), [disparity](double d) { return std::abs(d - disparity) < 1e-5; }) != disparities.end())
            {
                goodMatches.push_back(match);
            }
        }

        // Draw and display keypoint matches
        cv::drawMatches(*visImg, kptsPrev, *visImg, kptsCurr, goodMatches, visImage,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        std::string ttcText = "TTC Camera: " + std::to_string(TTC) + " s";
        cv::putText(visImage, ttcText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        cv::imshow("TTC Visualization", visImage);
        cv::waitKey(0);
    }


  /////////////////////
}

 


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
  //////////////////////

    // Time between two measurements in seconds
    double dT = 1.0 / frameRate;

    // Vector to hold distances
    std::vector<double> prevDistances;
    std::vector<double> currDistances;

    // Collect distances from previous frame
    for (const auto &point : lidarPointsPrev)
    {
        prevDistances.push_back(point.x);
    }

    // Collect distances from current frame
    for (const auto &point : lidarPointsCurr)
    {
        currDistances.push_back(point.x);
    }

    // Sort distances to find percentiles
    std::sort(prevDistances.begin(), prevDistances.end());
    std::sort(currDistances.begin(), currDistances.end());

    // Compute the 20th percentile (approximate closest distance)
    double minXPrev = prevDistances[prevDistances.size() / 5];
    double minXCurr = currDistances[currDistances.size() / 5];

    // Calculate TTC using the percentile distances
    if (minXPrev != minXCurr)
    {
        TTC = (minXCurr * dT) / (minXPrev - minXCurr);
    }
    else
    {
        TTC = std::numeric_limits<double>::infinity(); // Handle division by zero if distances are equal
    }

    // Output debug information
    std::cout << "Lidar TTC Calculation:" << std::endl;
    std::cout << "Previous frame min distance: " << minXPrev << std::endl;
    std::cout << "Current frame min distance: " << minXCurr << std::endl;
    std::cout << "TTC: " << TTC << std::endl;


  ///////////////////////////
}

////////////////
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <cmath>

using namespace cv;
///////////////////

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
 
  /////////////////////
  
	int prevBoxCount = prevFrame.boundingBoxes.size();
    int currBoxCount = currFrame.boundingBoxes.size();
    std::vector<std::vector<int>> matchCounts(prevBoxCount, std::vector<int>(currBoxCount, 0));

    // Iterate over matches to count box-to-box matches
    for (const auto &match : matches)
    {
        const cv::KeyPoint &prevKeypoint = prevFrame.keypoints[match.queryIdx];
        const cv::KeyPoint &currKeypoint = currFrame.keypoints[match.trainIdx];

        cv::Point prevPoint(static_cast<int>(prevKeypoint.pt.x), static_cast<int>(prevKeypoint.pt.y));
        cv::Point currPoint(static_cast<int>(currKeypoint.pt.x), static_cast<int>(currKeypoint.pt.y));

        std::vector<int> prevBoxIDs;
        std::vector<int> currBoxIDs;

        // Find the bounding box IDs for the previous frame
        for (int i = 0; i < prevBoxCount; ++i)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(prevPoint))
            {
                prevBoxIDs.push_back(i);
            }
        }

        // Find the bounding box IDs for the current frame
        for (int j = 0; j < currBoxCount; ++j)
        {
            if (currFrame.boundingBoxes[j].roi.contains(currPoint))
            {
                currBoxIDs.push_back(j);
            }
        }

        // Increment match counts for each box pair
        for (const int &prevBoxID : prevBoxIDs)
        {
            for (const int &currBoxID : currBoxIDs)
            {
                matchCounts[prevBoxID][currBoxID]++;
            }
        }
    }

    // Determine the best match for each bounding box in the previous frame
    for (int i = 0; i < prevBoxCount; ++i)
    {
        int maxMatches = 0;
        int bestMatchIndex = -1;

        for (int j = 0; j < currBoxCount; ++j)
        {
            if (matchCounts[i][j] > maxMatches)
            {
                maxMatches = matchCounts[i][j];
                bestMatchIndex = j;
            }
        }

        if (bestMatchIndex != -1)
        {
            bbBestMatches[i] = bestMatchIndex;
        }
    }
}
