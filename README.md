# SFND 3D Object Tracking

By completing this project, we will have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, we will know how to detect objects in an image using the YOLO deep-learning framework. And finally, know how to associate regions in a camera image with Lidar points in 3D space. 

**The program schematic:**

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, we will implement the missing parts in the schematic. To do this, we will complete four major tasks: 
1. Develop a way to match 3D objects over time by using keypoint correspondences. 
2. Compute the TTC based on Lidar measurements. 
3. Compute the TTC based on camera data, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. Conduct various tests with the framework. By identifying the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor.
   
## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## FP.1 : Match 3D Objects
In this task, we implemented the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the IDs of the matched regions of interest (i.e. the boxID property)“. Matches must be the ones with the highest number of keypoint correspondences.

```ruby

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

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

```

## FP.2 : Compute Lidar-based TTC
In this part of the final project we computed the time-to-collision for all matched 3D objects based on Lidar measurements alone. The estimation is designed to be robust against outliers that may be too close, which could otherwise result in inaccurate TTC estimates.

```ruby

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
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
}
```

## FP.3 : Associate Keypoint Correspondences with Bounding Boxes
Before we can compute the camera's time-to-collision (TTC) estimate in the next exercise, we need to identify all keypoint matches associated with each 3D object. This was implemented by checking if the corresponding keypoints fall within the region of interest in the camera image. Matches that meet this criterion were collected into a vector. To eliminate outliers among these matches, we calculated a robust mean of the Euclidean distances between keypoint matches, subsequently removing those that deviated significantly from this mean. Outlier matches were identified and discarded based on their Euclidean distance relative to all matches within the bounding box.

```ruby
 void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Variables to calculate the total distance and count of valid matches within the ROI
    double totalDistance = 0.0;
    int validMatchCount = 0;

    // Iterate through all keypoint matches to find those within the bounding box's ROI
    for (const auto &match : kptMatches)
    {
        cv::KeyPoint currentKeyPoint = kptsCurr[match.trainIdx];
        
        // Check if the current keypoint is within the bounding box's ROI
        if (boundingBox.roi.contains(currentKeyPoint.pt))
        {
            cv::KeyPoint previousKeyPoint = kptsPrev[match.queryIdx];
            double distance = cv::norm(currentKeyPoint.pt - previousKeyPoint.pt);
            
            // Accumulate distance and count valid matches
            totalDistance += distance;
            validMatchCount++;
        }
    }

    // Calculate the mean distance, avoiding division by zero
    double meanDistance = (validMatchCount > 0) ? (totalDistance / validMatchCount) : 0.0;

    // Clear previous keypoints and matches in the bounding box
    boundingBox.keypoints.clear();
    boundingBox.kptMatches.clear();

    // Iterate again to filter matches based on distance
    for (const auto &match : kptMatches)
    {
        cv::KeyPoint currentKeyPoint = kptsCurr[match.trainIdx];
        
        // Check again for the keypoint within the bounding box's ROI
        if (boundingBox.roi.contains(currentKeyPoint.pt))
        {
            cv::KeyPoint previousKeyPoint = kptsPrev[match.queryIdx];
            double currentDistance = cv::norm(currentKeyPoint.pt - previousKeyPoint.pt);

            // Check if the current distance is within the specified threshold
            if (currentDistance < meanDistance * 1.3)
            {
                boundingBox.keypoints.push_back(currentKeyPoint);
                boundingBox.kptMatches.push_back(match);
            }
        }
    }
}
```
## FP.4 : Compute Camera-based TTC
Once keypoint matches have been added to the bounding boxes, we thwn computed the TTC estimate. 
```ruby
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double> distRatios; // Stores the ratios of distances between matched keypoints
    const double minDistanceThreshold = 100.0; // Minimum required distance to consider a match

    for (size_t i = 0; i < kptMatches.size(); ++i)
    {
        const cv::KeyPoint &outerCurr = kptsCurr[kptMatches[i].trainIdx];
        const cv::KeyPoint &outerPrev = kptsPrev[kptMatches[i].queryIdx];

        for (size_t j = i + 1; j < kptMatches.size(); ++j)
        {
            const cv::KeyPoint &innerCurr = kptsCurr[kptMatches[j].trainIdx];
            const cv::KeyPoint &innerPrev = kptsPrev[kptMatches[j].queryIdx];

            double distanceCurr = cv::norm(outerCurr.pt - innerCurr.pt);
            double distancePrev = cv::norm(outerPrev.pt - innerPrev.pt);

            // Validate distances to avoid division by zero
            if (distancePrev > std::numeric_limits<double>::epsilon() && distanceCurr >= minDistanceThreshold)
            {
                double ratio = distanceCurr / distancePrev;
                distRatios.push_back(ratio);
            }
        }
    }

    // Ensure we have valid ratios for TTC calculation
    if (distRatios.empty())
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // Sort ratios and calculate the median
    std::sort(distRatios.begin(), distRatios.end());
    size_t midIndex = distRatios.size() / 2;
    double medianRatio = (distRatios.size() % 2 == 0) ? 
                         (distRatios[midIndex - 1] + distRatios[midIndex]) / 2.0 : 
                         distRatios[midIndex];

    // Compute the time-to-collision (TTC)
    double deltaTime = 1.0 / frameRate;
    TTC = (medianRatio != 1.0) ? -deltaTime / (1.0 - medianRatio) : std::numeric_limits<double>::infinity();

  
    // Output the TTC result
    std::cout << "Camera TTC: " << TTC << " seconds" << std::endl;

    // Optional visualization
    if (visImg != nullptr)
    {
        cv::Mat visImage;
        std::vector<cv::DMatch> goodMatches;

        // Visualize only valid matches
        for (const auto &match : kptMatches)
        {
            const cv::KeyPoint &currPoint = kptsCurr[match.trainIdx];
            const cv::KeyPoint &prevPoint = kptsPrev[match.queryIdx];
            double disparity = prevPoint.pt.x - currPoint.pt.x;

            if (std::find(distRatios.begin(), distRatios.end(), disparity) != distRatios.end())
            {
                goodMatches.push_back(match);
            }
        }

        // Draw matches and display the image
        cv::drawMatches(*visImg, kptsPrev, *visImg, kptsCurr, goodMatches, visImage,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        std::string ttcText = "TTC Camera: " + std::to_string(TTC) + " s";
        cv::putText(visImage, ttcText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      
        cv::imshow("TTC Visualization", visImage);
        cv::waitKey(0);
    }
}
```
## FP.5 : Performance Evaluation 1

<img width="1199" alt="image" src="https://github.com/user-attachments/assets/3974d315-7772-463b-8fd0-28869327d3a9">

<img width="1199" alt="image" src="https://github.com/user-attachments/assets/c6d33c34-bf2b-4cf6-a9e1-5dcbe940afc2">

<img width="1199" alt="image" src="https://github.com/user-attachments/assets/d76f8c72-6241-4950-ac97-80c0b575ca81">

<img width="1199" alt="image" src="https://github.com/user-attachments/assets/2213ded3-bc32-43c6-bfb7-66a539e3347e">

<img width="1199" alt="image" src="https://github.com/user-attachments/assets/5a4d91fa-ec74-433c-9eeb-53d22d7148a7">



https://github.com/user-attachments/assets/4d82cf5c-a0a6-498c-80db-c751ddaba51b





