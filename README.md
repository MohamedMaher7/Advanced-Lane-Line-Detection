# Advanced-Lane-Line-Detection
Simple Perception Stack for Self-Driving Cars


**The goals / steps of this project are the following:**

1.Compute the camera calibration matrix and distortion coefficients given a set of chessboard images in the camera_cal folder.
2.Apply a distortion correction to images.
3.Use color transforms to create a thresholded binary image.
4.Apply a perspective transform to rectify binary image .
5.Detect lane pixels and polynominal fit to find the lane boundary.
6.Determine the curvature of the lane and vehicle position with respect to center.
7.Warp the detected lane boundaries back onto the original image.
8.Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


# 1. Calibration and Distorion Correction:
The chessboard corners are the reference to generate objpoints and imgpoints.
I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function.
Matrix mtx and dist from camera calibration are applied to distortion correction to one of the test images like this one:
![Screenshot 2022-04-18 234559](https://user-images.githubusercontent.com/73904088/163897732-7707d1ca-20de-4513-9397-bb42fe87ec13.jpg)


# 2. Color Transformation
I tried RGB, HLS, LUV and LAB color space, in addition, gradient and magnitude, and their combination. 
RGB filter is very sensitive to the threshold and could not split lanes from lightful environment. 
S channel and gradient combination could split lines on the road, but too much disturbance also left.
Finally I found color space transformation method .
B channel from LAB space identified yellow lanes while L channel from LUV space could detect white lanes.

# 3. Bird-eye Perspective Transformation
I used Bird-eye Perspective to get the region of ineterst in the test images which is the lane

![Screenshot 2022-04-18 235749](https://user-images.githubusercontent.com/73904088/163897879-3647be37-9ca4-4138-ac7c-30519216118c.jpg)

# 4. Lane Detection and Polynominal Fit

Firstly, I calculated the histogram of non-zero x-axis in binary image. And based on the maximum sum of x position, I used sliding window search method to identify lane pixels. If it is previously detected in last frame, a quick search could be applied based on last detected x/y pixel positions with a proper margin.
Then I fitted lane lines with a 2nd order polynomial 

# 5. Curvature and Position Calculation
To connect pixel unit with real world meter unit, I defined conversions in x and y from pixels space to meters. In order to calculate precisely, I used detected lane width dynamically.

# 6. Identified Lane Transformation back
Last step is to transform the lane identified back onto the road image. 

**Output of previous steps:**
![download](https://user-images.githubusercontent.com/73904088/163897988-000ac93a-f167-411d-84c2-d725a000a270.png)

