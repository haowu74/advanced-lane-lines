## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Original"
[image2]: ./output_images/calibration1_undistorted.jpg "Undistorted"
[image3]: ./test_images/test1.jpg "Original Test1.jpg"
[image4]: ./output_images/test1_undistorted.jpg "Undistorted Test1.jpg"
[image5]: ./output_images/test1_binary.jpg "Binary Image"
[image6]: ./output_images/test1_warped.jpg "Warped"
[image7]: ./output_images/test2_result.jpg "Result"
[video1]: ./output_images/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function of `camera_calibration()` in the `project.py`  

I start by preparing "object points", which will be the (9, 6, 3) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (9, 6) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (9, 6) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original:
![alt text][image1]

Undistorted
![alt text][image2]

### Pipeline (single images)

#### 1. Undistorting image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Original Test1.jpg
![alt text][image3]

Undistorted Test1.jpg
![alt text][image4]

#### 2. Create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (function `pipeline()` in `project.py`).  
Here's an example of my output for this step. 

![alt text][image5]

#### 3. Perspective transform.

The code for my perspective transform includes a function called `warper()` in the file `project.py`. 
I use `straight_lines1.jpg` and `straight_lines2.jpg` as the source image, and mapping the source area to destination rectangle:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 209, 719      | 250, 719      | 
| 1095, 719     | 1030, 719     |
| 538, 492      | 250, 0        |
| 751, 492      | 1030, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Identify lane-line pixels and fit their positions with a polynomial

Then I fit my lane lines with a 2nd order polynomial using `numpy.polyfit` in `project.py`:

#### 5. Calculated the radius of curvature of the lane

I did this in lines 273 to 281 in function `process_image` in `project.py`. The curve radius and offset values are added into the image / video.

#### 6. Plot back down onto the road

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image `test2.jpg`:

![alt text][image7]

---

### Pipeline (video)

#### 1. Project Video

The result for the project video is at here: [link to my video result](./output_images/project_video.mp4)

---

### Discussion

I spent much time juggling with the color / gray gradient in this project. Actually a whole weekend trying to work out the 'best' combination.
Due to the time limitation, I can only make it work for the `project_video.mp4`, but failed in `challenge_video.mp4`. Once or twice,
when the performance on the `challenge_video.mp4` got better, the `project_video.mp4` became worse. When I used software to extract the
difficult frames in the `challenge_video.mp4` to analyse, I found the frames under the bridge is really hard to get the good gradient
combination. Even the HLS color space was not very helpful, the yellow line in the shadow has totally different Hue value compared 
to the yellow line in the sunshine. I tried a few combination of Sobel / Magnitude / Direction gradients but couldn't find the satisfying thresholds.

In terms of the optimisation, I used Look-ahead filter to reduce the sliding windows' searching area, based on the
result of the previous frame. But didn't do sanity check so if one frame is awfully wrong, the following frame would be a disaster.
Fortunately it doesn't happen in `project_video.mp4`, but obviously it is needed in challenge videos. 

Another thing to improve is: the program will crash when the program detects no line or only one line in the camera (as np.fillPoly
can't work on empty array). I think in that case maybe the program should use the lines in the previous frame (if just get lost in a couple of frames).
