# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images/solidYellowCurve.jpg

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

In this project, I built a computer vision pipeline to detect lane lines and creat extrapolated boundary lines. The pipeline is as follows:

- Convert frame to grayscale
- Create masks for yellow/white pixels
- Apply a Gaussian smoothing filter
- Apply a Canny edge detection
- Create an additional mask to focus on the "region of interest" in front of the vehicle
- Convert the points(i.e. pixels) in XY space to lines in Hough space
- Sort Left and Right Lane lines based on the calculated slope
- Apply Curve-Fitting to each Lane-Points to draw solid lines
- Vanishing-point was substitued by the deepst point detected [y_min]
- Using the extrema of the lines generated, create two averaged lane lines
- TODO: stabilize lane-lines - the potential works are to use HistoryBuffer and use Mean or weighted factors, or use tracking filters such as: kalman filter to predict/stabilize lane-lines. I think this out of scope of this simple-project.
- Draw the lane-lines for each frame

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by implementing:
- Sorting Left and Right Lane lines based on the calculated slope. For Right-Lane i used Slope-Range:[0.4:0.9], Left-Lane Slope-Range[-0.9:-0.4]
- Apply Curve-Fitting to each Lane-Points to draw solid lines

If you'd like to include images to show how the pipeline works, here is how to include an image: 

[//]: # (Image References)

[image2]: ./test_images_output/annotated_solidYellowCurve.jpg


### 2. Identify potential shortcomings with your current pipeline


Shortcomings would be:
- Using Line-Regression would not fit for curves
- Shadows on the Road/Lighthing-conditions, or damaged lane-lines affected the processing of lane-lines 
- Vanishing-Point calculation should be mathmatically be done
- Stabilizing Lane-Lines would help in maintaing the lane-lines, if the Images could not contain any lines
- Night-Driving may affect the output


### 3. Suggest possible improvements to your pipeline

Possible improvements would be:
- Use higher-degree of Polynomial fitting to match lines on curves
- Use Line-Interesection Algorithms to better calculate the Vanishing-Point
- Use Tracking - or Buffer to stabilize Lane-Line recognition

