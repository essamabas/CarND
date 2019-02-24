
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
## Code Section:
The Code is constructed using 4-Classes:
- Class `Camera`: This includes Calibration, undistortion section
- Class `Image` : This includes all Filtering Thresholds: white_threshold, sobel_treshold, hls_threshold, ...
The Class takes an Image with Instancing - and create Gray/HLS Copy
An Instance should be created for every Image/Frame while processing Video
- Class `search_window` : Search Window approach based on applying a convolution-filter, which will maximize the number of "hot" pixels in each window. The Code is based on Lesson-9: Section-6 from Lectures
- Class `Lane`: Lane Class performs:
    - polynomial line fitting of the current-detected right/left lane points
    - Buffer the Coefficents of the right/left polynomial
    - Apply **Sanity Check** of current-detected right/left lane polynomial coefficients:
        - current detected Left/Right should have the same slope sign
        - current detected Left/Right slopes should have +/-3 difference from the mean of the last saved n-Iterations
    - Calculate Road-Radius
    - Calculate Lane-Center Position Offset


## Calibration Matrix
In This Section Camera Calibration will be applied based on Chessboard Images
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

**Note**: Camera Object will be used for Processing Images afterwards, so this Section should be run at first.


```python
# Calibrate Camera

# Extract Calibration Matrix by passing Chessboard images 
calib_ret, calib_mtx, calib_dist = MyCamera.calibrateByChessboard(images_folder_name="camera_cal/calibration*.jpg",size=(9,6))
print("Calibration-Matrix=", calib_mtx)
print("Calibration-Distortion-Coefficents=", calib_dist)
```
    


![png](output_images\output_3_1.png)



![png](output_images\output_3_2.png)



![png](output_images\output_3_3.png)



![png](output_images\output_3_4.png)



![png](output_images\output_3_5.png)



![png](output_images\output_3_6.png)



![png](output_images\output_3_7.png)



![png](output_images\output_3_8.png)



![png](output_images\output_3_9.png)



![png](output_images\output_3_10.png)



![png](output_images\output_3_11.png)



![png](output_images\output_3_12.png)



![png](output_images\output_3_13.png)



![png](output_images\output_3_14.png)



![png](output_images\output_3_15.png)



![png](output_images\output_3_16.png)



![png](output_images\output_3_17.png)


### Undistort Images

Now, I have calculated the Calibration-Matrix and Distortion Coefficients. <br>
I will apply undistortion Image to a Test-Image


```python
img = mpimg.imread('test_images/test6.jpg')
# undistort images
undist_img = MyCamera.undistort(img)
MyImage = Image(undist_img, interact_plot=False, show_plot=False)

# Visualize unwarp
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undist_img)
ax2.set_title('Undistorted Image', fontsize=30)
```




![png](output_images\output_5_1.png)


## Test Filters-Thresholds

Test Filtering on Test images: 
By using the Interactive-Plot - Best Threshold values has been choosed for the following Filters:

* Soebl-x Filter Thresholds - $$[30,100]$$
* Magnitude of the Gradients Thresholds - $$[50,200]$$
* White-Mask - $$[200,250]$$
* HLS- S-Channel Threshold - $$[175,255]$$
* RGB -R-Channel Threshold - $$[200,255]$$

These best-values has been applied in `Image::process_image`


### Show Combined-Threshold

![png](output_images\output_14_1.png)


## Test perspective transform

`straight_lines1.jpg` was used to extract source and distination points that to be used in Warping Images.<br>
The Coordinates are saved in `Image::calc_warp_points`

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 150, 720      | 350, 720      | 
| 560, 450      | 350, 0        |
| 720, 450      | 1010, 720     |
| 1180, 720     | 1010, 720     |



![png](output_images\output_16_1.png)


## Test Lane Lines Fitting

1. Search Windows will be applied to the Warped Image using convolution Mask. By applying a window template across the image from left to right and creating a convolved signal. The peak of the convolved signal is where highest overlap of pixels, which indicates the position for the lane marker.
2. Apply 2nd Polynomial Fitting to the found `l_points` and `r_points`


```python
# apply Search-Window
img = mpimg.imread('test_images/test2.jpg')
# undistort images
undist_img = MyCamera.undistort(img)
MyImage = Image(undist_img, interact_plot=False, show_plot=False)
# extract ROI/ WARP
roi_img, warp_img, M, M_Inv = MyImage.process_image()
    
MySearchWindow = search_window(warp_img, window_width=10, window_height=60, margin=10)
# find window-centroids
output_img, l_points, r_points = MySearchWindow.find_line_points()

leftx =  l_points.nonzero()[1]
lefty =  l_points.nonzero()[0]
rightx = r_points.nonzero()[1]
righty = r_points.nonzero()[0]

# Extract left and right line pixel positions
ploty = np.linspace(0, 719, num=720)# to cover same y-range as image

# Fit a second order polynomial to pixel positions in each fake lane line
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    
# Visualize unwarp
f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(undist_img)
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(warp_img)
ax2.set_title('Unwarped Image', fontsize=30)

# Plot up the fitting data
mark_size = 3
ax3.plot(leftx, lefty, 'o', color='red', markersize=mark_size)
ax3.plot(rightx, righty, 'o', color='blue', markersize=mark_size)
ax3.plot(left_fitx, ploty, color='green', linewidth=3)
ax3.plot(right_fitx, ploty, color='green', linewidth=3)
ax3.set_xlim(0, 1280)
ax3.set_ylim(0, 720)
ax3.invert_yaxis() # to visualize as we do the images
ax3.set_title('Window Search and Polynomial Fitting', fontsize=30)
```



![png](output_images\output_18_1.png)


---
## Test Images

We can put all the algorithms together in `lane_detect_pipeline` and test the images located in `test_images`.<br>
Images will saved in `output_images` folder.<br><br><br>
Road-Curvature has been calculated based on the [Formula](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) <br><br>

Center-Offset is calculated using:<br>
    $ CenterOffset = (ImageCenter - LaneCenterPosition) * XmPerPix $
<br>**Where:**<br>
    $ XmPerPix=3.7/700$<br>
    $ LaneCenterPosition = (Right(y_max) + Left(y_max))/2 $


```python
# Loop to all Images in Test-Images Folder
for img_name in os.listdir("test_images/"):
    # extract frame-image
    img = mpimg.imread('test_images/'+ img_name)
    
    # undistort images
    undist_img = MyCamera.undistort(img)
    
    # apply Lane-Detection Pipeline
    un_warp_img = lane_detect_pipeline(img)
    
    mpimg.imsave("output_images/annotated_"+img_name, un_warp_img)

    # Visualize unwarp
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(img)
    ax1.set_title('Orignial Image', fontsize=30)
    ax2.imshow(un_warp_img)
    ax2.set_title('Lane Image', fontsize=30)

```


![png](output_images\output_20_0.png)



![png](output_images\output_20_1.png)



![png](output_images\output_20_2.png)



![png](output_images\output_20_3.png)



![png](output_images\output_20_4.png)



![png](output_images\output_20_5.png)



![png](output_images\output_20_6.png)



![png](output_images\output_20_7.png)


---
## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on :

`project_video.mp4`

### 1.project_video.mp4


```python
output_clip1_path = 'test_videos_output/project_video.mp4'
output_clip2_path = 'test_videos_output/challenge_video.mp4'
output_clip3_path = 'test_videos_output/harder_challenge_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
clip2 = VideoFileClip("challenge_video.mp4")
clip3 = VideoFileClip("harder_challenge_video.mp4")
output_clip1 = clip1.fl_image(lane_detect_pipeline) #NOTE: this function expects color images!!
output_clip2 = clip2.fl_image(lane_detect_pipeline)
output_clip3 = clip3.fl_image(lane_detect_pipeline)

%time output_clip1.write_videofile(output_clip1_path, audio=False)
#%time output_clip2.write_videofile(output_clip2_path, audio=False)
#%time output_clip3.write_videofile(output_clip3_path, audio=False)
```


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output_clip1_path))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/project_video.mp4">
</video>



## Reflection

### Justification:
- The current pipeline worked well on gray-scale images; thus i avoided using HSV/HSL images to extract the yellow lanes - i found it very error-prone based on the selected ranges.
- The current pipeline kept the algorithms simple and straightforward to meet the Project submission dead-line

### Shortcomings would be:
- Window Searching consumed alot of time
- Using different Filters with so many threshold-values, made it difficult to extract Lane-Lines
- Shadows on the Road/Lighthing-conditions, or damaged/missing lane-lines affected the processing of lane-lines 
- Using Buffers/ Average to stabilize results, did not work in many frames, where the lines were damaged, which led to totally missleading lane-lines
- Stabilizing Lane-Lines would help maintaing the lane-lines, if the Images could not contain any lines
- Road Radius varies signicantly between Left/Right Lane Lines - due to fitting problems


### Suggest possible improvements to your pipeline

Possible improvements would be:
- Use Lane-Model (4-Points Polly-Model) and try to fit it on the road - with this approach the Lane-Lines will be
Use Tracking to stabilize Lane-Line recognition

More description are also provided in the writeup `writeup.md` saved in the workspace


