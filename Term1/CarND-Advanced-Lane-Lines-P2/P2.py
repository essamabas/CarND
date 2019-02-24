
# coding: utf-8

# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---
# ## First, I'll compute the camera calibration using chessboard images


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# Configuration
plt.rcParams["figure.figsize"] = [16,9]

# Helper-Functions
# add-subplot 
def subplot_img(fig, img, col_max, col_idx, title):
    # suplot-parameters index nrow, ncolumns, index
    a = fig.add_subplot(1, col_max, col_idx)
    img_plot = plt.imshow(img)
    # set Ttile of subplot
    a.set_title(title)
    return a

# add-subplot 
def subplot_data(fig, data, col_max, col_idx, title):
    # suplot-parameters index nrow, ncolumns, index
    a = fig.add_subplot(1, col_max, col_idx)
    m_plot = plt.plot(data)
    # set Ttile of subplot
    a.set_title(title)
    return a
        
class Camera():
    # 
    def __init__(self):
        # Inital-Camera Values
        self.mtx =[]
        #distortion coefficients
        self.dist=[]
        # rotation vector
        self.rvecs=[]
        # translation vector
        self.tvecs=[]
    
    
    #images_folder_name = 'camera_cal/calibration*.jpg'
    #size = (9,6)
    def calibrateByChessboard(self, images_folder_name, size, draw_corners = True):
        # initialize variables
        col_max = 2
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((size[0]*size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(images_folder_name)
        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
                
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, size,None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                if draw_corners == True:
                    # define new figure
                    fig = plt.figure()
                    subplot_img(fig, img, col_max, 1, "original")
                    img_mod = cv2.drawChessboardCorners(img, size, corners, ret)
                    subplot_img(fig, img, col_max, 2, "Corners")
            else:
                print("No Corners have been found for:" + fname)
        
        # Check that object points, image points are not empty
        if(len(objpoints) > 0) and (len(imgpoints) > 0):
            # Use cv2.calibrateCamera() and cv2.undistort()
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    
    # returns the undistorted image
    def undistort(self,img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist


class Image():
    
    def __init__(self, img, interact_plot=False, show_plot=False):
        # Save-Image Structure
        self.img = img
        # Save Plot-Configuration for Test-Images
        self.show_plot = show_plot
        # Save Interact Configuration for Testing Thresholds
        self.interact_plot = interact_plot
        
        #Convert to Gray
        self.gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #convert to hls-image
        self.hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
    def white_thresh(self, w_thresh_min= 200, w_thresh_max = 255):
        # Apply White-color-mask
        mask_white = cv2.inRange(self.gray, w_thresh_min, w_thresh_max)
        white_image = cv2.bitwise_and(self.gray, mask_white)
        white_binary = np.zeros_like(white_image)
        white_binary[(white_image >= w_thresh_min) & (white_image <= w_thresh_max)] = 1
        
        if self.interact_plot == True:
            plt.imshow(white_binary)
            plt.show()
    
        return white_binary

    # img should be gray-image
    def abs_sobel_thresh(self, sobel_kernel=3, thresh_min=20, thresh_max=100):
        # Calculate directional gradient
        # Sobel x
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        if self.interact_plot == True:
            plt.imshow(sxbinary)
            plt.show()
        
        return sxbinary

    def mag_thresh(self, sobel_kernel=3, m_thresh_min= 0, m_thresh_max= 255):
        # Calculate gradient magnitude
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= m_thresh_min) & (gradmag <= m_thresh_max)] = 1

        if self.interact_plot == True:
            plt.imshow(mag_binary)
            plt.show()
            
        # Apply threshold
        return mag_binary

    def dir_threshold(self, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply threshold
        dir_binary = []
        return dir_binary
    
    # TODO: apply s_thresh_min = 170, s_thresh_max = 255
    def hls_threshold(self, s_thresh_min=170, s_thresh_max = 255):

        #extract s-Channel
        s_channel = self.hls[:,:,0]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        if self.interact_plot == True:
            plt.imshow(s_binary)
            plt.show()
            
        return s_binary
    
    def rgb_threshold(self, r_thresh_min = 200, r_thresh_max= 255):

        #extract s-Channel
        r_channel = self.img[:,:,2]

        # Threshold color channel
        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel >= r_thresh_min) & (r_channel <= r_thresh_max)] = 1

        if self.interact_plot == True:
            plt.imshow(r_binary)
            plt.show()
            
        return r_binary
    
    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # Save Vertices of ROI
        self.vertices = vertices
        
        #defining a blank mask to start with
        mask = np.zeros_like(img)   

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, np.int_([vertices]), ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    def hist(self, img):
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0]//2:,:]

        # Sum across image pixels vertically - make sure to set an `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        return histogram

    # Warp Image - based on Src-Points, Dst-Points
    # TODO: calculate src-points and distination points - may use OFFSET as in Exercise
    def warp(self, img, src, dst):
        img_size = (img.shape[1], img.shape[0])

        # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped, M, Minv
    
    # unWarp Image - based on saved MINV
    def unwarp(self, img, Minv):
        
        # get size of warped image
        img_size = (img.shape[1], img.shape[0])
        
        # use cv2.warpPerspective() to warp your image to a top-down view
        un_warp_img = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
        
        # Combine the result with the original image
        new_img = np.copy(self.img)
        un_warp_img = cv2.addWeighted(new_img, 1, un_warp_img, 0.3, 0)
        
        return un_warp_img
    
    def calc_warp_points(self, x_center_adj=0):    
        
        '''
        # calculator the vertices of the region of interest
        imshape = self.img.shape
        xcenter=imshape[1]/2+x_center_adj
        #     xfd=55
        #     yf=450
        #     xoffset=100
        xfd=54
        yf=imshape[0]/2
        xoffset=100

        src = np.float32(
            [(xoffset,imshape[0]),
             (xcenter-xfd, yf), 
             (xcenter+xfd,yf), 
             (imshape[1]-xoffset,imshape[0])])

        
        dst = np.float32(
            [(xcenter - 2*xoffset,imshape[0]),
             (xcenter - 2*xoffset,0),
             (xcenter + 2*xoffset, 0),
             (xcenter + 2*xoffset,imshape[1])])

        '''
        # let us assume that vertices represent rectangle in real-world
        src = np.float32([[150, 720],[560, 450] ,[720, 450] , [1180, 720]])
        dst = np.float32([[350, 720],[210, 0], [1010, 0],[870,720]])

        return src, dst
    
    def draw_lane_lines(self, warped, leftx, lefty, rightx, righty):
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([leftx, lefty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, righty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)
        
        return color_warp
    
    def draw_lane_data(self, img, curv_rad, center_dist):

        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
        cv2.putText(img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
        
        direction = ''
        if center_dist > 0:
            direction = 'right'
        elif center_dist < 0:
            direction = 'left'
        abs_center_dist = abs(center_dist)
        text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
        cv2.putText(img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
        
        return img

    def process_image(self, img_name=""):
        
        # check if show_plot is True
        if self.show_plot == True:
            col_max = 8 
            # define new figure
            fig = plt.figure()
            subplot_img(fig, self.img, col_max, 1, "undist_" + img_name)

        # filter 
        # apply Sobel-x: variant (10,160)
        gradx = self.abs_sobel_thresh(sobel_kernel= 3, thresh_min=30, thresh_max=100)
        if self.show_plot == True:
            subplot_img(fig, gradx, col_max, 2, "sobelx")

        mag_binary  = self.mag_thresh(sobel_kernel=9, m_thresh_min=50, m_thresh_max = 200)
        if self.show_plot == True:
            subplot_img(fig, mag_binary, col_max, 3, "magnitude")

        white_binary =  self.white_thresh(w_thresh_min= 200, w_thresh_max=250)
        if self.show_plot == True:
            subplot_img(fig, white_binary, col_max, 4, "white")

        hls_binary = self.hls_threshold(s_thresh_min=175, s_thresh_max=255)
        rgb_binary = self.rgb_threshold(r_thresh_min=200, r_thresh_max=255)
        if self.show_plot == True:
            subplot_img(fig, hls_binary, col_max, 5, "hls threshold")

        combined = np.zeros_like(gradx)
        combined[((gradx == 1) | (mag_binary == 1)) | ( (rgb_binary == 1) | (white_binary == 1)  | (hls_binary == 1) )] = 1
        if self.show_plot == True:
            subplot_img(fig, combined, col_max, 6, "combined")

        # Define a region-of-Interest
        imshape = img.shape
        vertices = np.array([[(0,imshape[0]),
                              ((imshape[1]/2)-10, (imshape[0]/2)+20), 
                              ((imshape[1]/2)+10, (imshape[0]/2)+20), 
                              (imshape[1],imshape[0])]], dtype=np.int32)


        # let us assume that vertices represent rectangle in real-world
        src_points, dst_points = self.calc_warp_points()
        roi_img = self.region_of_interest(combined, src_points)
        if self.show_plot == True:
            subplot_img(fig, roi_img, col_max, 7, "roi_img")

        warp_img, M, M_Inv = self.warp(roi_img, src_points,dst_points)
        if self.show_plot == True:
            subplot_img(fig, warp_img, col_max, 8 , "warp_img")

        # Create histogram of image binary activations
        histogram = self.hist(warp_img)
        #subplot_data(fig, histogram, col_max, 7 , "hist_img")
        
        return roi_img , warp_img,M, M_Inv 

class search_window():
    # Init
    def __init__(self, image, window_width=50, window_height=80, margin=100):
        # window settings
        self.window_width = 50 
        self.window_height = 80 # Break image into 9 vertical layers since image height is 720
        self.margin = 100 # How much to slide left and right for searching
        self.img = image

    def window_mask(self,center,level):
        output = np.zeros_like(self.img)
        output[int(self.img.shape[0]-(level+1)*self.window_height):int(self.img.shape[0]-level*self.window_height), \
        max(0,int(center-self.window_width/2)):min(int(center+self.window_width/2),self.img.shape[1])] = 1
        return output

    def find_window_centroids(self):

        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(self.img[int(3*self.img.shape[0]/4):,:int(self.img.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))- self.window_width/2
        r_sum = np.sum(self.img[int(3*self.img.shape[0]/4):,int(self.img.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))- self.window_width/2+int(self.img.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(self.img.shape[0]/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(self.img[int(self.img.shape[0]-(level+1)* self.window_height):\
            int(self.img.shape[0]-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = self.window_width/2
            l_min_index = int(max(l_center+offset-self.margin,0))
            l_max_index = int(min(l_center+offset+ self.margin,self.img.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-self.margin,0))
            r_max_index = int(min(r_center+offset+ self.margin,self.img.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    
    def find_line_points(self):
        # find certroids
        window_centroids = self.find_window_centroids()
        
        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(self.img)
            r_points = np.zeros_like(self.img)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(window_centroids[level][0],level)
                r_mask = self.window_mask(window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage= np.dstack((self.img, self.img, self.img))*255 # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((self.img,self.img,self.img)),np.uint8)
            
        return output, l_points, r_points
    
    
# Define a class to receive the characteristics of each line detection
class Lane():
    
    def __init__(self):
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
        # was the line detected in the last iteration?
        self.detected = False  
        
        # x values of the last n fits of the line
        self.buffer_left_fit = []
        self.buffer_right_fit = []
        
        # https://docs.python.org/2/library/collections.html
        #d.appendleft('f')                # add a new entry to the left side
        #d.pop()                          # return and remove the rightmost item
        # d.clear()                        # empty the deque
        # list(d)                          # list the contents of the deque
        
        #average x values of the fitted line over the last n iterations
        self.best_left_fitx = []  
        self.best_right_fitx = []
        
        #polynomial coefficients averaged over the last n iterations
        self.best_left_fit  = [None]
        self.best_right_fit = [None] 
        
        #polynomial coefficients for the most recent fit
        self.current_left_fit = [np.array([False])]  
        self.current_right_fit = [np.array([False])]
        
        #radius of curvature of the line in some units
        self.left_curverad = None 
        self.right_curverad = None
        
        #distance in meters of vehicle center from the line
        self.center_offset = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        self.det_leftx = None  
        self.det_rightx = None 
        #y values for detected line pixels
        self.det_lefty = None
        self.det_righty = None
        
        # Initial Detection-Error Counter - to be incremented/decremented everytime the check failed/passed
        self.det_error_counter = 0

    def find_lane_pixels(self, warp_img):

        # Initialize detected flag
        self.detected = False
            
        # get height
        self.h = warp_img.shape[0]
        self.w = warp_img.shape[1]
        
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, self.h-1, num=self.h)
        
        # create an instance of Search-window to search for left/right lanes
        MySearchWindow = search_window(warp_img, window_width=10, window_height=60, margin=10)
        
        # find window-centroids
        output_img, l_points, r_points = MySearchWindow.find_line_points()
        
        # extract x,y of left/right lanes
        self.det_leftx =  l_points.nonzero()[1]
        self.det_lefty =  l_points.nonzero()[0]
        self.det_rightx = r_points.nonzero()[1]
        self.det_righty = r_points.nonzero()[0]

        if len(self.det_leftx) != 0 and len(self.det_rightx) != 0:
            # lines are detected for this Image
            self.detected = True 
        
        return output_img, l_points, r_points

    
    def check_current_fit(self, left_fit, right_fit):
        
        tmp_check = False
        
        #check if left/right match the same slope margin from History
        if self.best_left_fit[0] != None:
            if np.absolute(self.best_left_fit[0] - left_fit[1]) < 3: 
                tmp_check = True
            else:
                tmp_check = False
                
            if np.absolute(self.best_right_fit[0] - right_fit[1]) < 3: 
                tmp_check = True
            else:
                tmp_check = False
        
        else:
            # 1st best_left_fit
            self.best_left_fit = left_fit
            self.best_right_fit = right_fit
        
        # check if left/right lines have the same slope margin
        if np.sign(left_fit[0]) == np.sign(right_fit[0]):
            tmp_check = True
        else:
            tmp_check = False
    
        # 
        if tmp_check == False:
            # check that best_left/right_fit are once defined
            if self.best_left_fit[0]!= None and self.best_right_fit[0]!= None:
                self.current_left_fit = self.best_left_fit
                self.current_right_fit = self.best_right_fit
                
            # Increment Error-Counter
            self.det_error_counter = self.det_error_counter + 1
            
            # if Error-Counter excceeds limit
            #if(self.det_error_counter > 20):
            
        else:
            if self.det_error_counter > 0:
                # Increment Error-Counter
                self.det_error_counter = self.det_error_counter - 1
            
            self.current_left_fit = left_fit
            self.current_right_fit = right_fit
            
            # Check that Buffer is not Empty
            if(len(self.buffer_left_fit) > 20):
                #Remove First Item
                self.buffer_left_fit.pop(1)
                
            # Check that Buffer is not Empty
            if(len(self.buffer_right_fit) > 20):
                #Remove First Item
                self.buffer_right_fit.pop(1)
            
            # Append to Buffer
            self.buffer_left_fit.append(self.current_left_fit)
            self.buffer_right_fit.append(self.current_right_fit)
            # get best_left/right_fit
            self.best_left_fit = np.mean(self.buffer_left_fit, axis=0)
            self.best_right_fit = np.mean(self.buffer_right_fit, axis=0)
                
        return self.current_left_fit, self.current_right_fit
    
    
    def fit_polynomial(self):
                
        # check if lines are detected
        if self.detected == True:
               
            # extract fitting paramters 
            left_fit = np.polyfit(self.det_lefty, self.det_leftx, 2)
            right_fit = np.polyfit(self.det_righty, self.det_rightx, 2)
            print("left_fit=", left_fit)
            print("right_fit=", right_fit)
            
            # Sanity Check 
            left_fit, right_fit = self.check_current_fit(left_fit, right_fit)
                
            try:
                left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
                right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
            except TypeError:
                # Avoids an error if `left` and `right_fit` are still none or incorrect
                print('The function failed to fit a line!')
                left_fitx = 1*self.ploty**2 + 1*self.ploty
                right_fitx = 1*self.ploty**2 + 1*self.ploty

        else:
            # use best_left_fitx/ best_right_fitx
            left_fitx = []
            right_fitx = []
            
        return self.ploty, left_fitx, right_fitx

    def calc_center_offset(self):
        '''
        Calculates the car center offset to the lane center
        ''' 
        image_center = self.w/2
        left_x_int = self.current_left_fit[0]*self.h**2 + self.current_left_fit[1]* self.h + self.current_left_fit[2]
        right_x_int = self.current_right_fit[0]*self.h**2 + self.current_right_fit[1]*self.h + self.current_right_fit[2]
        lane_center_position = (left_x_int + right_x_int) /2
        
        # Calculate Center-Offset
        self.center_offset = (image_center - lane_center_position) * self.xm_per_pix
        return self.center_offset
        
    def calc_road_curvature(self):
        '''
        Calculates the curvature of polynomial functions in pixels.
        ''' 
        # Initial Value of Curve-Radius
        left_curverad = 0
        right_curverad = 0
        
        # Fit a second order polynomial to pixel positions in each fake lane line
        if len(self.det_leftx) != 0 and len(self.det_rightx) != 0:
            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(self.det_lefty*self.ym_per_pix, self.det_leftx* self.xm_per_pix, 2)
            right_fit_cr = np.polyfit(self.det_righty*self.ym_per_pix, self.det_rightx* self.xm_per_pix, 2)
    
            # Define y-value where we want radius of curvature
            # We'll choose the maximum y-value, corresponding to the bottom of the image
            y_eval = np.max(self.ploty)
            y_eval_cr = y_eval*self.ym_per_pix

            ##### Calculate the R_curve (radius of curvature) #####
            self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_cr + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_cr + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        return self.left_curverad, self.right_curverad








       


# ## Calibration Matrix
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

# In[19]:


# Calibrate Camera
# Create an Camera-Object
MyCamera = Camera()
# Extract Calibration Matrix by passing Chessboard images 
MyCamera.calibrateByChessboard(images_folder_name="D:\\carND\\CarND-Advanced-Lane-Lines-P2\\camera_cal\\calibration*.jpg",size=(9,6))


# apply Search-Window
img = mpimg.imread('D:\\carND\\CarND-Advanced-Lane-Lines-P2\\test_images\\test6.jpg')

# undistort images
undist_img = MyCamera.undistort(img)
MyImage = Image(undist_img, interact_plot=False, show_plot=False)

# extract ROI/ WARP
roi_img, warp_img,M, M_Inv = MyImage.process_image("")

# instance line-Object
MyLane = Lane()
output_img, l_points, r_points = MyLane.find_lane_pixels(warp_img)
ploty, left_fitx, right_fitx = MyLane.fit_polynomial()
left_curverad, right_curverad =  MyLane.calc_road_curvature()
center_offset = MyLane.calc_center_offset()

# Unwarp Image with Lane-Lines
Warped_Lane_img = MyImage.draw_lane_lines(warp_img, left_fitx, ploty, right_fitx, ploty)
un_warp_img     = MyImage.unwarp(Warped_Lane_img,M_Inv)
un_warp_img     = MyImage.draw_lane_data(un_warp_img, (left_curverad+right_curverad)/2, center_offset)


# Visualize unwarp
f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(undist_img)
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(un_warp_img)
ax2.set_title('Unwarped Image', fontsize=30)

# Plot up the fitting data
mark_size = 3
leftx =  l_points.nonzero()[1]
lefty =  l_points.nonzero()[0]
rightx = r_points.nonzero()[1]
righty = r_points.nonzero()[0]
ax3.plot(leftx, lefty, 'o', color='red', markersize=mark_size)
ax3.plot(rightx, righty, 'o', color='blue', markersize=mark_size)
ax3.plot(left_fitx, ploty, color='green', linewidth=3)
ax3.plot(right_fitx, ploty, color='green', linewidth=3)
ax3.set_xlim(0, 1280)
ax3.set_ylim(0, 720)
ax3.invert_yaxis() # to visualize as we do the images