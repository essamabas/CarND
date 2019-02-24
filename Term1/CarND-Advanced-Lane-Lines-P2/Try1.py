import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

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
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((size[0]*size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(images_folder_name)
        print(images)
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
                    img = cv2.drawChessboardCorners(img, size, corners, ret)
                    cv2.imshow('img',img)
        
        if(len(objpoints) > 0) and (len(imgpoints) > 0):
            # Use cv2.calibrateCamera() and cv2.undistort()
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    
    # returns the undistorted image
    def undistort(self,img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist

    # Warp Image - based on Src-Points, Dst-Points
    # TODO: calculate src-points and distination points - may use OFFSET as in Exercise
    def Warp(self, img, src, dst):
        img_size = (img.shape[1],img.shape[0])

        # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped, M, Minv

    # img should be gray-image
    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel=3, thresh=(20, 100)):
        # Calculate directional gradient
        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # Apply threshold
        return sxbinary

    def mag_thresh(self, gray, sobel_kernel=3, thresh=(0, 255)):
        # Calculate gradient magnitude
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        # Apply threshold
        return mag_binary

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply threshold
        dir_binary = []
        return dir_binary
    
    # TODO: apply s_thresh_min = 170, s_thresh_max = 255
    def color_threshold(self, img , thresh=(170, 255)):
        #convert to hls-image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        #extract s-Channel
        s_channel = hls[:,:,2]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

        return s_binary
    
    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

# Calibrate Camera
MyCamera = Camera()
MyCamera.calibrateByChessboard(images_folder_name="camera_cal/calibration*.jpg",size=(8,6))