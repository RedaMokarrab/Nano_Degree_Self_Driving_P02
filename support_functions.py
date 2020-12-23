
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
import numpy as np
import math
import os

#function takes folder path with chessboard images and analyze the images to return the image points 
#and object points
def get_obj_image_points(path ,nx,ny):

    #lists to hold the current image obj points and imgpoints
    imgpoints = [] # 2d points in image plane.
    objpoints = [] # 3d point in real world space and and it will be based on chessboard dimensions 9*5

    #size variable
    size=[]

    #create object point array zeroed then fill with checss board coordinates
    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x and y coordinates

    #Loop over all images for calibration 
    chessboard_images=os.listdir(path)

    for filename in chessboard_images: 

      # find corners and undistort images    
        #read chessboard image 
        image = mpimg.imread(path+filename)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #update size variable 
        size= gray.shape[::-1]
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners ( note 2 images were not captured due to loss of corners (imge 4 and 5))
        if ret == True:
            # Draw and display the corners
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw corners and save in output folder
            image_corners = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            cv2.imwrite("camera_cal_output/"+filename[:-4]+"_01_Corners.jpg",image_corners)
            
    
    
    return objpoints,imgpoints,size
    
#function takes in image and returns the undistorted image based on image points and object points 
def get_dist_parameters(objpoints, imgpoints,size):
    
    #get the distorsion coff.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,size , None, None)

    return mtx,dist

#compute the M matrix for prospective transform for easier processing 
def compute_prespective_M(image ,src,dst):
    
    img_size = ( image.shape[1] , image.shape[0] )
    M = cv2.getPerspectiveTransform(src, dst)
    
    return M


#compute the inverse M matrix for prospective transform after processing to actual image 
def compute_inverse_M (image,src,dst ):
    
    img_size = ( image.shape[ 1] , image.shape[ 0 ] )
    I_M = cv2.getPerspectiveTransform(dst, src)

    
    return I_M


#return the HLS color of image 
def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


#select white and yellow colors of image 
def select_color(image ):
    
    # find colors that are in provided range and highlight it in red
    
    white_lo=    np.array([230,230,230])
    white_hi=    np.array([255,255,255])
    
    yellow_lo_RGB=   np.array([225,180,0])
    yellow_hi_RGB=   np.array([255,255,170]) 
    
    yellow_lo_HLS=   np.array([20,120,80])
    yellow_hi_HLS=   np.array([45,200,255]) 

    rgb_image = np.copy(image)
    hls_image = convert_hls(image)
    
    
    mask_1=cv2.inRange(rgb_image,white_lo,white_hi) #filter on rgb white
    mask_2=cv2.inRange(hls_image,yellow_lo_RGB,yellow_hi_RGB) #filter on rgb yellow 
    mask_3=cv2.inRange(hls_image,yellow_lo_HLS,yellow_hi_HLS) #filter on hls yellow 
    
    mask = mask_1+mask_2+mask_3
    
    result = cv2.bitwise_and(image,image, mask= mask)
    
    combined_binary = np.zeros_like(result)
    combined_binary[result>1] = 1
    
    combined_white = np.zeros_like(result)
    combined_white[result>1] = 255
    
    
    
    return combined_binary,combined_white



#Calculate the x gradient on specific channel and filter based on threshould 
def x_gradient(channel, threshould ):

    # Sobel x
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8((255*abs_sobelx)/(np.max(abs_sobelx)))
    # Threshold x gradient
    xbinary = np.zeros_like(scaled_sobel)
    xbinary[(scaled_sobel >= threshould[0]) & (scaled_sobel <= threshould[1])] = 1

    return xbinary

#function takes undistorted image and runs x gradient for v channel and l channel
#also runs color channel filter on s channel 
def lane_filter(undist_img, s_thresh,l_thresh,b_thresh,x_thresh):

    #Convert to HSV
    hsv = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HSV)
    #convert to LAB
    lab = cv2.cvtColor(undist_img, cv2.COLOR_RGB2LAB)

    v_channel = hsv[:,:,2]
    s_channel = hsv[:,:,1]
    l_channel = lab[:,:,0]
    b_channel = lab[:,:,2]


    #get gradient for both v and l channels
    vxbinary= x_gradient(v_channel,x_thresh)
    lxbinary= x_gradient(l_channel,x_thresh)

    # Threshold color channel s 
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold color channel l
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    # Threshold color channel b
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    
    combined_binary = np.zeros_like(lxbinary)
    #combined_binary[(vxbinary == 1)|(lxbinary==1)|(s_binary==1|(l_binary==1))] = 1
    combined_binary[(vxbinary == 1)|(lxbinary==1)|(s_binary==1)|(l_binary==1)|(b_binary==1)] = 1
    
   


    return combined_binary



