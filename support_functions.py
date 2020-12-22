import matplotlib.pyplot as plt
import cv2 
import numpy as np

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
def lane_filter(undist_img, s_thresh,l_thresh,x_thresh):

    #Convert to HSV
    hsv = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HSV)
    #convert to HLS 
    hls = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS)

    v_channel = hsv[:,:,2]
    s_channel = hsv[:,:,1]
    l_channel = hls[:,:,1]

    #get gradient for both v and l channels
    vxbinary= x_gradient(v_channel,x_thresh)
    lxbinary= x_gradient(l_channel,x_thresh)

    # Threshold color channel s 
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold color channel l 
    l_binary = np.zeros_like(l_channel)
    l_binary [((l_channel>= l_thresh[0])& (l_channel<= l_thresh[1]))]=1


    combined_binary = np.zeros_like(lxbinary)
    combined_binary[(vxbinary == 1)|(lxbinary==1)|(s_binary==1)|(l_binary==1)] = 1


    return combined_binary



