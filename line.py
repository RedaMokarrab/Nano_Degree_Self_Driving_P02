import matplotlib.pyplot as plt
import cv2 
import numpy as np
from collections import deque

#class lance to hold the lane global information 
#it contains data for both lines + the curvature and off center information
class Lane():
    def __init__(self,previous_margin,nwindows,windowmargin,minipix,n_iter):
        self.left_line = self.Line(n_iter)
        self.right_line = self.Line(n_iter)
        # was the lane detected in the last iteration?
        self.detected = False  
        #radius of curvature of the lane meters (average of curve for both lines)
        self.curve_m = 0 
        #calibration parameters
        #search from previous margin
        self.previous_margin = previous_margin
        # Choose the number of sliding windows
        self.nwindows = nwindows
        # Set the width of the windows +/- margin
        self.windowmargin = windowmargin
        # Set minimum number of pixels found to recenter window
        self.minpix = minipix

    
    # Define a line sub-class to receive the characteristics of each line detection
    class Line():
        def __init__(self,n_iter):
            #current x base for line 
            self.xbase_hist = deque(maxlen=n_iter)     
            #average x base location for last n frames 
            self.xbase_average=None
            #polynomial coefficients for the most recent n frame fits
            self.fit_hist = deque(maxlen=n_iter)
            #polynomial coefficients averaged over the last n iterations
            self.best_fit = None 
            #radius of curvature of the line in real units 
            self.radius_of_curvature = deque(maxlen=n_iter)
            #xfit for most recent fit
            self.current_fitx =None

    #function will be called only in first frame of if the current frame was not able to fit lines
    #search will start again
    def find_lane_pixels_sliding_window(self,binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        #append x base for each line then calculate average
        self.left_line.xbase_hist.append(leftx_base)
        self.right_line.xbase_hist.append(rightx_base)
        #average the xbase for best result
        self.left_line.xbase_average = np.mean(self.left_line.xbase_hist)
        self.right_line.xbase_average = np.mean(self.right_line.xbase_hist)
        
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated later for each window in nwindows
        #start position is taken to be the average
        leftx_current = self.left_line.xbase_average
        rightx_current = self.right_line.xbase_average


        # empty lists to receive new left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.windowmargin
            win_xleft_high = leftx_current + self.windowmargin
            win_xright_low = rightx_current - self.windowmargin
            win_xright_high = rightx_current + self.windowmargin
        
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

                # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            self.detected = True
        except ValueError:
            # Avoids an error if the above is not implemented fully
            self.detected = False
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        
        return leftx, lefty, rightx, righty,


    def find_lane_pixels_search_previous(self,binary_warped):
        
        #grab search margin outside the previous fit
        margin = self.previous_margin
        
        #grab the polyfit from best fit variable 
        left_fit= self.left_line.best_fit
        right_fit=self.right_line.best_fit
        
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
 
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        #update xbases to be used for the vehicle location calculation 
        #append x base for each line then calculate average.
        ymax=binary_warped.shape[0]
              
        leftx_base = (left_fit[0]*(ymax**2) )+ (left_fit[1]*ymax )+ left_fit[2] #assuming max y
        rightx_base = (right_fit[0]*(ymax**2)) +( right_fit[1]*ymax) + right_fit[2] #assuming max y
        self.left_line.xbase_hist.append(leftx_base)
        self.right_line.xbase_hist.append(rightx_base)
        #average the xbase for best result
        self.left_line.xbase_average = np.mean(self.left_line.xbase_hist)
        self.right_line.xbase_average = np.mean(self.right_line.xbase_hist)
        
        
        
        return leftx, lefty, rightx, righty 
    
    def fit_polynomial(self,binary_warped):
        #if first frame or last frame didn't detect a lane start searching from begining otherwise
        #search from within last frame data 
        if(self.detected == False):        
            # Find our lane pixels first
            leftx, lefty, rightx, righty = self.find_lane_pixels_sliding_window(binary_warped)
        else:
            leftx, lefty, rightx, righty = self.find_lane_pixels_search_previous(binary_warped)
       
        
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)   #retunrs A,B and C coffiecents
        right_fit = np.polyfit(righty, rightx, 2) #retunrs A,B and C coffiecents 
    
        
        return left_fit,right_fit
    
    
    def measure_curvature_real(self,left_fit,right_fit,ym_per_pix,xm_per_pix,size):

        y_eval = size

        # Calculation of R_curve (radius of curvature)
        left_curve_real = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curve_real = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        #save current curvature
        self.left_line.radius_of_curvature = left_curve_real
        self.right_line.radius_of_curvature = right_curve_real
       
        curvature_average=(left_curve_real+right_curve_real)/2
        curve_text="Current Curve is : "+(str(int(curvature_average)))+" m"
        
        return curve_text
    
    def measure_vehicle_offset(self,binary_warped,xm_per_pix):
        #save current vehicle offset text
        delta=(self.right_line.xbase_average-self.left_line.xbase_average )-(binary_warped.shape[1]/2)
        if(delta < 0):
            vehicle_offset_text="Vehicle is "+str(round(xm_per_pix*abs(delta),2))+"m left off the center"
        elif(delta>0):
            vehicle_offset_text="Vehicle is "+str(round(xm_per_pix*delta,2))+"m right off the center"  
        else:
            vehicle_offset_text="Vehicle is 0 m off the center"  
        
        return vehicle_offset_text
     
    #not completed yet, still need updates
    def lane_fit_sanity_check(self,left_fit,right_fit):
        status=True
        
        
        
        return status
    
    def get_lane_highlighted(self,binary_warped,ym_per_pix,xm_per_pix):
    
        
        left_fit,right_fit =self.fit_polynomial(binary_warped)
        
        curve_text = self.measure_curvature_real(left_fit,right_fit,ym_per_pix,xm_per_pix,binary_warped.shape[0])
        vehicle_offset_text = self.measure_vehicle_offset(binary_warped,xm_per_pix)

        #check if new fit makes sense before using it  
        if(self.lane_fit_sanity_check(left_fit,right_fit)):
            #save value in class and append to calculate the average on last n frames 
            self.left_line.fit_hist.append(left_fit)
            self.right_line.fit_hist.append(right_fit)        
            self.left_line.best_fit = np.mean(self.left_line.fit_hist, axis=0) 
            self.right_line.best_fit = np.mean(self.right_line.fit_hist, axis=0)
            left_fit=self.left_line.best_fit
            right_fit=self.right_line.best_fit 
        else:
            #overwrite with the best fit 
            left_fit=self.left_line.best_fit
            right_fit=self.right_line.best_fit 
            self.detected = False
            
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            #save current line xfit
            self.left_line.current_fitx = left_fitx
            self.right_line.current_fitx = right_fitx
            self.detected = True  #mark lane as detected
        except TypeError:
            # use last frame lines 
            print('The function failed to fit a line!')
            left_fitx = self.left_line.current_fitx
            right_fitx  = self.right_line.current_fitx
            self.detected = False #mark lane as not detected and start search from next frame and keep old data from previous n frames
       

        lane_highlight_left=  np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        lane_highlight_right=  np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_highlight_pts = np.hstack((lane_highlight_left, lane_highlight_right))
                
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img,np.int_([lane_highlight_pts]),(0,255,0))
     
        return window_img,curve_text,vehicle_offset_text