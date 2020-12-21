import matplotlib.pyplot as plt
import cv2 
import numpy as np



#class lance to hold the lane global information 
#it contains data for both lines + the curvature and off center information
class Lane():
    def __init__(self,previous_margin,nwindows,windowmargin,minipix,n_iter):
        self.left_line = self.Line(n_iter)
        self.right_line = self.Line(n_iter)
        # was the lane detected in the last iteration?
        self.detected = False  
        #lane midpoint in pixels
        self.midpoint = None
        #radius of curvature of the lane meters (average of curve for both lines)
        self.curve_m = 0 
        #calibration parameters
        #search from previous margin
        self.margin = previous_margin
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
            self.xbase = None     
            #polynomial coefficients for the most recent fit
            self.current_fit = [np.array([False])] 
            #xfit for most recent fit
            self.current_fitx =None
            #polynomial coefficients averaged over the last n iterations
            self.best_fit = collections.deque(maxlen=n_iter)  
            #average x values of the fitted line over the last n iterations
            self.bestx = collections.deque(maxlen=n_iter)       
            #radius of curvature of the line in some units
            self.radius_of_curvature = collections.deque(maxlen=n_iter)  

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

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base


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
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        #update class attributes
        #save lane mid point for off center calculation
        self.midpoint=midpoint
        self.left_line.xbase = leftx_base
        self.right_line.xbase =rightx_base

        
        return leftx, lefty, rightx, righty,


    
    def fit_polynomial(self,binary_warped):
        #if first frame or last frame didn't detect a lane start searching from begining otherwise
        #search from within last frame data 
        if(self.detected == False):        
            # Find our lane pixels first
            leftx, lefty, rightx, righty = self.find_lane_pixels_sliding_window(binary_warped)
 
        else:

            leftx, lefty, rightx, righty = self.find_lane_pixels_sliding_window(binary_warped)
       
        
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)   #retunrs A,B and C coffiecents
        right_fit = np.polyfit(righty, rightx, 2) #retunrs A,B and C coffiecents 
        
        return left_fit,right_fit
    
    
    def measure_curvature_real(self,left_fit,right_fit,size,ym_per_pix,xm_per_pix):

        y_eval = size

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        return left_curverad, right_curverad
    
    def lane_fit_sanity_check(self,left_curve,right_curve):
        status=True
        
        if((self.left_line.radius_of_curvature!=None) & (self.right_line.radius_of_curvature!=None)):
            #check line curve is almost the same 
            if(abs(left_curve-self.left_line.radius_of_curvature)>1000):
                status=False
            if(abs(right_curve-self.right_line.radius_of_curvature)>1000):
                status=False
                
            print("left_curve:" , left_curve)
            print("right_curve:" , right_curve)

        
        return status
    
    def get_lane_highlighted(self,binary_warped,ym_per_pix,xm_per_pix):
    
        
        left_fit,right_fit =self.fit_polynomial(binary_warped)
        left_curve_real,right_curve_real = self.measure_curvature_real(
            left_fit,right_fit,binary_warped.shape[0],ym_per_pix,xm_per_pix)

        #save current fit
        self.left_line.current_fit   =left_fit
        self.right_line.current_fit  =right_fit
            
        #save current curvature
        self.left_line.radius_of_curvature = left_curve_real
        self.right_line.radius_of_curvature = right_curve_real
       
        curvature_average=(left_curve_real+right_curve_real+self.curve_m)/3
        self.curve_m = curvature_average
        curve_text="Current Curve is : "+(str(int(curvature_average)))+" m"
        

        #save current vehicle offset text
        delta=(self.right_line.xbase-self.left_line.xbase )-(binary_warped.shape[1]/2)
        if(delta < 0):
            vehicle_offset_text="Vehicle is "+str(round(xm_per_pix*abs(delta),2))+"m left off the center"
        elif(delta>0):
            vehicle_offset_text="Vehicle is "+str(round(xm_per_pix*delta,2))+"m right off the center"  
        else:
            vehicle_offset_text="Vehicle is 0 m off the center"  
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            #save current line xfit
            self.left_line.current_fitx
            self.right_line.current_fitx
            
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