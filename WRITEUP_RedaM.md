## Project: 2: Advanced Lane Finding  
#### by Reda Mokarrab

---

#### Item list:

* "support_functions.py" : Contains functions needed to filter image for better lane detection
* "line.py" : Line class with member functions to process the lane and left/right line data. 
* "Part 1 Camera Calibration and Writeup preparation.ipynb" First part needs to be run once only to extract camera calibration parameters and also to generated the images to be used in the write up.
* "Part 2 Video pipeline processing.ipynb" main video pipeline stream that takes project video and process it to output the final video lane markings.

---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients, to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Chessboard"
[image2]: ./camera_cal_output/calibration2_01_Corners.jpg "Chessboard corners"
[image3]: ./camera_cal_output/calibration2_02_Undist.jpg "Chessboard undistorted"
[image4]: ./test_images/test2.jpg "Original image"
[image5]: ./output_images/test2_02_Undist.jpg "Undistorted image"
[image6]: ./output_images/test2_03_Wraped.jpg "Prospective transform"
[image7]: ./output_images/test2_04_Filtered.jpg "Color and Gradient filter"
[image8]: ./output_images/test2_05_Lane_Highlight.jpg "Sliding window search"
[image9]: ./output_images/test2_06_final.jpg "Output image"
[video1]: ./output_videos/project_video_output.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


#### The code for the following steps "Camera Calibration" and "Pipeline (single images)" is contained in IPython notebook `Part 1 Camera Calibration and Writeup preparation.ipynb` where we only have to run once each time calibration is needed.
---



### Camera Calibration

#### Camera calibration and distortion coefficients:

First step is reading the `camera_cal` folder and prcess the 20 image using function `get_obj_image_points()` and mark the chessboard corners using `cv2.findChessboardCorners()` on gray copy of the original image, after getting the `objpoints` and `imgpoints`, Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

As example following is chessboard image :
![alt text][image1]

and following is the detected corners highlighted: 
![alt text][image2]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function, then used `cv2.undistort()` function and obtained this result
Chessboard after undistortion: 
![alt text][image3]

---


### Pipeline (single images)

#### 1. undistort images :

Then I ran the same undist function over all the test images from `test_images` folder 

#Loop over all images for calibration 
test_images=os.listdir("test_images/")

```python
for filename in test_images: 

    img = cv2.imread("test_images/"+filename)

    imgsize = ( img.shape[1] , img.shape[0] )

    #undist the input image :
    undist= cv2.undistort(img, mtx, dist, None, mtx)

    #save undistorted images 
    cv2.imwrite("output_images/"+filename[:-4]+"_02_Undist.jpg",undist)
```

Example:
* Original image: 
![alt text][image4]

* Undist image:
![alt text][image5]



#### 2. Perspective transform:

In this part I read the two images with straight lines and then using paint I eyeballed the src points, 
then took the average of transoform matrix to be used later for prospective transform. 
I also calculated the Inverse matrix to be used later.
```python
#calculate M for images with straight lines
undist_1 = cv2.imread("output_images/straight_lines1_02_undist.jpg")
undist_2 = cv2.imread("output_images/straight_lines2_02_undist.jpg")

#image 1: 
src=  np.float32(((207, 720), (582, 460), (701, 460), (1090, 720)))
dst = np.float32( ((240, 720),(240, 0),(850, 0),(850, 720)))
M_1 = compute_prespective_M(undist_1,src,dst)
#image 2: 
src=  np.float32(((207, 720), (582, 460), (705, 460), (1090, 720)))
M_2 = compute_prespective_M(undist_2,src,dst)

M_mean = (M_1+M_2)/2
MInverse = compute_inverse_M (undist_1,src,dst )
```

By using the `cv2.warpPerspective` function and calculated M matrix I was able to transfor the image as follows: 
![alt text][image6]


#### 3. Image filter :


I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`). 
Used methods and thresholds in function `lane_filter`  were as follows: 

| Method                   | thresholds    | 
|:------------------------:|:-------------:| 
| xgradient v channel      | 20,100        | 
| xgradient l channel      | 20,100        | 
| s channel in HSV         | 150, 255      |
| l channel in HSL         | 200, 255      | 


Here's an example of my output for this step. 
![alt text][image7]


#### 4. Extract lane pixels: 

Then I used the sliding window method to mark the pixels that are existing in the expected location of a lane using function `fit_polynomial` and used the following paramaters 

```python
#  number of sliding windows
nwindows = 9
#  the width of the windows +/- margin
margin = 50
#  minimum number of pixels found to recenter window
minpix = 50
```

![alt text][image8]



#### 5. Add the detected lane as overlay:
 
 After taking back the fitted lines for left and right lines I colored the area in between and added to the original image before applying the inverse matrix calculated above in step 2 

```python
window_img,temp = fit_polynomial(filtered_binary)
window_img_unwarped =cv2.warpPerspective(window_img, MInverse, imgsize)
    
lane_image_final = cv2.addWeighted(undist, 1, window_img_unwarped, 0.3, 0)
```    
and following is the final image after doing it.

![alt text][image9]


### 6. Saving the calibration parameters:
Used the pickle library to save the calibration parameters needs to be used in part 2 as follows:

```python
calibration_parameters = {"mtx": mtx,"dist_coff": dist,
                         "matrix":M_mean, "inverse_matrix":MInverse,
                         "s_channel_thresh":s_thres,"gradient_thresh":x_thresh,
                         "l_channel_thresh":l_thresh}

pickle.dump(calibration_parameters, open( "calibration_parameters.p", "wb" ))
```

### 7. Vehicle position and curve rad:
##### This is done as part of the video pipeline in `line.py` file.

by using the following conversion parameters we will be able to transform from pixel dimensions to real world dimension:
```python
ym_per_pix = (30/720) # meters per pixel in y dimension
xm_per_pix = (3.7/700) # meters per pixel in x dimension
```

*curvature calculation :
```python
y_eval = size
# Calculation of R_curve (radius of curvature)
left_curve_real = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curve_real = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#save current curvature
self.left_line.radius_of_curvature = left_curve_real
self.right_line.radius_of_curvature = right_curve_real
curvature_average=(left_curve_real+right_curve_real)/2
curve_text="Current Curve is : "+(str(int(curvature_average)))+" m"
```

* vehicle position: 
```python
delta=(self.right_line.xbase_average-self.left_line.xbase_average )-(binary_warped.shape[1]/2)
if(delta < 0):
    vehicle_offset_text="Vehicle is "+str(round(xm_per_pix*abs(delta),2))+"m left off the center"
elif(delta>0):
    vehicle_offset_text="Vehicle is "+str(round(xm_per_pix*delta,2))+"m right off the center"  
else:
    vehicle_offset_text="Vehicle is 0 m off the center"      
```


---

### Pipeline (video)

#### The code for the following section is contained in IPython notebook `Part 2 Video pipeline processing.ipynb` ,`support_functions.py` and `line.py` 


I moved the functions used in part 1 after optimization to `line.py` and `support_functions.py` files to be used in the pipeline.

Class line is defined as follows: 


```python
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
    def find_lane_pixels_search_previous(self,binary_warped):
    def fit_polynomial(self,binary_warped):
    def measure_curvature_real(self,left_fit,right_fit,ym_per_pix,xm_per_pix,size):
    def measure_vehicle_offset(self,binary_warped,xm_per_pix):
    def get_lane_highlighted(self,binary_warped,ym_per_pix,xm_per_pix):
```    

Class in nutshell, 
Main class is Lane which has two subclass lines (left/right) and queue to hold last n fit coffiecients and some info related to line sucgh as curverature, 
When image provided to `get_lane_highlighted` function If first time to run I execute the `find_lane_pixels_sliding_window` function otherwise if second time I run the `find_lane_pixels_search_previous`  


Following is the output after running project video.

Here's a [video1]



---

### Discussion

#### Areas of improvement:

* Detection of lane sanity check such as the curve delta between both lines (left/right) and expected lane width is not done yet which didn't filter out the line fits that were wrong and affected the average process. funtion `lane_fit_sanity_check` added but didn't know the best way to do it  

* Taking vehicle CAN input from vehicle as a verification of what is being detected by camera can help alot (for ex. the yaw rate and wheel angle) 

* doing more cycles of sliding window before applying search from previous would be better specially if video started with bad lanes.


