import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
# from moviepy.editor import VideoFileClip
import os
import scipy.misc
import pickle
import collections
from functions import *

## Reading in Calibration Images and Getting Distortion Correction Values
# plot_it = True
plot_it = False

strCalibrationIn = ["camera_cal/" + x for x in os.listdir("camera_cal/")]
# Getting rid of "thumbs.db" on my machine
if "camera_cal/Thumbs.db" in strCalibrationIn: strCalibrationIn.remove("camera_cal/Thumbs.db")

calibration_path = "camera_calibration_saved.p"

# Checking if we already have calibration settings in the local pickle file. If not, then we create them
if os.path.exists(calibration_path) == False:
    camera_calibration(strCalibrationIn,img_size=(1280,720))
    
# Reading in existing calibration values
with open(calibration_path, mode='rb') as f:
    camera_calibration_values = pickle.load(f)
mtx = camera_calibration_values["mtx"]
dist = camera_calibration_values["dist"]

# Making a plot for the report comparing distorted and undistorted images
if plot_it == True:
    img = cv2.imread(strCalibrationIn[np.random.randint(low=0, high=len(strCalibrationIn))])
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)

## Reading in Test Images
# plot_it = True
plot_it = False

strTestIn = ["test_images/" + x for x in os.listdir("test_images/")]
# Getting rid of "thumbs.db" on my machine
if "test_images/Thumbs.db" in strTestIn: strTestIn.remove("test_images/Thumbs.db")

## Undistorting test image
img = cv2.imread(strTestIn[7]) # Test image
dst = cv2.undistort(img, mtx, dist, None, mtx) # Undistorted test image

if plot_it == True:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)

## Applying colour transformation
s_thresh=(170, 255)
sx_thresh=(20, 100)
colour_binary = pipeline(dst, s_thresh=s_thresh, sx_thresh=sx_thresh)

if plot_it == True:
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)
    
    ax2.imshow(colour_binary)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

## Masking
img_size = shape(colour_binary)
height_offset = 1000; # How the height of the warped image will change
lr_padding = 200; # Left-right padding 
ratio = (1060-270)/(680-605) # Ratio of horizon straight edge to hood straight edge

# This unmodified box fits well around the lanes on a flat straight road
src = np.float32([[670+round(lr_padding/ratio), 440],[1060+lr_padding, 675],[250-lr_padding, 675],[610-round(lr_padding/ratio), 440]])
dest = np.float32([[1060+lr_padding, 440-height_offset],[1060+lr_padding, 675],[270-lr_padding, 675],[270-lr_padding, 440-height_offset]])
colour_binary_masked = region_of_interest(colour_binary, np.int32([src]))

## Warp it
# plot_it = True
plot_it = False

warped = warper(colour_binary_masked, src, dest)

if plot_it == True:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.plot(src[:,0],src[:,1],'.-r')
    ax1.plot(src[0:4:3,0],src[0:4:3,1],'.-r')
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warped)
    ax2.set_title('Warped Image', fontsize=30)

## Fitting polynomial equations to lane lines
# plot_it = True
plot_it = False

nwindows = 9 # Choose the number of sliding windows
margin = 100 # Set the width of the windows +/- margin
minpix = 50 # Set minimum number of pixels found to recenter window

valu2 = find_lines(warped, nwindows, margin, minpix, plot_it)
left_fit = valu2[0]
right_fit = valu2[1]

## Finding curvature 
plot_it = True
# plot_it = False

curvature_values = find_curvature(left_fit, right_fit)
left_curve = curvature_values[0]
right_curve = curvature_values[1]
off_center = curvature_values[2]

if plot_it == True:
    print('Left Curvature: ', left_curve, 'm')
    print('Right Curvature: ', right_curve, 'm')
    print('Distance Off Center: ', off_center, 'm')

## Warp lines back into original image shape
plot_it = True
# plot_it = False

result = unwarp_add_lane(warped, img, left_fit, right_fit, dest, src)

if plot_it == True:
    plt.imshow(result)