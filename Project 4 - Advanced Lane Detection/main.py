import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
# from moviepy.editor import VideoFileClip
import os
import scipy.misc
import pickle

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def camera_calibration(strCalibrationIn, img_size):
    
    # prepare object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    for x in strCalibrationIn:
        img = cv2.imread(x)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["ret"] = ret
    dist_pickle["rvecs"] = rvecs
    dist_pickle["tvecs"] = tvecs
    pickle.dump(dist_pickle, open("camera_calibration_saved.p", "wb" ))

## Reading in Calibration Images and Getting Distortion Correction Values
# plot_it = True
plot_it = False

strCalibrationIn = ["camera_cal/" + x for x in os.listdir("camera_cal/")]
# Getting rid of "thumbs.db" on my machine
if "camera_cal/Thumbs.db" in strCalibrationIn: strCalibrationIn.remove("camera_cal/Thumbs.db")

calibration_path = "camera_calibration_saved.p"

# Checking if we already have calibration settings in the local pickle file
# If not, then we create them
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
# plot_it = False

strTestIn = ["test_images/" + x for x in os.listdir("test_images/")]
# Getting rid of "thumbs.db" on my machine
if "test_images/Thumbs.db" in strTestIn: strTestIn.remove("test_images/Thumbs.db")

img = cv2.imread(strTestIn[1]) # Test image
dst = cv2.undistort(img, mtx, dist, None, mtx) # Undistorted test image

if plot_it == True:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    










