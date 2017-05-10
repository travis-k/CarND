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
from helper_functions import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time
from scipy.ndimage.measurements import label

class Lines:
    def __init__(self):
    
        # self.final_img = p4_pipeline(img)
        # was the line detected in the last iteration?
        self.left_detected = False  
        self.right_detected = False
        #polynomial coefficients for the most recent fit
        self.current_left_fit = [np.array([False])]
        self.current_right_fit = [np.array([False])]  
        # Radius of curvature and off-center
        self.left_curvature = None
        self.right_curvature = None
        self.off_center = None
        
        self.recent_leftx = None 
        self.recent_rightx = None
        
        self.n = 0 # Number of fits we've stored in here
        
        #average x values of the fitted line over the last n iterations
        self.bestx_right = None 
        self.bestx_left = None
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None  
        self.best_fit_right = None
        
        #difference in fit coefficients between last and new fits
        self.diffs_left = None
        self.diffs_right = None
        
        ##
        self.heatmaps = 0
        # self.heat = np.zeros((1,720,1280))
        self.heat = np.array([]).reshape(0,720,1280)


def p5_pipeline(img, self):

    plot_it = False

    ## Reading in Calibration Images and Getting Distortion Correction Values
    
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
        img_cal = mpimg.imread(strCalibrationIn[np.random.randint(low=0, high=len(strCalibrationIn))])
        dst = mpimg.undistort(img_cal, mtx, dist, None, mtx)
        
        f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img_cal)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)

    ## Undistorting image
    dst = cv2.undistort(img, mtx, dist, None, mtx) # Undistorted test image
    
    if plot_it == True:
        f2, (ax3, ax4) = plt.subplots(1, 2, figsize=(20,10))
        ax3.imshow(img)
        ax3.set_title('Original Image', fontsize=30)
        ax4.imshow(dst)
        ax4.set_title('Undistorted Image', fontsize=30)
    
    
    ## Training Linear SVM classifier    

    model_path = "svc_model.p"
    
    # If svc_model.p does not exist, build,train and save a new model
    if os.path.exists(model_path) == False:
        # Reading in existing training data (made with data_processing.py)
        with open('train_data.p', mode='rb') as f:
            pickle_data = pickle.load(f)
        X = pickle_data["X"]
        y = np.squeeze(pickle_data["y"])
        X_scaler = pickle_data["X_scaler"] 
        
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
        
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC') 
        
        svc_pickle = {}
        svc_pickle["svc"] = svc
        svc_pickle["X_scaler"] = X_scaler
        pickle.dump(svc_pickle, open(model_path, "wb" ))
    
    # Reading in existing svc model
    with open(model_path, mode='rb') as f:
        pickle_data = pickle.load(f)
    svc = pickle_data["svc"]
    X_scaler = pickle_data["X_scaler"]
    
    ## Locating cars 

    # Values used in data preprocessing:
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (32,32)
    hist_bins=32

    # Far
    ystart = 400
    ystop = 500
    scale = 1
    out_img1, bbox1 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    if bbox1 == []:
        bbox1 = np.array([]).reshape(0,2,2)
            
    # Mid-Range
    ystart = 400
    ystop = 650
    scale = 2   
    out_img2, bbox2 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)   
    if bbox2 == []:
        bbox2 = np.array([]).reshape(0,2,2)

    # Close Range
    ystart = 500
    ystop = 700
    scale = 2.5   
    out_img3, bbox3 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins) 
    if bbox3 == []:
        bbox3 = np.array([]).reshape(0,2,2)

    bbox = np.concatenate((bbox1,bbox2,bbox3),axis=0)
    
    # plot_it = True
    if plot_it == True:
        f37, (ax37, ax38, ax39) = plt.subplots(1,3)
        ax37.imshow(out_img1)
        ax37.set_title('Small Scale Search (Far)')
        ax38.imshow(out_img2)
        ax38.set_title('Mid Range Search')
        ax39.imshow(out_img3)
        ax39.set_title('Close Range Search')
        
    ## Applying heat map
    plot_it = False
    print(self.heatmaps)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    draw_img = np.copy(img)
    
    if len(bbox) > 0:
        # Add heat to each box in box list
        heat = np.expand_dims(add_heat(heat,bbox.astype(int)),0)
        
        self.heat = np.append(self.heat,heat,axis=0)
        self.heatmaps += 1
        
        if self.heatmaps < 5:
            # heat_avg = np.mean(self.heat,axis=0)
            heat_avg = np.sum(self.heat,axis=0)
        else:
            # heat_avg = np.mean(self.heat[self.heatmaps-5:self.heatmaps,:,:],axis=0)
            heat_avg = np.sum(self.heat[self.heatmaps-5:self.heatmaps,:,:],axis=0)
        
        # Apply threshold to help remove false positives
        heat_avg = apply_threshold(heat_avg,11)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat_avg, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)    
        
        if plot_it == True:
            f35, (ax35, ax36) = plt.subplots(1, 2)
            ax35.imshow(draw_img)
            ax35.set_title('Car Positions')
            ax36.imshow(heatmap, cmap='hot')
            ax36.set_title('Heat Map')
            f35.tight_layout()
    
    ##
  
  
  
  
  
  
  
  
  
  
  
    
    
    
    final_img = draw_img
    # ## Applying colour transformation
    # s_thresh=(170, 255)
    # sx_thresh=(20, 100)
    # colour_binary = cb_pipeline(dst, s_thresh=s_thresh, sx_thresh=sx_thresh)

  #   ##   if plot_it == True:
    #     # Plot the result
    #     f3, (ax5, ax6) = plt.subplots(1, 2, figsize=(24, 9))
    #     f3.tight_layout()
    #     
    #     ax5.imshow(img)
    #     ax5.set_title('Original Image', fontsize=40)
    #     
    #     ax6.imshow(colour_binary)
    #     ax6.set_title('Colour Binary', fontsize=40)
    #     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # 
    # ## Masking
    # img_size = np.shape(colour_binary)
    # height_offset = 1500; # How the height of the warped image will change
    # lr_padding = 200; # Left-right padding 
    # ratio = (1060-270)/(680-605) # Ratio of horizon straight edge to hood straight edge
    # 
    # # This unmodified box fits well around the lanes on a flat straight road
    # src = np.float32([[670+round(lr_padding/ratio), 440],[1060+lr_padding, 675],[250-lr_padding, 675],[610-round(lr_padding/ratio), 440]])
    # dest = np.float32([[1060+lr_padding, 440-height_offset],[1060+lr_padding, 675],[270-lr_padding, 675],[270-lr_padding, 440-height_offset]])
    # colour_binary_masked = region_of_interest(colour_binary, np.int32([src]))
    # 
    # if plot_it == True:
    #     f10 = plt.figure()
    #     ax10 = f10.add_subplot(111)
    #     ax10.imshow(colour_binary_masked)
    #     ax10.set_title('Masked Colour Binary', fontsize=40)
    # 
    # ## Warp it
    # 
    # warped = warper(colour_binary_masked, src, dest)
    # 
    # if plot_it == True:
    #     f4, (ax7, ax8) = plt.subplots(1, 2, figsize=(20,10))
    #     ax7.imshow(colour_binary_masked)
    #     ax7.plot(src[:,0],src[:,1],'.-r')
    #     ax7.plot(src[0:4:3,0],src[0:4:3,1],'.-r')
    #     ax7.set_title('Unwarped Image', fontsize=30)
    #     ax8.imshow(warped)
    #     ax8.set_title('Warped Image', fontsize=30)
    #     
    # ## Fitting polynomial equations to lane lines
    # nwindows = 20 # Choose the number of sliding windows
    # margin = 50 # Set the width of the windows +/- margin
    # minpix = 100 # Set minimum number of pixels found to recenter window
    # 
    # if self.left_detected == False | self.right_detected == False: 
    #     self = find_lines(self, warped, nwindows, margin, minpix, plot_it=False)
    #     self.n = 1
    #     left_fit = self.current_left_fit
    #     right_fit = self.current_right_fit
    #     self.best_fit_left = self.current_left_fit
    #     self.best_fit_right = self.current_right_fit
    # else:
    #     basexleft = self.best_fit_left[0]*720**2 + self.best_fit_left[1]*720 + self.best_fit_left[2]
    #     basexright = self.best_fit_right[0]*720**2 + self.best_fit_right[1]*720 + self.best_fit_right[2]
    #     
    #     if (basexright - basexleft < 300) | (basexleft > 500) | (basexright < 700): # If the lane lines in the previous frame are too close together, recalculate from nothing
    #         self = find_lines(self, warped, nwindows, margin, minpix, plot_it=False)
    #         self.n = 1
    #         left_fit = self.current_left_fit
    #         right_fit = self.current_right_fit
    #         self.best_fit_left = self.current_left_fit
    #         self.best_fit_right = self.current_right_fit

  #   ##       else:
    #         self = find_lines_near(warped, self, self.current_left_fit, self.current_right_fit, margin)
    #         self.best_fit_left = np.average([[self.best_fit_left.squeeze()],[self.current_left_fit]],0,weights=[self.n, 1]).squeeze() 
    #         self.best_fit_right = np.average([[self.best_fit_right.squeeze()],[self.current_right_fit]],0,weights=[self.n, 1]).squeeze()
    #         if self.n < 10: 
    #             self.n += 1
    # 
    # ## Finding curvature 
    # 
    # self = find_curvature(self)

  #   ##   if plot_it == True:
    #     print('Left Curvature: ', self.left_curvature, 'm')
    #     print('Right Curvature: ', self.right_curvature, 'm')
    #     print('Distance Off Center: ', self.off_center, 'm')
    # 
    # ## Warp lines back into original image shape
    # 
    # result = unwarp_add_lane(warped, img, self.best_fit_left.squeeze(), self.best_fit_right.squeeze(), dest, src)

  #   ##   if plot_it == True:
    #     f20 = plt.figure()
    #     ax20 = f20.add_subplot(111)
    #     ax20.imshow(result)
    #     ax20.set_title('Pipeline Output', fontsize=30)
    # 
    # final_img = result
    # 
    # strleft = str('Left Line Curvature: ' + '{0:.2f}'.format(self.left_curvature) + ' m')
    # strright = str('Right Line Curvature: ' + '{0:.2f}'.format(self.right_curvature) + ' m')
    # stravg = str('Average Line Curvature: ' + '{0:.2f}'.format((self.right_curvature + self.left_curvature)/2) + ' m')
    # stroffset = str('Offset from Lane Center: ' '{0:.2f}'.format(self.off_center) + ' m')
    # 
    # # cv2.putText(final_img, strleft, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    # # cv2.putText(final_img, strright, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    # cv2.putText(final_img, stravg, (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    # cv2.putText(final_img, stroffset, (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    
    self.final_img = final_img
    
    return self