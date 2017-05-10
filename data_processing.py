import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import scipy.misc
import pickle
import collections
from helper_functions import *
from sklearn.preprocessing import StandardScaler

strVehicle = ["training_data/vehicles/" + x for x in os.listdir("training_data/vehicles/")]
strNotVehicle = ["training_data/non-vehicles/" + x for x in os.listdir("training_data/non-vehicles/")]

# Getting rid of "thumbs.db" on my machine
if "training_data/vehicles/Thumbs.db" in strVehicle: strVehicle.remove("training_data/vehicles/Thumbs.db")
if "training_data/non-vehicles/Thumbs.db" in strNotVehicle: strNotVehicle.remove("training_data/non-vehicles/Thumbs.db")

# Reading in features for cars and not-cars (color histogram and HOG feature vectors)
car_features = extract_features(strVehicle, color_space='YCrCb', spatial_size=(32,32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',spatial_feat=True, hist_feat=True, hog_feat=True)
notcar_features = extract_features(strNotVehicle, color_space='YCrCb', spatial_size=(32,32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',spatial_feat=True, hist_feat=True, hog_feat=True)

# Stacking and scaling the feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                    
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# Classifiers for the above feature vectors. 1 is car, 0 is not car
y_car = np.ones((len(car_features),1))
y_notcar = np.zeros((len(notcar_features),1))
y = np.vstack((y_car, y_notcar)).astype(np.int)

# Saving the data
data_pickle = {}
data_pickle["X"] = scaled_X
data_pickle["y"] = y
data_pickle["X_scaler"] = X_scaler
pickle.dump(data_pickle, open("train_data.p", "wb" ))

## Visualize Data
# car_ind = np.random.randint(0, len(strVehicle))
# notcar_ind = np.random.randint(0, len(strNotVehicle))
#     
# # Read in car / not-car images
# car_image = mpimg.imread(strVehicle[car_ind])
# notcar_image = mpimg.imread(strNotVehicle[notcar_ind])
# 
# 
# f200, (ax200, ax201) = plt.subplots(1, 2)
# ax200.imshow(car_image)
# ax200.set_title('Example Car Image')
# ax201.imshow(notcar_image)
# ax201.set_title('Example Not-car Image')
# f200.tight_layout()

## Visualize Hog Data

# # Generate a random index to look at a car image
# ind = np.random.randint(0, len(strVehicle))
# # Read in the image
# image = mpimg.imread(strVehicle[ind])
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# # Define HOG parameters
# orient = 9
# pix_per_cell = 8
# cell_per_block = 2
# # Call our function with vis=True to see an image output
# features, hog_image = get_hog_features(gray, orient, 
#                         pix_per_cell, cell_per_block, 
#                         vis=True, feature_vec=False)
# 
# f200, (ax200, ax201) = plt.subplots(1, 2)
# ax200.imshow(image)
# ax200.set_title('Car Image')
# ax201.imshow(hog_image)
# ax201.set_title('Hog Car Image')
# f200.tight_layout()













