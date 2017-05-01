import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
import os
import scipy.misc
import pickle
import collections
from helper_functions import *
from p4_pipeline import *

# Reading in Test Images
strTestImgIn = ["test_images/" + x for x in os.listdir("test_images/")]
strTestImgOut = ["test_images_output/" + x for x in os.listdir("test_images/")]

# Getting rid of "thumbs.db" on my machine
if "test_images/Thumbs.db" in strTestImgIn: strTestImgIn.remove("test_images/Thumbs.db")
if "test_images_out/Thumbs.db" in strTestImgOut: strTestImgOut.remove("test_images/Thumbs.db")

# Running test images and outputting to test_images_output
for x in range(0, len(strTestImgIn)):
    img = mpimg.imread(strTestImgIn[x])
    final_img = p4_pipeline(img)
    scipy.misc.imsave(strTestImgOut[x], final_img)
    
# Reading in Test Videos
strTestVideoIn = ["test_videos/" + x for x in os.listdir("test_videos/")]
strTestVideoOut = ["test_videos_output/" + x for x in os.listdir("test_videos/")]

# Getting rid of "thumbs.db" on my machine
if "test_videos/Thumbs.db" in strTestVideoIn: strTestVideoIn.remove("test_videos/Thumbs.db")
if "test_videos_out/Thumbs.db" in strTestVideoOut: strTestVideoOut.remove("test_videos/Thumbs.db")

# # Processing each video file and outputting to videos_output
# for i in range (0,len(strTestVideoIn)):
#     clip = VideoFileClip(strTestVideoIn[i])
#     
#     processed_clip = clip.fl_image(p4_pipeline)
#     processed_clip.write_videofile(strTestVideoOut[i], audio=False)
    


