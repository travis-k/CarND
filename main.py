import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip, ImageSequenceClip
import os
import scipy.misc
import pickle
import collections
from helper_functions import *
from p5_pipeline import Lines, p5_pipeline

## Test Images
strTestImgIn = ["test_images/" + x for x in os.listdir("test_images/")]
strTestImgOut = ["test_images_output/" + x for x in os.listdir("test_images/")]

# Getting rid of "thumbs.db" on my machine
if "test_images/Thumbs.db" in strTestImgIn: strTestImgIn.remove("test_images/Thumbs.db")
if "test_images_out/Thumbs.db" in strTestImgOut: strTestImgOut.remove("test_images/Thumbs.db")

history = Lines() # No history for single images
img = mpimg.imread(strTestImgIn[4])
history = p5_pipeline(img, history)

# # Running test images and outputting to test_images_output
# for x in range(0, len(strTestImgIn)):
#     history = Lines() # No history for single images
#     img = mpimg.imread(strTestImgIn[x])
#     history = p5_pipeline(img, history)
#     scipy.misc.imsave(strTestImgOut[x], history.final_img)

## Test Videos
strTestVideoIn = ["test_videos/" + x for x in os.listdir("test_videos/")]
strTestVideoOut = ["test_videos_output/" + x for x in os.listdir("test_videos/")]

# Getting rid of "thumbs.db" on my mac    bbox3 = np.array([]).reshape(0,2,2)hine
if "test_videos/Thumbs.db" in strTestVideoIn: strTestVideoIn.remove("test_videos/Thumbs.db")
if "test_videos_out/Thumbs.db" in strTestVideoOut: strTestVideoOut.remove("test_videos/Thumbs.db")
 
# # Processing each video file and outputting to videos_output
# for i in range (0,len(strTestVideoIn)):
#     history = Lines() # Resetting values in Lines class for this new video
#     clip = VideoFileClip(strTestVideoIn[i])
#     new_frames = []
#     for frame in clip.iter_frames():
#         history = p5_pipeline(frame,history)
#         new_frames.append(history.final_img)
#     new_clip = ImageSequenceClip(new_frames, fps=clip.fps)
#     new_clip.write_videofile(strTestVideoOut[i])    


