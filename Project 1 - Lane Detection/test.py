
# coding: utf-8

# **Run all the cells below to make sure everything is working and ready to go. All cells should run without error.**

# ### Test Matplotlib and Plotting

# In[1]:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# In[2]:

img = mpimg.imread('test.jpg')
plt.imshow(img)


# ### Test OpenCV

# In[3]:

import cv2


# In[ ]:

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='Greys_r')


# ## Test TensorFlow

# In[ ]:

import tensorflow as tf


# In[ ]:

with tf.Session() as sess:
    a = tf.constant(1)
    b = tf.constant(2)
    c = a + b
    # Should be 3
    print("1 + 2 = {}".format(sess.run(c)))


# ## Test Moviepy

# In[ ]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Create a new video with `moviepy` by processing each frame to [YUV](https://en.wikipedia.org/wiki/YUV) color space.

new_clip_output = 'test_output.mp4'
test_clip = VideoFileClip("test.mp4")
new_clip = test_clip.fl_image(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YUV)) #NOTE: this function expects color images!!
new_clip.write_videofile(new_clip_output, audio=False)




