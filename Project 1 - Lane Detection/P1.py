import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os
import scipy.misc

## Helper Functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, y_max, x_max, color=[255, 0, 0], thickness=20):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    slopes = (lines[:,0,3]-lines[:,0,1])/(lines[:,0,2] - lines[:,0,0])
    lengths = sqrt((lines[:,0,3]-lines[:,0,1])**2 + (lines[:,0,2] - lines[:,0,0])**2)
    
    left_side = np.logical_and(slopes > 0, lengths > 0.01)
    offside = np.logical_or(lines[:,0,0] < x_max/2, lines[:,0,2] < x_max/2)
    left_side[offside] = False
    
    right_side = np.logical_and(slopes < 0, lengths > 0.01)
    offside = np.logical_or(lines[:,0,0] > x_max/2, lines[:,0,2] > x_max/2)
    right_side[offside] = False
    
    
    if len(lengths[left_side]) > 0 and len(lengths[right_side]) > 0:

        slope_left = np.average(slopes[left_side], weights=lengths[left_side])
        slope_right = np.average(slopes[right_side], weights=lengths[right_side])
        
        x_mid_left = np.average(vstack((lines[left_side,0,0],lines[left_side,0,2])), weights=vstack((lengths[left_side],lengths[left_side])))
        y_mid_left = np.average(vstack((lines[left_side,0,1],lines[left_side,0,3])), weights=vstack((lengths[left_side],lengths[left_side])))
        
        x_mid_right = np.average(vstack((lines[right_side,0,0],lines[right_side,0,2])), weights=vstack((lengths[right_side],lengths[right_side])))
        y_mid_right = np.average(vstack((lines[right_side,0,1],lines[right_side,0,3])), weights=vstack((lengths[right_side],lengths[right_side])))
        
        x_lower_left = ((y_max - y_mid_left)/slope_left) + x_mid_left
        x_lower_right = ((y_max - y_mid_right)/slope_right) + x_mid_right
        
        y_height = y_max/1.6;
        
        x_upper_left = ((y_height - y_mid_left)/slope_left) + x_mid_left
        x_upper_right = ((y_height - y_mid_right)/slope_right) + x_mid_right
        
        cv2.line(img, (int(x_lower_left), y_max), (int(x_upper_left), int(y_height)), color, thickness)
        cv2.line(img, (int(x_lower_right), y_max), (int(x_upper_right), int(y_height)), color, thickness) 

    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, y_max, x_max):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, y_max, x_max)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(img):

    gray = grayscale(img)
    
    # Define a kernel size and apply Gaussian smoothing
    # kernel_size = 9
    kernel_size = 9
    blur_gray = gaussian_blur(img, kernel_size)
    
    # Define our parameters for Canny and apply
    # low_threshold = 200
    # high_threshold = 300
    low_threshold = 200
    high_threshold = 300
    edges = canny(img, low_threshold, high_threshold)
    
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    
    # We define a triangle for our polygon mask
    imshape = img.shape
    vertices = np.array([[(imshape[1]/30,imshape[0]),(imshape[1]/2, imshape[0]/1.9), (imshape[1]-(imshape[1]/30),imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    # rho = 12 # distance resolution in pixels of the Hough grid
    # theta = np.pi/80 # angular resolution in radians of the Hough grid
    # threshold = 100     # minimum number of votes (intersections in Hough grid cell)
    # min_line_len = 6 #minimum number of pixels making up a line
    # max_line_gap = 3    # maximum gap in pixels between connectable line segments
    
    rho = 12
    theta = 0.001
    threshold = 100
    min_line_len = 20
    max_line_gap = 10
    
    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap, imshape[0], imshape[1])
    
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    # Draw the lines on the edge image
    marked_lane = weighted_img(line_image, img)#, α=0.8, β=1., λ=0.)
    #plt.imshow(marked_lane)
    
    # return marked_lane
    return marked_lane

## Main Functions

# Loading test images
# strImage = 'test_images/solidWhiteCurve.jpg'
# strImage = 'test_images/solidWhiteRight.jpg'
# strImage = 'test_images/solidYellowCurve.jpg'
# strImage = 'test_images/solidYellowCurve2.jpg'
# strImage = 'test_images/solidYellowLeft.jpg'
# strImage = 'test_images/whiteCarLaneSwitch.jpg'

# strImageIn = ["test_images/" + x for x in os.listdir("test_images/")]
# strImageOut = ["test_images_output/" + x for x in os.listdir("test_images/")]
# 
# for i in range (0,5):
#     img = mpimg.imread(strImageIn[i])
#     marked_lane = process_image(img)
#     scipy.misc.imsave(strImageOut[i], marked_lane)

yellow_output = 'yellow.mp4'
clip1 = VideoFileClip("solidYellowLeft.mp4")

yellow_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
yellow_clip.write_videofile(yellow_output, audio=False)

white_output = 'white.mp4'
clip2 = VideoFileClip("solidWhiteRight.mp4")

white_clip = clip2.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

challenge_output = 'challenge-output.mp4'
clip2 = VideoFileClip("challenge.mp4")

challenge_clip = clip2.fl_image(process_image) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)




