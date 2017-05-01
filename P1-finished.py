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


def draw_lines(img, lines, y_max, x_max, AvgFlag, color=[255, 0, 0]):
    # This function draws the hough lines on the image, or the averaged lane line location
    # based on the AvgFlag (true = averaged line location, false = rough hough lines)
    
    if lines is not None: # Simple check to see if there are indeed lines
        if AvgFlag is True:
            
            thickness = 12
            
            # Slopes are found for all Hough lines
            slopes = (lines[:,0,3]-lines[:,0,1])/(lines[:,0,2] - lines[:,0,0])
            # The lengths of all the lines are found as Ill
            lengths = np.sqrt((lines[:,0,3]-lines[:,0,1])**2 + (lines[:,0,2] - lines[:,0,0])**2)
            
            # Left lane lines likely have a slope greater than zero, as they go upwards and to the right
            left_side = (slopes > 0)
            # If a point from the "left lane" is on the right half of the image, ignore it as it is probably incorrect or noise
            offside = np.logical_or(lines[:,0,0] < x_max/2, lines[:,0,2] < x_max/2) 
            left_side[offside] = False
            
            # Do the opposite of the above procedure to find the right lane lines
            right_side = (slopes < 0)
            offside = np.logical_or(lines[:,0,0] > x_max/2, lines[:,0,2] > x_max/2)
            right_side[offside] = False
        
            # Check to see if there are both left and right side lines
            if len(lengths[left_side]) > 0 and len(lengths[right_side]) > 0:
                
                # To find the slopes of the lane lines, take the weighted averages of all lines on that side
                # The hope is that the longer lines will be what we want. Shorter lines may be noise and
                # the weight will give it less of a negative impact
                slope_left = np.average(slopes[left_side], weights=lengths[left_side])
                slope_right = np.average(slopes[right_side], weights=lengths[right_side])
                
                # Average all the points on the left and right sides to find the "midpoint" (it won't be actually halfway)
                x_mid_left = np.average(np.vstack((lines[left_side,0,0],lines[left_side,0,2])), weights=np.vstack((lengths[left_side],lengths[left_side])))
                y_mid_left = np.average(np.vstack((lines[left_side,0,1],lines[left_side,0,3])), weights=np.vstack((lengths[left_side],lengths[left_side])))
                
                x_mid_right = np.average(np.vstack((lines[right_side,0,0],lines[right_side,0,2])), weights=np.vstack((lengths[right_side],lengths[right_side])))
                y_mid_right = np.average(np.vstack((lines[right_side,0,1],lines[right_side,0,3])), weights=np.vstack((lengths[right_side],lengths[right_side])))
                
                # Now, using the slope of the lines and the "midpoint", I find where it will intersect the bottom edge of the image
                x_lower_left = ((y_max - y_mid_left)/slope_left) + x_mid_left
                x_lower_right = ((y_max - y_mid_right)/slope_right) + x_mid_right
                
                # This is the height cuttof for the lines we will draw
                y_height = y_max/1.6;
                
                # Using the above cutoff and the slope, find the x-coordinate of the upper point
                x_upper_left = ((y_height - y_mid_left)/slope_left) + x_mid_left
                x_upper_right = ((y_height - y_mid_right)/slope_right) + x_mid_right
                
                # Plot the upper and lower (x,y) pair for both left and right side
                cv2.line(img, (int(x_lower_left), y_max), (int(x_upper_left), int(y_height)), color, thickness)
                cv2.line(img, (int(x_lower_right), y_max), (int(x_upper_right), int(y_height)), color, thickness) 
        
        # If not returning averaged lines, return the raw stuff
        else:
            thickness = 2
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, y_max, x_max, AvgFlag):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, y_max, x_max, AvgFlag)
    return line_img

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

def process_image(img, AvgFlag):

    # This function processes the image and returns the hough lines, or the averaged approximated lane location
    # AvgFlag = True returns the averaged lanes and False returns the rough Hough lines
    
    gray = grayscale(img)
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 9
    blur_gray = gaussian_blur(img, kernel_size)
    
    # Define our parameters for Canny and apply
    low_threshold = 200
    high_threshold = 300
    edges = canny(img, low_threshold, high_threshold)
    
    # Next I'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    
    # I define a triangle for our polygon mask
    imshape = img.shape
    # Masking out most of the frame, including the very bottom of the frame to avoid the hood of the car
    vertices = np.array([[(0,imshape[0]*0.9),(imshape[1]/2, imshape[0]/1.7), (imshape[1],imshape[0]*0.9)]], dtype=np.int32)

    
    masked_edges = region_of_interest(edges, vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 12 # distance resolution in pixels of the Hough grid
    theta = 0.001 # angular resolution in radians of the Hough grid
    threshold = 100     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels betIen connectable line segments
    
    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap, imshape[0], imshape[1], AvgFlag)
    
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    # Draw the lines on the edge image
    marked_lane = weighted_img(line_image, img)#, α=0.8, β=1., λ=0.)
    
    return marked_lane
    
def raw_lines(img):
    return process_image(img, False)

def averaged_lines(img):
    return process_image(img, True)

## Main Script

# Loading test images
strImageIn = ["test_images/" + x for x in os.listdir("test_images/")]
strImageOutRaw = ["test_images_rawlines_output/" + x for x in os.listdir("test_images/")]
strImageOutAvg = ["test_images_avglines_output/" + x for x in os.listdir("test_images/")]

# Running test images and outputting raw and averaged lane results to test_images_output
for i in range (0,6):
    img = mpimg.imread(strImageIn[i])
    
    marked_lane = raw_lines(img)
    scipy.misc.imsave(strImageOutRaw[i], marked_lane)
    
    marked_lane = averaged_lines(img)
    scipy.misc.imsave(strImageOutAvg[i], marked_lane)

# Processing videos with both raw Hough line output and avgeraged line output:
# Loading image files (all 3)
strVideoIn = ["videos/" + x for x in os.listdir("videos/")]
strVideoOutRaw = ["videos_rawlines_output/" + x for x in os.listdir("videos/")]
strVideoOutAvg = ["videos_avglines_output/" + x for x in os.listdir("videos/")]

# Processing each video file and outputting to videos_output
for i in range (0,3):
    clip = VideoFileClip(strVideoIn[i])
    
    # Raw Hough lines:
    processed_clip = clip.fl_image(raw_lines)
    processed_clip.write_videofile(strVideoOutRaw[i], audio=False)
    
    # Averaged lines:
    processed_clip = clip.fl_image(averaged_lines)
    processed_clip.write_videofile(strVideoOutAvg[i], audio=False)
    




