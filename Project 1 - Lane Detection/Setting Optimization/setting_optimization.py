import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os
import scipy

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


def draw_lines(img, lines, y_max, x_max, color=[255, 0, 0], thickness=2):


    if lines is not None:
        slopes = (lines[:,0,3]-lines[:,0,1])/(lines[:,0,2] - lines[:,0,0])
        lengths = sqrt((lines[:,0,3]-lines[:,0,1])**2 + (lines[:,0,2] - lines[:,0,0])**2)
        
        left_side = (slopes > 0)
        offside = np.logical_or(lines[:,0,0] < x_max/2, lines[:,0,2] < x_max/2)
        left_side[offside] = False
        
        right_side = (slopes < 0)
        offside = np.logical_or(lines[:,0,0] > x_max/2, lines[:,0,2] > x_max/2)
        right_side[offside] = False
        
        left_lines = lines[left_side,:,:]
        right_lines = lines[right_side,:,:]
        
        for line in left_lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
        for line in right_lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, y_max, x_max):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    # print(rho, theta, threshold, min_line_len, max_line_gap)
    
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
    
def mse(imageA, imageB):
    # Taken from http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):
    # Taken from http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f" % m)
    
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    
    # show the images
    plt.show()
    
def round_up_to_odd(f):
    # From http://stackoverflow.com/questions/31648729/python-round-a-float-up-to-next-odd-integer
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

def process_image(img, kernel_size, low_threshold, high_threshold, rho, theta, threshold, min_line_len, max_line_gap):
    
    kernel_size = round_up_to_odd(kernel_size)

    gray = grayscale(img)
    
    # Define a kernel size and apply Gaussian smoothing
    # kernel_size = 9
    blur_gray = gaussian_blur(img, kernel_size)
    
    # Define our parameters for Canny and apply
    # low_threshold = 200
    # high_threshold = 300
    edges = canny(img, low_threshold, high_threshold)
    
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    
    # We define a triangle for our polygon mask
    imshape = img.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]/2, imshape[0]/1.7), (imshape[1],imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    # rho = 12 # distance resolution in pixels of the Hough grid
    # theta = np.pi/80 # angular resolution in radians of the Hough grid
    # threshold = 100     # minimum number of votes (intersections in Hough grid cell)
    # min_line_len = 6 #minimum number of pixels making up a line
    # max_line_gap = 3    # maximum gap in pixels between connectable line segments
    
    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap, imshape[0], imshape[1])
    
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    # Draw the lines on the edge image
    marked_lane = weighted_img(line_image, img)#, α=0.8, β=1., λ=0.)
    #plt.imshow(marked_lane)
    
    # return marked_lane
    return line_image
    
def objective_function(design_vars):
    # This function takes in the above design variables and will return an image with a black background and red ([255 0 0]) lines. This will be compared to the desired image, created manually, and the end result of the optimization will be settings which provide the most similar image to the desired result.
    
    kernel_size = int(design_vars[0])
    low_threshold = int(design_vars[1])
    high_threshold = int(design_vars[2])
    rho = int(design_vars[3])
    theta = design_vars[4]
    threshold = int(design_vars[5])
    min_line_len = int(design_vars[6])
    max_line_gap = int(design_vars[7])
    
    strImageIn = ["test_images/" + x for x in os.listdir("test_images/")]
    strImageDesired = ["desired_images/" + x for x in os.listdir("test_images/")]
    
    likeness = []
    
    for i in range (0,5):
        img = mpimg.imread(strImageIn[i])
        img_desired = mpimg.imread(strImageDesired[i])
        
        marked_lane = process_image(img, kernel_size, low_threshold, high_threshold, rho, theta, threshold, min_line_len, max_line_gap)
        
        likeness.append(mse(marked_lane, img_desired))


    # avg_likeness = np.average(likeness)
    # print(avg_likeness, design_vars)
    
    worst_likeness = max(likeness)
    
    print(worst_likeness, design_vars)
    
    return worst_likeness
    
def check_result(design_vars):
# This function is just for me to compare the images visually
    kernel_size = int(design_vars[0])
    low_threshold = int(design_vars[1])
    high_threshold = int(design_vars[2])
    rho = int(design_vars[3])
    theta = np.ceil(design_vars[4])
    threshold = int(design_vars[5])
    min_line_len = int(design_vars[6])
    max_line_gap = int(design_vars[7])
    
    strImage = ["test_images/" + x for x in os.listdir("test_images/")]
    strDesired = ["desired_images/" + x for x in os.listdir("test_images/")]
    
    i = 4;
    
    img = mpimg.imread(strImage[i])
    marked_lane = process_image(img, kernel_size, low_threshold, high_threshold, rho, theta, threshold, min_line_len, max_line_gap)
    compare_images(marked_lane, mpimg.imread(strDesired[i]), '')

## Main Script


# bounds = [[1, 60],[2, 500],[2, 900],[2, 50],[0.0001, 2*pi],[1, 400],[2, 300],[2, 300]]
# results = scipy.optimize.differential_evolution(objective_function, bounds)

# check_result([  25.3505928 ,  317.14373338,  634.16902284,    2.03785962, 0.81923436,   10.06286094,    3.1415375 ,    2.19538521])

check_result([  2.83331786e+01,   3.36554604e+02,   1.69931028e+02,3.69596687e+00, 5.30782900e-03,   1.92756312e+02,6.33460354e+01,   4.19259984e+00])