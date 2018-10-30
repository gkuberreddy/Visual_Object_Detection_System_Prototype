# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:02:58 2018

@author: KUBER REDDY
"""

#%% Import the required modules
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#%% Retreive the Image path from Commandline argument
myImage = sys.argv[1]

#%% Defining the required functions

## Converting an RGB image to the gray-scale image
def color_to_gray(color_img):
    gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    return gray

## Converting the Original coordinates to the Normalized Coordinates in the image
def Normalized_coordinates(col_j, row_i, img):
    num_rows, num_cols = img.shape[:2]
    x = col_j/(num_cols-1)
    y = row_i/(num_rows-1)
    return x,y

#%%  Calculating the normalized coordinates of the test Image
    
## Reading the image from the command line argument
img = cv.imread(myImage) 

## Converting the image to gray
img = color_to_gray(img)       

template = cv.imread('PCA_features.jpg',0)

## Obtaining the width and height of the image
w, h = template.shape[::-1]     

# Template matching method 
methods = 'cv.TM_CCOEFF'
method = eval(methods)

# Apply template matching
res = cv.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

# Obtaining the localized rectangle coordinates
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv.rectangle(img,top_left, bottom_right, 255, 2)

# Plotting the matching result as a result of applying template matching 
# On the original image on which phone is detected
plt.ion()
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(method)
plt.show()
plt.pause(1)

# Saving the detected phone image in the current directory
plt.savefig('phone_found.jpg')

# Closes the window
plt.close("all")

## Plotting the normalize coordinates and original coordinates
X,Y = Normalized_coordinates(top_left[0]+(w/2),top_left[1]+(h/2),img)
print("Original Coordinates of pixels: ",top_left[0]+(w/2),top_left[1]+(h/2))
print("Normalized coordinates of the phone: ",str.format('{0:.4f}', X),str.format('{0:.4f}', Y))
