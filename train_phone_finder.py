# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:38:59 2018
@author: KUBER REDDY
"""

        
#%% Import the required modules
import sys
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

#%%
## Get the Current Working directory
mycwd = os.getcwd()

## Retrieving the folder path argument
arglist  = sys.argv[1]

## Change the directory to the folder path
os.chdir(arglist)

#%% Defining the required functions

## Normalization function and retreiving the orginal coordinates in an image
def original_coordinates(x, y, img):
    num_rows, num_cols = img.shape[:2]
    col_j = x*(num_cols-1)
    row_i = y*(num_rows-1)
    return col_j,row_i

## Function to convert to the Normalized coordinates in an image
def Normalized_coordinates(col_j, row_i, img):
    num_rows, num_cols = img.shape[:2]
    x = col_j/(num_cols-1)
    y = row_i/(num_rows-1)
    return x,y

## Define the function to Convert to gray scale image
def color_to_gray(color_img):
    gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    return gray

#%% For Training Images, Extracting the feature vectors
def feature_extraction_train(image_list):
    num = 0
    all_features = np.zeros((3600,1))
    for i in range(0,len(image_list)-20):
        each_image = cv.imread(image_list[i])
        X_center,Y_center = image_XY['X'][i],image_XY['Y'][i]
        
    # Finding the original coordinates of the center of phone for Image
        row_num,col_num = original_coordinates(X_center, Y_center, each_image)
        
    # Converting to gray scale image for every image
        gray_image = color_to_gray(each_image)
        
    # Cropping the image to visualize the phone (Optimized the localization)
        try:
            cropped_image = gray_image[int(col_num)-30:int(col_num)+30,int(row_num)-30:int(row_num)+30]
            feature_Vectors = cropped_image
            
    # Reshaping the cropped image to a n-dimensional vector
            feature_Vectors = np.array(feature_Vectors).reshape(3600,1)
            
    # Appending to the list of feature_Vectors for all the images
            all_features = np.append(all_features,feature_Vectors, axis=1)
        except Exception as e:
            print(image_list[i])
            num += 1
            continue
    return all_features

## Preprocessing the feature vectors to obtain the right features using Principal Component Analysis

def PCA_features(all_features):
    ## Applying Principal Component Analysis(PCA) using scikit learn
    
    # Taking the first Principal component with the maximum variance
    pca = PCA(1)   
    all_features_new = pca.fit_transform(all_features)
    
    # Reshaping the features to original cropped image shape
    all_features_new = all_features_new.reshape((60,60))
    all_features_new.shape
    
    # Plotting the image using the PCA features
    plt.ion()
    plt.imshow(all_features_new)
    plt.show()
    plt.pause(1)
    
    # Saving the image in the current directory of the python script file
    os.chdir(mycwd)
    cv.imwrite('PCA_features.jpg',all_features_new)
    
    # Closing all the plots
    plt.close("all") 
    
    return all_features_new

#%% 
# =============================================================================
# Template matching method to check detection % on labeled dataset
# =============================================================================
def percent_true(image_list):
    perfect = 0
    imperfect = 0
    os.chdir(mycwd)
    
    ## Loading the template image
    template = cv.imread('PCA_features.jpg',0)
    
    ## Changing the directory to folder path 
    os.chdir(arglist)
    # Checking the method for all images
    for i in range(len(image_list)-20,len(image_list)):
        img = cv.imread(image_list[i])
        img = color_to_gray(img)
        
    ## Obtaining the width and height of the template image
        w, h = template.shape[::-1] 
    
        # Introducing the Template matching method
        methods = 'cv.TM_CCOEFF'
        
# =============================================================================
# TM_CCOEFF method :
# I----> image, T----> template, R ----> result
# R(x,y)=∑x′,y′(T′(x′,y′)⋅I′(x+x′,y+y′))
# where
# T′(x′,y′)=T(x′,y′)−1/(w⋅h)⋅∑x′′,y′′T(x′′,y′′)
# I′(x+x′,y+y′)=I(x+x′,y+y′)−1/(w⋅h)⋅∑x′′,y′′I(x+x′′,y+y′′)
# =============================================================================
        
        method = eval(methods)
    
        # Apply template matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
        top_left = min_loc
    
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        ## Obtaining the Normalized coordinates of the center
        X,Y = Normalized_coordinates(top_left[0]+(w/2),top_left[1]+(h/2),img)

        
        ## Checking the error from the original coordinates
        X_center,Y_center = image_XY['X'][i],image_XY['Y'][i]

        if np.sqrt((X_center-X)**2+(Y_center-Y)**2)<0.05:
            perfect+=1
        else:
            imperfect+=1
    # Detection % according to the given radius 0.05 (normalized distance) for labeled dataset
    correct_perc = (perfect/(perfect+imperfect))*100
    return correct_perc


#%% 
def main():
    global image_XY
    
    ## Creating a pandas data frame from the labels.txt file in the folder
    image_XY = pd.read_csv('labels.txt', sep=" ", header=None, names=["IMG","X","Y"])
    
    ## Intoducing the list of strings for the images in the folder
    image_list = []
    for i in range(0,len(image_XY['IMG'])):
        image_list.append(image_XY['IMG'][i])
    print(image_list) # print the image list
    
    ## Feature Extraction
    all_features = feature_extraction_train(image_list)
    all_features = np.delete(all_features,0,1)
    print(all_features.shape)
    
    ## Feature Preprocessing
    all_features_new = PCA_features(all_features)
    print(all_features_new)
    
    ## Calculating the detection % on labeled data set
    correct_perc = percent_true(image_list)
    print("Detection % on provided labeled dataset:", correct_perc)
    return all_features_new
    

if __name__ =='__main__':
    main()




