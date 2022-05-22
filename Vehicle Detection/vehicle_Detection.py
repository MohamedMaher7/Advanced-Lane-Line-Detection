import sys
import os.path as path
import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
import pickle


# Extact the HOG of the image
def GetFeaturesFromHog(image, orient, cellsPerBlock, pixelsPerCell, visualise=False, feature_vector_flag=True):
    if (visualise == True):
        hog_features, hog_image = hog(image, orientations=orient,
                                      pixels_per_cell=(pixelsPerCell, pixelsPerCell),
                                      cells_per_block=(cellsPerBlock, cellsPerBlock),
                                      visualize=True, feature_vector=feature_vector_flag)
        return hog_features, hog_image
    else:
        hog_features = hog(image, orientations=orient,
                           pixels_per_cell=(pixelsPerCell, pixelsPerCell),
                           cells_per_block=(cellsPerBlock, cellsPerBlock),
                           visualize=False, feature_vector=feature_vector_flag)
        return hog_features


# Convert Image Color Space. Note the colorspace parameter is like cv2.COLOR_RGB2YUV
def ConvertImageColorspace(image, colorspace):
    return cv2.cvtColor(image, colorspace)
