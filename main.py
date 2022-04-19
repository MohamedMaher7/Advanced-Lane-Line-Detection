import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque


def camera_calibration():
    global mtx,dist
    # Prepare object points
    obj_pts = np.zeros((6 * 9, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []

    # Get directory for all calibration images
    images = glob.glob('./camera_cal/*.jpg')
    img = None

    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # MUST be 8-bit grayscale or color image
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)

    # Get Image Size
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

def undistort(img, mtx, dist):
    #Matrix mtx and dist from camera calibration are applied to distortion correction
    #img: input img is RGB (imread by mpimg)
    # transform to BGR to fit cv2.imread
    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #mtx: camera calibration parameter
    # dist: camera calibration parameter
    dst_img = cv2.undistort(img_BGR, mtx, dist, None, mtx)
    #return: Undistorted img
    return cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)