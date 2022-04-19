import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # x and y values in last frame
        self.x = None
        self.y = None

        # x intercepts for average smoothing
        self.bottom_x = deque(maxlen=frame_num)
        self.top_x = deque(maxlen=frame_num)

        # Record last x intercept
        self.current_bottom_x = None
        self.current_top_x = None

        # Polynomial coefficients: x = A*y**2 + B*y + C
        self.A = deque(maxlen=frame_num)
        self.B = deque(maxlen=frame_num)
        self.C = deque(maxlen=frame_num)
        self.fit = None
        self.fitx = None
        self.fity = None

    def get_intercepts(self):
        bottom = self.fit[0] * 720 ** 2 + self.fit[1] * 720 + self.fit[2]
        top = self.fit[2]
        return bottom, top

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
    

def car_pos(left_fit, right_fit):

    #Calculate the position of car based on left and right lanes
    #convert to real unit meter
    #return: distance (meters) of car offset from the middle of left and right lane
    xleft_eval = left_fit[0] * np.max(ploty) ** 2 + left_fit[1] * np.max(ploty) + left_fit[2]
    xright_eval = right_fit[0] * np.max(ploty) ** 2 + right_fit[1] * np.max(ploty) + right_fit[2]
    ym_per_pix = 18 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / abs(xleft_eval - xright_eval)  # meters per pixel in x dimension
    xmean = np.mean((xleft_eval, xright_eval))
    offset = (img_shape[1]/2 - xmean) * xm_per_pix  # +: car in right; -: car in left side

    y_eval = np.max(ploty)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                    np.absolute(2 * left_fit_cr[0])

    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                    np.absolute(2 * right_fit_cr[0])

    mean_curv = np.mean([left_curverad, right_curverad])

    return offset, mean_curv
    
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    #concatenate images vertically
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    #Concatenate images of the same width vertically
    return cv2.vconcat(im_list_resize)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    #When concatenating images of the same height horizontally
    return cv2.hconcat(im_list_resize)


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    # concatenating images of different sizes in vertical and horizontal tiles, use the resizing and concatenating function.
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)