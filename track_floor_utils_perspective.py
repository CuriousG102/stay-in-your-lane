import math

import cv2
import numpy as np

CAR_CAM_FOV = 80  # degrees

# All measurements are in meters.
CAR_CAM_POS_RELATIVE_Y = .08255 

CAR_CAM_POS_RELATIVE_Z = .07112

CAR_HEIGHT = .10795

# Funny discovery, this really doesn't matter for our transform, because both L_MAX and
# L_MIN have it in the numerator. Nevertheless left in because it _is_ 
# used by path_planning_perspective.
CAR_CAM_HEIGHT = CAR_CAM_POS_RELATIVE_Y + CAR_HEIGHT

CAR_CAM_FORWARD = CAR_CAM_POS_RELATIVE_Z

CAR_CAM_ROTATION_X = 15  # degrees off of level, towards ground

CAR_IMG_PORTION = .6

CAR_CAM_TOP_ANGLE = CAR_CAM_ROTATION_X + CAR_CAM_FOV * (1 / 2 - CAR_IMG_PORTION)

L_MIN = (
    CAR_CAM_HEIGHT 
    / math.tan(math.radians(CAR_CAM_ROTATION_X) 
               + math.radians(CAR_CAM_FOV) / 2))
L_MAX = CAR_CAM_HEIGHT / math.tan(math.radians(CAR_CAM_TOP_ANGLE))
X_U = L_MAX * math.tan(math.radians(CAR_CAM_FOV) / 2)
Z_U = L_MAX
X_B = L_MIN * math.tan(math.radians(CAR_CAM_FOV) / 2)
Z_B = L_MIN

X_OVER_Z_RATIO = (2 * X_U) / (Z_U - Z_B)

IMG_SIZE_Z = 400
IMG_SIZE_X = int(X_OVER_Z_RATIO * IMG_SIZE_Z)

CAR_CAM_RESOLUTION = 1088 // 2

X_REMAP = None

Z_REMAP = None

def _init_remaps():
    global X_REMAP, Z_REMAP
    z_imax, x_imax = CAR_CAM_RESOLUTION, CAR_CAM_RESOLUTION
    X_REMAP = np.zeros((int(IMG_SIZE_Z), int(IMG_SIZE_X)), dtype=np.float32)
    Z_REMAP = np.zeros((int(IMG_SIZE_Z), int(IMG_SIZE_X)), dtype=np.float32)
    for z_img in range(int(z_imax * CAR_IMG_PORTION)):
        z_ifrac = z_img / z_imax
        z_plot = (
            CAR_CAM_HEIGHT 
            / math.tan(math.radians(CAR_CAM_ROTATION_X)
                       + math.radians(CAR_CAM_FOV) / 2 
                       - z_ifrac * math.radians(CAR_CAM_FOV)))
        z_pfrac = (z_plot - L_MIN) / (L_MAX - L_MIN)
        z_w = int(IMG_SIZE_Z) - int(z_pfrac * int(IMG_SIZE_Z)) - 1
        for x_img in range(x_imax):
            x_ifrac = x_img / x_imax
            x_plot = 2 * (x_ifrac - .5) * math.tan(math.radians(CAR_CAM_FOV) / 2) * z_plot
            x_pfrac = .5 * (x_plot / X_U + 1)
            x_w = int(x_pfrac * int(IMG_SIZE_X)) - 1
            
            X_REMAP[z_w, x_w] = x_img
            Z_REMAP[z_w, x_w] = z_imax - z_img - 1

    X_REMAP, Z_REMAP = cv2.convertMaps(X_REMAP, Z_REMAP, cv2.CV_32FC1)

_init_remaps()

def car_img_to_top_down_perspective(car_img):
    car_img[0, 0] = 0
    return cv2.remap(car_img, X_REMAP, Z_REMAP, cv2.INTER_NEAREST)
