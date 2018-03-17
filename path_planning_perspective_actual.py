import functools
import math
import multiprocessing as mp
from multiprocessing.pool import Pool

import cv2
import numpy as np

import cv2
import numpy as np

import image_utils
import prediction
import track_floor_utils_perspective

MAX_STEERING = 16
# MAX_STEERING = 14
STEERING_ACTION_INCREMENTS = 4
PATH_THICKNESS = 40
MOMENTUM_PREFERENCE = 450
OUTSIDE_MULTIPLIER = 0.5

STEERING_OVERLAY_INDEXING_OFFSET = int(
    MAX_STEERING / STEERING_ACTION_INCREMENTS)

LINE_DILATION_KERNEL = np.ones((5, 5))

IMG_OVER_REAL_RATIO = (
    track_floor_utils_perspective.IMG_SIZE_Z 
    / (track_floor_utils_perspective.Z_U - track_floor_utils_perspective.Z_B))

NUM_WORKERS = 4

worker_pool = None

STEERING_OVERLAYS = None

def _initialize_steering_overlays():
    global STEERING_OVERLAYS
    STEERING_OVERLAYS = []

    s_angles = range(
        -MAX_STEERING, MAX_STEERING + 1, STEERING_ACTION_INCREMENTS)
    for s_angle in s_angles:
        steering_overlay = np.zeros(
            (track_floor_utils_perspective.IMG_SIZE_Z, 
             track_floor_utils_perspective.IMG_SIZE_X,), 
            dtype=np.uint8)
        path = prediction.telemetry_for_steering_pure(s_angle)
        if isinstance(path, prediction.Line):
            left = int(
                track_floor_utils_perspective.IMG_SIZE_X/2
                - PATH_THICKNESS / 2)
            right = int(
                track_floor_utils_perspective.IMG_SIZE_X/2
                + PATH_THICKNESS / 2)
            steering_overlay[:,left:right] = 1
        elif isinstance(path, prediction.Circle):
            r = path.radius
            # Position of the camera from the bottom 
            # of the image.
            car_cam_z_pos = -track_floor_utils_perspective.L_MIN
            # Position of the back of the car from 
            # the bottom of the image.
            car_back_z_pos = (
                car_cam_z_pos 
                - track_floor_utils_perspective.CAR_CAM_POS_RELATIVE_Z
                - abs(prediction.CAR_AXLE_BACK))
            # Z position of back of car in image coordinates.
            # car_back_img_z = int(
            #     track_floor_utils_perspective.IMG_SIZE_Z + 
            #     abs(IMG_OVER_REAL_RATIO * car_back_z_pos))

            # Z position of the center of the circle 
            # off of the bottom of the image.
            circle_center_z = car_back_z_pos

            # X position of the center of the circle
            # off of the center X of the image.
            circle_center_x = (
                r if path.side is prediction.Circle.CircleSide.LEFT else -r)

            # Convert to image coordinates.
            circle_center_img_z = int(
                track_floor_utils_perspective.IMG_SIZE_Z 
                + abs(IMG_OVER_REAL_RATIO * circle_center_z))

            circle_center_img_x = int(
                track_floor_utils_perspective.IMG_SIZE_X / 2
                + IMG_OVER_REAL_RATIO * circle_center_x)

            img_r = int(IMG_OVER_REAL_RATIO * r)

            cv2.circle(
                steering_overlay, 
                (circle_center_img_x, circle_center_img_z,),
                img_r, 1, PATH_THICKNESS)
        STEERING_OVERLAYS.append(steering_overlay)

_initialize_steering_overlays()

def index_for_s_angle(s_angle):
    return (int(s_angle / STEERING_ACTION_INCREMENTS)
            + STEERING_OVERLAY_INDEXING_OFFSET)

def get_overlay_for_steering(s_angle):
    return STEERING_OVERLAYS[index_for_s_angle(s_angle)]

def get_top_down(undist_image):
    return track_floor_utils_perspective.car_img_to_top_down_perspective(
        undist_image[:,208:-208,:])

def get_top_down_hue(undist_image):
    return cv2.cvtColor(get_top_down(undist_image), cv2.COLOR_BGR2HLS) 

def get_dilated_top_down_thresholded_img_white(top_down_hue):
    # Hacky and should be controlled by constant
    top_down_thresholded = image_utils.threshold_for_white(top_down_hue)
    return cv2.dilate(top_down_thresholded, LINE_DILATION_KERNEL)

def get_dilated_top_down_thresholded_img_yellow(top_down_hue):
    # Hacky and should be controlled by constant
    top_down_thresholded = image_utils.threshold_for_yellow(top_down_hue)
    return cv2.dilate(top_down_thresholded, LINE_DILATION_KERNEL)

def score_s_angle(
    top_down_thresholded_yellow, #top_down_thresholded_white, 
    previous_s_angle, prospective_s_angle):
    steering_overlay = get_overlay_for_steering(prospective_s_angle)
    overlap_sum_yellow = (
        top_down_thresholded_yellow 
        & steering_overlay).sum()
    #overlap_sum_white = (
    #    top_down_thresholded_white 
    #    & steering_overlay).sum()
    score = (
        overlap_sum_yellow
        # - OUTSIDE_MULTIPLIER * overlap_sum_white
        + (MOMENTUM_PREFERENCE 
           if prospective_s_angle == previous_s_angle 
           else 0))
    return score

def get_s_angle_score_pair(top_down_thresh, previous_s_angle, prospective_s_angle):
    return (
        prospective_s_angle, 
        score_s_angle(top_down_thresh, previous_s_angle, prospective_s_angle))

def get_best_s_angle(undist_image, previous_s_angle):
    s_angles = range(
        MAX_STEERING, 
        -MAX_STEERING - 1, 
        -STEERING_ACTION_INCREMENTS)
    top_down_hue = get_top_down_hue(undist_image)
    top_down_thresholded_yellow = get_dilated_top_down_thresholded_img_yellow(top_down_hue)
    #top_down_thresholded_white = get_dilated_top_down_thresholded_img_white(top_down_hue)
    
    get_s_angle_score_pair_partial = functools.partial(
            get_s_angle_score_pair, top_down_thresholded_yellow, previous_s_angle)

    # s_angle_scores = worker_pool.imap_unordered(get_s_angle_score_pair_partial, s_angles)
    # s_angle_scores = worker_pool.map(get_s_angle_score_pair_partial, s_angles)
    s_angle_scores = map(get_s_angle_score_pair_partial, s_angles)
    best_scoring_angle, score = max(s_angle_scores, key=lambda pair: pair[1])
    best_angle = min(abs(best_scoring_angle), MAX_STEERING)
    if best_scoring_angle < 0:
        best_angle *= -1
    return best_angle, score

# worker_pool = mp.pool.Pool(NUM_WORKERS)
