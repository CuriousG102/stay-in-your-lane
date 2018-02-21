import math

import cv2
import numpy as np

import image_utils
import prediction
import track_floor_utils_perspective

MAX_STEERING = 25
MAX_STEERING_TO_ATTEMPT = 18
STEERING_ACTION_INCREMENTS = 1
MOMENTUM_PREFERENCE = 40
STEERING_PENALTY_EXPONENT = 0.7
PATH_THICKNESS = 4
OVERSTEER = 1.1

STEERING_OVERLAY_INDEXING_OFFSET = int(
    MAX_STEERING / STEERING_ACTION_INCREMENTS)

LINE_DILATION_KERNEL = np.ones((10, 10))

IMG_OVER_REAL_RATIO = (
    track_floor_utils_perspective.IMG_SIZE_Z 
    / (track_floor_utils_perspective.Z_U - track_floor_utils_perspective.Z_B))

STEERING_OVERLAYS = None

def _initialize_steering_overlays():
    global STEERING_OVERLAYS
    STEERING_OVERLAYS = []

    s_angles = range(
        -MAX_STEERING, MAX_STEERING + 1, STEERING_ACTION_INCREMENTS)
    for s_angle in s_angles:
        steering_overlay = np.zeros(
            (track_floor_utils_perspective.IMG_SIZE_Z, 
             track_floor_utils_perspective.IMG_SIZE_X, 3), 
            dtype=np.uint8)
        path = prediction.telemetry_for_steering_pure(s_angle)
        if isinstance(path, prediction.Line):
            left = int(
                track_floor_utils_perspective.IMG_SIZE_X/2
                - PATH_THICKNESS / 2)
            right = int(
                track_floor_utils_perspective.IMG_SIZE_X/2
                + PATH_THICKNESS / 2)
            steering_overlay[
                :,left:right,0] = 255
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
                img_r, (255, 0, 0), PATH_THICKNESS)
        STEERING_OVERLAYS.append(steering_overlay)

_initialize_steering_overlays()

def get_overlay_for_steering(s_angle):
    return STEERING_OVERLAYS[
        int(s_angle / STEERING_ACTION_INCREMENTS)
        + STEERING_OVERLAY_INDEXING_OFFSET]

def get_dilated_top_down_thresholded_img(telemetry):
    img = image_utils.get_cv2_from_tel_field(telemetry, 'front_camera_image')
    replot_img = (
        track_floor_utils_perspective.car_img_to_top_down_perspective(img))
    thresh_replot_img = image_utils.simple_threshold_ternary(replot_img)
    return cv2.dilate(thresh_replot_img, LINE_DILATION_KERNEL)

def get_dilated_top_down_thresholded_outside_img(telemetry):
    img = image_utils.get_cv2_from_tel_field(telemetry, 'front_camera_image')
    replot_img = (
        track_floor_utils_perspective.car_img_to_top_down_perspective(img))
    thresh_replot_img = image_utils.simple_threshold_ternary(replot_img)
    thresh_replot_img[:, :, :2] = 0
    return cv2.dilate(thresh_replot_img, LINE_DILATION_KERNEL)

def score_s_angle(top_down_thresh, prospective_s_angle, current_s_angle):
    img = (
        top_down_thresh 
        + get_overlay_for_steering(prospective_s_angle))
    b, g, r = (img[:, :, i] for i in range(3))
    overlap_sum = (r & b).sum() /255
    print(prospective_s_angle, current_s_angle, abs(prospective_s_angle - current_s_angle))
    momentum_penalty = (
        MOMENTUM_PREFERENCE 
        * min(1, abs(prospective_s_angle - current_s_angle)))
    angle_softener = math.sqrt(abs(prospective_s_angle)+3)
    steering_penalty = abs(prospective_s_angle)**STEERING_PENALTY_EXPONENT
    score = -(((overlap_sum) / angle_softener) + steering_penalty)
    print('(', overlap_sum, ') / ', angle_softener, ' + ', steering_penalty, ' = ', -score)
    # score = -(((overlap_sum + momentum_penalty) / angle_softener) + steering_penalty)
    # print('(', overlap_sum, ' + ', momentum_penalty, ') / ', angle_softener, ' + ', steering_penalty, ' = ', -score)
    return score

def get_best_s_angle(telemetry):
    # Should add in relevance decay at some point.
    s_angles = range(
        MAX_STEERING_TO_ATTEMPT, 
        -MAX_STEERING_TO_ATTEMPT-1, 
        -STEERING_ACTION_INCREMENTS)
    top_down_thresh = get_dilated_top_down_thresholded_outside_img(telemetry)
    ranking_key = (
        lambda s_angle: score_s_angle(
            top_down_thresh, s_angle, telemetry.steering))
    best_scoring_angle = max(s_angles, key=ranking_key)
    best_angle = min(abs(best_scoring_angle)**OVERSTEER, MAX_STEERING_TO_ATTEMPT)
    if best_scoring_angle < 0:
        best_angle *= -1
    return best_angle

def drive_loop(s, times):
    EVERY_DELTA = 0.12
    THROTTLE = 1
    elapsed_time = 0
    current_steering = 0

    curr_time = 0
    while True:
        t = s.get_telemetry()
        curr_time += t.delta_time
        if t.finished:
            print('VICTORY in {0} seconds'.format(
                curr_time))
            s.reset_instruction()
            times.append(curr_time)
            curr_time = 0
            continue
        elapsed_time += t.delta_time
    if elapsed_time >= EVERY_DELTA:
        elapsed_time = 0
        current_steering = get_best_s_angle(t)

    s.send_instructions(current_steering, THROTTLE)
