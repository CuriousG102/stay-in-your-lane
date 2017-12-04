import collections
import os
import pickle
import random
random.seed(1234)

import cv2
import numpy as np
from scipy import ndimage

import prediction
import track_floor_utils 

FINISH_LINE_Z_SCALE = 0.25 * 10
FINISH_LINE_POS = (0.35, -74.39)
FINISH_LINE_ROT_Y = -80.22701
MAX_STEERING = 25
STEERING_ACTION_INCREMENTS = 5

OFF_TRACK_VALUE = np.iinfo(np.int32).min
MAX_DIST_VALUE = np.iinfo(np.int32).max
FINISH_LINE_VALUE = OFF_TRACK_VALUE + 1
TRACK_FLOOR_DISTANCE = None 
MIN_DIST_COLOR, MAX_DIST_COLOR = 255, 30

PICKLE_FILE_NAME = 'pickle_dist'

def get_track_floor_visual():
    track_img = np.zeros(TRACK_FLOOR_DISTANCE.shape + (3,), dtype=np.uint8)
    # make non-points red
    track_img[:, :, 2] = (255 * 
        (TRACK_FLOOR_DISTANCE == OFF_TRACK_VALUE).astype(np.uint8))
    # make finish line blue
    track_img[:, :, 0] = (255 * 
        (TRACK_FLOOR_DISTANCE == FINISH_LINE_VALUE).astype(np.uint8))
    
    dist_points_selector = (
        (TRACK_FLOOR_DISTANCE != OFF_TRACK_VALUE)
        & (TRACK_FLOOR_DISTANCE != MAX_DIST_VALUE)
        & (TRACK_FLOOR_DISTANCE != FINISH_LINE_VALUE))
    track_floor_distances_only = (
        (TRACK_FLOOR_DISTANCE * dist_points_selector).astype(np.float64))
    min_dist, max_dist = (
        track_floor_distances_only.min(), track_floor_distances_only.max())
    dist_slope = (MIN_DIST_COLOR - MAX_DIST_COLOR) / (min_dist - max_dist)
    dist_intercept = MIN_DIST_COLOR
    track_floor_distances_only *= dist_slope
    track_floor_distances_only += dist_intercept
    track_floor_distances_only *= dist_points_selector
    track_img[:, :, 1] = track_floor_distances_only.astype(np.uint8)

    return track_img

def _initialize_track_floor_distance():
    # set distances for all that aren't known to what is 
    # effectively +infinity
    global TRACK_FLOOR_DISTANCE
    TRACK_FLOOR_DISTANCE = (cv2.imread('TrackFloorInverseFill.png')[:, :, 0]
        .astype(np.bool).astype(np.int32) * OFF_TRACK_VALUE)
    hash_of_precursor = hash(TRACK_FLOOR_DISTANCE.tostring())
    if os.path.exists(PICKLE_FILE_NAME):
        with open(PICKLE_FILE_NAME, 'rb') as f:
            hash_of_pickle_precursor, pickled_dist_matrix = pickle.load(f)
            if hash_of_precursor == hash_of_pickle_precursor:
                TRACK_FLOOR_DISTANCE = pickled_dist_matrix
                return

    print('Have to build dist matrix for path_planning. This will take a while, '
          'but only has to run when the temp file is not present or when the '
          'precursor to the distance matrix changes')
    zero_mask = TRACK_FLOOR_DISTANCE == 0
    TRACK_FLOOR_DISTANCE += zero_mask.astype(np.int32) * MAX_DIST_VALUE

    # draw the finish line in two steps:
    # first as an impenetrable wall one point up on the z axis.
    # second as zero distance points
    pos_x, pos_z = FINISH_LINE_POS
    rot_y_rad = np.deg2rad(FINISH_LINE_ROT_Y)
    x1 = FINISH_LINE_Z_SCALE / 2 * np.sin(rot_y_rad) + pos_x
    x2 = -(FINISH_LINE_Z_SCALE / 2 * np.sin(rot_y_rad)) + pos_x
    z1 = FINISH_LINE_Z_SCALE / 2 * np.cos(rot_y_rad) + pos_z
    z2 = -(FINISH_LINE_Z_SCALE / 2 * np.cos(rot_y_rad)) + pos_z
    img_pt_1 = track_floor_utils.img_point_bottom_left_to_top_left(
        track_floor_utils.unity_plane_point_to_img_point((x1, z1)))
    img_pt_2 = track_floor_utils.img_point_bottom_left_to_top_left(
        track_floor_utils.unity_plane_point_to_img_point((x2, z2)))
    img_x1, img_z1 = img_pt_1
    img_x2, img_z2 = img_pt_2
    if (img_x1 > img_x2):
        img_pt_1, img_pt_2 = img_pt_2, img_pt_1
    img_x1, img_z1 = img_pt_1
    img_x2, img_z2 = img_pt_2
    z_slope = (img_z2 - img_z1) / (img_x2 - img_x1)
    z_intercept = img_z1
    for img_x in range(int(img_x1), int(img_x2)):
        img_z = z_intercept + z_slope * (img_x - img_x1)
        img_x, img_z = int(img_x), int(img_z)
        if TRACK_FLOOR_DISTANCE[img_z - 1, img_x] != OFF_TRACK_VALUE:
            TRACK_FLOOR_DISTANCE[img_z - 1, img_x] = FINISH_LINE_VALUE
        if TRACK_FLOOR_DISTANCE[img_z - 2, img_x] != OFF_TRACK_VALUE:
            TRACK_FLOOR_DISTANCE[img_z - 2, img_x] = FINISH_LINE_VALUE
        if TRACK_FLOOR_DISTANCE[img_z, img_x] != OFF_TRACK_VALUE:
            TRACK_FLOOR_DISTANCE[img_z, img_x] = 0

    # Fill by starting with values of 0, coloring all pixels adjacent
    # to them with 1. Start with values of 1, color all pixels adjacent
    # to them with 2 ... Stop when all pixels are colored.
    for i in range(0, 99999999999):
        t_i_mask = TRACK_FLOOR_DISTANCE == i
        if not t_i_mask.any():
            break
        t_i_dilation = ndimage.morphology.binary_dilation(
            t_i_mask, np.ones((3,3)))
        t_i_dilation &= ~(
            (TRACK_FLOOR_DISTANCE == OFF_TRACK_VALUE)
            | (TRACK_FLOOR_DISTANCE == FINISH_LINE_VALUE)
            | (TRACK_FLOOR_DISTANCE <= i))
        TRACK_FLOOR_DISTANCE *= ~t_i_dilation
        TRACK_FLOOR_DISTANCE += (i + 1) * t_i_dilation.astype(np.int32)

    with open('pickle_dist', 'wb') as f:
        pickle.dump((hash_of_precursor, TRACK_FLOOR_DISTANCE), f)

_initialize_track_floor_distance()

# track_max_dist = None
# def heuristic_cost(position, speed):
#     global track_max_dist
#     if track_max_dist is None:
#         dist_points_selector = (
#             (TRACK_FLOOR_DISTANCE != OFF_TRACK_VALUE)
#             & (TRACK_FLOOR_DISTANCE != MAX_DIST_VALUE)
#             & (TRACK_FLOOR_DISTANCE != FINISH_LINE_VALUE))
#         track_max_dist = (TRACK_FLOOR_DISTANCE * dist_points_selector).max()
#     img_point = track_floor_utils.img_point_bottom_left_to_top_left(
#         track_floor_utils.unity_plane_point_to_img_point(position))
#     img_dist = TRACK_FLOOR_DISTANCE[img_point[::-1]]
#     # rough approximation

def bfs_construct_path_helper(state, meta):
    actions = []
    while True:
        if state in meta:
            state, action = meta[state]
            actions.append(action)
        else:
            break
    actions.reverse()
    return actions


def bfs_across_finish_helper(state, meta):
    '''
    Above finish line within some distance 
    and the path is plausibly long. Super hacky.
    '''
    pos, rot_y = state
    pos_x, pos_z = track_floor_utils.img_point_bottom_left_to_top_left(
        track_floor_utils.unity_plane_point_to_img_point(pos))
    t_val = TRACK_FLOOR_DISTANCE[int(pos_z), int(pos_x)]
    return (
        t_val > 5800 
        and len(bfs_construct_path_helper(state, meta)) > 90)


STEERING_ACTIONS = list(range(
    -MAX_STEERING, MAX_STEERING + 1, STEERING_ACTION_INCREMENTS))

# too slow :-(
def get_steering_angle_bfs(tel, pos, rot_y, delta_time):
    open_set = collections.deque()
    open_set_helper = set()
    closed_set = set()
    meta = dict()
    round_n = lambda n: np.around(n, decimals=1)
    round_pos = lambda p: (round_n(p[0]), round_n(p[1]))
    round_state = lambda s: (round_pos(s[0]), s[1]//5 * 5)

    start = round_state((pos, rot_y))
    open_set.append(start)
    open_set_helper.add(start)

    while open_set:
        parent_state = open_set.popleft()
        open_set_helper.remove(parent_state)
        parent_pos, parent_rot_y = parent_state

        # if across finish line, return start steering angle
        # We detect this in a hacky way
        if bfs_across_finish_helper(parent_state, meta):
            ations = bfs_construct_path_helper(parent_state, meta)
            return actions[0]

        children_generator = (
            (round_state(prediction.telemetry_after_delta_time_pure(
                max(tel.speed, 4), s_angle, parent_pos, parent_rot_y, delta_time)), 
             s_angle)
            for s_angle in STEERING_ACTIONS)
        for child_state, action in children_generator:
            print(child_state, action)
            child_pos, child_rot_y = child_state
            child_img_x, child_img_z = (
                track_floor_utils.img_point_bottom_left_to_top_left(
                    track_floor_utils.unity_plane_point_to_img_point(child_pos)))
            track_floor_height, track_floor_width = TRACK_FLOOR_DISTANCE.shape
            # check that the point is viable. Otherwise pretend we never saw
            # it at all :p
            if (child_img_x < 0 or child_img_x >= track_floor_width or 
                child_img_z < 0 or child_img_z >= track_floor_height):
                continue
            t_val = TRACK_FLOOR_DISTANCE[int(child_img_z), int(child_img_x)]
            if t_val == OFF_TRACK_VALUE or t_val == MAX_DIST_VALUE:
                continue

            if child_state in closed_set:
                continue

            if child_state not in open_set_helper:
                meta[child_state] = (parent_state, action)
                open_set.append(child_state)
                open_set_helper.add(child_state)

        closed_set.add(parent_state)

# tried it out with various smoothing and time intervals with true state,
# but it does appear that this is a bit too dumb
def get_steering_angle_greedy(tel, pos, rot_y, delta_time):
    children_generator = (
        (prediction.telemetry_after_delta_time_pure(
            max(tel.speed, 1), s_angle, pos, rot_y, delta_time), 
         s_angle)
        for s_angle in STEERING_ACTIONS)
    best_angle = None
    best_t_val = float('+inf')
    for child_state, s_angle in children_generator:
        child_pos, child_rot_y = child_state
        child_img_x, child_img_z = (
            track_floor_utils.img_point_bottom_left_to_top_left(
                track_floor_utils.unity_plane_point_to_img_point(child_pos)))
        # check that the point is viable. Otherwise pretend we never saw
        # it at all :p
        track_floor_height, track_floor_width = TRACK_FLOOR_DISTANCE.shape
        if (child_img_x < 0 or child_img_x >= track_floor_width or 
            child_img_z < 0 or child_img_z >= track_floor_height):
            continue
        t_val = TRACK_FLOOR_DISTANCE[int(child_img_z), int(child_img_x)]
        if t_val == OFF_TRACK_VALUE or t_val == MAX_DIST_VALUE:
            continue
        if t_val < best_t_val:
            best_t_val = t_val
            best_angle = s_angle
    return best_angle


SAFETY_MARGIN = 60
SAFETY_ADDER = 200
CHILD_SAFETY_ADDER = 100
CHANGE_LIMIT = 6
CHANGE_INTVL = 2
def get_steering_angle_limited_horizon_helper(tel, pos, rot_y, delta_time, h, change_limiter=True):
    img_x, img_z = (
            track_floor_utils.img_point_bottom_left_to_top_left(
                track_floor_utils.unity_plane_point_to_img_point(pos)))
    track_floor_height, track_floor_width = TRACK_FLOOR_DISTANCE.shape
    if (img_x - SAFETY_MARGIN < 0 or img_x + SAFETY_MARGIN >= track_floor_width or 
        img_z - SAFETY_MARGIN < 0 or img_z + SAFETY_MARGIN >= track_floor_height):
        return (None, float('+inf'))
    t_val = TRACK_FLOOR_DISTANCE[int(img_z), int(img_x)]
    if t_val == OFF_TRACK_VALUE or t_val == MAX_DIST_VALUE:
        return (None, float('+inf'))
    safety_box = TRACK_FLOOR_DISTANCE[int(img_z - SAFETY_MARGIN):int(img_z + SAFETY_MARGIN),
                                      int(img_x - SAFETY_MARGIN):int(img_x + SAFETY_MARGIN)]
    safety_adder = 0
    if OFF_TRACK_VALUE in safety_box or MAX_DIST_VALUE in safety_box:
        safety_adder = SAFETY_ADDER
    if (h == 0):
        return (None, safety_adder + t_val)

    if change_limiter:
        possible_angles = range(max(-MAX_STEERING, int(tel.steering) - CHANGE_LIMIT), 
                                min(MAX_STEERING + 1, int(tel.steering) + CHANGE_LIMIT + 1), 
                                CHANGE_INTVL)
    else:
        possible_angles = range(-MAX_STEERING, MAX_STEERING + 1, STEERING_ACTION_INCREMENTS)

    children_generator = (
        (prediction.telemetry_after_delta_time_pure(
            max(tel.speed, 1), s_angle, pos, rot_y, delta_time), 
         s_angle)
        for s_angle in possible_angles)
    best_angle = None
    best_t_val = float('+inf')
    for child_state, s_angle in children_generator:
        child_pos, child_rot_y = child_state
        child_img_x, child_img_z = (
            track_floor_utils.img_point_bottom_left_to_top_left(
                track_floor_utils.unity_plane_point_to_img_point(pos)))
        child_safety_box = TRACK_FLOOR_DISTANCE[
            int(child_img_z - SAFETY_MARGIN):int(child_img_z + SAFETY_MARGIN),
            int(child_img_x - SAFETY_MARGIN):int(child_img_x + SAFETY_MARGIN)]
        child_safety_adder = 0
        if OFF_TRACK_VALUE in safety_box or MAX_DIST_VALUE in child_safety_box:
            child_safety_adder = CHILD_SAFETY_ADDER
        future_t_val = child_safety_adder + get_steering_angle_limited_horizon_helper(tel, child_pos, child_rot_y, delta_time, h - 1)[1]
        # keep it from trying to hop the track ...
        # we can do this more elegantly in the future by looking 
        # at the trajectory and seeing if it crosses any areas marked
        # OFF_TRACK_VALUE
        if (t_val - future_t_val) > 400:
            return (None, float('+inf'))
        if future_t_val < best_t_val:
            best_angle = s_angle
            best_t_val = future_t_val

    return (best_angle, best_t_val)

def get_steering_angle_limited_horizon(tel, pos, rot_y, delta_time, h):
    return get_steering_angle_limited_horizon_helper(tel, pos, rot_y, delta_time, h)[0]