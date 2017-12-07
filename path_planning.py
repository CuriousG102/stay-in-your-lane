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
TRACK_FLOOR_DISTANCE_EDGE_DECAY = None
# for deubgging
TRACK_FLOOR_EDGE_PRECURSOR = None
MIN_DIST_COLOR, MAX_DIST_COLOR = 255, 30

PICKLE_FILE_NAME = 'pickle_dist'
PICKLE_EDGE_FILE_NAME = 'pickled_edge_dist'

def get_track_floor_visual(in_arg):
    track_img = np.zeros(in_arg.shape + (3,), dtype=np.uint8)
    # make non-points red
    track_img[:, :, 2] = (255 * 
        (in_arg == OFF_TRACK_VALUE).astype(np.uint8))
    # make finish line blue
    track_img[:, :, 0] = (255 * 
        (in_arg == FINISH_LINE_VALUE).astype(np.uint8))
    
    dist_points_selector = (
        (in_arg != OFF_TRACK_VALUE)
        & (in_arg != MAX_DIST_VALUE)
        & (in_arg != FINISH_LINE_VALUE))
    track_floor_distances_only = (
        (in_arg * dist_points_selector).astype(np.float64))
    min_dist, max_dist = (
        track_floor_distances_only.min(), track_floor_distances_only.max())
    print(min_dist, max_dist)
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


DECAY_ADD = 500
DECAY_RATE = 0.993
def _initialize_track_floor_distance_edge_decay():
    # set distances for all that aren't known to what is 
    # effectively +infinity
    global TRACK_FLOOR_DISTANCE_EDGE_DECAY
    global TRACK_FLOOR_EDGE_PRECURSOR
    hash_of_precursor = hash(
        (TRACK_FLOOR_DISTANCE.tostring(), DECAY_ADD, DECAY_RATE))
    if os.path.exists(PICKLE_EDGE_FILE_NAME):
        with open(PICKLE_EDGE_FILE_NAME, 'rb') as f:
            hash_of_pickle_precursor, pickled_dist_matrix = pickle.load(f)
            if hash_of_precursor == hash_of_pickle_precursor:
                TRACK_FLOOR_DISTANCE_EDGE_DECAY = pickled_dist_matrix
                return
    print('Have to build dist decay matrix for path_planning. '
          'This will take a while, '
          'but only has to run when the temp file is not present or when the '
          'precursors to the dist decay distance matrix change.')

    TRACK_FLOOR_DISTANCE_EDGE_DECAY = np.zeros_like(TRACK_FLOOR_DISTANCE)
    # we don't want to decay values around the finish line or modify
    # finish line value
    track_floor_distance_mask = TRACK_FLOOR_DISTANCE != OFF_TRACK_VALUE
    finish_line_mask = TRACK_FLOOR_DISTANCE == FINISH_LINE_VALUE

    track_floor_starter_dilation = ndimage.morphology.binary_dilation(
        ~track_floor_distance_mask, np.ones((3,3)))
    TRACK_FLOOR_DISTANCE_EDGE_DECAY += (
        track_floor_starter_dilation 
        & track_floor_distance_mask 
        & ~finish_line_mask)
    for i in range(1, 9999999999):
        print(i)
        current_propagation_mask = TRACK_FLOOR_DISTANCE_EDGE_DECAY == i
        if not current_propagation_mask.any():
            break
        dilation = ndimage.morphology.binary_dilation(
            current_propagation_mask, np.ones((3,3)))
        dilation &= (
            track_floor_distance_mask # is in track
            # and is not an already filled in value
            & ~((TRACK_FLOOR_DISTANCE_EDGE_DECAY >= 1) 
                & (TRACK_FLOOR_DISTANCE_EDGE_DECAY <= i))
            & ~finish_line_mask) # and is not finish line 
        TRACK_FLOOR_DISTANCE_EDGE_DECAY += (i + 1) * dilation.astype(np.int32)

    TRACK_FLOOR_DISTANCE_EDGE_DECAY = ((
        DECAY_ADD * DECAY_RATE ** TRACK_FLOOR_DISTANCE_EDGE_DECAY) 
    * track_floor_distance_mask * ~finish_line_mask)

    TRACK_FLOOR_DISTANCE_EDGE_DECAY = (
        TRACK_FLOOR_DISTANCE_EDGE_DECAY.astype(np.int32))

    TRACK_FLOOR_EDGE_PRECURSOR = (
        TRACK_FLOOR_DISTANCE 
        * (~(track_floor_distance_mask) | finish_line_mask)
        + TRACK_FLOOR_DISTANCE_EDGE_DECAY)

    TRACK_FLOOR_DISTANCE_EDGE_DECAY += TRACK_FLOOR_DISTANCE

    # We have to correct integer overflows...
    max_dist_mask = TRACK_FLOOR_DISTANCE == MAX_DIST_VALUE
    TRACK_FLOOR_DISTANCE_EDGE_DECAY *= ~max_dist_mask
    TRACK_FLOOR_DISTANCE_EDGE_DECAY += (max_dist_mask.astype(np.int32) * MAX_DIST_VALUE)

    with open(PICKLE_EDGE_FILE_NAME, 'wb') as f:
        pickle.dump((hash_of_precursor, TRACK_FLOOR_DISTANCE_EDGE_DECAY), f)

_initialize_track_floor_distance_edge_decay()

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


CHANGE_LIMIT = 10
CHANGE_INTVL = 2
NUM_BREAK_DOWNS = 10
def get_steering_angle_limited_horizon_helper(tel, pos, rot_y, delta_time, projection_multiplier, h, dist_matrix, change_limiter=True):
    img_x, img_z = (
            track_floor_utils.img_point_bottom_left_to_top_left(
                track_floor_utils.unity_plane_point_to_img_point(pos)))
    track_floor_height, track_floor_width = dist_matrix.shape
    if (img_x < 0 or img_x >= track_floor_width or 
        img_z < 0 or img_z >= track_floor_height):
        return (None, float('+inf'))
    t_val = dist_matrix[int(img_z), int(img_x)]
    if t_val == OFF_TRACK_VALUE or t_val == MAX_DIST_VALUE:
        return (None, float('+inf'))
    if (h == 0):
        return (None, t_val)

    if change_limiter:
        possible_angles = range(max(-MAX_STEERING, int(tel.steering) - CHANGE_LIMIT), 
                                min(MAX_STEERING + 1, int(tel.steering) + CHANGE_LIMIT + 1), 
                                CHANGE_INTVL)
    else:
        possible_angles = range(-MAX_STEERING, MAX_STEERING + 1, STEERING_ACTION_INCREMENTS)

    children_generator = (
        (prediction.break_down_into_times_pure(
            max(tel.speed, 2), s_angle, pos, rot_y, delta_time, NUM_BREAK_DOWNS), 
         s_angle)
        for s_angle in possible_angles)
    best_angle = None
    best_t_val = float('+inf')
    for child_state, s_angle in children_generator:
        child_pos, child_rot_y = child_state
        child_img_x, child_img_z = (
            track_floor_utils.img_point_bottom_left_to_top_left(
                track_floor_utils.unity_plane_point_to_img_point(pos)))
        future_t_val = (
            get_steering_angle_limited_horizon_helper(
                tel, child_pos, child_rot_y, 
                delta_time * projection_multiplier, 
                projection_multiplier, h - 1, dist_matrix)[1])
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

def get_steering_angle_limited_horizon(tel, pos, rot_y, delta_time, projection_multiplier, h, dist_matrix):
    return get_steering_angle_limited_horizon_helper(tel, pos, rot_y, delta_time, projection_multiplier, h, dist_matrix)[0]

def drive_loop(s):
    EVERY_DELTA = 0.25
    THROTTLE = 1
    elapsed_time = 0
    current_steering = 0
    while True:
        t = s.get_telemetry()
        if t.finished:
            print('VICTORY')
            s.reset_instruction()
            continue
        elapsed_time += t.delta_time
        if elapsed_time >= EVERY_DELTA:
            elapsed_time = 0
            pos = t.x, t.z
            rot_y = t.rot_y
            #if abs(t.steering) > 12:
            #    print('\n\n!!!!!!!HIGH STEERING FALLBACK!!!!!!!')
            #    s_angle = get_steering_angle_limited_horizon(
            #        t, pos, rot_y, .25, 1.8, 4,
            #    TRACK_FLOOR_DISTANCE_EDGE_DECAY)
            if False:
                pass
            else:
                s_angle = get_steering_angle_limited_horizon(
                    t, pos, rot_y, .2, 1.0, 3,
                TRACK_FLOOR_DISTANCE_EDGE_DECAY)
            if s_angle is None:
                print('\n\n!!!!!!!Fallback precision!!!!!!!\n\n')
                s_angle = get_steering_angle_limited_horizon(
                    t, pos, rot_y, .1, 1.0, 4,
                TRACK_FLOOR_DISTANCE_EDGE_DECAY)
                if s_angle is None:
                    print('No path forward')
                    s.reset_instruction()
                    continue
            current_steering = 0.0 * current_steering +  1.0 * (s_angle / 25)
            print('T at decision: ', t)
            print('New steering: ', current_steering)
            print('Recommended s_angle: ', s_angle)
        
        s.send_instructions(current_steering, THROTTLE)
