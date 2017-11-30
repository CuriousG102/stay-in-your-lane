import pandas as pd
import numpy as np
import random
from sim_client import SimClient
import time
import cv2
from PIL import Image
import matplotlib
matplotlib.use("QT5Agg")
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import image_utils
from random import randint
import math


def choose_action(Q_row, er):
    action = 0
    actions = np.array([-.75, -.5, 0, .5, .75])
    ep_random = er
    if random.random() < ep_random:
        action = randint(0, len(actions) - 1)
    else:
        action = np.argmax(Q_row)
    return actions[action], action


def average_slope_intercept(lines):

    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # ignore a vertical line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            left_lines.append((slope, intercept))
            left_weights.append((length))
    # add more weight to longer lines
    left_lane = np.dot(left_weights,  left_lines) / \
        np.sum(left_weights) if len(left_weights) > 0 else None
    #right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None

    return left_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    if slope == 0:
        x1 = 0
        x2 = 0
        y1 = int(y1)
        y2 = int(y2)
    else:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    """
    Determines the lane lines
    function found at: and modified to 
    """
    if lines is not None:
        left_lane = average_slope_intercept(lines)
    else:
        left_lane = 0

    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6         # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    #right_line = make_line_points(y1, y2, right_lane)
    return left_line  # , right_line

    """    
    def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
        # make a separate image to draw lines and combine with the orignal later
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                #print(*line)
                cv2.line(line_image, (line[0][0], line[0][1] ),( line[0][2] ,line[0][3]),  color, thickness)
        # image1 * α + image2 * β + λ
        # image1 and image2 must be the same shape.
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)"""


def draw_lane_lines(image, line, color=[255, 0, 0], thickness=20):
    """
    Draws lane lines according to the input line, returns the image with the lines draw on it
    """
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    if line is not None:
        # print(*line)
        cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)


def average_line_slope(lines):
    """
    Average the slope of all the lines in an image
    """

    slope_sum = 0
    for line in lines:
        if line[0][2] == line[0][0]:
            continue
        if line is not None:
            slope_sum += (line[0][1] - line[0][3]) / (line[0][0] - line[0][2])
    average = slope_sum / len(lines)
    return average


def determine_state(slope, buckets):
    """
    Takes in the buckets and the current average slop of all lines in an image, returns the state of the car

    """

    if slope > 0:
        state = np.digitize(slope, list(reversed(buckets)))
    if slope < 0:
        state = np.digitize(abs(slope), buckets) + len(buckets)
    return state - 1


def determine_reward(slope, buckets):
    """
    Takes in the buckets and the current average slop of all lines in an image, returns the state of the car

    """

    if slope > 0:
        state = np.digitize(slope, list(reversed(buckets)))
    if slope < 0:
        state = np.digitize(abs(slope), buckets) + len(buckets)
    return state


def ROI(image):
    """
    keeps only the center of the image and returns the center third of the image
    """
    empty = np.zeros_like(image)
    empty[:] = (0, 0, 0)


def get_explore_rate(ep):
    MIN_EXPLORE_RATE = .1
    return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - math.log10((ep + 1.) / 25)))


def main():

    # hardcode number of actions
    # Adjust to represent the states
    STATE_BOUNDS = np.array([50, 15, 10, 3, 1.5, 0])
    REWARDS = np.array([-5, -4, -3, -2, 5, 10, 5, -2, -3, -4, -5])
    goal_state = 2
    NUM_states = (STATE_BOUNDS.size) * 2 - 1
    NUM_actions = 5  # left right or none
    episode_number = 500
    t_max = 7
    y = .95
    alpha = .25
    action_step = .2
    control_step = .5
    dt = .02
    min_threshold = 25
    max_threshold = 2 * min_threshold
    kernel_size = 5
    lower = np.uint8([70, 110, 75])
    upper = np.uint8([126, 209, 215])  # 180,228, 228
    crop_bounds_left = .2
    crop_bounds_right = .8
    epsilon = .75
    learning_rate = .5
    MIN_EXPLORE_RATE = 0
    discount_factor = .5

    average_slope_array = []

    s = SimClient()
    s.start()
    # Intialize Q table based on number of rows and actions
    Q = np.zeros([NUM_states, NUM_actions])
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Yellow', cv2.WINDOW_NORMAL)
    cv2.namedWindow('blur', cv2.WINDOW_NORMAL)
    g = open('moves.txt', 'wb')
    qtable = open('Qtable.txt', 'wb')

    for episode in range(episode_number):
        # Intialize the sim also put a reset in here somehwere
        total_reward = 0
        act_index = 2
        action = 0
        state = 0
        next_state = 0
        current_steering = 0
        constant_throttle = .75
        control_time = 0
        t = 0
        # Reset Episode

        flag = 0
        er = get_explore_rate(episode)
        prev_action = 2
        prev_state = 4
        act_index = 0
        reward = 0
        # time.sleep(.5)

        while(t < t_max):
            # determine current state using function and input from sim.
            # Only take an action every control_step time
            average_slope = None
            observation = s.get_telemetry()
            #image = Image.open(observation.front_camera_image)
            # image.show()
            # observation.front_camera_image.show()
            open_cv_image = np.array(observation.cheater_camera_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            #open_cv_image = np.array(observation.overhead_camera_image)
            blurred = cv2.GaussianBlur(
                open_cv_image, (kernel_size, kernel_size), 0)

            #yellow_only = cv2.inRange(blurred, lower, upper)
            #canny_edges = cv2.Canny(blurred, min_threshold, max_threshold)
            #ret, thresh = cv2.threshold(yellow_only, 120, 255, cv2.THRESH_BINARY)
            h, w, c = blurred.shape

            left_bound = int(round(crop_bounds_left * w, 0))
            right_bound = int(round(crop_bounds_right * w, 0))
            top_bound = int(round(.5 * h, 0))
            print(left_bound, right_bound)

            thresh = image_utils.simple_threshold(
                blurred)[0:top_bound, left_bound:right_bound]
            canny_thresh_edges = cv2.Canny(
                thresh, min_threshold, max_threshold)
            # update below code
            lines = cv2.HoughLinesP(canny_thresh_edges, rho=1, theta=1 *
                                    np.pi / 180, threshold=30, minLineLength=25, maxLineGap=40)
            # average line

            if lines is not None:
                lines = lane_lines(canny_thresh_edges, lines)
                output = draw_lane_lines(thresh, lines)
                if lines[0][0] == lines[1][0]:
                    average_slope = 100
                    average_slope_array.append(100)
                else:
                    average_slope = (
                        lines[0][1] - lines[1][1]) / (lines[0][0] - lines[1][0])
                    average_slope_array.append(
                        (lines[0][1] - lines[1][1]) / (lines[0][0] - lines[1][0]))
                average_slope = round(average_slope, 2)
            # print(left_lane,right_lane)
                cv2.imshow('blur', blurred)
                cv2.imshow('Yellow', thresh)
                cv2.imshow('Display', output)
            # average_slope=average_slope_intercept(lines)
            #cv2.imshow('Display', canny_thresh_edges)
                cv2.waitKey(1)
            # plt.imshow(canny_edges)
            # plt.show()

            if average_slope:
                state = determine_state(average_slope, STATE_BOUNDS)
            else:
                state = 0
            print(state)

            if(observation.colliding == True or lines == None):
                Q[prev_state, act_index] += learning_rate * \
                    (-10 + discount_factor * (best_q) -
                     Q[prev_state, act_index])
                break
            if(observation.finished == True):
                Q[prev_state, act_index] += learning_rate * \
                    (1000 + discount_factor *
                     (best_q) - Q[prev_state, act_index])
                break

            print(control_time)
            if (control_time > control_step):
                average_slope_interval = np.sum(
                    average_slope_array) / len(average_slope_array)
                average_slope_interval = round(average_slope_interval, 2)
                if average_slope_interval:
                    state = determine_state(
                        average_slope_interval, STATE_BOUNDS)
                reward = REWARDS[state - 1]
                # update reward based on previous action taken
                best_q = np.amax(Q[state - 1])
                Q[prev_state, act_index] += learning_rate * \
                    (reward + discount_factor *
                     (best_q) - Q[prev_state, act_index])
                # this assigns your award based on your previous action taken
                # print(state)
                # print(reward)
                print(Q)
                control_time = 0
                average_slope_array = []
                action, act_index = choose_action(Q[state, :], er)
                # print(action)
                s.send_instructions(action, constant_throttle)
                prev_action = act_index
                prev_state = state
                np.savetxt(g, [episode, t, state, action, average_slope_interval,
                               reward, prev_action, prev_state], newline=" ")
                np.savetxt(qtable, Q)
                g.write(b'\n')

            else:
                # Increment the sim without action
                s.send_instructions(action, constant_throttle)
            # print(t)
            # print(observation)
            # print(episode)
            print(average_slope)

            # print(observation.colliding)

            # Determine new state
            # next_state=assign_state(distance_from_center_line)
            # Determine reward of new state
            # reward=determine_reward(state
            # total_reward+= reward #add on the reward for the episode

            state = next_state
            control_time += observation.delta_time
            t += observation.delta_time
        # s.get_telemetry()
        s.reset_instruction()
    g.close()
    qtable.close()

    s.stop()

if __name__ == '__main__':
    main()
