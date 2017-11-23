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


def assign_state(distance_from_centerline):
    pass
    state = 0
    # if:
    return state


def choose_action(Q_row):
    action = 1
    actions = np.array([-1, 0, 1])
    if True:
        pass
    else:
        action = np.argmax(Q_row)
    return actions[action]

    # Implement a policy here to choose an action balanced between random and
    # best action.


def determine_reward(state):
    rewards = np.array([-5, -2, 1, -2. - 5])
    reward = rewards[state]
    return reward

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            print(*line)
            cv2.line(line_image, (line[0][0], line[0][1] ),( line[0][2] ,line[0][3]),  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)
             
    


def main():

    # hardcode number of actions
    # Adjust to represent the states
    STATE_BOUNDS = np.array([-1, -.5, -.1, .1, .5, 1])
    goal_state = 2
    NUM_states = STATE_BOUNDS.size - 1
    NUM_actions = 3  # left right or none, need to decide increment though
    episode_number = 200
    t_max = 7
    y = .95
    alpha = .25
    action_step = .2
    control_step = .1
    dt = .02
    min_threshold = 25
    max_threshold = 2 * min_threshold
    kernel_size = 5
    lower = np.uint8([ 90, 120,120])
    upper = np.uint8([ 126,209, 215]) # 180,228, 228


    s = SimClient()
    s.start()
    # Intialize Q table based on number of rows and actions
    Q = np.zeros([NUM_states, NUM_actions])
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Yellow', cv2.WINDOW_NORMAL)
    cv2.namedWindow('blur', cv2.WINDOW_NORMAL)




    for episode in range(episode_number):
        # Intialize the sim also put a reset in here somehwere
        total_reward = 0
        action = 0
        state = 0
        next_state = 0
        current_steering = -.03
        constant_throttle = 1
        control_time = 0
        t = 0
        # Reset Episode

        flag = 0
        # time.sleep(.5)

        while(t < t_max):
            # determine current state using function and input from sim.
            # Only take an action every control_step time
            observation = s.get_telemetry()
            #image = Image.open(observation.front_camera_image)
            # image.show()
            # observation.front_camera_image.show()
            open_cv_image = np.array(observation.front_camera_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            #open_cv_image = np.array(observation.overhead_camera_image)
            
            
            blurred = cv2.GaussianBlur(open_cv_image, (kernel_size, kernel_size), 0)
            yellow_only = cv2.inRange(blurred, lower, upper)
            cv2.imshow('blur', blurred)
            cv2.imshow('Yellow', yellow_only)
            #canny_edges = cv2.Canny(blurred, min_threshold, max_threshold)
            ret, thresh = cv2.threshold(yellow_only, 120, 255, cv2.THRESH_BINARY)
            canny_thresh_edges = cv2.Canny(thresh, min_threshold, max_threshold)
            #update below code
            lines = cv2.HoughLinesP(canny_thresh_edges,rho = 1,theta = 1*np.pi/180,threshold = 30,minLineLength = 25,maxLineGap = 40)

            output=draw_lane_lines(canny_thresh_edges, lines)
            #print(left_lane,right_lane)
            cv2.imshow('Display', output)
            #cv2.imshow('Display', canny_thresh_edges)
            cv2.waitKey(1)

            # plt.imshow(canny_edges)
            # plt.show()

            if(observation.colliding == True):
                break
            if (control_time > control_step):
                action = choose_action(Q[state, :])
                # sign times action step
                action_input = action * action_step
                # increment sim with action
                new_steering = current_steering + action_input
                s.send_instructions(new_steering, constant_throttle)
                # Update Q Table
                #Q[state,action] += alpha*(reward + y*np.max(Q[(next_state),:]) - Q[state,action])
                control_time = 0
                current_steering = new_steering
            else:
                # Increment the sim without action
                s.send_instructions(current_steering, constant_throttle)

            print(observation)
            print(episode)
            print(observation.colliding)
            print(t)
            # Determine new state
            # next_state=assign_state(distance_from_center_line)
            # Determine reward of new state
            # reward=determine_reward(state
            # total_reward+= reward #add on the reward for the episode
            state = next_state
            control_time += dt
            t += dt

        s.reset_instruction()

    s.stop()

if __name__ == '__main__':
    main()
