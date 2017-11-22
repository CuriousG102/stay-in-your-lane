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
    state=0;
    #if:
    return state


def choose_action(Q_row):
    action=1
    actions=np.array([-1, 0 , 1])
    if True:
        pass
    else:
        action = np.argmax(Q_row)
    return actions[action]

    #Implement a policy here to choose an action balanced between random and best action. 

def determine_reward(state):
    rewards=np.array([-5, -2, 1, -2. -5])
    reward=rewards[state]
    return reward

    #Send to the simulator


def main():

    #hardcode number of actions
    STATE_BOUNDS = np.array([-1, -.5, -.1, .1, .5, 1]) #Adjust to represent the states
    goal_state=2
    NUM_states = STATE_BOUNDS.size - 1
    NUM_actions=3 #left right or none, need to decide increment though
    episode_number=200;
    t_max=7
    y=.95
    alpha=.25
    action_step=.2
    control_step=.1
    dt=.02
    min_threshold=35
    max_threshold=2*min_threshold
    kernel_size=5

    s=SimClient()
    s.start()
    Q = np.zeros([NUM_states,NUM_actions]) #Intialize Q table based on number of rows and actions
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    for episode in range(episode_number): 
        #Intialize the sim also put a reset in here somehwere
        total_reward=0
        action=0
        state=0
        next_state=0
        current_steering= 0
        constant_throttle=1
        control_time=0
        t=0
        #Reset Episode

        
        
        flag=0
        #time.sleep(.5)

        while(t<t_max): 
            #determine current state using function and input from sim.
            #Only take an action every control_step time
            observation=s.get_telemetry()
            #image = Image.open(observation.front_camera_image)
            #image.show()
            #observation.front_camera_image.show()
            open_cv_image = np.array(observation.front_camera_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            #open_cv_image = np.array(observation.overhead_camera_image)
            
            blurred = cv2.GaussianBlur(open_cv_image, (kernel_size, kernel_size), 0)
            canny_edges = cv2.Canny(blurred, min_threshold, max_threshold)
            ret,thresh = cv2.threshold(open_cv_image,120,255,cv2.THRESH_BINARY)
            #canny_thresh_edges = cv2.Canny(thresh, min_threshold, max_threshold)
            #cv2.imshow('Display', canny_edges)
            cv2.imshow('Display', thresh)
            #cv2.imshow('Display', canny_thresh_edges)
            cv2.waitKey(1)

            #plt.imshow(canny_edges)
            #plt.show()

            if(observation.colliding== True):
                break
            if (control_time>control_step):
                action=choose_action(Q[state,:])
                #sign times action step
                action_input=action*action_step
                #increment sim with action 
                new_steering=current_steering+action_input
                s.send_instructions(new_steering,constant_throttle)
                #Update Q Table
                #Q[state,action] += alpha*(reward + y*np.max(Q[(next_state),:]) - Q[state,action])
                control_time=0
                current_steering=new_steering
            else:
                s.send_instructions(current_steering, constant_throttle) #Increment the sim without action
        
            print(observation)
            print(episode)
            print(observation.colliding)
            print(t)
            #Determine new state
            #next_state=assign_state(distance_from_center_line)
            #Determine reward of new state
            #reward=determine_reward(state
            #total_reward+= reward #add on the reward for the episode
            state=next_state
            control_time+=dt
            t+=dt

        s.reset_instruction()

    s.stop()

if __name__ == '__main__':
    main()