import numpy as np


def determine_state(slope,buckets):
    """
    Takes in the buckets and the current average slop of all lines in an image, returns the state of the car

    """

    if slope>0:
        state=np.digitize(slope, list(reversed(buckets)))
    if slope<0:
        state=np.digitize(abs(slope), buckets)+len(buckets)

    return state


buckets=[50, 20, 10, 3, 1.5, 0]


print(determine_state(-30,buckets))

"""
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
"""




