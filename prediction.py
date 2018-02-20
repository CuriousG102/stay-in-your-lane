import math
import numpy as np
import track_floor_utils

CAR_AXLE_FRONT = .127
CAR_AXLE_BACK = - .16

def telemetry_after_delta_time_pure(speed, s_angle, pos, rot_y, delta_time):
    x, z = pos
    rot_y = math.radians(rot_y)
    s_angle_original = s_angle
    s_angle = math.radians(s_angle) 

    d = -CAR_AXLE_BACK + CAR_AXLE_FRONT
    initial_back_x = x + math.sin(rot_y) * CAR_AXLE_BACK
    initial_back_z = z + math.cos(rot_y) * CAR_AXLE_BACK

    if abs(s_angle_original) > 0.1:
        final_back_x = (
            d * 1 / math.tan(s_angle) 
            * (math.cos(rot_y - speed * delta_time * math.tan(s_angle) / d) 
               - math.cos(rot_y)) * .1
            + x)
        final_back_z = (
            d * 1 / math.tan(s_angle) 
            * (math.sin(rot_y) 
               - math.sin(rot_y - speed * delta_time * math.tan(s_angle) / d)) * .1
            + z)
        final_rot_y = math.tan(s_angle) / d * speed * delta_time + rot_y

    else:
        delta_back_x = speed * delta_time * math.sin(rot_y)
        final_back_x = initial_back_x + delta_back_x
        delta_back_z = speed * delta_time * math.cos(rot_y)
        final_back_z = initial_back_z + delta_back_z
        delta_rot_y = 0
        final_rot_y = rot_y + delta_rot_y

    final_pos = (final_back_x + abs(CAR_AXLE_BACK) * math.sin(final_rot_y),
                 final_back_z + abs(CAR_AXLE_BACK) * math.cos(final_rot_y),)

    return (final_pos, np.rad2deg(final_rot_y))

def telemetry_after_delta_time(telemetry, pos, rot_y, delta_time):
    '''
    Returns ((x, z), rot_y) predicted on track floor after delta time.

    Takes:
    t: Telemetry
    x: Known or guessed position of car
    z: Known or guessed position of car
    rot_y: Known or guessed rotation of car on y axis
    delta_time: Amount of time that passes while producing our estimate.
    '''
    
    return telemetry_after_delta_time_pure(
        telemetry.speed, telemetry.steering, pos, rot_y, delta_time)
