import math
import numpy as np
import track_floor_utils

CAR_AXLE_FRONT = track_floor_utils.CAR_SCALE * 1.27
CAR_AXLE_BACK = track_floor_utils.CAR_SCALE * - 1.6

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
    t = telemetry
    x, z = pos
    rot_y = math.radians(rot_y)
    s_angle = math.radians(t.steering)
    speed = t.speed
    d = -CAR_AXLE_BACK + CAR_AXLE_FRONT
    initial_front = np.array([
        x + math.sin(rot_y) * CAR_AXLE_FRONT, 
        z + math.cos(rot_y) * CAR_AXLE_FRONT])
    initial_back = np.array([
        x + math.sin(rot_y) * CAR_AXLE_BACK, 
        z + math.cos(rot_y) * CAR_AXLE_BACK])
    v_b_vector = (np.array([math.sin(rot_y), 
                            math.cos(rot_y)]) 
                  * speed)
    v_b = np.sqrt((v_b_vector**2).sum())
    final_back = initial_back + v_b_vector * delta_time


    final_front_y = (
        np.cos(s_angle)**2 
        * (-d+delta_time*v_b
           + np.sqrt(d**2 + 
                     delta_time * v_b 
                     * (2 * d - delta_time * v_b) 
                     * math.tan(s_angle)**2)))
    final_front_x = final_front_y*math.tan(s_angle)

    final_front_y_rotated = (
        final_front_y * math.cos(rot_y) 
        - final_front_x * math.sin(rot_y))
    final_front_x_rotated = (
        final_front_x * math.cos(rot_y) + final_front_y * math.sin(rot_y))

    final_front = np.array([final_front_x_rotated, final_front_y_rotated])

    final_front = final_front + initial_front

    final_direction = final_front - final_back
    change_in_angle = math.atan2(final_direction[0], final_direction[1]) - rot_y

    fin_rot_y = rot_y + change_in_angle

    direction = (
        (final_front - final_back) 
        / np.sqrt(((final_front-final_back)**2).sum()))

    from_back = abs(CAR_AXLE_BACK) * direction

    car_pos = final_back + from_back

    return (tuple(car_pos),
            np.rad2deg(fin_rot_y))


def telemetry_after_delta_time2(telemetry, pos, rot_y, delta_time):
    t = telemetry
    x, z = pos
    rot_y = math.radians(rot_y)
    s_angle = math.radians(t.steering)
    speed = t.speed
    d = -CAR_AXLE_BACK + CAR_AXLE_FRONT
    if s_angle>.01: #You're going in a straight line otherwise
        R = d*math.tan(math.pi/2-s_angle)
    else:
        car_pos = (z+speed*math.sin(rot_y),x+speed*math.cos(rot_y))
        return(tuple(car_pos),np.rad2deg(rot_y))
    phi = speed*delta_time/R
    delta_x = R-R*math.cos(phi)
    delta_z = R*math.sin(phi)
    if s_angle>0:
        delta_x = -delta_x

    delta_x = delta_x*math.cos(rot_y) + delta_z*math.sin(rot_y)
    delta_z = -delta_z*math.sin(rot_y)+delta_z*math.cos(rot_y)
    x = x+delta_x
    z = z+delta_z
    car_pos = (x,z)
    new_angle = np.rad2deg(rot_y+phi)
    return(tuple(car_pos),np.rad2deg(rot_y))



def break_down_into_times(telemetry, pos, rot_y, delta_time,desired_break_down):
    '''
    Just in case the resolution isn't good enough you good do an interpolation.
    :param telemetry:
    :param pos:
    :param rot_y:
    :param delta_time:
    :param desired_break_down:
    :return:
    '''
    time = delta_time/desired_break_down
    for _ in range(desired_break_down):

        pos,rot_y = telemetry_after_delta_time(telemetry, pos, rot_y, time)
        #Produce new telemetry
    return pos,rot_y
