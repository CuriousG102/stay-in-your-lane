import enum
import math

import numpy as np

CAR_AXLE_FRONT = .0904875
CAR_AXLE_BACK = - .0968375
CAR_LENGTH = abs(CAR_AXLE_BACK) + abs(CAR_AXLE_FRONT)

class Circle:
    class CircleSide(enum.Enum):
        LEFT = 1
        RIGHT = 2
    def __init__(self, radius, side):
        self.radius = radius
        self.side = side

class Line:
    pass

def telemetry_for_steering_pure(s_angle):
    '''
    Returns a relative representation of the car's path if it continues at 
    a given
    speed and s_angle indefinitely. This representation can take the form of
    objects of two different types: Circle and Line. Line simply represents
    that the car will remain travelling straight. Circle shows that, given 
    enough time in the current configuration, the car will travel in a circle
    of radius radius. Side on the circle represents the starting point of the 
    car. LEFT means that that the first derivative of theta is negative and the
    car is considered to start at theta=pi. 
    RIGHT means that the first derivative
    of theta is positive and the car is considered to start at theta=0. 
    '''
    s_angle_original = s_angle
    s_angle = math.radians(abs(s_angle))

    if abs(s_angle_original) <= 0.1:
        return Line()

    radius = CAR_LENGTH / math.tan(s_angle)
    side = (Circle.CircleSide.LEFT 
            if s_angle_original > 0 
            else Circle.CircleSide.RIGHT)

    return Circle(radius, side)

def telemetry_after_delta_time_pure(speed, s_angle, pos, rot_y, delta_time):
    ''' 
    Returns a single point prediction of where a car will end up in a global coordinate space
    given speed, steering angle, position, and rotation of the car in that space, after a time 
    delta_time. The prediction is returned as a tuple containing a tuple and a float:
    ((predicted_x, predicted_z,), predicted_rot_y,)
    '''
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
