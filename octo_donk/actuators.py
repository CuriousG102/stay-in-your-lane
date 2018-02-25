"""
actuators.py
Classes to control the motors and servos. These classes 
are wrapped in a mixer class before being used in the drive loop.
"""

import time

import Adafruit_PCA9685
from scipy.interpolate import interp1d
        
class PCA9685:
    ''' 
    PWM motor controler using PCA9685 boards. 
    This is used for most RC Cars
    '''
    def __init__(self, channel, frequency=60):
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(frequency)
        self.channel = channel

    def set_pulse(self, pulse):
        self.pwm.set_pwm(self.channel, 0, pulse) 

    def run(self, pulse):
        self.set_pulse(pulse)
        
class PWMSteering:
    """
    Wrapper over a PWM motor cotnroller to convert angles to PWM pulses.
    """
    LEFT_ANGLE = -1 
    RIGHT_ANGLE = 1

    def __init__(self, controller=None,
                       left_pulse=290,
                       right_pulse=490):

        self.controller = controller
        self.angle_mapping = interp1d([self.LEFT_ANGLE, self.RIGHT_ANGLE],
                                      [left_pulse, right_pulse])

    def run(self, angle):
        #map absolute angle to angle that vehicle can implement.
        self.controller.set_pulse(int(self.angle_mapping(angle)))

    def shutdown(self):
        self.run(0) #set steering straight



class PWMThrottle:
    """
    Wrapper over a PWM motor cotnroller to convert -1 to 1 throttle
    values to PWM pulses.
    """
    MIN_THROTTLE = -1
    MAX_THROTTLE =  1

    def __init__(self, controller=None,
                       max_pulse=300,
                       min_pulse=490,
                       zero_pulse=350):

        self.controller = controller
        self.forward_mapping = interp1d([0, self.MAX_THROTTLE],
                                        [zero_pulse, max_pulse])
        self.reverse_mapping = interp1d([self.MIN_THROTTLE, 0],
                                        [min_pulse, zero_pulse])
        
        #send zero pulse to calibrate ESC
        self.controller.set_pulse(zero_pulse)
        time.sleep(1)


    def run(self, throttle):
        if throttle > 0:
            self.controller.set_pulse(int(self.forward_mapping(throttle)))
        else:
            self.controller.set_pulse(int(self.reverse_mapping(throttle)))
        
    def shutdown(self):
        self.run(0) #stop vehicle

