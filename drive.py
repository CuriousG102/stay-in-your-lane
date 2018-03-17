from matplotlib.pyplot import imshow
import octo_donk.actuators
import octo_donk.camera
import octo_donk.camera_calibration
import path_planning_perspective_actual
import track_floor_utils_perspective
import image_utils
import time

c = octo_donk.camera.CorrectedOnDemandStream()
steering_controller = octo_donk.actuators.PCA9685(1)
steering = octo_donk.actuators.PWMSteering(steering_controller)
throttle_controller = octo_donk.actuators.PCA9685(0)
throttle = octo_donk.actuators.PWMThrottle(throttle_controller)
c.start()

throttle.run(-0.23)
previous_s_angle = 0
counter = 0
start = time.time()
PERIOD = 10
while time.time() - start < PERIOD:
    undist_image = c.get_latest_undist_image()
    s_angle, score = path_planning_perspective_actual.get_best_s_angle(undist_image, previous_s_angle)
    previous_s_angle = s_angle
    print(s_angle, score)
    steering.run(s_angle / 16 * -1)
    counter += 1
throttle.run(0)
steering.run(0)
counter/PERIOD

