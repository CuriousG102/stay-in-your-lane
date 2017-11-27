import cv2
import numpy as np

import prediction
import track_floor_utils

def create_viz(actual_telemetry, delta_t, dot_size=5):
    '''
    Produces opencv image to visualize the actual course
    of the car in a drive vs. that predicted by our approximation
    function. Prediction is generated as follows:

    1.  We get actual starting position and rotation
        from first telemetry.
    2.  After time delta_t or more has passed, we feed 
        our speed and steering angle at the first telemetry 
        as well as the time passed to 
        prediction.get_telemetry, along with our 
        position and rotation. We replace our stored
        position and rotation with the returned
        predicted value.
    3. We repeat step 2 until we run out of telemetries.

    actual_telemetry: List of all telemetries from our run.
    delta_t: Time to wait till next prediction.
    dot_size: Radius of dots indicating our predictions. If
              you have set a small delta_t predictions will
              essentially be a line so this should be small.
              As delta_t gets larger you may want to make
              dots larger so they are individually 
              distinguishable.

    Returns: 
    Three channel opencv image of track with red line for
    actual telemetry and green for predicted.

    '''
    # Get the track we're drawing on
    track = track_floor_utils.TRACK_FLOOR
    track = np.repeat(track, 3).reshape(track.shape+(3,))

    # get our starting point
    first_telemetry = actual_telemetry[0]
    estimated_pos = first_telemetry.x, first_telemetry.z
    estimated_rot_y = first_telemetry.rot_y
    start_tel_i = 0

    passed_time = 0
    for i, tel in enumerate(actual_telemetry[1:]):
        i += 1
        print('Number ', i)
        tel_img_pos = track_floor_utils.img_point_bottom_left_to_top_left(
            track_floor_utils.unity_plane_point_to_img_point(
                (tel.x, tel.z)))
        print('Tel ((x,z), rot_y)', ((tel.x, tel.z), tel.rot_y))
        track = cv2.circle(
            track, tuple(int(i) for i in tel_img_pos), 5, (0, 255, 0), 5)
        passed_time += tel.delta_time
        if passed_time > delta_t:
            estimated_pos, estimated_rot_y = (
                prediction.telemetry_after_delta_time(
                    actual_telemetry[start_tel_i], 
                    estimated_pos, estimated_rot_y, passed_time))
            print('Est ((x,z), rot_y)', (estimated_pos, estimated_rot_y))
            passed_time = 0
            start_tel_i = i
            tel_img_est_pos = track_floor_utils.img_point_bottom_left_to_top_left(
                track_floor_utils.unity_plane_point_to_img_point(
                    (estimated_pos[0], estimated_pos[1])))
            track = cv2.circle(
                track, tuple(int(i) for i in tel_img_est_pos), 
                dot_size, (0, 0, 255), 5)

    return track
