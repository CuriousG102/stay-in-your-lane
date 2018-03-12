import cv2
import numpy as np
import os
from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera.renderers import PiOverlayRenderer
import time

DEFAULT_CALIBRATION_DIRECTORY = 'calibration_files_fisheye3'

def get_calibration_file_path(base_name, calibration_directory, width, height):
    if calibration_directory is None:
        calibration_directory = DEFAULT_CALIBRATION_DIRECTORY
    return os.path.join(
            calibration_directory, 
            base_name + '_{0}_{1}.npy'.format(width, height))

def get_obj_points_file_path(width, height, calibration_directory=None):
    return get_calibration_file_path('obj_points', calibration_directory,
                                     width, height)

def get_img_points_file_path(width, height, calibration_directory=None):
    return get_calibration_file_path('img_points', calibration_directory,
                                     width, height)

def get_cv2_maps_file_paths(width, height, calibration_directory=None):
    rest_args = calibration_directory, width, height
    return (get_calibration_file_path('map_x', *rest_args), 
            get_calibration_file_path('map_y', *rest_args))

#def get_cv2_maps(width, height, calibration_directory=None):
#    CALIBRATION_FLAGS = (
#        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
#        cv2.fisheye.CALIB_CHECK_COND +
#        cv2.fisheye.CALIB_FIX_SKEW)
#
#    obj_points = np.load(get_obj_points_file_path(width, height))
#    img_points = np.load(get_img_points_file_path(width, height))
#    map_x_path, map_y_path = get_cv2_maps_file_paths(
#            width, height, calibration_directory)
#    if os.path.exists(map_x_path) and os.path.exists(map_y_path):
#        return np.load(map_x_path), np.load(map_y_path)
#
#    resolution = width, height
#    N_OK = len(obj_points)
#    K = np.zeros((3,3))
#    D = np.zeros((4,1))
#    rvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
#    tvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
#    rms, _, _, _, _ = cv2.fisheye.calibrate(
#        obj_points,
#        img_points,
#        resolution,
#        K,
#        D,
#        rvecs,
#        tvecs,
#        CALIBRATION_FLAGS,
#        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
#    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
#        K, D, np.eye(3), K, resolution, cv2.CV_16SC2)
#    
#    np.save(map_x_path, map_x)
#    np.save(map_y_path, map_y)
#
#    return map_x, map_y

def get_cv2_maps(width, height, output_width, output_height, calibration_directory=None):
    CALIBRATION_FLAGS = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW)

    obj_points = np.load(get_obj_points_file_path(width, height))
    img_points = np.load(get_img_points_file_path(width, height))
    map_x_path, map_y_path = get_cv2_maps_file_paths(
            width, height, calibration_directory)
#    if os.path.exists(map_x_path) and os.path.exists(map_y_path):
#        return np.load(map_x_path), np.load(map_y_path)
    
    resolution = width, height
    undistort_resolution = output_width, output_height
    N_OK = len(obj_points)
    K = np.zeros((3,3))
    D = np.zeros((4,1))
    rvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        obj_points,
        img_points,
        resolution,
        K,
        D,
        rvecs,
        tvecs,
        CALIBRATION_FLAGS,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    print(K)
    new_K = K.copy()
    new_K[0,0] = new_K[0,0]/2
    new_K[1,1] = new_K[1,1]/2
    new_K[0,2] = width/2
    new_K[1,2] = height/2
    print(new_K)
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, undistort_resolution, cv2.CV_16SC2)
    
#    np.save(map_x_path, map_x)
#    np.save(map_y_path, map_y)

    return map_x, map_y
    

def capture_for_calibration(width, height, calibration_directory=None):
    camera = PiCamera()
    resolution = width, height
    camera.resolution = resolution 
    raw_capture = PiRGBArray(camera)
    time.sleep(0.1)
    camera.start_preview(alpha=128)
    try:
        obj_points = []
        img_points = []
        # width, height
        BOARD_SIZE = (6, 9)
        objp = np.zeros((1, BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:BOARD_SIZE[0],0:BOARD_SIZE[1]].T.reshape(-1,2)
        TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        while 'exit' not in input():
            with PiRGBArray(camera) as stream:
                camera.capture(stream, format='bgr')
                img = stream.array
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print('Image captured, searching for corners...')
                ret, corners = cv2.findChessboardCorners(
                        gray, BOARD_SIZE, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK 
                        + cv2.CALIB_CB_NORMALIZE_IMAGE)
                
                if ret:
                    print('Good shot!: %i samples acquired' % (len(obj_points) + 1))
                    obj_points.append(objp)
                    cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), TERMINATION_CRITERIA)
                    img_points.append(corners)
                else:
                    print('Corners not acquired. Try again')
    finally:
        camera.stop_preview()
        camera.close()
    np.save(get_obj_points_file_path(*resolution, calibration_directory), 
            obj_points)
    np.save(get_img_points_file_path(*resolution, calibration_directory), 
            img_points)
    return obj_points, img_points, gray.shape[::-1]
