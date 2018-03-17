import cv2
import io
import multiprocessing as mp
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

from .camera_calibration import get_cv2_maps

def get_np_buffer_wrapper(raw_arr, resolution, num_channels):
    arr = np.frombuffer(raw_arr, dtype=np.uint8)
    arr.shape = resolution[::-1] + (num_channels,)
    return arr

def capture_and_undistort_stream_continuous(
    undist_img_arr_wrapper, undist_update_event, resolution,undistort_resolution, scale_factor, framerate):
    
    mapx, mapy = get_cv2_maps(*(resolution+undistort_resolution), scale_factor=scale_factor)
    undist_img_raw_arr = undist_img_arr_wrapper.get_obj()
    undist_img_arr = get_np_buffer_wrapper(
        undist_img_raw_arr, 
        (undistort_resolution[0]//scale_factor, undistort_resolution[1]//scale_factor,), 3)
    with PiCamera() as camera:
        camera.resolution = (resolution[0]//scale_factor, resolution[1]//scale_factor)
        camera.framerate = framerate
        # Camera is a rolling shutter. Setting exposure mode to sports
        # makes it prefer increases to gain over increases to exposure
        # time, reducing motion blur and artifacts related to 
        # rolling shutter.
        camera.exposure_mode = 'sports'
        raw_capture = io.BytesIO()
        frame_iter = camera.capture_continuous(
            raw_capture, format='bgra', use_video_port=True)
        print(camera.framerate)
        for frame in frame_iter:
            with raw_capture.getbuffer() as raw_frame:
                cam_img_arr = get_np_buffer_wrapper(
                    raw_frame,
                    (resolution[0]//scale_factor, resolution[1]//scale_factor), 4)
                undist_img_cpy_arr = cv2.remap(cam_img_arr[:, :, :3], mapx, mapy, 
                         cv2.INTER_LINEAR)
                with undist_img_arr_wrapper.get_lock():
                    np.copyto(undist_img_arr, undist_img_cpy_arr)
                    undist_update_event.set()
            raw_capture.seek(0)

class CorrectedVideoStream:
    def __init__(
        self, 
        resolution=(1920, 1088,), 
        undistort_resolution=(1920, 1088), 
        scale_factor=2,
        framerate=60):
        self.new_frame_update_event = mp.Event()
        self.undist_img_arr_wrapper = mp.Array(
            'B', 
            (undistort_resolution[0] // scale_factor) * 
            (undistort_resolution[1] // scale_factor) * 3)
        self.resolution = resolution
        self.undistort_resolution = undistort_resolution
        self.framerate = framerate
        self.scale_factor = scale_factor

    def _init_processes(self):
        self.undistort_stream_process = mp.Process(
                target=capture_and_undistort_stream_continuous,
                name='Camera Stream Capture and Undistort',
                daemon=True,
                args=(self.undist_img_arr_wrapper, 
                      self.new_frame_update_event, 
                      self.resolution,
                      self.undistort_resolution,
                      self.scale_factor, self.framerate))
    def start(self):
        self._init_processes()
        self.undistort_stream_process.start()

    def stop(self):
        self.undistort_stream_process.terminate()

    def get_latest_undist_image(self):
        undist_img_cpy_arr = np.ndarray(
            (self.undistort_resolution[1] // self.scale_factor, 
             self.undistort_resolution[0] // self.scale_factor, 
             3,), dtype=np.uint8)
        undist_img_raw_arr = self.undist_img_arr_wrapper.get_obj()
        undist_img_arr = get_np_buffer_wrapper(
            undist_img_raw_arr, 
            (self.undistort_resolution[0] // self.scale_factor, 
             self.undistort_resolution[1] // self.scale_factor,), 3)
        with self.undist_img_arr_wrapper.get_lock():
            np.copyto(undist_img_cpy_arr, undist_img_arr)
        
        return undist_img_cpy_arr
