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

def undistort_stream_continuous(
        cam_img_arr_wrapper, cam_update_event, undist_img_arr_wrapper,
        undist_update_event, resolution):
    mapx, mapy = get_cv2_maps(*resolution)
    cam_img_raw_arr = cam_img_arr_wrapper.get_obj()
    cam_img_arr = get_np_buffer_wrapper(cam_img_raw_arr, resolution, 4)
    cam_img_cpy_arr = np.ndarray(resolution[::-1] + (4,), dtype=np.uint8)
    undist_img_raw_arr = undist_img_arr_wrapper.get_obj()
    undist_img_arr = get_np_buffer_wrapper(undist_img_raw_arr, resolution, 3)
    while True:
        cam_update_event.wait()
        with cam_img_arr_wrapper.get_lock():
            np.copyto(cam_img_cpy_arr, cam_img_arr)
            cam_update_event.clear()
        undist_img_cpy_arr = cv2.remap(cam_img_cpy_arr[:, :, :3], mapx, mapy, 
                 cv2.INTER_LINEAR)
        with undist_img_arr_wrapper.get_lock():
            np.copyto(undist_img_arr, undist_img_cpy_arr)
            undist_update_event.set()

def capture_stream_continuous(img_arr_wrapper, updated,
                              resolution, framerate):
    img_raw_arr = img_arr_wrapper.get_obj()
    img_arr = get_np_buffer_wrapper(img_raw_arr, resolution, 4)
    with PiCamera() as camera:
        camera.resolution = resolution
        camera.framerate = framerate
        # Camera is a rolling shutter. Setting exposure mode to sports
        # makes it prefer increases to gain over increases to exposure
        # time, reducing motion blur and artifacts related to 
        # rolling shutter.
        camera.exposure_mode = 'sports'
        raw_capture = io.BytesIO()
        frame_iter = camera.capture_continuous(
            raw_capture, format='bgra', use_video_port=True)
        for frame in frame_iter:
            with img_arr_wrapper.get_lock():
                with raw_capture.getbuffer() as raw_frame:
                    frame = get_np_buffer_wrapper(raw_frame, resolution, 4)
                    np.copyto(img_arr, frame)
                    updated.set()
            raw_capture.seek(0)

class CorrectedVideoStream:
    def __init__(self, resolution=(1920, 1200,), framerate=32):
        self._cam_update_event = mp.Event()
        self.new_frame_update_event = mp.Event()
        self._cam_img_arr_wrapper = mp.Array('B', 
                                             resolution[0] * resolution[1] * 4)
        self.undist_img_arr_wrapper = mp.Array(
                'B', resolution[0] * resolution[1] * 3)
        self.resolution = resolution
        self.framerate = framerate

    def _init_processes(self):
        self.capture_stream_process = mp.Process(
                target=capture_stream_continuous, 
                name='Camera Stream Capture', 
                daemon=True,
                args=(self._cam_img_arr_wrapper, self._cam_update_event,
                      self.resolution, self.framerate,))
        self.undistort_stream_process = mp.Process(
                target=undistort_stream_continuous,
                name='Camera Stream Undistort',
                daemon=True,
                args=(self._cam_img_arr_wrapper, self._cam_update_event,
                      self.undist_img_arr_wrapper, 
                      self.new_frame_update_event, self.resolution,))
    def start(self):
        self._init_processes()
        self.capture_stream_process.start()
        self.undistort_stream_process.start()

    def stop(self):
        self.capture_stream_process.terminate()
        self.undistort_stream_process.terminate()

    def get_latest_undist_image(self):
        undist_img_cpy_arr = np.ndarray(self.resolution[::-1] + (3,), dtype=np.uint8)
        undist_img_raw_arr = self.undist_img_arr_wrapper.get_obj()
        undist_img_arr = get_np_buffer_wrapper(undist_img_raw_arr, self.resolution, 3)
        with self.undist_img_arr_wrapper.get_lock():
            np.copyto(undist_img_cpy_arr, undist_img_arr)
        
        return undist_img_cpy_arr
    
    def get_latest_cam_image(self):
        cam_img_cpy_arr = np.ndarray(self.resolution[::-1] + (3,), dtype=np.uint8)
        cam_img_raw_arr = self._cam_img_arr_wrapper.get_obj()
        cam_img_arr = get_np_buffer_wrapper(cam_img_raw_arr, self.resolution, 4)
        with self._cam_img_arr_wrapper.get_lock():
            np.copyto(cam_img_cpy_arr, cam_img_arr[:,:,:3])
        
        return cam_img_cpy_arr
