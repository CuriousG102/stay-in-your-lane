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

DEFAULT_RESOLUTION = (1920,1088,)
DEFAULT_SCALE_FACTOR = 2
mapx, mapy = get_cv2_maps(*(DEFAULT_RESOLUTION+DEFAULT_RESOLUTION), scale_factor=DEFAULT_SCALE_FACTOR)
def undistort_image(cam_img):
    return cv2.remap(cam_img, 
                     mapx, mapy, 
                     cv2.INTER_LINEAR)

DEFAULT_FRAMERATE = 60
class CorrectedOnDemandStream:
    def __init__(
        self, 
        resolution = DEFAULT_RESOLUTION, 
        scale_factor = DEFAULT_SCALE_FACTOR,
        framerate = DEFAULT_FRAMERATE):
        self.resolution = resolution
        self.undistort_resolution = resolution
        self.framerate = framerate
        self.scale_factor = scale_factor
        self.camera = None
        self.frame_iter = None
        self.raw_capture = None
    
    def start(self):
        if self.camera is not None:
            raise Exception('Already started')
        self.camera = PiCamera()
        self.camera.resolution = (self.resolution[0]//self.scale_factor, self.resolution[1]//self.scale_factor)
        self.camera.framerate = self.framerate
        # Camera is a rolling shutter. Setting exposure mode to sports
        # makes it prefer increases to gain over increases to exposure
        # time, reducing motion blur and artifacts related to 
        # rolling shutter.
        self.camera.exposure_mode = 'sports'
        self.raw_capture = io.BytesIO()
        self.frame_iter = self.camera.capture_continuous(
            self.raw_capture, format='bgra', use_video_port=True)
    
    def stop(self):
        self.camera.close()
        self.camera = None
        self.frame_iter = None
        self.raw_capture = None 
    
    def get_latest_cam_image(self):
        next(self.frame_iter)
        img = get_np_buffer_wrapper(
            self.raw_capture.getbuffer(), 
            (self.resolution[0]//self.scale_factor, self.resolution[1]//self.scale_factor), 
            4).copy()[:,:,:3]
        self.raw_capture.seek(0)
        return img
    
    def get_latest_undist_image(self):
        next(self.frame_iter)
        img = get_np_buffer_wrapper(
            self.raw_capture.getbuffer(), 
            (self.resolution[0]//self.scale_factor, self.resolution[1]//self.scale_factor), 
            4)[:,:,:3]
        self.raw_capture.seek(0)
        return undistort_image(img)
