import numpy as np
import cv2 

def pil_image_to_open_cv_format(pil_image):
    return np.array(pil_image.convert('RGB'))[:, :, ::-1]

def get_cv2_from_tel_field(tel, field_name):
    return pil_image_to_open_cv_format(getattr(tel, field_name))

# sample use of above:
# while True:
#     cv2.imshow('a', get_cv2_from_tel_field(s.get_telemetry(), 'front_camera_image'))
#     cv2.waitKey(1)
