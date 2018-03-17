import numpy as np
import cv2 

def pil_image_to_open_cv_format(pil_image):
    return np.array(pil_image.convert('RGB'))[:, :, ::-1]

def get_cv2_from_tel_field(tel, field_name):
    return pil_image_to_open_cv_format(getattr(tel, field_name))

def simple_threshold(img):
    b,g,r = (img[:, :, i] for i in range(3))
    bt,gt,rt = (cv2.threshold(color_channel,
                              150, 255, cv2.THRESH_BINARY)[1]
                for color_channel in (b, g, r))
    return bt | gt | rt

class ThresholdImages:
    def __init__(self, outside, inside):
        self.outside = outside
        self.inside = inside

def simple_threshold_sides(img):
    b,g,r = (img[:, :, i] for i in range(3))
    bt,gt,rt = (cv2.threshold(color_channel,
                              120, 255, cv2.THRESH_BINARY)[1]
                for color_channel in (b, g, r))
    outside = bt | gt | rt
    return ThresholdImages(outside, (bt | gt | rt) & ~outside)

def simple_threshold_ternary(img):
    ternary = np.zeros(img.shape, dtype=np.uint8)
    thresh_sides = simple_threshold_sides(img)
    ternary[:, :, 2] = thresh_sides.outside
    ternary[:, :, 1] = thresh_sides.inside
    return ternary

def crop_img_from_below(img, num_rows):
    return img[num_rows:, :, :]

# sample use of above:
# while True:
#     img = get_cv2_from_tel_field(s.get_telemetry(),
#                                  'cheater_camera_image')
#     cv2.imshow('a', image_utils.simple_threshold(img))
#     cv2.waitKey(1)

BOTTOM_SATURATION_YELLOW = np.array([20, 20, 20])
TOP_SATURATION_YELLOW = np.array([40, 255, 255])
def threshold_for_yellow(hue):
    return cv2.inRange(
        hue, BOTTOM_SATURATION_YELLOW, TOP_SATURATION_YELLOW)
    #hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0] 
    #return hue
    #b,g,r = (img[:, :, i] for i in range(3))
    #hue = np.arctan2(SQRT_3 * (g - b), 2 * r - g - b)
    #return (hue <= DEGREES_UPPER) & (hue >= DEGREES_LOWER)
    
BOTTOM_SATURATION_WHITE = np.array([0, 248, 0])
TOP_SATURATION_WHITE = np.array([179, 255, 255])
def threshold_for_white(hue):
    return cv2.inRange(
        hue, BOTTOM_SATURATION_WHITE, TOP_SATURATION_WHITE)
