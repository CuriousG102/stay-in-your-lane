import numpy as np
import cv2
import time

def pil_image_to_open_cv_format(pil_image):
    return np.array(pil_image.convert('RGB'))[:, :, ::-1]

def get_cv2_from_tel_field(tel, field_name):
    return pil_image_to_open_cv_format(getattr(tel, field_name))

def simple_threshold(img):
    b,g,r = (img[:, :, i] for i in range(3))
    bt,gt,rt = (cv2.threshold(color_channel,
                              120, 255, cv2.THRESH_BINARY)[1]
                for color_channel in (b, g, r))
    return bt | gt | rt

def crop_img_from_below(img, num_rows):
    return img[num_rows:, :, :]

def make_movie(telemetries):
    for tel in telemetries[4:]:
        cv2.imshow('blah',np.array(tel.front_camera_image.convert('RGB')))
        if (cv2.waitKey(1)& 0xFF)==ord('q'):
            break
    cv2.destroyAllWindows()

def make_movie_general(images):
    for image in images:
        image = cv2.resize(image,(600,600))
        time.sleep(.03)
        cv2.imshow('blah', np.array(image))
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cv2.destroyAllWindows()


# sample use of above:
# while True:
#     img = get_cv2_from_tel_field(s.get_telemetry(),
#                                  'cheater_camera_image')
#     cv2.imshow('a', image_utils.simple_threshold(img))
#     cv2.waitKey(1)
