import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

CAMERA_ANGLE = np.radians(15)
FIELD_OF_VIEW = np.radians(60)
HEIGHT = 2
def image_to_bird(img):
    IMAGEX,IMAGEY,_ = img.shape
    CENTER = (int(IMAGEX/2),int(IMAGEY/2))
    HORIZON = CENTER[1]* CAMERA_ANGLE/(FIELD_OF_VIEW/2)
    L_mid = HEIGHT / np.tan(CAMERA_ANGLE)
    bottom_Angle = CAMERA_ANGLE + np.radians(FIELD_OF_VIEW / 2)
    L_bot = HEIGHT / np.tan(bottom_Angle)
    ALPHA = L_mid / L_bot
    BORDER_SIZE = int(np.ceil(IMAGEX * (ALPHA - 1) / 2))
    border = cv2.copyMakeBorder(img, top=0, bottom=0, left=BORDER_SIZE, right=BORDER_SIZE,
                                borderType=cv2.BORDER_CONSTANT)
    BORDERY, BORDERX, _ = border.shape
    x_1, y_1 = (BORDER_SIZE, CENTER[1])
    x_2, y_2 = (BORDER_SIZE + CENTER[0], HORIZON)
    x_3, y_3 = (BORDER_SIZE + IMAGEX, CENTER[1])
    y_0 = BORDERY
    y_4 = BORDERY
    def x_0(y_0):
        return (int((y_0 - y_1) * (x_2 - x_1) / (y_2 - y_1)) + x_1)


    def x_4(y_4):
        return (int((y_4 - y_1) * (x_2 - x_3) / (y_2 - y_3)) + x_3)
    horizontal_of_transform = HORIZON + 30

    tl, tr, br, bl = [[x_0(horizontal_of_transform), horizontal_of_transform],
                      [x_4(horizontal_of_transform), horizontal_of_transform], [x_4(BORDERY), BORDERY],
                      [x_0(BORDERY), BORDERY]]

    rect = np.array([tl, tr, br, bl], dtype="float32")
    maxWidth = max(abs(br[0] - bl[0]), abs(tr[0] - tl[0]))
    maxHeight = max(abs(br[0] - tr[0]), abs(bl[0] - tl[0]))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(border, M, (maxWidth, maxHeight))
    return warped

def make_video(telemetry):
    video = []
    for elem in telemetry:
        Image = cv2.imread(telemetry.front_camera_image)
        video.append(Image)
    return video


if __name__  == '__main__':
    Image = cv2.imread('example.png')
    print(1)
    image_to_bird(Image)