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

def convert_front_to_bird(telemetry):
    elem = telemetry.front_camera_image
    lower_white = np.array([140,140,140])
    upper_white = np.array([255,255,255])
    lower_yellow = np.array([0,70,100])
    upper_yellow = np.array([80,255,190])
    img= cv2.cvtColor(np.array(elem.convert('RGB')), cv2.COLOR_RGB2BGR)
    yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(img, lower_white, upper_white)
    res = cv2.bitwise_and(img, img, mask=white_mask+yellow_mask)
    return res


def convert_front_to_bird_video(telemetries):
    A = []
    lower_white = np.array([140,140,140])
    upper_white = np.array([255,255,255])
    lower_yellow = np.array([0,70,100])
    upper_yellow = np.array([80,255,190])
    for elem in telemetries:
        img= cv2.cvtColor(np.array(elem.front_camera_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)  # I have the Green threshold image.

        # Threshold the HSV image to get only blue colors
        white_mask = cv2.inRange(img, lower_white, upper_white)
        res = cv2.bitwise_and(img, img, mask=white_mask+yellow_mask)
        A.append(image_to_bird(res))

    return A

def convert_front_to_bird_video_with_lines(telemetries):
    A = []
    lower_white = np.array([140,140,140])
    upper_white = np.array([255,255,255])
    lower_yellow = np.array([0,70,100])
    upper_yellow = np.array([80,255,190])
    for elem in telemetries:
        img= cv2.cvtColor(np.array(elem.front_camera_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)  # I have the Green threshold image.

        # Threshold the HSV image to get only blue colors
        white_mask = cv2.inRange(img, lower_white, upper_white)
        res = cv2.bitwise_and(img, img, mask=white_mask+yellow_mask)
        img = image_to_bird(res)
        _, thresh = cv2.threshold(yellow_mask, 250, 255, cv2.THRESH_BINARY)
        V = []

        for i, row in enumerate(thresh):
            for j, column in enumerate(row):
                if column == 255:
                    V.append([i, j])
                    break
        for j in range(len(thresh), 0, -1):
            for j, column in enumerate(thresh[j]):
                if column == 255:
                    V.append([i, j])
                    break
        maxHeight,maxWidth = img.shape
        cv2.line(img, (V[0][1], 0), (V[-1][1], maxHeight), (0, 0, 255), 10)
        cv2.line(img, (V[0][1] + 50, 0), (V[-1][1] - 50, maxHeight), (255, 0, 0), 10)
        cv2.line(img, (V[0][1] - 50, 0), (V[-1][1] + 50, maxHeight), (255, 0, 0), 10)
        A.append(img)
    return A



# def convert_front_to_thresh_bird(telemetries):
#     A = []
#     for elem in telemetries:
#         img = image_to_bird(np.array(elem.front_camera_image.convert('RGB')))
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         ret, thresh = cv2.threshold(gray_image, 95, 255, cv2.THRESH_BINARY)
#         A.append(thresh)
#
#     return A


if __name__  == '__main__':
    Image = cv2.imread('example.png')
    print(1)
    image_to_bird(Image)