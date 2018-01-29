import math

import cv2 
from PIL import Image, ImageDraw
import numpy as np

import image_utils

TRACK_FLOOR = (cv2.imread('TrackFloor.png')[:, :, 0]
    .astype(np.bool).astype(np.uint8) * 255)

TRACK_FLOOR_FILLED = (cv2.imread('TrackFloorFilled.png')[:, :, 0]
    .astype(np.bool))

TRACK_CENTER_POS = (3.45, -70.714)

TRACK_SCALE = (16.28928, 28.50579)

TRACK_BOTTOM_LEFT_POS = tuple(
    center_axis_point - scale_axis_point / 2
    for center_axis_point, scale_axis_point
    in zip(TRACK_CENTER_POS, TRACK_SCALE)
)

CAR_START_CENTER_POS = (0.711504, -73.78844)

CAR_ROTATION = (0, 10, 0)

CAR_SCALE = 0.1

CAR_CAM_FOV = 60  # degrees

CAR_CAM_POS_RELATIVE_Y = 1.5

CAR_CAM_POS_RELATIVE_Z = 1.5

CAR_CAM_HEIGHT = CAR_SCALE * CAR_CAM_POS_RELATIVE_Y

CAR_CAM_FORWARD = CAR_SCALE * CAR_CAM_POS_RELATIVE_Z

CAR_CAM_ROTATION_X = 15  # degrees off of level, towards ground

# want to capture .7 of image based on simulated drive
CAR_CAM_TOP_ANGLE = CAR_CAM_ROTATION_X - 12

L_MIN = (
    CAR_CAM_HEIGHT 
    / math.tan(math.radians(CAR_CAM_ROTATION_X) 
               + math.radians(CAR_CAM_FOV) / 2))
L_MAX = CAR_CAM_HEIGHT / math.tan(math.radians(CAR_CAM_TOP_ANGLE))
X_U = L_MAX * math.tan(math.radians(CAR_CAM_FOV) / 2)
Z_U = L_MAX
X_B = L_MIN * math.tan(math.radians(CAR_CAM_FOV) / 2)
Z_B = L_MIN

TRACK_SCALE_X, TRACK_SCALE_Z = TRACK_SCALE
IMG_SCALE_Z, IMG_SCALE_X = TRACK_FLOOR.shape
TRACK_TO_IMG_RATIO_X = IMG_SCALE_X / TRACK_SCALE_X
TRACK_TO_IMG_RATIO_Z = IMG_SCALE_Z / TRACK_SCALE_Z

IMG_SIZE_Z = (Z_U - Z_B) * TRACK_TO_IMG_RATIO_Z
IMG_SIZE_X = (2 * X_U) * TRACK_TO_IMG_RATIO_X

# CAR_CAM_RESOLUTION = 720

IMG_VISIBLE_MASK = None

def _init_non_visible_mask():
    global IMG_VISIBLE_MASK

    # Triangle details for mask
    x_diff = (X_U - X_B) * TRACK_TO_IMG_RATIO_X
    left_triangle_points = (
        (0, int(IMG_SIZE_Z) - 1),
        (x_diff, int(IMG_SIZE_Z) - 1),
        (0, 0))
    right_triangle_points = (
        (int(IMG_SIZE_X) - 1, int(IMG_SIZE_Z) - 1),
        (int(IMG_SIZE_X) - 1 - x_diff, int(IMG_SIZE_Z) - 1),
        (int(IMG_SIZE_X) - 1, 0))

    # Draw triangles and invert for mask
    mask = np.zeros((int(IMG_SIZE_Z), int(IMG_SIZE_X)), dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    draw.polygon(left_triangle_points, fill=255)
    draw.polygon(right_triangle_points, fill=255)
    mask = np.asarray(mask)
    IMG_VISIBLE_MASK = ~mask

_init_non_visible_mask()

def unity_plane_point_to_img_point(unity_point):
    '''
    Convert a unity plane point (x, z) on track to an image point with
    a bottom left origin.
    '''
    unity_point_from_bottom_left = np.array(unity_point) - np.array(TRACK_BOTTOM_LEFT_POS)
    track_scale_x, track_scale_z = TRACK_SCALE
    img_scale_z, img_scale_x = TRACK_FLOOR.shape
    scale = np.array([img_scale_x / track_scale_x, img_scale_z / track_scale_z])
    return tuple(unity_point_from_bottom_left * scale)

def img_point_bottom_left_to_top_left(img_point):
    img_point_x, img_point_z = img_point
    img_scale_z, img_scale_x = TRACK_FLOOR.shape
    return (img_point_x,
            img_scale_z - 1 - img_point_z)

def pos_in_track(position):
    '''
    Returns boolean indicating whether a unity coordinate is within
    or outside of the track
    '''
    top_left_point = tuple(
        int(i) 
        for i in img_point_bottom_left_to_top_left(
            unity_plane_point_to_img_point(position)))[::-1]
    track_shape = TRACK_FLOOR_FILLED.shape
    if (top_left_point[0] >= track_shape[0] 
        or top_left_point[1] >= track_shape[1]):
        return False
    return TRACK_FLOOR_FILLED[top_left_point]

# def get_car_view_box_center_pos(position, rotation):
    # x, z = position
    # rotation = math.radians(rotation)
    # return (x + CAM_DIST_FROM_CAR * math.sin(rotation),
    #         z + CAM_DIST_FROM_CAR * math.cos(rotation))

def get_car_view_points_unity(position, rotation):
    '''
    Returns top left origin img points of trapezoid 
    for unwarped car view.
    '''
    # Car pose
    x, z = position
    rotation = math.radians(rotation)

    # Camera pose
    car_cam_x = x + CAR_CAM_FORWARD * math.sin(rotation)
    car_cam_z = z + CAR_CAM_FORWARD * math.cos(rotation)

    # Camera view points math

    # Bottom of trapezoid
    bottom_left_x = (
        car_cam_x 
        - X_B * math.sin(rotation + math.pi / 2)
        + Z_B * math.sin(rotation))
    bottom_right_x = (
        car_cam_x 
        + X_B * math.sin(rotation + math.pi / 2)
        + Z_B * math.sin(rotation))
    bottom_left_z = (
        car_cam_z
        - X_B * math.cos(rotation + math.pi / 2)
        + Z_B * math.cos(rotation))
    bottom_right_z = (
        car_cam_z
        + X_B * math.cos(rotation + math.pi / 2)
        + Z_B * math.cos(rotation))
    bottom_left = bottom_left_x, bottom_left_z
    bottom_right = bottom_right_x, bottom_right_z

    # Top of trapezoid
    upper_left_x = (
        car_cam_x 
        - X_U * math.sin(rotation + math.pi / 2)
        + Z_U * math.sin(rotation))
    upper_right_x = (
        car_cam_x 
        + X_U * math.sin(rotation + math.pi / 2)
        + Z_U * math.sin(rotation))
    upper_left_z = (
        car_cam_z
        - X_U * math.cos(rotation + math.pi / 2)
        + Z_U * math.cos(rotation))
    upper_right_z = (
        car_cam_z
        + X_U * math.cos(rotation + math.pi / 2)
        + Z_U * math.cos(rotation))
    upper_left = upper_left_x, upper_left_z
    upper_right = upper_right_x, upper_right_z

    return (bottom_left, bottom_right, upper_right, upper_left)

def get_car_view_points_img(position, rotation):
    cam_view_points = get_car_view_points_unity(position, rotation)

    img_cam_view_points_bottom_left = [
        unity_plane_point_to_img_point(point) 
        for point in cam_view_points]
    img_cam_view_points_top_left = [
        img_point_bottom_left_to_top_left(point)
        for point in img_cam_view_points_bottom_left]

    return img_cam_view_points_top_left

def get_car_view_selection_matrix(position, rotation):
    '''
    Returns uint8 matrix size of track floor image representing 
    the area of the image that is visible by elements with value 255.
    All other elements are zero.
    '''
    # c is center point of circle, r is radius,
    # n is "normal" theta of radius being rotated. 
    # Returns (x, z) of point on circle after applying 
    # rotation.

    img_cam_points_top_left = get_car_view_points_img(position, rotation)
    selection_matrix = np.zeros(TRACK_FLOOR.shape).astype(np.uint8)
    selection_matrix = Image.fromarray(selection_matrix)
    draw = ImageDraw.Draw(selection_matrix)
    draw.polygon(img_cam_points_top_left, fill=255)

    selection_matrix = np.asarray(selection_matrix)

    return selection_matrix

def get_car_view_show_matrix(position, rotation):
    '''
    Returns uint8 matrix which mirrors TRACK_FLOOR but where all elements
    not currently visible by car are set to zero.
    '''
    selection_matrix = get_car_view_selection_matrix(position, rotation)
    return selection_matrix & TRACK_FLOOR

def get_car_view_img_box_points(position, rotation):
    view_points = get_car_view_points_img(position, rotation)
    _, __, upper_right, upper_left = view_points
    
    numerator = upper_left[1] - upper_right[1]
    denominator = upper_left[0] - upper_right[0]
    rot = math.atan2(numerator, denominator)

    box_points = [
        upper_right, upper_left,
        (upper_left[0] + (Z_U - Z_B) * TRACK_TO_IMG_RATIO_X * math.cos(rot - math.pi/2),
         upper_left[1] + (Z_U - Z_B) * TRACK_TO_IMG_RATIO_Z * math.sin(rot - math.pi/2))]

    return box_points

def get_car_view_img(position, rotation):
    box_points = get_car_view_img_box_points(position, rotation)
    transform = cv2.getAffineTransform(
        np.float32(box_points[:3]), 
        np.float32([[IMG_SIZE_X, 0], 
                    [0, 0],
                    [0, IMG_SIZE_Z]]))
    cropped_img = cv2.warpAffine(
        TRACK_FLOOR, transform, 
        (int(IMG_SIZE_X), int(IMG_SIZE_Z)))
    return cropped_img & IMG_VISIBLE_MASK

def car_img_to_top_down_perspective(car_img):
    z_imax, x_imax, _ = car_img.shape
    assert(z_imax == x_imax)
    replot_img = np.zeros((int(IMG_SIZE_Z), int(IMG_SIZE_X), 3), dtype=np.uint8)
    for z_img in range(int(z_imax*.7)):
        z_ifrac = z_img / z_imax
        z_plot = (
            CAR_CAM_HEIGHT 
            / math.tan(math.radians(CAR_CAM_ROTATION_X)
                       + math.radians(CAR_CAM_FOV) / 2 
                       - z_ifrac * math.radians(CAR_CAM_FOV)))
        z_pfrac = (z_plot - L_MIN) / (L_MAX - L_MIN)
        # print('z: ', z_pfrac)
        assert z_pfrac >= 0, z_pfrac <= 1
        z_w = int(IMG_SIZE_Z) - int(z_pfrac * int(IMG_SIZE_Z)) - 1
        for x_img in range(x_imax):
            x_ifrac = x_img / x_imax
            x_plot = 2 * (x_ifrac - .5) * math.tan(math.radians(CAR_CAM_FOV) / 2) * z_plot
            x_pfrac = .5 * (x_plot / X_U + 1)
            # print('x: ', x_pfrac)
            assert x_pfrac >= 0, x_pfrac <= 1
            x_w = int(x_pfrac * int(IMG_SIZE_X))

            print(x_w, z_w)
            replot_img[z_w, x_w] = car_img[z_imax - z_img - 1, x_img]

    return replot_img

# def get_img_equality_fraction(t, cam_name, location_candidates):
    # car_img = image_utils.simple_threshold(
    #     image_utils.get_cv2_from_tel_field(t, cam_name))
    # car_img_resized = None
    # car_img_resized_sum = None
    # for ((x, z), rot_y) in location_candidates:
    #     map_img = get_car_view_img((x, z), rot_y).astype(np.bool)
    #     if car_img_resized is None:
    #         car_img_resized = cv2.resize(car_img, map_img.shape[::-1])
    #         car_img_resized = car_img_resized.astype(np.bool)
    #         car_img_resized_sum = car_img_resized.sum()
    #     cmp_stat = (
    #         (map_img & car_img_resized).sum() 
    #         / max(map_img.sum(), car_img_resized_sum))
    #     yield (((x, z), rot_y), cmp_stat)
