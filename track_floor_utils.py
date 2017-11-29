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

CAR_CAM_POS_RELATIVE = (0, 19, 9)  # 19 isn't quite right as car center 
                                   # is slightly elevated off track, but 
                                   # don't think it's a big deal

CAM_VIEW_BOX_SIDE_SIZE = (         # Global scale, be careful to note that
    2 * CAR_SCALE * CAR_CAM_POS_RELATIVE[1] 
    * math.tan(math.radians(CAR_CAM_FOV / 2))) 

CORNER_DIST_FROM_CAM_CENTER = math.sqrt(2 * (CAM_VIEW_BOX_SIDE_SIZE / 2)**2)

# convert to global scale
CAM_DIST_FROM_CAR = CAR_SCALE * CAR_CAM_POS_RELATIVE[2] 

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
    top_left_point_x, top_left_point_z = top_left_point
    track_shape_height, track_shape_width = TRACK_FLOOR_FILLED.shape
    if (top_left_point_x < 0 or top_left_point_z < 0
        or top_left_point_x >= track_shape_width  
        or top_left_point_y >= track_shape_height):
        return False
    return TRACK_FLOOR_FILLED[top_left_point]

def get_car_view_box_center_pos(position, rotation):
    get_plane_points = lambda c, r, n: (c[0] + r * math.cos(math.radians(n - rotation)), 
                                        c[1] + r * math.sin(math.radians(n - rotation)))
    return get_plane_points(position, CAM_DIST_FROM_CAR, 90)

def get_car_view_box_points(position, rotation):
    '''
    Returns top left origin img points for car view.
    '''
    get_plane_points = lambda c, r, n: (c[0] + r * math.cos(math.radians(n - rotation)), 
                                        c[1] + r * math.sin(math.radians(n - rotation)))

    # get cam viewbox points in unity coordinate system
    cam_center_pos = get_car_view_box_center_pos(position, rotation)
    cam_box_points = [
        get_plane_points(cam_center_pos, CORNER_DIST_FROM_CAM_CENTER, angle)
        for angle in (45, 135, 225, 315,)]

    img_cam_box_points_bottom_left = [
        unity_plane_point_to_img_point(point) 
        for point in cam_box_points]
    img_cam_box_points_top_left = [
        img_point_bottom_left_to_top_left(point)
        for point in img_cam_box_points_bottom_left]

    return img_cam_box_points_top_left

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

    img_cam_box_points_top_left = get_car_view_box_points(position, rotation)
    selection_matrix = np.zeros(TRACK_FLOOR.shape).astype(np.uint8)
    selection_matrix = Image.fromarray(selection_matrix)
    draw = ImageDraw.Draw(selection_matrix)
    draw.polygon(img_cam_box_points_top_left, fill=255)
    selection_matrix = np.asarray(selection_matrix)

    return selection_matrix

def get_car_view_show_matrix(position, rotation):
    '''
    Returns uint8 matrix which mirrors TRACK_FLOOR but where all elements
    not currently visible by car are set to zero.
    '''
    selection_matrix = get_car_view_selection_matrix(position, rotation)
    return selection_matrix & TRACK_FLOOR

def get_car_view_img(position, rotation):
    box_points = get_car_view_box_points(position, rotation)
    track_scale_x, track_scale_z = TRACK_SCALE
    img_scale_z, img_scale_x = TRACK_FLOOR.shape
    selection_size = int(
        img_scale_x / track_scale_x * CAM_VIEW_BOX_SIDE_SIZE)
    transform = cv2.getAffineTransform(
        np.float32(box_points[:3]), 
        np.float32([[selection_size, 0], 
                    [0, 0],
                    [0, selection_size]]))
    return cv2.warpAffine(TRACK_FLOOR, transform, 
                          (selection_size, selection_size))

def get_img_equality_fraction(t, cam_name, location_candidates):
    car_img = image_utils.simple_threshold(
        image_utils.get_cv2_from_tel_field(t, cam_name))
    car_img_resized = None
    car_img_resized_sum = None
    for ((x, z), rot_y) in location_candidates:
        map_img = get_car_view_img((x, z), rot_y).astype(np.bool)
        if car_img_resized is None:
            car_img_resized = cv2.resize(car_img, map_img.shape[::-1])
            car_img_resized = car_img_resized.astype(np.bool)
            car_img_resized_sum = car_img_resized.sum()
        cmp_stat = (
            (map_img & car_img_resized).sum() 
            / max(map_img.sum(), car_img_resized_sum))
        yield (((x, z), rot_y), cmp_stat)
