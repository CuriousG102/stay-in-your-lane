import pip

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def Flatten_Car_Image(Image,Pixels):
    '''
    Input:
    Image: A jpeg or png image file location... In our case the front_car_image
    Pixels: An array of 4 2-element arrays that enclose the area of interest.
    The area that contains the lines. Go counter-clockwise.
    '''
    img = cv2.imread(Image)
    rows,cols,ch = img.shape
    pts1 = np.float32(Pixels)
    pts2 = np.float32([[300,300],[0,300],[300,0],[0,0]]) #Not exactly the best transform yet
    M = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_Image = cv2.warpPerspective(img,M,(300,300))
    return transformed_Image



def Find_Pixels(Image):
    '''
    :param Image: Takes a cv2.Image object
    :return:  the points of interest for the transformation
    '''

def show_Image(Image):
    '''
    Shows an image. Just for testing
    :param Image: An Image of class cv2.Image
    :return:
    '''
    plt.plot(Image)

def make_squares(n,m):
    B = np.zeros((n,n))
    for X in range(n):
        for Y in range(n):
            B[X,Y] = np.floor((np.floor(X/(n/m))+np.floor(Y/(n/m)))%2)+1
    #img = Image.fromarray(B)
    B = np.array(B)
    # rows, cols, ch = img.shape
    pts1 = np.float32([[n-1, n-1], [0, n-1], [n-1, 0], [0, 0]])
    pts2 = np.float32([[n/2-n/5, n/5], [n/2+n/5, n/5], [n-1, 0], [0, 0]])  # Not exactly the best transform yet
    M = cv2.getPerspectiveTransform(pts2, pts1)
    transformed_Image = cv2.warpPerspective(B, M, (n, n))
    plt.imshow(transformed_Image)
    plt.show()

def detect_horizon(Image_Loc):
    '''
    Use Canny Edge detection and Hough Transform to find horizontal lines. If we
    want to use this we need to adjust lighting.
    :param Image_Loc: Location of image
    :return:
    '''
    img = cv2.imread(Image_Loc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 40)
    cv2.imshow('edges', edges)
    tol = .8
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)
    for rho, theta in lines[0]:
        if theta > np.pi / 2 - tol and theta < np.pi / 2 + tol:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('gray', gray)



if __name__ == '__main__':
    Image = 'rail.jpg'
    Pixels = [[1000,730],[100,730],[700,500],[300,500]]
    Example = Flatten_Car_Image(Image,Pixels)
    img = cv2.imread(Image)
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(Example), plt.title('Output')
    plt.show()
   # make_squares(512,8)



