import pip

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    pts2 = pts2 = np.float32([[300,300],[0,300],[300,0],[0,0]]) #Not exactly the best transform yet
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


if __name__ == '__main__':
    Image = 'rail.jpg'
    Pixels = [[1000,730],[100,730],[700,500],[300,500]]
    Example = Flatten_Car_Image(Image,Pixels)
    img = cv2.imread(Image)
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(Example), plt.title('Output')
    plt.show()



