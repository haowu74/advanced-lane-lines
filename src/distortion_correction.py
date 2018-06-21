import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def distortion_correction(img, mtx, dist):
    img = mpimg.imread('../test_images/straight_lines1.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst