import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def camera_calibration():

    images = glob.glob('../camera_cal/calibration*.jpg')
    #img = mpimg.imread('../camera_cal/calibration12.jpg')
    #plt.imshow(img)
    #plt.show()

    objpoints = []
    imgpoints = []

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    gray = None
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #plt.imshow(img)
            #plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

#print(ret)
#print(mtx)
#print(dist)
#print(rvecs)
#print(tvecs)

#test
#img = mpimg.imread('../test_images/straight_lines1.jpg')
#ret, mtx, dist, rvecs, tvecs = camera_calibration()
#dst = cv2.undistort(img, mtx, dist, None, mtx)

#plt.imshow(dst)
#plt.show()