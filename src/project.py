import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from moviepy.editor import VideoFileClip

def camera_calibration():

    images = glob.glob('../camera_cal/calibration*.jpg')

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

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def pipeline(img):

    r_channel = img[: ,:, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x Gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary_1 = np.zeros_like(scaled_sobel)
    sxbinary_1[(scaled_sobel >= 20) & (scaled_sobel <= 120)] = 1

    #Magnitude of Gradient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=9)
    abs_sobel = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary_2 = np.zeros_like(scaled_sobel)
    sxbinary_2[(scaled_sobel >= 30) & (scaled_sobel <= 100)] = 1

    #Direction of Gradient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=15)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=15)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
    sxbinary_3 = np.zeros_like(scaled_sobel)
    sxbinary_3[(scaled_sobel >= 0.8) & (scaled_sobel <= 1.0)] = 1

    #S channel threshold
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 100) & (s_channel <= 255)] = 1

    #H channel threshold
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel <= 40) & (h_channel > 20)] = 1

    #R channel threshold
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > 180) & (r_channel <= 255)] = 1

    #B channel threshold
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel < 100) | (b_channel > 230)] = 1

    #G channel threshold
    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel > 100) & (g_channel <= 255)] = 1

    #L channel threshold
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 180)] = 1

    #Stack the thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(r_binary == 1) | (h_binary == 1) | ((sxbinary_1 == 1) & (sxbinary_2 == 1)) & (sxbinary_3 == 1)] = 1

    return combined_binary

def warper():

    # straight_lines1.jpg
    src = np.float32([[209, 719], [1095, 719], [538, 492], [751, 492]])
    dst = np.float32([[250, 719], [1030, 719], [250, 0], [1030, 0]])

    M1 = cv2.getPerspectiveTransform(src, dst)
    Minv1 = cv2.getPerspectiveTransform(dst, src)

    # straight_lines2.jpg
    src = np.float32([[228, 719], [1109, 719], [537, 492], [757, 492]])
    dst = np.float32([[250, 719], [1030, 719], [250, 0], [1030, 0]])

    M2 = cv2.getPerspectiveTransform(src, dst)
    Minv2 = cv2.getPerspectiveTransform(dst, src)

    M = (M1 + M2) / 2
    Minv = (Minv1 + Minv2) / 2
    return M, Minv

def  process_image(img):

    global M, Minv, mtx, dist, frame, left_fit, right_fit
    img = cv2.undistort(img, mtx, dist, None, mtx)

    #Put the warp before the pipeline because Sobel Gradient will work better from top view perspective
    height, width, channels = img.shape
    img_warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
    binary_warped = pipeline(img_warped)

    #fn = '../draft_images/frame' + str(n) + '.jpg'
    #plt.imsave(fn, binary_warped)
    #fn = '../draft_images/origin' + str(n) + '.jpg'
    #plt.imsave(fn, img)

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    if frame == 1:
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    else:

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 50
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                       left_fit[1] * nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                        right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    curverad = (left_curverad + right_curverad) / 2
    offset = (width / 2 - (left_fitx[-1]+right_fitx[-1]) / 2) * xm_per_pix
    if offset > 0:
        offset_text = 'Vehicle is ' + str(round(offset,2)) + 'm left of centre'
    elif offset < 0:
        offset_text = 'Vehicle is ' + str(round(-offset,2)) + 'm right of centre'
    else:
        offset_text = 'Vehicle is at the centre'
    curverad_text = 'Radius of Curvature = ' + str(int(curverad)) + '(m)'

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    result = cv2.putText(result, curverad_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    result = cv2.putText(result, offset_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    frame += 1
    return result

if __name__ == "__main__":
    frame = 1
    ret, mtx, dist, rvecs, tvecs = camera_calibration()
    M, Minv = warper()
    video_output = '../output_images/project_video.mp4'
    clip1 = VideoFileClip("../project_video.mp4")
    #video_output = '../output_images/challenge_video.mp4'
    #clip1 = VideoFileClip("../challenge_video.mp4")
    video_clip = clip1.fl_image(process_image)
    video_clip.write_videofile(video_output, audio=False)

    #img = mpimg.imread('../test_images/test2.jpg')
    #fn = '../output_images/test2_result.jpg'

    #plt.imsave(fn, process_image(img))
