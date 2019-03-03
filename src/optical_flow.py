import os
import cv2 as cv
import numpy as np

def exhaustive_optical_flow(img_data):
    zero_img = np.zeros(img_data[0].shape)
    zero_img = zero_img[...,np.newaxis]
    rgb_img = cv.cvtColor(img_data[0],cv.COLOR_GRAY2RGB)
    hsv_img = cv.cvtColor(rgb_img,cv.COLOR_BGR2HSV)
    hsv_img[...,1] = 255
    contour_imgs = 0
    prev_img = img_data[0]
    for i in range(1, len(img_data)):
        next_img = img_data[i]
        flow = cv.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv_img[...,0] = ang*180/np.pi/2
        hsv_img[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv_img,cv.COLOR_HSV2BGR)
        bgr_mean = cv.medianBlur(bgr,5)
        gray = cv.cvtColor(bgr_mean,cv.COLOR_BGR2GRAY)
        gray_img_thr = cv.threshold(gray,100,255, cv.THRESH_BINARY)
        contour_imgs += np.asarray(gray_img_thr)
        previous_img = next_img
    
    retval,final_contour = cv.threshold(contour_imgs[1],30,255,cv.THRESH_BINARY)
    return final_contour
