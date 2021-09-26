import cv2 as cv


def convert_RGB(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def convert_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)