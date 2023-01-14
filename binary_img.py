import numpy as np
import cv2

# read the fingerprint image as grayscale
def extractSkin(path):
    image = cv2.imread(path)
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)


    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

def task(path):
    img = extractSkin(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    min_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    #cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    max_img = cv2.convertScaleAbs(max_img)
    min_img = cv2.convertScaleAbs(min_img)

    # create a binary mask with the local maximum and minimum values
    mask = np.zeros_like(img)
    mask[max_img > min_img] = 255

    # apply the mask to the original image to create a binary image
    binary = cv2.bitwise_and(img, mask)


    # display the resulting image
    cv2.imshow("Fingerprint", binary)
    cv2.waitKey(0)
task("i4.jpeg")
task("i5.jpeg")