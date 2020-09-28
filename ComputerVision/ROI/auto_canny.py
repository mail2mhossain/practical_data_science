# USAGE
# python auto_canny.py --image teacup.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
clone = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# apply Canny edge detection using automatically determined threshold
auto = imutils.auto_canny(blurred)
#thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

# perform a series of dilations and erosions to close holes
# in the shapes
thresh = cv2.dilate(auto, None, iterations=4)
thresh = cv2.erode(thresh, None, iterations=2)

# detect contours in the edge map
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    # create an empty mask for the contour and draw it
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # extract the bounding box ROI from the mask
    (x, y, w, h) = cv2.boundingRect(c)

    box = cv2.minAreaRect(c)
    box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))

    cv2.drawContours(clone, [box], -1, (0, 0, 255), 2)
    cv2.imshow("Orig", clone)

    masked = cv2.bitwise_and(image, image, mask=mask)
    roi = masked[y:y + h, x:x + w]

    cv2.imshow("Mask Applied to Image", roi)

    cv2.waitKey(0)



# show the images
#cv2.imshow("Original", image)
#cv2.imshow("Wide", wide)
#cv2.imshow("Tight", tight)
#cv2.imshow("Auto", auto)
#cv2.imshow("Threshold", thresh)
#cv2.waitKey(0)
