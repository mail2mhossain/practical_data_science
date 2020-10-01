# USAGE
# python detect_faces_haarcascades.py --detector haarcascade_frontalface_default.xml --image test1.jpg 

# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to Haar cacscade face detector")
args = vars(ap.parse_args())

# load our image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector and detect faces in the image
detector = cv2.CascadeClassifier(args["detector"])
rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=9,
	minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
print("[INFO] detected {} faces".format(len(rects)))

# loop over the bounding boxes and draw a rectangle around each face
for (x, y, w, h) in rects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
