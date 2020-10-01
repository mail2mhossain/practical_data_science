# USAGE
# python detect_faces_face_recognition.py --image test1.jpg 

# import the necessary packages
import face_recognition
import dlib
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


# load our image and convert it to grayscale
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load the face detector and detect faces in the image
# detect the (x, y)-coordinates of the bounding boxes
# corresponding to each face in the input image
boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
print("[INFO] detected {} faces".format(len(boxes)))

# loop over the bounding boxes and draw a rectangle around each face
for (x, y, w, h) in boxes:
    print (x, y, w, h)
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

# show the detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
