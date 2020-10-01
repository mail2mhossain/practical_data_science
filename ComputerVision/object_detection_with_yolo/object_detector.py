# USAGE
# python object_detector.py --conf config/config.json --image florida_trip.png

# import the necessary packages

from config.conf import Conf
from yolo_parser.parseyolooutput import ParseYOLOOutput
from datetime import datetime
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

#imagePaths = list(paths.list_images(args["dataset"]))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="Path to the input configuration file")
ap.add_argument("-i", "--image", required=True,
	help="Path of the image")
args = vars(ap.parse_args())

# load the image configuration file
conf = Conf(args["conf"])
image = cv2.imread(args["image"])

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([conf["yolo_path"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([conf["yolo_path"], "yolov3.weights"])
configPath = os.path.sep.join([conf["yolo_path"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the YOLO output parsing object
pyo = ParseYOLOOutput(conf)

# resize the image, convert it to grayscale, and blur it
image = imutils.resize(image, width=400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# construct a blob from the input image and then perform
# a forward pass of the YOLO object detector, giving us
# our bounding boxes and associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0,
            (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

(H, W) = image.shape[:2]

# parse YOLOv3 output
(boxes, confidences, classIDs) = pyo.parse(layerOutputs, LABELS, H, W)

# apply non-maxima suppression to suppress weak,
# overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences,
				conf["confidence"], conf["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the frame
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
        y = (y - 15) if (y - 15) > 0 else h - 15
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # show the frame
        cv2.imshow("show", image)
        cv2.waitKey(0)

# do a bit of cleanup
cv2.destroyAllWindows()
