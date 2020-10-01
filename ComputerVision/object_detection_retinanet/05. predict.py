# USAGE
# python predict.py --model output.h5 --labels weapons/retinanet_classes.csv \
#	--image weapons/images/armas_1.jpg --confidence 0.5

# import the necessary packages
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to class labels")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the class label mappings
LABELS = open(args["labels"]).read().strip().split("\n")
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

# load the model from disk
model = models.load_model(args["model"], backbone_name="resnet50")

# load the input image (in BGR order), clone it, and preprocess it
image = read_image_bgr(args["image"])
output = image.copy()
image = preprocess_image(image)
(image, scale) = resize_image(image)
image = np.expand_dims(image, axis=0)

# detect objects in the input image and correct for the image scale
(boxes, scores, labels) = model.predict_on_batch(image)
boxes /= scale

# loop over the detections
for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
	# filter out weak detections
	if score < args["confidence"]:
		continue

	# convert the bounding box coordinates from floats to integers
	box = box.astype("int")

	# build the label and draw the label + bounding box on the output
	# image
	label = "{}: {:.2f}".format(LABELS[label], score)
	cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
		(0, 255, 0), 2)
	cv2.putText(output, label, (box[0], box[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)