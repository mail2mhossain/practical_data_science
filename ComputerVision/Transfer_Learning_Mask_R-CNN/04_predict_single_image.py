from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
		help="path to input image to segment")
args = vars(ap.parse_args())


# initialize the class names dictionary
CLASS_NAMES = {1: "kangaroo"}


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "kangaroo_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
    # set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.9

config = PredictionConfig()
# initialize the Mask R-CNN model for inference
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')

# load model weights
model_path = 'mask_rcnn_kangaroo_cfg_0005.h5'
model.load_weights(model_path, by_name=True)

# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=1024)

# perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=1)[0]

# loop over of the detected object's bounding boxes and
# masks, drawing each as we go along
for i in range(0, r["rois"].shape[0]):
	mask = r["masks"][:, :, i]
	image = visualize.apply_mask(image, mask,
				(1.0, 0.0, 0.0), alpha=0.5)
	image = visualize.draw_box(image, r["rois"][i],
				(1.0, 0.0, 0.0))

# convert the image back to BGR so we can use OpenCV's
# drawing functions
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# loop over the predicted scores and class labels
for i in range(0, len(r["scores"])):
	# extract the bounding box information, class ID, label,
	# and predicted probability from the results
	(startY, startX, endY, end) = r["rois"][i]
	classID = r["class_ids"][i]
	label = CLASS_NAMES[classID]
	score = r["scores"][i]

	# draw the class label and score on the image
	text = "{}: {:.4f}".format(label, score)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# resize the image so it more easily fits on our screen
image = imutils.resize(image, width=512)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)