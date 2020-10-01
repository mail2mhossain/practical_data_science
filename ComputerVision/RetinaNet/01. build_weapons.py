# USAGE
# python build_weapons.py

# import the necessary packages
from config import weapons_config as config
from bs4 import BeautifulSoup
from imutils import paths
import random
import os

# initialize the set of classes we have encountered so far
CLASSES = set()

# grab all image paths then construct a training and testing split
# from them
imagePaths = list(paths.list_files(config.IMAGES_PATH))
random.shuffle(imagePaths)
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainImagePaths = imagePaths[:i]
testImagePaths = imagePaths[i:]

# create the list of datasets to build
datasets = [
	("train", trainImagePaths, config.TRAIN_CSV),
	("test", testImagePaths, config.TEST_CSV),
]

# loop over the datasets
for (dType, imagePaths, outputCSV) in datasets:
	# load the contents of the input data split
	print("[INFO] creating '{}' set...".format(dType))
	print("[INFO] {} total images in '{}' set".format(
		len(imagePaths), dType))

	# open the output CSV file
	csv = open(outputCSV, "w")

	# loop over the image paths
	for imagePath in imagePaths:
		# build the corresponding annotation path
		fname = imagePath.split(os.path.sep)[-1]
		fname = "{}.xml".format(fname[:fname.rfind(".")])
		annotPath = os.path.sep.join([config.ANNOT_PATH, fname])

		# load the contents of the annotations file and build the soup
		contents = open(annotPath).read()
		soup = BeautifulSoup(contents, "html.parser")

		# extract the image dimensions
		w = int(soup.find("width").string)
		h = int(soup.find("height").string)

		# loop over all object elements
		for o in soup.find_all("object"):
			# extract the label and bounding box coordinates
			label = o.find("name").string
			xMin = int(o.find("xmin").string)
			yMin = int(o.find("ymin").string)
			xMax = int(o.find("xmax").string)
			yMax = int(o.find("ymax").string)

			# truncate any bounding box coordinates that may fall outside
			# the boundaries of the image
			xMin = max(0, xMin)
			yMin = max(0, yMin)
			xMax = min(w, xMax)
			yMax = min(h, yMax)

			# due to errors in annotation, it may be possible that
			# the minimum values are larger than the maximum values;
			# in this case, treat it as an error during annotation
			# and ignore the bounding box
			if xMin >= xMax or yMin >= yMax:
				continue

			# similarly, we could run into the opposite case where
			# the max values are smaller than the minimum values
			elif xMax <= xMin or yMax <= yMin:
				continue

			# write the image path, bounding box coordinates, and label
			# to the output CSV file
			row = [os.path.abspath(imagePath), str(xMin), str(yMin),
				str(xMax), str(yMax), label]
			csv.write("{}\n".format(",".join(row)))

			# update the set of unqiue class labels
			CLASSES.add(label)

	# close the CSV file
	csv.close()

# write the classes to file
print("[INFO] writing classes...")
csv = open(config.CLASSES_CSV, "w")
rows = [",".join([c, str(i)]) for (i, c) in enumerate(CLASSES)]
csv.write("\n".join(rows))
csv.close()