# import the necessary packages
from scipy.spatial import distance as dist
from imutils import paths
import numpy as np
import cv2


#image = cv2.imread("raptors_02.png")
#(means, stds) = cv2.meanStdDev(image)

#features = np.concatenate([means, stds]).flatten()

#print (features)

d = dist.euclidean([33.40, 29.97, 39.20, 27.05, 23.14, 24.45] , [32.91, 31.70, 41.27, 32.24, 24.12, 24.84] )

print (d)
