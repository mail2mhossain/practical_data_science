# import the necessary packages
from sklearn.svm import LinearSVC
import argparse
import mahotas
import glob
import cv2

image = cv2.imread("sand.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

features = mahotas.features.haralick(image).mean(axis=0)

print(features)
