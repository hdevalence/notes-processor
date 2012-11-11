#/usr/bin/env python

# Python script to automatically convert poor-quality 
# photos of paper with writing on them into duotone images.

import sys
import cv2
import numpy as np

def processImage(fname):
	source = cv2.imread(fname,cv2.CV_LOAD_IMAGE_GRAYSCALE)
	# Posterize the image using adaptive thresholding with
	# a fairly large neighbourhood.
	output = cv2.adaptiveThreshold(source, 255,
	                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	                               cv2.THRESH_BINARY,
	                               11, 12)
	cv2.imwrite("p_%s.png" %fname, output)

if __name__ == "__main__":
	map(processImage,sys.argv[1:])

