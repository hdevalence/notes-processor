#/usr/bin/env python

# Python script to automatically convert poor-quality 
# photos of paper with writing on them into duotone images.

import sys
import cv

def processImage(fname):
	source = cv.LoadImageM(fname,cv.CV_LOAD_IMAGE_GRAYSCALE)
	output = cv.CreateImage(cv.GetSize(source),cv.IPL_DEPTH_8U,1)
	# Posterize the image using adaptive thresholding with
	# a fairly large neighbourhood.
	cv.AdaptiveThreshold(source,output,255,blockSize=11,param1=12)
	cv.SaveImage("p_%s.png" %fname, output)

if __name__ == "__main__":
	map(processImage,sys.argv[1:])

