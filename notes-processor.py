#!/usr/bin/env python

# Python script to automatically convert poor-quality 
# photos of paper with writing on them into duotone images.

import sys
import cv2
import numpy as np

def processImage(fname):
	print "Processing %s" % fname
	source = cv2.imread(fname,cv2.CV_LOAD_IMAGE_GRAYSCALE)
	source = warpSheet(source)
	# Posterize the image using adaptive thresholding with
	# a fairly large neighbourhood.
	output = cv2.adaptiveThreshold(source, 255,
	                               cv2.ADAPTIVE_THRESH_MEAN_C,
	                               cv2.THRESH_BINARY,
	                               19, 8)
	cv2.imwrite("p_%s.png" %fname, output)

def findSquares(image):
	squares = []
	# Blur image to emphasize bigger features.
	blur = cv2.blur(image,(2,2))
	retval, edges = cv2.threshold(blur,0,255,
	                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	#cv2.imwrite("tmp.png", edges)
	contours, hierarchy = cv2.findContours(edges,
	                                cv2.RETR_LIST,
	                                cv2.CHAIN_APPROX_SIMPLE)
	#cdraw = np.zeros(blur.shape)
	#cv2.drawContours(cdraw,contours,-1,(255,0,0),thickness=5)
	for c in contours:
		clen = cv2.arcLength(c,True)
		c = cv2.approxPolyDP(c,0.02*clen,True)
		#cv2.drawContours(cdraw,[c],-1,(0,0,255),thickness=5)
		area = abs(cv2.contourArea(c))
		if len(c) == 4 and \
		   0.1*edges.size <= area <= 0.9*edges.size and \
		   cv2.isContourConvex(c):
			# Omit angle check.
			squares.append(c)
	#cv2.imwrite("tmp__b.png", cdraw)
	return squares

def warpSheet(image):
	squares = findSquares(image)
	sheet = max(squares,key=lambda s: cv2.arcLength(s,True))
	sourcePoints = sheet[::,0,::].astype('float32')
	r,c = image.shape
	destPoints = np.array([[c,0],
	                       [0,0],
	                       [0,r],
	                       [c,r]],dtype='float32')
	transform = cv2.getPerspectiveTransform(sourcePoints,destPoints)
	return cv2.warpPerspective(image,transform,(c,r))

if __name__ == "__main__":
	map(processImage,sys.argv[1:])

