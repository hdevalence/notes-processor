#!/usr/bin/env python

# Python script to automatically convert poor-quality 
# photos of paper with writing on them into duotone images.

import sys
import cv2
import numpy as np
import os.path
from itertools import starmap

NBHD_SIZE = 19
UNSHARP_T = 48
ADAPT_T   = 24

def processImage(fname):
    print "Processing %s" % fname
    source = cv2.imread(fname,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return bitone(warpSheet(source))

def bitone(image):
    '''
    Convert a greyscale image to a bitone image,
    in such a way that we preserve as much detail as possible,
    and have the least amount of speckles.
    '''
    # First, sharpen the image: unsharp mask w/ threshold.
    blur = cv2.blur(image,(NBHD_SIZE,NBHD_SIZE))
    diff = cv2.absdiff(image,blur)
    # Apparently OpenCV doesn't have a way to
    # apply a mask to a weighted sum, so we do it ourselves.
    _,mask = cv2.threshold(blur,UNSHARP_T,1,cv2.THRESH_BINARY)
    blur = cv2.multiply(blur,mask)
    sharpened = cv2.addWeighted(image,2,blur,-1,0)
    cv2.imwrite('sharp.png',sharpened)
    # Now threshold the sharpened image.
    thresh = cv2.adaptiveThreshold(sharpened, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   NBHD_SIZE, ADAPT_T)
    return thresh

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
            squares.append(c)
    #cv2.imwrite("tmp__b.png", cdraw)
    return squares


def warpSheet(image):
    h, w = image.shape
    squares = findSquares(image)
    sheet = max(squares,key=lambda s: cv2.arcLength(s,True))
    #We don't know how the points of the square we found are ordered. 
    #So we just study it out, and we'll see.
    src = sheet[::,0,::].astype('float32')
    # Compute distances from topleft corner (0,0)
    # to find topleft and bottomright
    d = np.sum(np.abs(src)**2,axis=-1)**0.5
    t_l = np.argmin(d)
    b_r = np.argmax(d)
    # Compute distances from topright corner (w,0)
    # to find topright and bottomleft
    y = np.array([[w,0],]*4)
    d = np.sum(np.abs(src-y)**2,axis=-1)**0.5
    t_r = np.argmin(d)
    b_l = np.argmax(d)
    #Now assemble these together
    destH, destW = h, int(h*8.5/11.0)
    dest = np.zeros(src.shape,dtype='float32')
    dest[t_l] = np.array([0,0])
    dest[t_r] = np.array([destW,0])
    dest[b_l] = np.array([0,destH])
    dest[b_r] = np.array([destW,destH])
    transform = cv2.getPerspectiveTransform(src,dest)
    return cv2.warpPerspective(image,transform,(destW,destH))

def rename(originalName):
    d,f = os.path.split(originalName)
    f,ext = os.path.splitext(f)
    return os.path.join(d,'p_%s.png' %f)

if __name__ == "__main__":
    processed = map(processImage,sys.argv[1:])
    newnames = map(rename,sys.argv[1:])
    for n,i in zip(newnames,processed):
	cv2.imwrite(n,i)
