This is (the beginnings of) a little script to process photos of
handwritten notes and make them better for viewing/reproduction.

Currently, it does two things:

1.  Tries to detect a sheet of paper against a background, and
    then automatically crop and perform a perspective
    correction.
2.  Applies an adaptive thresholding function to posterize the
    image, giving black writing on white background.

Todo
----

1.  The dewarping is somewhat broken, in the following ways:
    *   It assumes that the sheet will *always* be on a
        differently-coloured background, which may not be true.
    *   It assumes that the sheet is 8.5*11 instead of
        trying to determine its true aspect ratio.

2.  The file handling is really brittle and only works enough to
    test the image processing.

3.  The output is saved as an 8bit greyscale png, when it is
    really only 1bit data.

Creating PDFs
-------------

You can create a PDF from the processed images using Graphicsmagick:

    gm convert p_*.png document.pdf


