
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import cv
import os
import sys
from pathlib import Path   
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--path", required=False,
	help="path to output image")
args = vars(ap.parse_args())

def remove_background(foreground):
	BLUR = 21
	CANNY_THRESH_1 = 10
	CANNY_THRESH_2 = 100
	MASK_DILATE_ITER = 10
	MASK_ERODE_ITER = 10
	MASK_COLOR = (0.0,0.0,0.0) # In BGR format


#-- Read image
	img=foreground
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection 
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)

#-- Find contours in edges, sort by area 
	contour_info = []
	contours,abc = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for c in contours:
		contour_info.append((
        	c,
        	cv2.isContourConvex(c),
        	cv2.contourArea(c),
    	))
	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
	max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
	mask = np.zeros(edges.shape)
	# cv2.fillConvexPoly(mask, max_contour[0], (255))
	for c in contour_info:
		cv2.fillConvexPoly(mask, c[0], (255, 0, 0))

#-- Smooth mask, then blur it
	mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
	mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
	mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
	mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#-- Blend masked img into MASK_COLOR background
	mask_stack  = mask_stack.astype('float32') / 255.0         
	img         = img.astype('float32') / 255.0    
	masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)  
	masked = (masked * 255).astype('uint8')                    
	#masked[mask == 255] = [0, 0, 0]
	cv2.imshow('img', masked) 
	cv2.imwrite('savedImage.jpg',masked)                                  # Display
	return masked


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=512)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])

result_name=Path(args["image"]).name[:-4]
print(sys.path[0])
print(sys.argv[6])
print(result_name)

image = imutils.resize(image, width=800)
background_rem=remove_background(image)
cv2.imshow("Input", image)
#image=background_rem
gray = cv2.cvtColor(background_rem, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
# show the original input image and detect faces in the grayscale
# image
cv2.imshow("Input", image)
rects = detector(gray, 1)



# loop over the face detections
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	print(rect)
	faceOrig = imutils.resize(gray[y:y + h, x:x + w], width=256)
	faceAligned = fa.align(image, gray, rect)

	import uuid
	f = str(uuid.uuid4())
	#cv2.imwrite("savedpic.png", faceAligned)

	# display the output images
	cv2.imshow("Original", faceOrig)
	cv2.imshow("Aligned", faceAligned)
	crop_img = faceAligned[100:400, 100:400]
	crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("ddd", crop_img)
	cv2.imwrite(sys.argv[6]+"//"+result_name+".png", crop_img)
	#faceAligned_resize=imutils.resize(faceAligned[y:y + h, x:x + w], width=256)
	#cv2.imshow("faceAligned_resize", faceAligned_resize)
	print(faceAligned.shape)
	#background_rem=remove_background22(faceOrig)
	#cv2.waitKey(0)



