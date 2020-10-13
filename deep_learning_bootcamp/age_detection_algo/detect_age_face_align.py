# import the necessary packages 
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse 
import imutils
import dlib
import cv2
import numpy as np 
import pandas as pd


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's faec detector (HOG-based) and then create the facial landmark predictor and the face aligner 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"] + "/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)


# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale image 

#cv2.imshow("input", image)
rects = detector(gray, 2)
#print(rects[1])

i = 0
img_name_list = []

# loop over the face detections
for rect in rects:
    # extract the ROI of the *orifinal* face, then align the face using facial landmark 
    (x, y, w, h) = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect)
    #display the output images 
    #cv2.imshow("Original", faceOrig)
    #cv2.imwrite(args['image'].split("/")[-1], faceAligned)
    img_name = args['image'].split(".")[0].split('/')[-1] + str(i) + '.png'
    i += 1
    img_name_list.append(img_name)
    cv2.imwrite(img_name, faceAligned)
    cv2.waitKey(0)
print(img_name_list)
image_merge = cv2.hconcat([cv2.imread(i) for i in img_name_list])

image_name = args['image'].split("/")[-1].split('.')[0] + '_merge.png'

cv2.imwrite(image_name,image_merge) 
#`cv2.waitKey(0)

