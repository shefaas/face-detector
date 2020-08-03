# Detecting faces on images and video streams using OpenCV and deep learning.
# Resources: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# import the necessary packages
import numpy as np
import argparse
import cv2


def argParsing():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="path to input image")
	ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

	return vars(ap.parse_args())


def imagePreprocessing(image):
	# get the height and width, ignore no. of channels.
	(h, w) = image.shape[:2]

	resizedImage = cv2.resize(image, (300, 300))
	scaleFactor = 1.0
	size = (300, 300)
	# Mean can be a 3-tuple of the RGB means or they can be a single value
	# to subtract the mean pixel value of the training dataset ILSVRC_2012
	# (B: 104.0069879317889, G: 116.66876761696767, R: 122.6789143406786)
	mean = (104.0, 177.0, 123.0)

	blob = cv2.dnn.blobFromImage(resizedImage, scaleFactor, size, mean)

	return blob


def drawBoxes(detections, args, image):
	(h, w) = image.shape[:2]
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			startX, startY, endX, endY = box.astype("int")

			# draw the bounding box of the face along with the associated probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY), (50, 0, 200), 2)
			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 0, 200), 2)


args = argParsing()

# load our serialized model from disk
print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])

blob = imagePreprocessing(image)

# pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

#print(detections)

resizedImage = cv2.resize(image, (int(image.shape[:2][1]/4), int(image.shape[:2][0]/4)))
drawBoxes(detections, args, resizedImage)

# show the output image
cv2.imshow("Output", resizedImage)
cv2.waitKey(0)