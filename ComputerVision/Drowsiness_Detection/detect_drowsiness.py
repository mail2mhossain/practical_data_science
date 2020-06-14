# USAGE
# python detect_drowsiness.py --conf config/config.json

# import the necessary packages
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
from imutils import face_utils
from datetime import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


MOUTH_AR_THRESH = 0.65
YAWN_THRESH_COUNT =  120
YAWN_THRESH_TIME = 600
EYE_AR_THRESH =  0.3
EYE_AR_CONSEC_FRAMES = 8
display = True
alarm = False
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
cascade_path = "haarcascade_frontalface_default.xml"


def euclidean_dist(ptA, ptB):

	return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	a = euclidean_dist(eye[1], eye[5])
	b = euclidean_dist(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	c = euclidean_dist(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (a + b) / (2.0 * c)

	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the three sets of
	# vertical mouth landmarks (x, y)-coordinates
	a = euclidean_dist(mouth[1], mouth[7])
	b = euclidean_dist(mouth[2], mouth[6])
	c = euclidean_dist(mouth[3], mouth[5])

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	d = euclidean_dist(mouth[0], mouth[4])

	# compute the mouth aspect ratio
	mar = (a + b + c) / (2.0 * d)

	# return the mouth aspect ratio
	return mar


if alarm:
	from gpiozero import TrafficHat
	th = TrafficHat()
	print("[INFO] using TrafficHat alarm...")

# initialize the frame center coordinates
centerX = None
centerY = None

# initialize the blink counter, yawn counter, a boolean used to
# indicate if the alarm is going off, and start time
blinkCounter = 0
yawnCounter = 0
alarmOn = False
startTime = None

# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier(cascade_path)
predictor = dlib.shape_predictor(shape_predictor_path)

# grab the indexes of the facial landmarks for the left, right eye,
# and inner part of the mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()

# FOR WINDOWS
#vs = cv2.VideoCapture(0)

time.sleep(2.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize,
	# flip horizontally, and convert to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	frame = cv2.flip(frame, 1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# set the frame center coordinates
	if centerX is None and centerY is None:
		(centerX, centerY) = (frame.shape[1] // 2,
			frame.shape[0] // 2)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# loop over the detected faces
	for rect in rects:
		# draw a bounding box surrounding the face
		(x, y, w, h) = rect
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

	# check if the number of faces detected is greater than zero
	if len(rects) > 0:
		# sort the detected rectangles by their position relative to
		# the center and grab the face that's closest to the center
		centerRect = sorted(rects, key=lambda r: abs((
			r[0] + (r[2] / 2)) - centerX) + abs((
			r[1] + (r[3] / 2)) - centerY))[0]

		# get the coordinates of the rectangle in the center and
		# construct a dlib rectangle object from the Haar cascade
		# bounding box
		(x, y, w, h) = centerRect
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold
		if ear < EYE_AR_THRESH:
			# increment the blink frame counter
			blinkCounter += 1

			# if the eyes were closed for a sufficient number of
			# frames, then sound the alarm
			if blinkCounter >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not alarmOn:
					alarmOn = True

					# check to see if the TrafficHat buzzer should
					# be sounded and red light set to blink
					if alarm:
						th.buzzer.blink(0.1, 0.1, 30,
							background=True)
						th.lights.red.blink(0.1, 0.1, 30,
							background=True)

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT! - eyes", (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			blinkCounter = 0
			alarmOn = False

		# extract the inner mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio for the mouth
		mouth = shape[mStart:mEnd]
		mar = mouth_aspect_ratio(mouth)

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

		# check to see if the mouth aspect ratio is above the yawn
		# threshold
		if mar > MOUTH_AR_THRESH:
			# increment the yawn frame counter and set the start time
			yawnCounter += 1
			startTime = datetime.now() if startTime == None else \
				startTime

			# check to see if yawn frame counter is greater than yawn
			# frame threshold and if the difference between current
			# time and start time is less than or equal to yawn
			# threshold time (in which case the person is drowsy)
			if yawnCounter >= YAWN_THRESH_COUNT and \
				(datetime.now() - startTime).seconds <= \
				YAWN_THRESH_TIME:
				# if the alarm is not on, turn it on
				if not alarmOn:
					alarmOn = True

					# check to see if the TrafficHat buzzer should
					# be sounded and red light set to blink
					if alarm:
						th.buzzer.blink(0.1, 0.1, 10,
							background=True)
						th.lights.red.blink(0.1, 0.1, 30,
							background=True)

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT! - yawning",
					(10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
					(0, 0, 255), 2)

		# check to see if the start time is set
		elif startTime != None:
			# check if the difference between current time and start
			# time is greater than yawn threshold time
			if (datetime.now() - startTime).seconds > \
				YAWN_THRESH_TIME:
				# reset yawn counter, alarm flag and start time
				yawnCounter = 0
				alarmOn = False
				startTime = None

		# draw the computed aspect ratios on the frame
		cv2.putText(frame, "EAR: {:.3f} MAR: {:.3f}".format(
			ear, mar), (175, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
			(0, 0, 255), 2)

	# if the 'display flag is set, then display the current frame
	# to the screen and record if a user presses a key
	if display:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# check to see if we have any open windows, and if so, close them
if display:
	cv2.destroyAllWindows()

# release the video stream pointer
vs.stop()
