# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 --output output/output_01.avi

# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --output output/webcam_output.avi


from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# CLASSES = ["background", "person"]
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

writer = None


W = None
H = None



# while True:
# 	# grab the frame from the threaded video stream and resize it
# 	# to have a maximum width of 400 pixels
# 	frame = vs.read()
# 	frame = imutils.resize(frame, width=400)

# 	# grab the frame dimensions and convert it to a blob
# 	(h, w) = frame.shape[:2]
# 	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
# 		0.007843, (300, 300), 127.5)

# 	# pass the blob through the network and obtain the detections and
# 	# predictions
# 	net.setInput(blob)
# 	detections = net.forward()

# 	# loop over the detections
# 	for i in np.arange(0, detections.shape[2]):
# 		# extract the confidence (i.e., probability) associated with
# 		# the prediction
# 		confidence = detections[0, 0, i, 2]

# 		# filter out weak detections by ensuring the `confidence` is
# 		# greater than the minimum confidence
# 		if confidence > args["confidence"]:
# 			# extract the index of the class label from the
# 			# `detections`, then compute the (x, y)-coordinates of
# 			# the bounding box for the object
# 			idx = int(detections[0, 0, i, 1])
# 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 			(startX, startY, endX, endY) = box.astype("int")
# 			# draw the prediction on the frame
# 			label = "{}: {:.2f}%".format(CLASSES[idx],
# 				confidence * 100)
# 			cv2.rectangle(frame, (startX, startY), (endX, endY),
# 				COLORS[idx], 2)
# 			y = startY - 15 if startY - 15 > 15 else startY + 15
# 			cv2.putText(frame, label, (startX, y),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# 	# show the output frame
# 	cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(1) & 0xFF

# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break

# 	# update the FPS counter
# 	fps.update()

# blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
# 		0.007843, (300, 300), 127.5)

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}


totalFrames = 0
totalDown = 0
totalUp = 0

fps = FPS().start()

while True:

	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	if args["input"] is not None and frame is None:
		break


	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if W is None or H is None:
		(H, W) = frame.shape[:2]


	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)



	status = "Waiting"
	rects = []



	if totalFrames % args["skip_frames"] == 0:

		status = "Detecting"
		trackers = []


		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):

			confidence = detections[0, 0, i, 2]


			if confidence > args["confidence"]:

				idx = int(detections[0, 0, i, 1])

				if CLASSES[idx] != "person":
					continue


				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")


				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)


				trackers.append(tracker)



	else:
		# loop over the trackers
		for tracker in trackers:

            
			status = "Tracking"


			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())


			rects.append((startX, startY, endX, endY))


	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)


	objects = ct.update(rects)


	for (objectID, centroid) in objects.items():


		to = trackableObjects.get(objectID, None)


		if to is None:
			to = TrackableObject(objectID, centroid)


		else:

            
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)


			if not to.counted:

                
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True

				
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True


		trackableObjects[objectID] = to


		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)



	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]


	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


	if writer is not None:
		writer.write(frame)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("q"):
		break


	totalFrames += 1
	fps.update()
# if writer is None:
# 		# initialize our video writer
# 		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# 		writer = cv2.VideoWriter(args["output"], fourcc, 30,
# 			(frame.shape[1], frame.shape[0]), True)

# 		# some information on processing single frame
# 		if total > 0:
# 			elap = (end - start)
# 			print("[INFO] single frame took {:.4f} seconds".format(elap))
# 			print("[INFO] estimated total time to finish: {:.4f}".format(
# 				elap * total))


fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


if writer is not None:
	writer.release()


if not args.get("input", False):
	vs.stop()


else:
	vs.release()


cv2.destroyAllWindows()
