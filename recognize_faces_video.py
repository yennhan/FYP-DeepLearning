# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
#Source: https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
# import the necessary packages
from imutils.video import VideoStream
from collections import deque
from threading import Thread
from firebase_admin import db
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np

def explicit():
    from google.cloud import storage
    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        'minimumviableproduct-f493e-firebase-adminsdk-v8z4z-9bb9769ad7.json')

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    root = db.reference("users_images")
    snapshot = root.order_by_key().get()
    #print(len(snapshot))
    object_name = 'unknown_person_%s.jpg' %len(snapshot)
    bucket = storage_client.get_bucket('minimumviableproduct-f493e.appspot.com')
    Blob = bucket.blob('/')
    the_image = bucket.blob(object_name)
    the_image.upload_from_filename('the_pic.png' , content_type='image/jpeg')
    #print(the_image.public_url)
    return the_image.public_url

def write_to_database(url,strange_face):
    root = db.reference()
    new_user = root.child('users_images').push({
        'url': url,
        'time': int(time.time()),
        'stranger': strange_face
    })

def call_screenshot(f):
    cv2.imwrite("the_pic.png", f)
    ur = explicit()
    url = face_upload()
    write_to_database(ur,url)
    return

def face_upload():
    from google.cloud import storage
    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        'minimumviableproduct-f493e-firebase-adminsdk-v8z4z-9bb9769ad7.json')
    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    root = db.reference("users_images")
    snapshot = root.order_by_key().get()
    #print(len(snapshot))
    object_name = 'stranger_face_%s.jpg' %len(snapshot)
    bucket = storage_client.get_bucket('minimumviableproduct-f493e.appspot.com')
    Blob = bucket.blob('/')
    the_image = bucket.blob(object_name)
    the_image.upload_from_filename('face.png' , content_type='image/jpeg')
    #print(the_image.public_url)
    return the_image.public_url

def call_face(f):
    cv2.imwrite("face.png",f)
    url = face_upload()
    return url

# construct the argument parser and parse the arguments
argument = argparse.ArgumentParser()
argument.add_argument("-e", "--encodings", required=True,
                      help="In order to deserialized the db")
argument.add_argument("-d", "--detection-method", type=str, default="hog",
                      help="hog or cnn detection model")
argument.add_argument("-o", "--output", type=str,
                      help="the output required video")
argument.add_argument("-y", "--display", type=int, default=1,
                      help="dispaly on screen or otherwise")
argument.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(argument.parse_args())

# load the known faces and embeddings

data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("Video loading")
import firebase_admin
from firebase_admin import credentials
cred = credentials.Certificate('minimumviableproduct-f493e-firebase-adminsdk-v8z4z-9bb9769ad7.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://minimumviableproduct-f493e.firebaseio.com/'
})

vs = VideoStream(src=0).start()
writer = None
time.sleep(0.5)
start=0
total_duration = 60
# loop over frames from the video file stream

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None


    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, "The_object", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # update the points queue
    pts.appendleft(center)
    '''
    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        #if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        #thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    '''
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces

    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        sub_face = frame[top:bottom, left:right]
        #print(name)
        end = int(time.time())
        total_duration = end-start
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        if name =="Unknown" and total_duration>=30:
            call_screenshot(frame)
            call_face(sub_face)
            start = int(time.time())
            total_duration = end-start


    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces todisk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()

