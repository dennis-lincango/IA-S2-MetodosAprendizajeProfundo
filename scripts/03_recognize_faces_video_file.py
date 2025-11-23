# USAGE
# python scripts/03_recognize_faces_video_file.py --encodings models/face_encodings.pkl --input test/files/videos/example_01.mp4
# python scripts/03_recognize_faces_video_file.py --encodings models/face_encodings.pkl --input test/files/videos/example_01.mp4 --output test/output/example_01_output.avi --display 0

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", type=str,
                help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# Parámetro de tolerancia (más bajo = más estricto)
TOLERANCE = 0.45

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the pointer to the video file and the video writer
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
writer = None

# loop over frames from the video file stream
while True:
    # grab the next frame
    (grabbed, frame) = stream.read()

    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break

    # resize the frame to have a width of 750px (to speed up processing)
    frame_resized = imutils.resize(frame, width=750)

    # convert the resized frame from BGR to RGB
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # ratio to scale back the face locations to original frame size
    r = frame.shape[1] / float(frame_resized.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
                                            model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(
            data["encodings"],
            encoding,
            tolerance=TOLERANCE
        )

        # default name is Unknown
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                person_name = data["names"][i]
                counts[person_name] = counts.get(person_name, 0) + 1

            # determine the recognized face with the largest number
            # of votes
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates back to original frame size
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)

        # Punto Medio
        xMedio = int(left + abs((left - right) / 2))
        yMedio = int(top + abs((top - bottom) / 2))
        cv2.line(frame, (xMedio - 1, yMedio), (xMedio + 1, yMedio), (0, 255, 0), 2)
        cv2.line(frame, (xMedio, yMedio - 1), (xMedio, yMedio + 1), (0, 255, 0), 2)
        # Fin Punto MEDIO

        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces to disk
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

# close the video file pointers
stream.release()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()

cv2.destroyAllWindows()