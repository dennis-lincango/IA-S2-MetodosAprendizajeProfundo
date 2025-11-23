# USAGE
# python scripts/01_encode_faces.py --dataset data --encodings models/face_encodings.pkl

# import the necessary packages
from imutils import paths
import face_recognition

import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# maximum width to resize images (to speed processing)
MAX_WIDTH = 800

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    try:
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

        # extract the person name
        name = imagePath.split(os.path.sep)[-2]

        # load image
        image = cv2.imread(imagePath)
        if image is None:
            print(f"[WARN] No se pudo leer la imagen: {imagePath}. Saltando...")
            continue

        # resize if too large
        if image.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / image.shape[1]
            new_dim = (MAX_WIDTH, int(image.shape[0] * scale))
            image = cv2.resize(image, new_dim)
            print(f"[INFO] Imagen redimensionada por ser muy grande.")

        # convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect face locations
        boxes = face_recognition.face_locations(
            rgb,
            model=args["detection_method"]
        )

        # compute facial encodings
        encodings = face_recognition.face_encodings(rgb, boxes)

        # store encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    except Exception as e:
        print(f"[ERROR] Problema con la imagen {imagePath}: {e}. Saltando...")
        continue

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))