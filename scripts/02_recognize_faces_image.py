# USAGE
# python scripts/02_recognize_faces_image.py --encodings models/face_encodings.pkl --image test/files/images/example_01.jpeg

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
    model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# umbral de decisión para aceptar un match
# más bajo = más estricto (más Unknown), más alto = más permisivo
THRESHOLD = 0.48

# loop over the facial embeddings
for encoding in encodings:
    # calcular distancias entre esta cara y todas las caras conocidas
    face_distances = face_recognition.face_distance(data["encodings"], encoding)

    # por defecto es desconocido
    name = "unknown"

    if len(face_distances) > 0:
        # índice del mejor match (menor distancia)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        # si la mejor distancia es menor al umbral, aceptamos el match
        if best_distance < THRESHOLD:
            name = data["names"][best_match_index]

    # actualizar la lista de nombres
    names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 255, 0), 2)

def resize_with_aspect_ratio(image, max_width=1000, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:  # solo reducir, nunca aumentar
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

# show the output image
image = resize_with_aspect_ratio(image)
cv2.imshow("Image", image)
cv2.waitKey(0)
