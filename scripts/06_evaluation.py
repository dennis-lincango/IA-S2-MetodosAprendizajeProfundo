import os
import cv2
import numpy as np
import face_recognition
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import os

# ============================
# CONFIG
# ============================
ENCODINGS_PATH = "../models/face_encodings.pkl"
TEST_DIR = "../test_dataset"
# FOLDER = "../test_dataset/dennis_lincango"


DETECTION_METHOD = "hog"
THRESHOLD = 0.8

# for file in os.listdir(FOLDER):
#     path = os.path.join(FOLDER, file)
#     img = Image.open(path).convert("RGB")
#     img.save(path.replace(".jpeg", "_fixed.jpg"), "JPEG")
#

# ============================
# LOAD MODEL
# ============================
print("[INFO] Cargando encodings...")
data = pickle.loads(open(ENCODINGS_PATH, "rb").read())

y_true = []
y_pred = []

# ============================
# PROCESS TEST IMAGES
# ============================
print("[INFO] Evaluando dataset...")

if not os.path.isdir(TEST_DIR):
    print(f"[ERROR] La carpeta '{TEST_DIR}' no existe.")
    exit()

for person_name in os.listdir(TEST_DIR):
    person_folder = os.path.join(TEST_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] No se pudo leer la imagen: {img_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # No detecta rostro → unknown
        if len(encodings) == 0:
            y_true.append(person_name)
            y_pred.append("unknown")
            continue

        encoding = encodings[0]

        distances = face_recognition.face_distance(data["encodings"], encoding)
        name = "unknown"

        if len(distances) > 0:
            idx = np.argmin(distances)
            if distances[idx] < THRESHOLD:
                name = data["names"][idx]

        y_true.append(person_name)
        y_pred.append(name)

# ============================
# VALIDACIONES IMPORTANTES
# ============================
if len(y_true) == 0:
    print("[ERROR] No se procesó ninguna imagen. Revisa el dataset.")
    print("Estructura esperada:")
    print("""
    test_dataset/
        Persona1/
            img1.jpg
            img2.jpg
        Persona2/
            imgX.jpg
        unknown/
            cualquierImagen.jpg
    """)
    exit()

if len(set(y_true)) == 0:
    print("[ERROR] No hay clases válidas en el dataset.")
    exit()

# ============================
# MÉTRICAS
# ============================
print("\n===== REPORT =====")
print(classification_report(y_true, y_pred, zero_division=0))

# print("\n===== CONFUSION MATRIX =====")
# print(confusion_matrix(y_true, y_pred))
