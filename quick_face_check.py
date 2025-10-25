 # quick_face_check.py (v2 robusta)
import argparse
import sys
import time
import os
import glob

import cv2
import numpy as np

# ---- Opcional: usar Pillow como lector alternativo para JPEGs problemáticos
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# ---- face_recognition (dlib)
try:
    import face_recognition
    HAS_FR = True
except Exception as e:
    print("[AVISO] No se pudo importar face_recognition (dlib). "
          "Se usará el detector Haar de OpenCV como fallback.\n"
          f"Detalle: {e}\n")
    HAS_FR = False


def to_rgb_uint8(img_any):
    """
    Asegura que la imagen sea RGB de 3 canales, dtype=uint8 y contigua en memoria.
    Maneja casos: GRAY, BGR, BGRA, PIL->RGB, etc.
    """
    if img_any is None:
        raise RuntimeError("Imagen nula.")

    # Si vino como PIL Image
    if HAS_PIL and isinstance(img_any, Image.Image):
        if img_any.mode != "RGB":
            img_any = img_any.convert("RGB")
        rgb = np.array(img_any, dtype=np.uint8)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    # Debe ser ndarray
    if not isinstance(img_any, np.ndarray):
        raise RuntimeError("Tipo de imagen no soportado (no es numpy array).")

    if img_any.dtype != np.uint8:
        img_any = img_any.astype(np.uint8)

    if len(img_any.shape) == 2:
        # GRAY -> RGB
        rgb = cv2.cvtColor(img_any, cv2.COLOR_GRAY2RGB)
    else:
        c = img_any.shape[2]
        if c == 3:
            # BGR -> RGB
            rgb = cv2.cvtColor(img_any, cv2.COLOR_BGR2RGB)
        elif c == 4:
            # BGRA -> RGB
            rgb = cv2.cvtColor(img_any, cv2.COLOR_BGRA2RGB)
        else:
            raise RuntimeError(f"Cantidad de canales no soportada: {c}")

    return np.ascontiguousarray(rgb, dtype=np.uint8)


def detect_with_face_recognition(bgr_like, model="hog", upsample=1):
    rgb = to_rgb_uint8(bgr_like)
    boxes = face_recognition.face_locations(rgb,
                                            number_of_times_to_upsample=upsample,
                                            model=model)
    # (top, right, bottom, left)
    return boxes


def detect_with_haar(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(60, 60))
    boxes = []
    for (x, y, w, h) in rects:
        top, left, bottom, right = y, x, y + h, x + w
        boxes.append((top, right, bottom, left))
    return boxes


def detect_faces(bgr, backend="hog"):
    """
    backend: 'hog' | 'cnn' | 'haar'
    """
    if backend == "haar" or not HAS_FR:
        return detect_with_haar(bgr), "OpenCV Haar cascade"

    # dlib (hog/cnn)
    try:
        boxes = detect_with_face_recognition(bgr, model=backend)
        return boxes, f"face_recognition ({backend})"
    except Exception as e:
        # Fallback inmediato a Haar
        print(f"[AVISO] Problema con face_recognition ({backend}): {e}. Usando Haar.")
        return detect_with_haar(bgr), "OpenCV Haar cascade"


def draw_boxes(img, boxes, color=(0, 255, 0), label="face"):
    for (top, right, bottom, left) in boxes:
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, label, (left, max(0, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def robust_imread(path):
    """
    Lee imagen de forma robusta:
    - Primero con OpenCV.
    - Si falla o da None, intenta con Pillow (si disponible).
    Devuelve ndarray BGR (como cv2.imread) o lanza error.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img

    if HAS_PIL:
        try:
            im = Image.open(path)
            # Convertimos a RGB y luego a BGR para mantener coherencia visual con OpenCV
            if im.mode != "RGB":
                im = im.convert("RGB")
            rgb = np.array(im, dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception as e:
            raise RuntimeError(f"No pude abrir con Pillow: {e}")

    raise RuntimeError("cv2.imread devolvió None y Pillow no está disponible.")


def run_on_image(path, backend="hog"):
    # Normaliza y ayuda si no hay extensión
    path = os.path.expanduser(path)
    path = os.path.normpath(path)

    root, ext = os.path.splitext(path)
    tried = []
    if ext == "":
        for candidate_ext in [".jpg", ".jpeg", ".png"]:
            cand = root + candidate_ext
            tried.append(cand)
            if os.path.isfile(cand):
                path = cand
                break

    if not os.path.isfile(path):
        if os.path.isdir(path):
            imgs = []
            for pat in ("*.jpg", "*.jpeg", "*.png"):
                imgs.extend(glob.glob(os.path.join(path, pat)))
            if imgs:
                path = imgs[0]
            else:
                print(f"[ERROR] Carpeta sin imágenes soportadas: {path}")
                sys.exit(1)
        else:
            print(f"[ERROR] No encontré el archivo: {path}")
            if tried:
                print("       Intenté también:", ", ".join(tried))
            sys.exit(1)

    try:
        img = robust_imread(path)
    except Exception as e:
        print(f"[ERROR] No pude leer la imagen: {path}\n       {e}")
        print("       Posibles causas: JPEG corrupto/CMYK, HEIC sin códecs, OneDrive solo en la nube.")
        sys.exit(1)

    t0 = time.time()
    boxes, used_backend = detect_faces(img, backend=backend)
    dt = (time.time() - t0) * 1000.0

    draw_boxes(img, boxes, label="face")
    cv2.putText(img, f"{used_backend} | {len(boxes)} rostro(s) | {dt:.1f} ms",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

    print(f"[OK] Detectados {len(boxes)} rostro(s) con {used_backend} en {dt:.1f} ms")
    print(f"[INFO] Imagen abierta: {path}")
    cv2.imshow("Resultado (presiona una tecla para cerrar)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_camera(src=0, backend="hog"):
    # En Windows, CAP_DSHOW suele ser más estable
    try:
        # Comprueba el sistema operativo
        if sys.platform.system() == 'Windows':
            # Usa CAP_DSHOW solo en Windows
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            # Usa el backend predeterminado en macOS/Linux
            cap = cv2.VideoCapture(src)
    except Exception:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] No pude abrir la cámara/fuente: {src}")
        sys.exit(1)

    # Fuerza conversión a RGB en backend cuando sea posible
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    # Resolución opcional
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Presiona 'q' para salir.")
    fps_avg = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] No se pudo leer frame.")
            break

        t0 = time.time()
        boxes, used_backend = detect_faces(frame, backend=backend)
        dt = (time.time() - t0)
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_avg = fps if fps_avg is None else (0.9 * fps_avg + 0.1 * fps)

        draw_boxes(frame, boxes, label="face")
        cv2.putText(frame, f"{used_backend} | faces: {len(boxes)} | FPS: {fps_avg:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("Webcam - reconocimiento/detección rápida", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Chequeo rápido de librerías de visión (inspirado en PyImageSearch)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", type=str, help="Ruta a imagen para prueba.")
    group.add_argument("--camera", "-c", nargs="?", const="0", help="Índice de cámara o ruta de video (por defecto 0).")
    parser.add_argument("--backend", "-b", type=str, default="hog",
                        choices=["hog", "cnn", "haar"],
                        help="Backend de detección: dlib 'hog' (CPU), dlib 'cnn' (más preciso), o 'haar' (OpenCV).")
    args = parser.parse_args()

    print(f"[INFO] OpenCV: {cv2.__version__}")
    if HAS_FR:
        print("[INFO] face_recognition disponible (dlib).")
    else:
        print("[INFO] face_recognition NO disponible. Usando OpenCV Haar cascade.")

    if args.image:
        run_on_image(args.image, backend=args.backend)
    else:
        src = args.camera
        try:
            src = int(src)
        except:
            pass
        run_on_camera(src, backend=args.backend)


if __name__ == "__main__":
    main()
