import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

def load_face_app(root=".", det_size=640, det_thresh=0.5, modules=("detection", "recognition")):
    """
    Crea el FaceAnalysis de InsightFace. Se llama una sola vez y se reutiliza.

    La primera vez descarga buffalo_l en <root>/models/buffalo_l/.
    modules: "detection" (SCRFD, caja + 5 landmarks) y, opcionalmente,
    "recognition" (ArcFace, embedding de 512 dimensiones).
    """
    app = FaceAnalysis(root=str(root), allowed_modules=list(modules), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(det_size, det_size))
    return app

def detect_faces(app, image_bgr, pad_ratio=0.0):
    """
    Detecta todos los rostros de la imagen, ordenados de mayor a menor área.

    Devuelve una lista de diccionarios con:
      bbox (4,), kps (5, 2), det_score (float) y embedding (512,) normalizado L2,
      o None si "recognition" no está cargado.

    pad_ratio agrega un borde negro (proporcional al lado mayor) antes de detectar,
    útil cuando el rostro ocupa casi toda la imagen. Las coordenadas devueltas
    ya descuentan ese borde.
    """
    pad = int(round(pad_ratio * max(image_bgr.shape[:2])))
    img = image_bgr
    if pad > 0:
        img = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    results = []
    for face in app.get(img):
        results.append({
            "bbox": face.bbox.astype(np.float32) - pad,
            "kps": face.kps.astype(np.float32) - pad,
            "det_score": float(face.det_score),
            "embedding": face.normed_embedding,
        })
    results.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
    return results

def select_main_face(faces, ambiguity_ratio=0.5):
    """
    Elige el rostro principal entre los detectados (ya ordenados por área).

    Devuelve (índice o None, estado):
      "sin_rostro": no hay detecciones.
      "unico":      exactamente un rostro.
      "principal":  varios, pero el más grande domina claramente.
      "ambiguo":    el segundo rostro tiene al menos ambiguity_ratio del área
                    del primero, así que no se sabe a quién corresponde la etiqueta.
    """
    if len(faces) == 0:
        return None, "sin_rostro"
    if len(faces) == 1:
        return 0, "unico"
    areas = [(f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]) for f in faces]
    if areas[1] >= ambiguity_ratio * areas[0]:
        return None, "ambiguo"
    return 0, "principal"

def align_face(image_bgr, kps, face_size=112):
    """
    Recorte alineado face_size x face_size (BGR uint8).

    Transformación de similitud (rotación, escala uniforme y traslación) que lleva
    los 5 landmarks a la plantilla de ArcFace. Es la misma alineación que usa
    FaceAnalysis internamente para calcular el embedding.
    """
    return face_align.norm_crop(image_bgr, landmark=np.asarray(kps, dtype=np.float32), image_size=face_size)