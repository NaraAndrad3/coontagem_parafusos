import cv2
import numpy as np
from pathlib import Path

from src.config import CFG
from src.visualization import annotate_image, draw_heads

def load_and_resize(path: str) -> np.ndarray:
    """
    Carrega e redimensiona a imagem mantendo proporção.
    
    Parâmetro 800px: mantém detalhe suficiente para detectar parafusos
    pequenos enquanto mantém processamento viável em smartphone.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    h, w = img.shape[:2]
    mx = CFG["resize_max"]
    if max(h, w) > mx:
        scale = mx / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Gaussian Blur 5×5 para remoção de ruído de superfície metálica.
    
    Nota: CLAHE foi testado e removido pois em imagens com fundo
    branco muito claro (img3) ele une o fundo com os objetos em
    um único blob. Gaussian Blur preserva contraste global sem
    introduzir artefatos de equalização local.
    """
    return cv2.GaussianBlur(img, CFG["blur_kernel"], 0)


def binarize(img_proc: np.ndarray) -> np.ndarray:
    """
    Binarização por Otsu invertido.
    
    Diagnóstico das 5 imagens: parafusos são consistentemente mais
    escuros que o fundo → THRESH_BINARY_INV mapeia objetos para 255.
    
    Otsu: algoritmo de maximização de variância inter-classe que
    encontra automaticamente o limiar ótimo para cada imagem.
    Elimina necessidade de ajuste manual de parâmetros.
    """
    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def apply_morphology(binary):

    def k(size):
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )

    # apenas limpeza leve

    mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        k(3),
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        k(5),
        iterations=1
    )

    return mask


def extract_contours(mask: np.ndarray, img_area: int) -> list[dict]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []

    h, w = mask.shape[:2]

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        if area < CFG["min_area"]:
            continue

        if area > 0.12 * img_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Remove regiões grudadas nas bordas da imagem
        margin = 3

        touches_border = (
            x <= margin or
            y <= margin or
            x + bw >= w - margin or
            y + bh >= h - margin
        )
       
        if touches_border:
            continue

        if bw < 20 or bh < 20:
            continue

        if len(cnt) < 5:
            continue

        _, (minor, major), _ = cv2.fitEllipse(cnt)

        if minor < 1:
            continue

        aspect = major / minor

        if not (1.4 <= aspect <= 12.0):
            continue

        hull_area = cv2.contourArea(cv2.convexHull(cnt))

        if hull_area < 1:
            continue

        solidity = area / hull_area

        if solidity < 0.35:
            continue

        rect_area = bw * bh
        extent = area / rect_area

        if extent < 0.18 or extent > 0.85:
            continue

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        blobs.append({
            "contour": cnt,
            "area": area,
            "aspect": round(aspect, 2),
            "solidity": round(solidity, 2),
            "extent": round(extent, 2),
            "bbox": (x, y, bw, bh),
            "center": (cx, cy),
        })

    return blobs


def watershed_count(mask_roi: np.ndarray) -> int:
    dist = cv2.distanceTransform(
        mask_roi,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

    if dist.max() == 0:
        return 1

    thresh_val = CFG["ws_dist_frac"] * dist.max()

    _, sure_fg = cv2.threshold(
        dist,
        thresh_val,
        255,
        cv2.THRESH_BINARY
    )

    sure_fg = sure_fg.astype(np.uint8)

    n_labels, _ = cv2.connectedComponents(sure_fg)

    watershed_estimate = max(1, n_labels - 1)

    return watershed_estimate

def estimate_total(blobs: list[dict], mask: np.ndarray) -> tuple[int, list[dict]]:
    if not blobs:
        return 0, []

    areas = np.array([b["area"] for b in blobs])
    median_area = float(np.median(areas))

    CFG["median_area_reference"] = median_area

    total_blobs = len(blobs)

    # Caso tipo img1:
    # Se já existem muitos blobs válidos, assumimos que cada blob é 1 parafuso.
    # Isso evita supercontagem por Watershed.
    if total_blobs >= 6:
        for b in blobs:
            b["estimated"] = 1

        return total_blobs, blobs

    total = 0
    annotated = []

    for b in blobs:
        ratio = b["area"] / median_area if median_area > 0 else 1.0

        if ratio >= CFG["overlap_ratio"]:
            x, y, w, h = cv2.boundingRect(b["contour"])

            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(mask.shape[1], x + w + pad)
            y2 = min(mask.shape[0], y + h + pad)

            roi = mask[y1:y2, x1:x2].copy()

            n = watershed_count(roi)
        else:
            n = 1

        b["estimated"] = n
        total += n
        annotated.append(b)

    return total, annotated

def detect_heads(img: np.ndarray) -> list[dict]:
    """
    Detecta cabeças de parafusos.

    Estratégia:
    - usa tons de cinza;
    - melhora contraste local;
    - detecta regiões metálicas compactas;
    - filtra por área, proporção e circularidade.

    A cabeça do parafuso é uma evidência forte porque:
    1 cabeça ≈ 1 parafuso.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    edges = cv2.Canny(blurred, 40, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = img.shape[:2]
    heads = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        if area < CFG.get("head_min_area", 120):
            continue

        if area > CFG.get("head_max_area", 3500):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < 10 or bh < 10:
            continue

        aspect = max(bw, bh) / max(1, min(bw, bh))

        if aspect > CFG.get("head_max_aspect", 2.2):
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity < CFG.get("head_min_circularity", 0.35):
            continue

        # Remove regiões grudadas nas bordas
        margin = 5
        if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
            continue

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        heads.append({
            "contour": cnt,
            "center": (cx, cy),
            "bbox": (x, y, bw, bh),
            "area": area,
            "aspect": round(aspect, 2),
            "circularity": round(circularity, 2),
        })

    return heads

def process_image(path: str) -> dict:
    img = load_and_resize(path)

    h, w = img.shape[:2]
    img_area = h * w

    processed = preprocess(img)
    binary = binarize(processed)
    mask = apply_morphology(binary)

    blobs = extract_contours(mask, img_area)
    body_count, ann = estimate_total(blobs, mask)

    heads = detect_heads(img)
    head_count = len(heads)

    # Regra de decisão híbrida
    final_count = max(body_count, head_count)

    result_img = annotate_image(img, ann, final_count)
    result_img = draw_heads(result_img, heads)

    cv2.putText(
        result_img,
        f"Corpos: {body_count} | Cabecas: {head_count}",
        (12, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2
    )

    return {
        "image": Path(path).name,
        "count": final_count,
        "body_count": body_count,
        "head_count": head_count,
        "blobs": len(ann),
        "heads": len(heads),
        "result_img": result_img,
        "stages": {
            "original": img,
            "processed": processed,
            "binary": binary,
            "mask_clean": mask,
            "result": result_img,
        }
    }

def process_uploaded_image(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    img_area = h * w

    processed = preprocess(img)
    binary = binarize(processed)
    mask = apply_morphology(binary)

    blobs = extract_contours(mask, img_area)
    body_count, ann = estimate_total(blobs, mask)

    heads = detect_heads(img)
    head_count = len(heads)

    final_count = max(body_count, head_count)

    result_img = annotate_image(img, ann, final_count)
    result_img = draw_heads(result_img, heads)

    cv2.putText(
        result_img,
        f"Corpos: {body_count} | Cabecas: {head_count}",
        (12, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2
    )

    return {
        "image": "uploaded_image",
        "count": final_count,
        "body_count": body_count,
        "head_count": head_count,
        "blobs": len(ann),
        "heads": len(heads),
        "result_img": result_img,
        "stages": {
            "original": img,
            "processed": processed,
            "binary": binary,
            "mask_clean": mask,
            "result": result_img,
        }
    }
