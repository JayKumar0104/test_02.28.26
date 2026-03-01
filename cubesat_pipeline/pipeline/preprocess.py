import cv2
import numpy as np
from typing import Optional


def preprocess_frame(img_bgr, cfg: dict, force_resize_width: Optional[int] = None):

    resize_w = (
        int(force_resize_width)
        if force_resize_width is not None
        else int(cfg["preprocess"].get("resize_width", 0))
    )

    if resize_w and img_bgr.shape[1] != resize_w:
        h = int(img_bgr.shape[0] * (resize_w / img_bgr.shape[1]))
        img_bgr = cv2.resize(img_bgr, (resize_w, h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clip = float(cfg["preprocess"].get("clahe_clip", 3.0))
    grid = cfg["preprocess"].get("clahe_grid", [8, 8])

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(int(grid[0]), int(grid[1]))
    )

    enhanced = clahe.apply(gray)

    k = int(cfg["preprocess"].get("blur_ksize", 5))
    if k % 2 == 0:
        k += 1

    denoised = cv2.GaussianBlur(enhanced, (k, k), 0)

    sigma = float(cfg["hazards"].get("canny_sigma", 0.35))
    v = float(np.median(denoised))

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(denoised, lower, upper)

    return img_bgr, gray, enhanced, denoised, edges, (lower, upper)
