import cv2
import numpy as np
from typing import List, Tuple

def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img_gray: np.ndarray,
           method: str = "gaussian",
           ksize: int = 5) -> np.ndarray:
    if method == "gaussian":
        return cv2.GaussianBlur(img_gray, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(img_gray, ksize)
    else:
        raise ValueError(f"Unknown denoise method: {method}")

def subtract_background(img_gray: np.ndarray,
                        blur_size: int = 51) -> np.ndarray:
    bg = cv2.medianBlur(img_gray, blur_size)
    return cv2.subtract(img_gray, bg)

def equalize_contrast(img_gray: np.ndarray,
                      clip_limit: float = 1.0,
                      grid_size: Tuple[int,int] = (16,16)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img_gray)

def resize_with_padding(img_gray: np.ndarray,
                        target_size: Tuple[int,int] = (512,512),
                        pad_value: int = 128) -> np.ndarray:
    h, w = img_gray.shape
    th, tw = target_size
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_w, pad_h = tw - nw, th - nh
    top, bottom = pad_h//2, pad_h - pad_h//2
    left, right = pad_w//2, pad_w - pad_w//2
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=pad_value)

def preprocess_image(img: np.ndarray,
                     denoise_method: str = "gaussian",
                     do_bg_subtract: bool = False,
                     clahe_clip: float = 1.0,
                     clahe_grid: Tuple[int,int] = (16,16),
                     target_size: Tuple[int,int] = (512,512)) -> np.ndarray:
    gray = to_grayscale(img)
    den = denoise(gray, method=denoise_method)
    fg = subtract_background(den) if do_bg_subtract else den
    cla = equalize_contrast(fg, clip_limit=clahe_clip, grid_size=clahe_grid)
    final = resize_with_padding(cla, target_size)
    return final

def preprocess_batch(images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    return [preprocess_image(img, **kwargs) for img in images]
