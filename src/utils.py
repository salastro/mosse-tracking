import cv2
import numpy as np
from scipy.signal import windows


def window2(N, M, w_func):
    wc = w_func(N)
    wr = w_func(M)
    maskr, maskc = np.meshgrid(wr, wc)
    return maskr * maskc


def preprocess(img):
    r, c = img.shape
    win = window2(r, c, windows.hann)
    eps = 1e-5
    img_out = np.log(img.astype(np.float32) + 1)
    img_out = (img_out - img_out.mean()) / (img_out.std() + eps)
    img_out = img_out * win
    return img_out


def gaussC(x, y, sigma, center):
    xc, yc = center
    exponent = ((x - xc) ** 2 + (y - yc) ** 2) / (2 * sigma)
    return np.exp(-exponent)


def fft_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude


def clamp_rect(rect, frame_shape):
    x, y, w, h = rect
    height, width = frame_shape[:2]

    w = max(1, min(int(w), width))
    h = max(1, min(int(h), height))
    x = max(0, min(int(x), width - w))
    y = max(0, min(int(y), height - h))
    return (x, y, w, h)
