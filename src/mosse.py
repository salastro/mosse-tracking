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
    """Compute log magnitude spectrum of FFT."""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude

# ---- Main script ----
video_path = "data/1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"result.mp4", fourcc, fps, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

ret, firstFrame = cap.read()
if not ret:
    raise RuntimeError("Failed to read video")

# Select ROI manually
rect = cv2.selectROI("Select object to track", firstFrame, fromCenter=False)
cv2.destroyAllWindows()

center = (rect[1] + rect[3] / 2, rect[0] + rect[2] / 2)

gsize = firstFrame.shape[:2]
R, C = np.meshgrid(np.arange(gsize[0]), np.arange(gsize[1]), indexing="ij")
sigma = 100
g = gaussC(R, C, sigma, center)
g = cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX)

if firstFrame.ndim == 3:
    grayFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
else:
    grayFrame = firstFrame

x, y, w, h = rect
patch = grayFrame[y:y+h, x:x+w]
g_patch = g[y:y+h, x:x+w]
G = np.fft.fft2(g_patch)
hT, wT = g_patch.shape
fi = preprocess(cv2.resize(patch, (wT, hT)))
Ai = G * np.conj(np.fft.fft2(fi))
Bi = np.fft.fft2(fi) * np.conj(np.fft.fft2(fi))

eta = 0.125
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frameIdx in range(1, frame_count + 1):
    ret, frame = cap.read()
    if not ret:
        break

    displayImg = frame.copy()
    if frame.ndim == 3:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grayFrame = frame

    if frameIdx == 1:
        Ai = eta * Ai
        Bi = eta * Bi
    else:
        H = Ai / Bi
        x, y, w, h = rect
        currPatch = grayFrame[y:y+h, x:x+w]
        if currPatch.size == 0:
            break

        fi = preprocess(cv2.resize(currPatch, (wT, hT)))
        response = np.real(np.fft.ifft2(H * np.fft.fft2(fi)))
        respNorm = cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)

        maxR, maxC = np.unravel_index(np.argmax(respNorm), respNorm.shape)
        dx = int(maxR - hT / 2)
        dy = int(maxC - wT / 2)
        rect = (x + dy, y + dx, w, h)

        newPatch = grayFrame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        if newPatch.size == 0:
            break

        fi = preprocess(cv2.resize(newPatch, (wT, hT)))
        Ai = eta * (G * np.conj(np.fft.fft2(fi))) + (1 - eta) * Ai
        Bi = eta * (np.fft.fft2(fi) * np.conj(np.fft.fft2(fi))) + (1 - eta) * Bi

    # Draw rectangle + frame index
    annotated = cv2.rectangle(displayImg, rect, (0, 255, 0), 2)
    annotated = cv2.putText(annotated, f"Frame: {frameIdx}", (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Compute FFT spectrum of current patch
    spectrum = fft_spectrum(fi)

    # Resize spectrum to match display height
    h_display = annotated.shape[0]
    spectrum = cv2.resize(spectrum, (h_display, h_display))

    # Convert to 3-channel for stacking
    spectrum = cv2.cvtColor(spectrum, cv2.COLOR_GRAY2BGR)

    # Combine: left = annotated video, right = FFT spectrum
    combined = np.hstack((annotated, spectrum))

    cv2.imshow("MOSSE Tracking + FFT", combined)
    out.write(annotated)  # still saving only the tracking stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Tracking complete. Saved to result.mp4")
