import argparse
import cv2
import numpy as np
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="MOSSE tracking with CLI args")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", default="result.mp4", help="Output video path (default: result.mp4)")
    parser.add_argument("-e", "--eta", type=float, default=0.125, help="Learning rate eta (default: 0.125)")
    parser.add_argument("-s", "--sigma", type=float, default=100.0, help="Gaussian sigma (default: 100)")
    parser.add_argument("--show-spectrum", action="store_true", help="Show FFT spectrum alongside video")
    return parser.parse_args()

def main():
    args = parse_args()
    video_path = args.input
    eta = args.eta
    sigma = args.sigma
    output_path = args.output
    show_spectrum = args.show_spectrum

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    ret, firstFrame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read video")

    # ---- Multiple ROI selection ----
    rois = cv2.selectROIs("Select objects to track", firstFrame, fromCenter=False)
    cv2.destroyAllWindows()
    if len(rois) == 0:
        raise RuntimeError("No ROI selected")

    gsize = firstFrame.shape[:2]
    R, C = np.meshgrid(np.arange(gsize[0]), np.arange(gsize[1]), indexing="ij")

    if firstFrame.ndim == 3:
        grayFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    else:
        grayFrame = firstFrame

    trackers = []
    for rect in rois:
        x, y, w, h = rect
        center = (rect[1] + rect[3] / 2, rect[0] + rect[2] / 2)

        g = gaussC(R, C, sigma, center)
        g = cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX)

        patch = grayFrame[y:y+h, x:x+w]
        g_patch = g[y:y+h, x:x+w]
        G = np.fft.fft2(g_patch)
        hT, wT = g_patch.shape
        fi = preprocess(cv2.resize(patch, (wT, hT)))
        Ai = G * np.conj(np.fft.fft2(fi))
        Bi = np.fft.fft2(fi) * np.conj(np.fft.fft2(fi))

        trackers.append({
            "rect": rect,
            "Ai": Ai,
            "Bi": Bi,
            "G": G,
            "wT": wT,
            "hT": hT,
            "fi": fi
        })

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

        for tracker in trackers:
            rect = tracker["rect"]
            Ai, Bi, G = tracker["Ai"], tracker["Bi"], tracker["G"]
            wT, hT = tracker["wT"], tracker["hT"]

            if frameIdx == 1:
                tracker["Ai"] = eta * Ai
                tracker["Bi"] = eta * Bi
            else:
                H = Ai / Bi
                x, y, w, h = rect
                currPatch = grayFrame[y:y+h, x:x+w]
                if currPatch.size == 0:
                    continue

                fi = preprocess(cv2.resize(currPatch, (wT, hT)))
                response = np.real(np.fft.ifft2(H * np.fft.fft2(fi)))
                respNorm = cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)

                maxR, maxC = np.unravel_index(np.argmax(respNorm), respNorm.shape)
                dx = int(maxR - hT / 2)
                dy = int(maxC - wT / 2)
                rect = (x + dy, y + dx, w, h)
                tracker["rect"] = rect

                newPatch = grayFrame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                if newPatch.size == 0:
                    continue

                fi = preprocess(cv2.resize(newPatch, (wT, hT)))
                tracker["Ai"] = eta * (G * np.conj(np.fft.fft2(fi))) + (1 - eta) * Ai
                tracker["Bi"] = eta * (np.fft.fft2(fi) * np.conj(np.fft.fft2(fi))) + (1 - eta) * Bi
                tracker["fi"] = fi

            # Draw each tracker rect
            displayImg = cv2.rectangle(displayImg, tracker["rect"], (0, 255, 0), 2)

        # Frame index overlay
        annotated = cv2.putText(displayImg, f"Frame: {frameIdx}", (5, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if show_spectrum and len(trackers) > 0:
            spectrum = fft_spectrum(trackers[0]["fi"])  # show spectrum of first tracker
            h_display = annotated.shape[0]
            spectrum = cv2.resize(spectrum, (h_display, h_display))
            spectrum = cv2.cvtColor(spectrum, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((annotated, spectrum))
            cv2.imshow("MOSSE Tracking + FFT", combined)
        else:
            cv2.imshow("MOSSE Tracking", annotated)

        out.write(annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Tracking complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
