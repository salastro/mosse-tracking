# MOSSE Tracker with CLI

This project implements a **Minimum Output Sum of Squared Error (MOSSE) correlation filter tracker** in Python using OpenCV and NumPy. It allows tracking of one or multiple user-selected objects in a video. The tracker operates in real time, using FFT-based correlation for efficiency.

---

## Features

* Multiple object tracking (select multiple ROIs in the first frame).
* Adaptive filter update for robustness to appearance changes.
* Real-time performance using FFT-based convolution.
* Optional visualization of the FFT spectrum alongside the video.
* Command-line interface with customizable parameters.

---

## Requirements

* Python 3.8+
* OpenCV
* NumPy
* SciPy

---

## Usage

Run the tracker from the command line:

```bash
python src/main.py -i input.mp4 -o output.mp4 --eta 0.125 --sigma 100 --show-spectrum
```

### Arguments

| Argument          | Description                                                           | Default      |
| ----------------- | --------------------------------------------------------------------- | ------------ |
| `-i`, `--input`   | Path to the input video (required).                                   | —            |
| `-o`, `--output`  | Path to save the output video.                                        | `result.mp4` |
| `-e`, `--eta`     | Learning rate for adaptive filter update.                             | `0.125`      |
| `-s`, `--sigma`   | Gaussian sigma used in filter initialization.                         | `100.0`      |
| `--show-spectrum` | Display FFT spectrum of the first tracker alongside the video stream. | `False`      |

---

## How it Works

1. **Initialization**

   * The first frame is shown and you manually select one or more objects (ROIs).
   * Each ROI is used to train an initial correlation filter in the frequency domain.

2. **Tracking**

   * For each new frame, the filter is applied via FFT to produce a response map.
   * The peak in the response map gives the new object location.

3. **Update**

   * The filter is updated with a running average using `eta`.
   * This allows adaptation to gradual changes in lighting and appearance.

4. **Visualization**

   * Bounding boxes are drawn around tracked objects.
   * If `--show-spectrum` is enabled, the FFT spectrum of the first object is shown alongside the video.

---

## Controls

* **ROI selection**: Select bounding boxes with the mouse in the first frame window. Press Enter to confirm selection. Escape when done.
* **Quit**: Press `q` during playback to stop tracking early.

---

## Output

* Annotated video with tracked objects is saved to the specified output path.
* Console output confirms when tracking is complete.

---

## Reference

This implementation is based on:
**D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui. “Visual Object Tracking using Adaptive Correlation Filters.” CVPR 2010.**

