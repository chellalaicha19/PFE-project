# PV Fusion Console

Local ground-station demo: point it at a folder of anomaly frames from the
Pi 5 (binary stage already run onboard) and it runs the multiclass + fusion
stage and shows results in a browser instead of a terminal.

## Setup (on your Mac)

```bash
cd pv_webapp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open http://127.0.0.1:5001

## Usage

Paste a folder path into the box, e.g.:

```
/Users/mac/Documents/good/fly
```

It accepts either:
- a parent folder full of `frame_XXXXXX/` subfolders, or
- a single `frame_XXXXXX/` folder directly

Each frame folder must contain `rgb.jpg` and `thermal.jpg` (`metadata.txt`
and `annotated.jpg` are read if present, but not required).

## What it does per frame

1. Runs YOLO panel detection (`YOLO_PATH` in `core_inference.py`) on
   `rgb.jpg`, crops up to 2 panels (falls back to the full frame if nothing
   is detected)
2. Runs the RGB EfficientNet-B1 multiclass model on each crop
3. Runs the thermal EfficientNet-B1 (TFLite) multiclass model once on the
   full `thermal.jpg`
4. Fuses each panel's RGB reading with the frame's thermal reading through
   the same `FUSION_RULES` table as `fusion2.py`, including the low-confidence
   and thermal near-tie checks

## Model paths

Edit the constants at the top of `core_inference.py` if any model moves:

```python
YOLO_PATH           = "/Users/mac/Documents/PFE/panel_detection/yolov11-1000/best.pt"
RGB_MULTI_PATH       = "/Users/mac/Documents/PFE/rgb_multicalssification/best_rgb_multiclass_final_last.pt"
THERMAL_MULTI_PATH   = "/Users/mac/Documents/PFE/thermal_multiclassification/thermal_multiclass_efficientnetb1_f16.tflite"
```

Models load once, lazily, on the first "Process folder" click — the first
run will be slow (loading YOLO + torch + TFLite), subsequent folders on the
same server run are fast.

## Notes / things worth checking before your demo

- Binary models are **not loaded** here — but the binary *verdicts* from the
  Pi are read straight out of `metadata.txt` (`Thermal Label:`, and the
  per-panel `Panel N: Healthy/Anomaly (...)` lines). A modality is only sent
  through its multiclass model if the Pi flagged it as anomaly; if it was
  Healthy / NO_ANOMALY, that label is used directly in fusion with the Pi's
  own confidence — matching your two-tier architecture exactly (multiclass
  only runs on what tripped the binary stage).
- Each panel card shows a small `binary` tag next to a reading when that
  modality was skipped rather than multiclassified, so it's visible during
  the demo which path each panel/modality took.
- Thermal binary is frame-level (one thermal image per frame), so it's
  evaluated once per frame, not per panel.
- RGB panel numbering assumes this ground-station YOLO pass finds the same
  panels in the same confidence order the Pi did (both sort by descending
  YOLO confidence). If YOLO here detects a different number of panels than
  `metadata.txt` lists, the unmatched panel falls back to running RGB
  multiclass rather than guessing — check the console output on frames where
  that happens.
- If `metadata.txt` is missing or unparsable for a frame, everything falls
  back to always running both multiclass models (the old behavior), so the
  tool still works on frames without proper metadata.
- If YOLO finds no panel at all, the RGB multiclass model gets the full
  `rgb.jpg` instead of a crop — same fallback as `fusion2.py`.
