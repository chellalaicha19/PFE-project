#!/usr/bin/env python3
"""
Pi5 onboard pipeline stress test.

Runs continuous (or flight-paced) dual-camera capture + onboard inference on
the Raspberry Pi 5 for a sustained duration, while independently logging
thermal/CPU/memory metrics. This is meant to surface real hardware behavior
that synthetic per-stage benchmarks can't show: USB bandwidth contention
between the RGB and TC002C feeds, sustained FPS under live I/O (not
pre-recorded frames), memory growth over time, and whether thermal
throttling (>80C) occurs under sustained load now that cooling is in place.

Usage:
    python3 stress_test.py --duration 1200 --mode flight
    python3 stress_test.py --duration 1200 --mode maxrate

--mode flight   paces captures at your real flight trigger interval (1.875s)
--mode maxrate  captures as fast as possible -- a worse-case thermal/load test

Two CSV logs land in ./logs/<timestamp>/:
    system_metrics.csv   -- sampled every SYSTEM_SAMPLE_INTERVAL seconds,
                             independent of the inference loop
    pipeline_metrics.csv -- one row per processed frame pair

Fill in the MODEL LOADING and INFERENCE sections below with your actual
model paths and call signatures (YOLO/NCNN, RGB binary ONNX, thermal binary).
Everything else should run as-is.
"""

import argparse
import csv
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter
# ------------------------------------------------------------------
# CONFIG -- EDIT THESE
# ------------------------------------------------------------------
RGB_DEVICE = "/dev/video2"          # confirm with `v4l2-ctl --list-devices`
THERMAL_DEVICE = "/dev/video0"      # TC002C, YUYV, 256x192
THERMAL_W, THERMAL_H = 256, 192

YOLO_MODEL_PATH = "/home/pi5/Documents/panel_detection2/best_ncnn_model"        # NCNN
RGB_BINARY_MODEL_PATH = "/home/pi5/Documents/binary_class/onnx_128/classifier_opt.onnx"     # ONNX Runtime
THERMAL_BINARY_MODEL_PATH = "/home/pi5/Documents/thermal_binary/thermal_binary_mobilenetv2_f16.tflite"

FLIGHT_TRIGGER_INTERVAL_S = 1.875   # matches your 1.5m / 0.8m/s flight plan
SYSTEM_SAMPLE_INTERVAL_S = 2.0      # independent watchdog sampling rate

THROTTLE_TEMP_C = 80.0
# ------------------------------------------------------------------


def get_cpu_temp_c():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return float("nan")


def get_throttle_flags():
    """Decode vcgencmd get_throttled bitmask.
    bit0 under-voltage now, bit1 freq capped now, bit2 throttled now,
    bit3 soft temp limit now, bit18 throttled since boot,
    bit19 soft temp limit since boot."""
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()
        val = int(out.split("=")[1], 16)
        return {
            "under_voltage_now": bool(val & 0x1),
            "freq_capped_now": bool(val & 0x2),
            "throttled_now": bool(val & 0x4),
            "soft_temp_limit_now": bool(val & 0x8),
            "throttled_since_boot": bool(val & 0x40000),
            "soft_temp_limit_since_boot": bool(val & 0x80000),
        }
    except Exception:
        return {}


def get_cpu_freq_mhz():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_clock", "arm"]).decode()
        return int(out.strip().split("=")[1]) / 1e6
    except Exception:
        return float("nan")


class SystemMonitor(threading.Thread):
    """Background watchdog: samples temp/throttle/freq/memory independent of
    whatever the inference loop is doing, so you still get clean data even
    if the pipeline stalls or crashes mid-run."""

    def __init__(self, log_path, interval_s, stop_event):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.interval_s = interval_s
        self.stop_event = stop_event
        self.proc = psutil.Process(os.getpid())

    def run(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "elapsed_s", "cpu_temp_c", "cpu_freq_mhz",
                "throttled_now", "soft_temp_limit_now", "throttled_since_boot",
                "process_rss_mb", "system_mem_used_pct", "system_cpu_pct",
            ])
            start = time.time()
            while not self.stop_event.is_set():
                t = time.time() - start
                temp = get_cpu_temp_c()
                freq = get_cpu_freq_mhz()
                flags = get_throttle_flags()
                rss_mb = self.proc.memory_info().rss / (1024 * 1024)
                mem_pct = psutil.virtual_memory().percent
                cpu_pct = psutil.cpu_percent(interval=None)
                writer.writerow([
                    f"{t:.1f}", temp, freq,
                    flags.get("throttled_now"), flags.get("soft_temp_limit_now"),
                    flags.get("throttled_since_boot"),
                    f"{rss_mb:.1f}", mem_pct, cpu_pct,
                ])
                f.flush()
                if temp >= THROTTLE_TEMP_C:
                    print(f"[WARN] t={t:.0f}s  CPU temp {temp:.1f}C >= {THROTTLE_TEMP_C}C threshold")
                self.stop_event.wait(self.interval_s)


# ------------------------------------------------------------------
# MODEL LOADING / INFERENCE -- replace bodies with your real code
# ------------------------------------------------------------------
def load_models():
    """Load YOLO (NCNN), RGB binary (ONNX), thermal binary (TFLite).
    Keep the same return shape (a 3-tuple) when you swap in real loaders."""
    yolo = YOLO(YOLO_MODEL_PATH)             # e.g. ncnn.Net(); yolo.load_param(...); yolo.load_model(...)
    rgb_binary = ort.InferenceSession(RGB_BINARY_MODEL_PATH)       # e.g. onnxruntime.InferenceSession(RGB_BINARY_MODEL_PATH)
    thermal_binary = Interpreter(model_path=THERMAL_BINARY_MODEL_PATH)
    thermal_binary.allocate_tensors()   # e.g. tflite.Interpreter(model_path=THERMAL_BINARY_MODEL_PATH)
    return yolo, rgb_binary, thermal_binary


def run_yolo_detect_crop(yolo, rgb_frame):
    """Replace with real NCNN inference + crop extraction.
    Must return (list_of_crops, elapsed_seconds)."""
    t0 = time.time()
    results = yolo(rgb_frame, verbose=False)
    # TODO: real inference
    crops = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)
        crops.append(rgb_frame[y1:y2, x1:x2])
    return crops, time.time() - t0


def run_rgb_binary(rgb_binary, crop):
    """Must return (result, elapsed_seconds)."""
    t0 = time.time()
    # TODO: real inference
    img = cv2.resize(crop, (128, 128)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # adjust if your training used HWC, not CHW
    input_name = rgb_binary.get_inputs()[0].name
    result = rgb_binary.run(None, {input_name: img})
    return result, time.time() - t0


def run_thermal_binary(thermal_binary, thermal_frame):
    """Must return (result, elapsed_seconds)."""
    t0 = time.time()
    # TODO: real inference
    img = cv2.resize(thermal_frame, (224, 224)).astype(np.float32) / 255.0  # confirm size matches training
    img = img[None, ...]
    input_details = thermal_binary.get_input_details()
    output_details = thermal_binary.get_output_details()
    thermal_binary.set_tensor(input_details[0]['index'], img)
    thermal_binary.invoke()
    result = thermal_binary.get_tensor(output_details[0]['index'])
    return result, time.time() - t0
# ------------------------------------------------------------------


def open_cameras():
    rgb_cap = cv2.VideoCapture(RGB_DEVICE)
    thermal_cap = cv2.VideoCapture(THERMAL_DEVICE)
    thermal_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    thermal_cap.set(cv2.CAP_PROP_FRAME_WIDTH, THERMAL_W)
    thermal_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, THERMAL_H)

    if not rgb_cap.isOpened():
        raise RuntimeError(f"Could not open RGB camera at {RGB_DEVICE}")
    if not thermal_cap.isOpened():
        raise RuntimeError(f"Could not open thermal camera at {THERMAL_DEVICE}")
    return rgb_cap, thermal_cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1200,
                         help="Test duration in seconds (default 1200 = 20 min)")
    parser.add_argument("--mode", choices=["flight", "maxrate"], default="flight",
                         help="flight = pace captures at the real trigger interval; "
                              "maxrate = capture as fast as possible (worst-case load)")
    args = parser.parse_args()

    run_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {run_dir}/  (mode={args.mode}, duration={args.duration}s)")

    stop_event = threading.Event()
    monitor = SystemMonitor(run_dir / "system_metrics.csv", SYSTEM_SAMPLE_INTERVAL_S, stop_event)
    monitor.start()

    rgb_cap, thermal_cap = open_cameras()
    yolo, rgb_binary, thermal_binary = load_models()

    pipeline_log_path = run_dir / "pipeline_metrics.csv"
    with open(pipeline_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elapsed_s", "frame_idx", "rgb_capture_ms", "thermal_capture_ms",
            "yolo_ms", "n_crops", "rgb_binary_ms", "thermal_binary_ms",
            "loop_total_ms", "instant_fps",
        ])

        start = time.time()
        frame_idx = 0
        try:
            while time.time() - start < args.duration:
                loop_t0 = time.time()

                t0 = time.time()
                ok_rgb, rgb_frame = rgb_cap.read()
                rgb_capture_ms = (time.time() - t0) * 1000

                t0 = time.time()
                ok_thermal, thermal_frame = thermal_cap.read()
                thermal_capture_ms = (time.time() - t0) * 1000

                if not ok_rgb or not ok_thermal:
                    print(f"[WARN] frame grab failed at t={time.time()-start:.1f}s "
                          f"(rgb={ok_rgb}, thermal={ok_thermal})")
                    continue

                crops, yolo_s = run_yolo_detect_crop(yolo, rgb_frame)

                rgb_binary_ms_total = 0.0
                for crop in crops:
                    _, s = run_rgb_binary(rgb_binary, crop)
                    rgb_binary_ms_total += s * 1000

                _, thermal_binary_s = run_thermal_binary(thermal_binary, thermal_frame)

                loop_total_ms = (time.time() - loop_t0) * 1000
                instant_fps = 1000.0 / loop_total_ms if loop_total_ms > 0 else float("inf")

                writer.writerow([
                    f"{time.time()-start:.2f}", frame_idx,
                    f"{rgb_capture_ms:.1f}", f"{thermal_capture_ms:.1f}",
                    f"{yolo_s*1000:.1f}", len(crops),
                    f"{rgb_binary_ms_total:.1f}", f"{thermal_binary_s*1000:.1f}",
                    f"{loop_total_ms:.1f}", f"{instant_fps:.2f}",
                ])
                f.flush()

                if frame_idx % 50 == 0:
                    print(f"t={time.time()-start:6.1f}s  frame={frame_idx:5d}  "
                          f"fps={instant_fps:5.2f}  loop={loop_total_ms:6.1f}ms")

                frame_idx += 1

                if args.mode == "flight":
                    elapsed = time.time() - loop_t0
                    sleep_left = FLIGHT_TRIGGER_INTERVAL_S - elapsed
                    if sleep_left > 0:
                        time.sleep(sleep_left)

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            stop_event.set()
            monitor.join(timeout=5)
            rgb_cap.release()
            thermal_cap.release()

    print(f"\nDone. Logs in {run_dir}/")
    print(f"Next: python3 analyze_run.py {run_dir}/")


if __name__ == "__main__":
    main()
