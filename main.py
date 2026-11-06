"""
================================================================================
AI Object Detection and Tracking System
CodeAlpha Artificial Intelligence Internship - Task 4
================================================================================
Technologies:
    - Python 3.x
    - Ultralytics YOLOv8 (Lightweight YOLOv8n pre-trained on COCO dataset)
    - DeepSORT (Deep Simple Online and Realtime Tracking)
    - OpenCV & Pillow for image processing and visualization
    - Tkinter for a responsive, modern Dark-Themed GUI

Features:
    - Real-time webcam capture or video file input
    - Multi-object detection for common COCO classes (person, car, dog, etc.)
    - DeepSORT tracking with Kalman filtering and persistent unique track IDs
    - Smooth motion trajectory trails
    - Class filtering & adjustable confidence threshold
    - Live metrics: Active object count, total tracked IDs, and FPS counter
    - Dual execution mode: Modern GUI or high-performance CLI OpenCV window
================================================================================
"""

import os
import sys
import time
import argparse
import threading
from collections import deque
import urllib.request
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Base project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_videos")
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, YOLO_MODEL_NAME)
SAMPLE_VIDEO_PATH = os.path.join(SAMPLE_DIR, "sample_traffic.mp4")

# Vibrant distinct color palette for tracking IDs
COLOR_PALETTE = [
    (56, 189, 248),   # Sky Blue
    (52, 211, 153),   # Emerald
    (251, 191, 36),   # Amber
    (244, 114, 182),  # Pink
    (167, 139, 250),  # Purple
    (248, 113, 113),  # Red
    (74, 222, 128),   # Green
    (96, 165, 250),   # Blue
    (251, 146, 60),   # Orange
    (45, 212, 191),   # Teal
    (232, 121, 249),  # Fuchsia
    (250, 204, 21),   # Yellow
    (129, 140, 248),  # Indigo
    (255, 128, 128),  # Coral
]

# COCO Class Groups for Filtering
CLASS_GROUPS = {
    "All Classes": None,
    "People Only": ["person"],
    "Vehicles (Cars, Buses, Bikes)": [
        "car", "motorcycle", "bus", "truck", "bicycle", "train", "boat", "airplane"
    ],
    "Animals (Dogs, Cats, etc.)": [
        "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
    ],
    "Common Daily Objects": [
        "person", "car", "bus", "truck", "motorcycle", "bicycle", "dog", "cat",
        "backpack", "handbag", "cell phone", "laptop"
    ],
}


# ==============================================================================
# Helper Functions: Model & Sample Video Downloader
# ==============================================================================

def ensure_model_exists(model_path=YOLO_MODEL_PATH):
    """
    Ensure the lightweight YOLOv8n model exists locally in models/.
    Downloads it automatically from official Ultralytics release assets if missing.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        return model_path

    print(f"[INFO] YOLOv8n model not found in {model_path}.")
    print("[INFO] Automatically downloading pre-trained YOLOv8n model...")
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{YOLO_MODEL_NAME}"
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"[INFO] Download complete! Model saved to: {model_path}")
        return model_path
    except Exception as e:
        print(f"[WARNING] Direct model download failed: {e}")
        print("[INFO] Ultralytics will attempt fallback download on model initialization.")
        return YOLO_MODEL_NAME


def ensure_sample_video():
    """
    Check if a sample video exists in sample_videos/.
    If missing, attempts to download or generate a synthetic video.
    """
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    if os.path.exists(SAMPLE_VIDEO_PATH) and os.path.getsize(SAMPLE_VIDEO_PATH) > 10000:
        return SAMPLE_VIDEO_PATH

    print("[INFO] Sample video not found. Checking generator script...")
    generator_script = os.path.join(SAMPLE_DIR, "generate_sample.py")
    if os.path.exists(generator_script):
        try:
            import subprocess
            subprocess.run([sys.executable, generator_script], check=True)
            if os.path.exists(SAMPLE_VIDEO_PATH):
                return SAMPLE_VIDEO_PATH
        except Exception as e:
            print(f"[WARNING] Could not run sample generator: {e}")

    # Fallback: create a quick synthetic video
    _create_fallback_video(SAMPLE_VIDEO_PATH)
    return SAMPLE_VIDEO_PATH


def _create_fallback_video(output_path):
    """Create a lightweight synthetic demo video for instant verification."""
    print(f"[INFO] Generating quick demonstration video at: {output_path}")
    width, height = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cars = [
        {"x": 40, "y": 280, "vx": 4, "w": 90, "h": 50, "color": (56, 189, 248), "name": "car"},
        {"x": 550, "y": 350, "vx": -5, "w": 100, "h": 55, "color": (244, 114, 182), "name": "car"},
        {"x": 120, "y": 140, "vx": 2, "w": 35, "h": 80, "color": (52, 211, 153), "name": "person"},
    ]

    for frame_i in range(150):
        frame = np.full((height, width, 3), (35, 38, 46), dtype=np.uint8)
        # Sidewalk & Road
        cv2.rectangle(frame, (0, 100), (width, 220), (60, 64, 75), -1)
        cv2.rectangle(frame, (0, 220), (width, height), (28, 30, 36), -1)
        # Lane divider
        dash = (frame_i * 6) % 60
        for lx in range(-60 + dash, width + 60, 60):
            cv2.line(frame, (lx, 330), (lx + 30, 330), (200, 200, 200), 2)

        for c in cars:
            cx, cy, cw, ch = int(c["x"]), int(c["y"]), c["w"], c["h"]
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), c["color"], -1)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 255, 255), 2)
            cv2.putText(frame, c["name"].upper(), (cx + 5, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            c["x"] += c["vx"]
            if c["vx"] > 0 and c["x"] > width:
                c["x"] = -cw
            elif c["vx"] < 0 and c["x"] < -cw:
                c["x"] = width

        out.write(frame)
    out.release()
    print("[INFO] Fallback demo video created successfully.")


# ==============================================================================
# Unified DeepSORT Tracker Adapter
# ==============================================================================

class UnifiedTracker:
    """
    Unified Tracker Interface supporting:
    1. Built-in DeepSORT implementation (`deep_sort` package)
    2. `deep-sort-realtime` package (if installed)
    Maintains track history, persistent IDs, and trajectory paths.
    """

    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.3):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.backend = None
        self.history = {}  # {track_id: deque([(cx, cy), ...])}
        self.class_votes = {}  # {track_id: {cls_name: accumulated_score}}
        self.total_unique_ids = set()

        # Attempt to load deep-sort-realtime first, fallback to internal deep_sort
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_cosine_distance=max_cosine_distance,
                nn_budget=100,
                embedder="mobilenet",
                half=False,
                bgr=True,
                embedder_gpu=False,
            )
            self.backend = "deep-sort-realtime"
            print("[INFO] Using 'deep-sort-realtime' tracker backend.")
        except Exception as e:
            print(f"[INFO] deep-sort-realtime unavailable ({e}). Using internal DeepSORT tracker.")
            from deep_sort.tracker import DeepSortTracker
            self.tracker = DeepSortTracker(
                max_cosine_distance=max_cosine_distance,
                nn_budget=100,
                max_iou_distance=0.7,
                max_age=max_age,
                n_init=n_init,
            )
            self.backend = "internal-deepsort"

    def update(self, raw_detections, frame):
        """
        Update tracker with detections from current frame.

        Parameters
        ----------
        raw_detections : list
            List of [([x1, y1, w, h], confidence, class_name)]
        frame : np.ndarray
            Current BGR video frame

        Returns
        -------
        list of dict:
            Tracked objects with 'id', 'ltrb', 'class', 'confidence', 'history'
        """
        tracked_results = []

        if self.backend == "deep-sort-realtime":
            tracks = self.tracker.update_tracks(raw_detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                ltrb = track.to_ltrb()
                cls_name = track.get_det_class() or "object"
                conf = track.get_det_conf() or 1.0

                self.total_unique_ids.add(tid)
                cx = int((ltrb[0] + ltrb[2]) / 2)
                cy = int((ltrb[1] + ltrb[3]) / 2)
                if tid not in self.history:
                    self.history[tid] = deque(maxlen=30)
                self.history[tid].append((cx, cy))

                if tid not in self.class_votes:
                    self.class_votes[tid] = {}
                self.class_votes[tid][cls_name] = self.class_votes[tid].get(cls_name, 0.0) + float(conf)
                stable_class = max(self.class_votes[tid].items(), key=lambda x: x[1])[0]

                tracked_results.append({
                    "id": tid,
                    "ltrb": [int(v) for v in ltrb],
                    "class": str(stable_class),
                    "confidence": float(conf),
                    "history": list(self.history[tid]),
                })
        else:
            # Internal DeepSORT backend
            tracks = self.tracker.update_tracks(raw_detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                ltrb = track.to_tlbr()
                self.total_unique_ids.add(tid)

                cx = int((ltrb[0] + ltrb[2]) / 2)
                cy = int((ltrb[1] + ltrb[3]) / 2)
                if tid not in self.history:
                    self.history[tid] = deque(maxlen=30)
                self.history[tid].append((cx, cy))

                if tid not in self.class_votes:
                    self.class_votes[tid] = {}
                self.class_votes[tid][track.class_name] = (
                    self.class_votes[tid].get(track.class_name, 0.0) + float(track.confidence)
                )
                stable_class = max(self.class_votes[tid].items(), key=lambda x: x[1])[0]

                tracked_results.append({
                    "id": tid,
                    "ltrb": [int(v) for v in ltrb],
                    "class": str(stable_class),
                    "confidence": float(track.confidence),
                    "history": list(self.history[tid]),
                })

        # Cleanup old trajectory history for non-active tracks
        active_ids = {t["id"] for t in tracked_results}
        for tid in list(self.history.keys()):
            if tid not in active_ids and len(self.history[tid]) > 0:
                self.history[tid].popleft()
                if len(self.history[tid]) == 0:
                    del self.history[tid]

        return tracked_results

    def reset(self):
        """Reset tracking state and unique counters."""
        self.history.clear()
        self.class_votes.clear()
        self.total_unique_ids.clear()
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()


# ==============================================================================
# Visualization: Bounding Boxes, Badges, Trajectory Trails & HUD Overlay
# ==============================================================================

def draw_corner_rect(img, pt1, pt2, color, thickness=2, corner_len=15):
    """Draw stylish corner brackets around the bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw main rectangle with subtle line
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Top-Left Corner
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)

    # Top-Right Corner
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)

    # Bottom-Left Corner
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y1 - corner_len), color, thickness, cv2.LINE_AA)

    # Bottom-Right Corner
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)


def render_tracking_visuals(
    frame,
    tracked_objects,
    fps=0.0,
    total_unique_ids=0,
    draw_trails=True,
    show_conf=True,
):
    """
    Renders bounding boxes, unique track ID badges, trajectory motion trails,
    and a translucent HUD overlay onto the frame.
    """
    h_img, w_img = frame.shape[:2]

    # 1. Draw trajectory motion trails
    if draw_trails:
        for obj in tracked_objects:
            pts = obj["history"]
            if len(pts) < 2:
                continue
            color = COLOR_PALETTE[hash(str(obj["id"])) % len(COLOR_PALETTE)]
            for i in range(1, len(pts)):
                thickness = int(np.sqrt(float(i + 1) / len(pts)) * 3) + 1
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

    # 2. Draw bounding boxes, centroid markers, and ID badges
    for obj in tracked_objects:
        tid = obj["id"]
        x1, y1, x2, y2 = obj["ltrb"]
        cls_name = obj["class"]
        conf = obj["confidence"]

        # Clamp box coordinates inside frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

        color = COLOR_PALETTE[hash(str(tid)) % len(COLOR_PALETTE)]

        # Stylish corner-bracket bounding box
        draw_corner_rect(frame, (x1, y1), (x2, y2), color, thickness=3, corner_len=18)

        # Draw centroid marker
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Label badge text: [ID: #1] person (0.92)
        if show_conf and conf > 0:
            label = f"ID: #{tid} | {cls_name} {int(conf * 100)}%"
        else:
            label = f"ID: #{tid} | {cls_name}"

        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        font_thick = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        # Position badge above bounding box (or inside if at top edge)
        badge_y1 = max(0, y1 - label_h - 10)
        badge_y2 = badge_y1 + label_h + 8
        badge_x2 = min(w_img, x1 + label_w + 12)

        # Solid background for label
        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Contrasting text (dark text on bright badge)
        cv2.putText(
            frame,
            label,
            (x1 + 6, badge_y2 - baseline - 3),
            font,
            font_scale,
            (10, 10, 15),
            font_thick + 1,
            cv2.LINE_AA,
        )

    # 3. Translucent Top HUD Bar
    hud_h = 42
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, hud_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # HUD Text items
    status_text = f"FPS: {fps:.1f}  |  Active Objects: {len(tracked_objects)}  |  Total Unique IDs: {total_unique_ids}"
    cv2.putText(
        frame,
        status_text,
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "YOLOv8 + DeepSORT",
        (w_img - 200, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (56, 189, 248),
        2,
        cv2.LINE_AA,
    )

    return frame


# ==============================================================================
# Object Detection & Tracking Engine
# ==============================================================================

class TrackingEngine:
    """
    Main processing pipeline:
    - Loads pre-trained YOLOv8 model
    - Initializes DeepSORT tracker
    - Processes video stream from webcam or file
    - Emits processed frames and stats
    """

    def __init__(self, model_path=YOLO_MODEL_PATH):
        ensure_model_exists(model_path)
        from ultralytics import YOLO

        print(f"[INFO] Initializing YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        print("[INFO] YOLOv8 model loaded successfully.")

        self.tracker = UnifiedTracker()
        self.is_running = False
        self.cap = None

        # Statistics
        self.current_fps = 0.0
        self.active_count = 0
        self.total_unique_ids = 0
        self.class_counts = {}

    def start_stream(
        self,
        source,
        conf_threshold=0.40,
        class_filter=None,
        draw_trails=True,
        show_conf=True,
        on_frame_callback=None,
        on_error_callback=None,
    ):
        """
        Runs the tracking loop synchronously or inside a thread.
        """
        self.is_running = True
        self.tracker.reset()

        # Handle webcam vs video file source
        is_webcam = False
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            source = int(source)
            is_webcam = True
            print(f"[INFO] Opening Webcam device index: {source}")
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        else:
            print(f"[INFO] Opening Video file: {source}")
            if not os.path.exists(source):
                err_msg = f"Video file not found at: {source}"
                if on_error_callback:
                    on_error_callback(err_msg)
                self.is_running = False
                return
            self.cap = cv2.VideoCapture(source)

        if not self.cap or not self.cap.isOpened():
            err_msg = (
                f"Could not open video source: '{source}'. "
                "Ensure the camera is connected and not used by another application, "
                "or verify the video file format."
            )
            if on_error_callback:
                on_error_callback(err_msg)
            self.is_running = False
            return

        fps_calc_time = time.time()
        fps_frame_counter = 0

        try:
            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if is_webcam:
                        print("[WARNING] Webcam frame stream stopped.")
                        break
                    else:
                        # Auto-loop video file for seamless continuous demo
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            break

                fps_frame_counter += 1
                now = time.time()
                elapsed = now - fps_calc_time
                if elapsed >= 0.5:
                    self.current_fps = fps_frame_counter / elapsed
                    fps_frame_counter = 0
                    fps_calc_time = now

                # 1. Run YOLOv8 object detection
                results = self.model(frame, conf=conf_threshold, verbose=False)[0]

                # 2. Extract bounding boxes and filter by class
                raw_detections = []
                frame_class_counts = {}

                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0].item())

                    # Apply class filter if specified
                    if class_filter is not None and cls_name not in class_filter:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1
                    raw_detections.append(([x1, y1, w, h], conf, cls_name))
                    frame_class_counts[cls_name] = frame_class_counts.get(cls_name, 0) + 1

                self.class_counts = frame_class_counts

                # 3. Update DeepSORT tracker
                tracked_objects = self.tracker.update(raw_detections, frame=frame)
                self.active_count = len(tracked_objects)
                self.total_unique_ids = len(self.tracker.total_unique_ids)

                # 4. Render visuals onto frame
                processed_frame = render_tracking_visuals(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    fps=self.current_fps,
                    total_unique_ids=self.total_unique_ids,
                    draw_trails=draw_trails,
                    show_conf=show_conf,
                )

                if on_frame_callback:
                    on_frame_callback(
                        processed_frame,
                        self.active_count,
                        self.total_unique_ids,
                        self.current_fps,
                        self.class_counts,
                    )

                # Yield small CPU time slice
                time.sleep(0.001)

        except Exception as e:
            if on_error_callback:
                on_error_callback(f"Tracking loop error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop video processing and release hardware resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[INFO] Video capture released.")


# ==============================================================================
# Modern Dark-Themed GUI (Tkinter + TTK + Pillow)
# ==============================================================================

class ObjectTrackingGUI:
    """
    Modern Dark-Themed User Interface for the Object Detection & Tracking System.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("AI Object Detection & Tracking | CodeAlpha Internship Task 4")
        self.root.geometry("1200x780")
        self.root.minsize(1050, 680)
        self.root.configure(bg="#0b0f19")

        # Initialize tracking backend
        self.engine = TrackingEngine()
        self.worker_thread = None
        self.latest_frame = None

        # State Variables
        self.source_mode = tk.StringVar(value="webcam")
        self.webcam_index = tk.IntVar(value=0)
        self.video_path = tk.StringVar(value="")
        self.conf_threshold = tk.DoubleVar(value=0.40)
        self.selected_class_group = tk.StringVar(value="All Classes")
        self.draw_trails = tk.BooleanVar(value=True)
        self.show_conf = tk.BooleanVar(value=True)

        self._apply_dark_style()
        self._build_layout()

        # Keyboard shortcuts
        self.root.bind("<q>", lambda e: self.on_press_quit())
        self.root.bind("<Q>", lambda e: self.on_press_quit())
        self.root.bind("<Escape>", lambda e: self.stop_tracking())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start Tkinter refresh loop
        self.root.after(30, self._gui_refresh_loop)

    def _apply_dark_style(self):
        """Configure modern dark styles for ttk widgets."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0b0f19")
        style.configure("Card.TFrame", background="#151d2f", relief="flat")
        style.configure("TLabel", background="#0b0f19", foreground="#f8fafc", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#151d2f", foreground="#f8fafc", font=("Segoe UI", 10))
        style.configure("Heading.TLabel", background="#0b0f19", foreground="#38bdf8", font=("Segoe UI", 16, "bold"))
        style.configure("SubHeading.TLabel", background="#0b0f19", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("Section.TLabel", background="#151d2f", foreground="#38bdf8", font=("Segoe UI", 11, "bold"))
        style.configure("MetricVal.TLabel", background="#151d2f", foreground="#38bdf8", font=("Segoe UI", 18, "bold"))
        style.configure("MetricLbl.TLabel", background="#151d2f", foreground="#94a3b8", font=("Segoe UI", 8))

        style.configure("TRadiobutton", background="#151d2f", foreground="#e2e8f0", font=("Segoe UI", 9))
        style.map("TRadiobutton", background=[("active", "#151d2f")], foreground=[("active", "#38bdf8")])

        style.configure("TCheckbutton", background="#151d2f", foreground="#e2e8f0", font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", "#151d2f")], foreground=[("active", "#38bdf8")])

        style.configure("TCombobox", fieldbackground="#1e293b", background="#1e293b", foreground="#ffffff")

    def _build_layout(self):
        """Construct the sidebar and video canvas."""
        # Top Header Bar
        header = ttk.Frame(self.root, padding="15 10 15 10")
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = ttk.Label(header, text="AI Object Detection & Tracking", style="Heading.TLabel")
        title_lbl.pack(anchor=tk.W)

        desc_lbl = ttk.Label(
            header,
            text="CodeAlpha AI Internship - Task 4  |  Real-time Multi-Object Detection (YOLOv8) & DeepSORT Tracking with Persistent IDs",
            style="SubHeading.TLabel",
        )
        desc_lbl.pack(anchor=tk.W)

        # Main Split Body
        body = ttk.Frame(self.root, padding="10")
        body.pack(fill=tk.BOTH, expand=True)

        # Left Column: Controls (Width: 340)
        sidebar = ttk.Frame(body, style="Card.TFrame", padding="12", width=340)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # --- Card 1: Video Source ---
        src_label = ttk.Label(sidebar, text="1. VIDEO SOURCE", style="Section.TLabel")
        src_label.pack(anchor=tk.W, pady=(0, 6))

        rb_webcam = ttk.Radiobutton(
            sidebar,
            text="Live Webcam",
            value="webcam",
            variable=self.source_mode,
            command=self._on_source_change,
        )
        rb_webcam.pack(anchor=tk.W, pady=2)

        cam_box = ttk.Frame(sidebar, style="Card.TFrame")
        cam_box.pack(fill=tk.X, padx=18, pady=(0, 6))
        ttk.Label(cam_box, text="Camera Index:", style="Card.TLabel").pack(side=tk.LEFT)
        self.cam_spin = ttk.Spinbox(cam_box, from_=0, to=5, width=4, textvariable=self.webcam_index)
        self.cam_spin.pack(side=tk.LEFT, padx=6)

        rb_video = ttk.Radiobutton(
            sidebar,
            text="Video File",
            value="video",
            variable=self.source_mode,
            command=self._on_source_change,
        )
        rb_video.pack(anchor=tk.W, pady=2)

        # Video path display & browse
        vid_btn_box = ttk.Frame(sidebar, style="Card.TFrame")
        vid_btn_box.pack(fill=tk.X, padx=18, pady=(2, 6))
        self.btn_browse = tk.Button(
            vid_btn_box,
            text="📁 Browse Video",
            command=self._browse_video,
            bg="#1e293b",
            fg="#f8fafc",
            activebackground="#334155",
            activeforeground="#38bdf8",
            relief="flat",
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=3,
        )
        self.btn_browse.pack(side=tk.LEFT)

        self.btn_sample = tk.Button(
            vid_btn_box,
            text="🎬 Demo Video",
            command=self._use_sample_video,
            bg="#1e293b",
            fg="#38bdf8",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=3,
        )
        self.btn_sample.pack(side=tk.LEFT, padx=6)

        self.lbl_selected_file = ttk.Label(sidebar, text="No file selected", style="Card.TLabel", font=("Segoe UI", 8))
        self.lbl_selected_file.pack(anchor=tk.W, padx=18, pady=(0, 8))

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=8)

        # --- Card 2: Model & Filters ---
        filt_label = ttk.Label(sidebar, text="2. DETECTION & TRACKING", style="Section.TLabel")
        filt_label.pack(anchor=tk.W, pady=(0, 6))

        # Confidence Slider
        ttk.Label(sidebar, text="Confidence Threshold:", style="Card.TLabel").pack(anchor=tk.W)
        self.slider_conf = tk.Scale(
            sidebar,
            from_=0.10,
            to=0.95,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.conf_threshold,
            bg="#151d2f",
            fg="#38bdf8",
            troughcolor="#1e293b",
            highlightthickness=0,
            font=("Segoe UI", 8),
        )
        self.slider_conf.pack(fill=tk.X, pady=(0, 6))

        # Class Category Filter
        ttk.Label(sidebar, text="Target Class Filter:", style="Card.TLabel").pack(anchor=tk.W)
        self.cb_classes = ttk.Combobox(
            sidebar,
            values=list(CLASS_GROUPS.keys()),
            textvariable=self.selected_class_group,
            state="readonly",
        )
        self.cb_classes.pack(fill=tk.X, pady=(2, 8))

        # Visual Toggles
        cb_trails = ttk.Checkbutton(sidebar, text="Show Motion Trajectory Trails", variable=self.draw_trails)
        cb_trails.pack(anchor=tk.W, pady=2)
        cb_conf = ttk.Checkbutton(sidebar, text="Show Confidence Scores", variable=self.show_conf)
        cb_conf.pack(anchor=tk.W, pady=2)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=8)

        # --- Card 3: Action Buttons ---
        ctrl_label = ttk.Label(sidebar, text="3. CONTROLS", style="Section.TLabel")
        ctrl_label.pack(anchor=tk.W, pady=(0, 6))

        btn_box = ttk.Frame(sidebar, style="Card.TFrame")
        btn_box.pack(fill=tk.X, pady=4)

        self.btn_start = tk.Button(
            btn_box,
            text="▶  START TRACKING",
            command=self.start_tracking,
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=6,
            cursor="hand2",
        )
        self.btn_start.pack(fill=tk.X, pady=3)

        self.btn_stop = tk.Button(
            btn_box,
            text="⏹  STOP TRACKING (Q)",
            command=self.stop_tracking,
            bg="#ef4444",
            fg="#ffffff",
            activebackground="#dc2626",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=6,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.btn_stop.pack(fill=tk.X, pady=3)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=8)

        # --- Card 4: Live Statistics Metrics ---
        stat_label = ttk.Label(sidebar, text="4. REAL-TIME METRICS", style="Section.TLabel")
        stat_label.pack(anchor=tk.W, pady=(0, 6))

        metrics_grid = ttk.Frame(sidebar, style="Card.TFrame")
        metrics_grid.pack(fill=tk.X)

        # Active Objects
        m1 = ttk.Frame(metrics_grid, style="Card.TFrame", padding="4")
        m1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.lbl_active_val = ttk.Label(m1, text="0", style="MetricVal.TLabel")
        self.lbl_active_val.pack()
        ttk.Label(m1, text="ACTIVE OBJECTS", style="MetricLbl.TLabel").pack()

        # Total Unique IDs
        m2 = ttk.Frame(metrics_grid, style="Card.TFrame", padding="4")
        m2.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.lbl_unique_val = ttk.Label(m2, text="0", style="MetricVal.TLabel")
        self.lbl_unique_val.pack()
        ttk.Label(m2, text="TOTAL TRACK IDs", style="MetricLbl.TLabel").pack()

        # FPS
        m3 = ttk.Frame(metrics_grid, style="Card.TFrame", padding="4")
        m3.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.lbl_fps_val = ttk.Label(m3, text="0.0", style="MetricVal.TLabel")
        self.lbl_fps_val.pack()
        ttk.Label(m3, text="PROCESSING FPS", style="MetricLbl.TLabel").pack()

        # Status Badge
        m4 = ttk.Frame(metrics_grid, style="Card.TFrame", padding="4")
        m4.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        self.lbl_status_val = ttk.Label(m4, text="READY", style="MetricVal.TLabel", foreground="#10b981")
        self.lbl_status_val.pack()
        ttk.Label(m4, text="SYSTEM STATUS", style="MetricLbl.TLabel").pack()

        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)

        self.lbl_classes_breakdown = ttk.Label(
            sidebar,
            text="Detected: None",
            style="Card.TLabel",
            font=("Segoe UI", 8),
            wraplength=310,
        )
        self.lbl_classes_breakdown.pack(anchor=tk.W, pady=(8, 0))

        # Right Column: Video Display Canvas
        canvas_card = ttk.Frame(body, style="Card.TFrame", padding="4")
        canvas_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_card, bg="#070a13", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Draw sleek initial placeholder on Canvas
        self.canvas.bind("<Configure>", self._draw_placeholder)

    def _on_source_change(self):
        """Update source selection UI states."""
        mode = self.source_mode.get()
        if mode == "webcam":
            self.cam_spin.configure(state="normal")
            self.btn_browse.configure(state="disabled")
            self.btn_sample.configure(state="disabled")
        else:
            self.cam_spin.configure(state="disabled")
            self.btn_browse.configure(state="normal")
            self.btn_sample.configure(state="normal")

    def _browse_video(self):
        """Open file dialog for video selection."""
        file_path = filedialog.askopenfilename(
            title="Select Video File for Object Tracking",
            filetypes=[
                ("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.video_path.set(file_path)
            self.source_mode.set("video")
            filename = os.path.basename(file_path)
            self.lbl_selected_file.configure(text=f"Selected: {filename}")
            self._on_source_change()

    def _use_sample_video(self):
        """Quickly load the demo video."""
        sample_path = ensure_sample_video()
        self.video_path.set(sample_path)
        self.source_mode.set("video")
        self.lbl_selected_file.configure(text=f"Selected: {os.path.basename(sample_path)}")
        self._on_source_change()

    def _draw_placeholder(self, event=None):
        """Draw sleek idle screen when no video is playing."""
        if self.latest_frame is not None or self.engine.is_running:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 100 or h < 100:
            return

        self.canvas.delete("all")

        # Dark grid background
        for x in range(0, w, 40):
            self.canvas.create_line(x, 0, x, h, fill="#0e1424")
        for y in range(0, h, 40):
            self.canvas.create_line(0, y, w, y, fill="#0e1424")

        # Decorative bounding box
        bw, bh = min(420, w - 80), min(220, h - 80)
        bx1 = (w - bw) // 2
        by1 = (h - bh) // 2
        bx2 = bx1 + bw
        by2 = by1 + bh

        self.canvas.create_rectangle(bx1, by1, bx2, by2, outline="#1e293b", width=2)
        # Corner brackets
        clen = 20
        self.canvas.create_line(bx1, by1, bx1 + clen, by1, fill="#38bdf8", width=3)
        self.canvas.create_line(bx1, by1, bx1, by1 + clen, fill="#38bdf8", width=3)
        self.canvas.create_line(bx2, by1, bx2 - clen, by1, fill="#38bdf8", width=3)
        self.canvas.create_line(bx2, by1, bx2, by1 + clen, fill="#38bdf8", width=3)
        self.canvas.create_line(bx1, by2, bx1 + clen, by2, fill="#38bdf8", width=3)
        self.canvas.create_line(bx1, by2, bx1, by2 - clen, fill="#38bdf8", width=3)
        self.canvas.create_line(bx2, by2, bx2 - clen, by2, fill="#38bdf8", width=3)
        self.canvas.create_line(bx2, by2, bx2, by2 - clen, fill="#38bdf8", width=3)

        self.canvas.create_text(
            w // 2,
            by1 + 55,
            text="🎯 AI VISION SYSTEM READY",
            fill="#38bdf8",
            font=("Segoe UI", 16, "bold"),
        )
        self.canvas.create_text(
            w // 2,
            by1 + 95,
            text="Select 'Live Webcam' or 'Video File' from the left panel",
            fill="#e2e8f0",
            font=("Segoe UI", 11),
        )
        self.canvas.create_text(
            w // 2,
            by1 + 130,
            text="Click '▶ START TRACKING' to begin detection",
            fill="#10b981",
            font=("Segoe UI", 11, "bold"),
        )
        self.canvas.create_text(
            w // 2,
            by1 + 165,
            text="Press 'Q' or Esc anytime to stop processing",
            fill="#94a3b8",
            font=("Segoe UI", 9),
        )

    def start_tracking(self):
        """Start the detection and tracking worker thread."""
        if self.engine.is_running:
            return

        mode = self.source_mode.get()
        if mode == "webcam":
            source = self.webcam_index.get()
        else:
            source = self.video_path.get().strip()
            if not source:
                # If no video selected, prompt to load sample video
                sample = ensure_sample_video()
                self.video_path.set(sample)
                source = sample
                self.lbl_selected_file.configure(text=f"Selected: {os.path.basename(sample)}")

        conf = float(self.conf_threshold.get())
        selected_grp = self.selected_class_group.get()
        class_filter = CLASS_GROUPS.get(selected_grp, None)
        draw_tr = self.draw_trails.get()
        show_c = self.show_conf.get()

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.lbl_status_val.configure(text="RUNNING", foreground="#38bdf8")

        # Launch tracking loop in background daemon thread
        self.worker_thread = threading.Thread(
            target=self.engine.start_stream,
            kwargs={
                "source": source,
                "conf_threshold": conf,
                "class_filter": class_filter,
                "draw_trails": draw_tr,
                "show_conf": show_c,
                "on_frame_callback": self._on_frame_received,
                "on_error_callback": self._on_tracking_error,
            },
            daemon=True,
        )
        self.worker_thread.start()

    def _on_frame_received(self, frame, active_count, total_count, fps, class_counts):
        """Thread-safe callback invoked on each processed frame."""
        self.latest_frame = frame
        self._update_stats(active_count, total_count, fps, class_counts)

    def _update_stats(self, active_count, total_count, fps, class_counts):
        """Update metrics labels in sidebar."""
        self.lbl_active_val.configure(text=str(active_count))
        self.lbl_unique_val.configure(text=str(total_count))
        self.lbl_fps_val.configure(text=f"{fps:.1f}")

        if class_counts:
            items = [f"{k}: {v}" for k, v in class_counts.items()]
            summary = "Detected: " + ", ".join(items[:4])
        else:
            summary = "Detected: Searching..."
        self.lbl_classes_breakdown.configure(text=summary)

    def _gui_refresh_loop(self):
        """Periodically blit the latest frame to the Tkinter Canvas."""
        if self.latest_frame is not None and self.engine.is_running:
            try:
                frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                canvas_w = max(50, self.canvas.winfo_width())
                canvas_h = max(50, self.canvas.winfo_height())

                fh, fw = frame_rgb.shape[:2]
                scale = min(canvas_w / fw, canvas_h / fh)
                new_w = max(1, int(fw * scale))
                new_h = max(1, int(fh * scale))

                resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(resized)
                self.photo = ImageTk.PhotoImage(image=img)

                self.canvas.delete("all")
                # Center frame on canvas
                cx = (canvas_w - new_w) // 2
                cy = (canvas_h - new_h) // 2
                self.canvas.create_image(cx, cy, anchor=tk.NW, image=self.photo)
            except Exception as e:
                pass
        elif not self.engine.is_running and self.latest_frame is None:
            self._draw_placeholder()

        # Schedule next refresh at ~30 FPS
        self.root.after(33, self._gui_refresh_loop)

    def _on_tracking_error(self, message):
        """Handle errors from the tracking thread."""
        print(f"[ERROR] {message}")
        self.root.after(0, lambda: messagebox.showerror("Tracking Error", message))
        self.root.after(0, self.stop_tracking)

    def stop_tracking(self):
        """Stop tracking session gracefully."""
        self.engine.stop()
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.lbl_status_val.configure(text="STOPPED", foreground="#ef4444")
        self.latest_frame = None
        self._draw_placeholder()

    def on_press_quit(self):
        """Triggered when pressing 'Q' or 'q' key."""
        if self.engine.is_running:
            self.stop_tracking()
        else:
            self.on_close()

    def on_close(self):
        """Handle window close event."""
        self.stop_tracking()
        self.root.destroy()


# ==============================================================================
# CLI / Headless OpenCV Execution Mode
# ==============================================================================

def run_cli_mode(source, conf_threshold=0.40, class_filter=None, draw_trails=True, show_conf=True):
    """
    Run the object detection and tracking pipeline directly in an OpenCV window.
    Ideal for headless servers, quick benchmarking, or low-latency testing.
    """
    print("=" * 65)
    print("AI Object Detection & Tracking System - CLI OpenCV Mode")
    print("=" * 65)
    print(f"Source: {source} | Confidence: {conf_threshold}")
    print("Press 'q' or 'ESC' in the video window to quit.")
    print("=" * 65)

    engine = TrackingEngine()
    window_name = "AI Object Detection & Tracking (YOLOv8 + DeepSORT) | Press 'Q' to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_frame(frame, active_count, total_count, fps, class_counts):
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            engine.stop()

    def on_error(msg):
        print(f"[ERROR] {msg}")

    try:
        engine.start_stream(
            source=source,
            conf_threshold=conf_threshold,
            class_filter=class_filter,
            draw_trails=draw_trails,
            show_conf=show_conf,
            on_frame_callback=on_frame,
            on_error_callback=on_error,
        )
    finally:
        cv2.destroyAllWindows()


# ==============================================================================
# Main Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Object Detection and Tracking System using YOLOv8 & DeepSORT (CodeAlpha Task 4)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: '0' for default webcam, or path to video file (e.g., sample_videos/sample_traffic.mp4).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold for YOLOv8 object detection (default: 0.40).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run directly in an OpenCV window without the Tkinter GUI.",
    )
    parser.add_argument(
        "--class-filter",
        type=str,
        default="All Classes",
        choices=list(CLASS_GROUPS.keys()),
        help="Filter detection by class group.",
    )
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable motion trajectory trails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If --no-gui or explicit --source is specified, launch CLI OpenCV mode
    if args.no_gui or args.source is not None:
        src = args.source if args.source is not None else 0
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        filter_classes = CLASS_GROUPS.get(args.class_filter, None)
        run_cli_mode(
            source=src,
            conf_threshold=args.conf,
            class_filter=filter_classes,
            draw_trails=not args.no_trails,
            show_conf=True,
        )
    else:
        # Default: Launch Modern Dark-Themed GUI
        root = tk.Tk()
        app = ObjectTrackingGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
