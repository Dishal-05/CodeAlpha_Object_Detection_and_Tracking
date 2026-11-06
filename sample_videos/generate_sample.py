"""
Sample Video Generator & Downloader for Object Detection & Tracking.
Provides a ready-to-use sample video in the sample_videos/ directory.
Attempts to download a sample clip; if offline, generates a synthetic video.
"""

import os
import sys
import urllib.request

SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_VIDEO_PATH = os.path.join(SAMPLE_DIR, "sample_traffic.mp4")

# Direct public download URLs for sample traffic/street videos
SAMPLE_URLS = [
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
]


def download_sample_video():
    """Try to download a real sample video from public repositories."""
    if os.path.exists(SAMPLE_VIDEO_PATH) and os.path.getsize(SAMPLE_VIDEO_PATH) > 10000:
        print(f"Sample video already exists at: {SAMPLE_VIDEO_PATH}")
        return True

    print("Attempting to download sample video...")
    for url in SAMPLE_URLS:
        try:
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, SAMPLE_VIDEO_PATH)
            if os.path.exists(SAMPLE_VIDEO_PATH) and os.path.getsize(SAMPLE_VIDEO_PATH) > 10000:
                print(f"Sample video downloaded successfully to: {SAMPLE_VIDEO_PATH}")
                return True
        except Exception as e:
            print(f"Download from {url} failed: {e}")

    return False


def generate_synthetic_video():
    """Generate a synthetic multi-object tracking test video if offline."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV/NumPy not yet installed, cannot generate synthetic video.")
        return False

    print(f"Generating synthetic test video at: {SAMPLE_VIDEO_PATH}...")
    width, height = 854, 480
    fps = 30
    duration_sec = 10
    total_frames = fps * duration_sec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(SAMPLE_VIDEO_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(SAMPLE_VIDEO_PATH, fourcc, fps, (width, height))

    # Define moving objects: [x, y, vx, vy, w, h, color, label]
    objects = [
        {"x": 50, "y": 280, "vx": 4, "vy": 0, "w": 110, "h": 60, "color": (40, 180, 240), "name": "car"},
        {"x": 750, "y": 360, "vx": -5, "vy": 0, "w": 120, "h": 65, "color": (230, 80, 80), "name": "car"},
        {"x": 100, "y": 140, "vx": 2, "vy": 0, "w": 40, "h": 90, "color": (80, 220, 80), "name": "person"},
        {"x": 600, "y": 150, "vx": -2, "vy": 0, "w": 40, "h": 90, "color": (200, 100, 220), "name": "person"},
        {"x": 350, "y": 220, "vx": 3, "vy": 0, "w": 60, "h": 50, "color": (60, 240, 180), "name": "bicycle"},
    ]

    for frame_idx in range(total_frames):
        # Background: Road & Sidewalk
        frame = np.full((height, width, 3), (45, 45, 50), dtype=np.uint8)

        # Sidewalk
        cv2.rectangle(frame, (0, 100), (width, 240), (80, 85, 90), -1)
        # Road
        cv2.rectangle(frame, (0, 240), (width, height), (35, 35, 40), -1)

        # Lane markings (dashed)
        dash_offset = (frame_idx * 6) % 80
        for lx in range(-80 + dash_offset, width + 80, 80):
            cv2.line(frame, (lx, 350), (lx + 40, 350), (220, 220, 220), 3)

        # Draw moving objects
        for obj in objects:
            ox = int(obj["x"])
            oy = int(obj["y"])
            ow = obj["w"]
            oh = obj["h"]

            # Draw body
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), obj["color"], -1)
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 255, 255), 2)
            cv2.putText(
                frame,
                obj["name"].upper(),
                (ox + 5, oy + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

            # Update position
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]

            # Wrap around screen
            if obj["vx"] > 0 and obj["x"] > width:
                obj["x"] = -ow
            elif obj["vx"] < 0 and obj["x"] < -ow:
                obj["x"] = width

        # Frame counter & instructions
        cv2.putText(
            frame,
            f"Test Video Feed | Frame: {frame_idx + 1}/{total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        out.write(frame)

    out.release()
    print(f"Synthetic sample video created: {SAMPLE_VIDEO_PATH}")
    return True


if __name__ == "__main__":
    if not download_sample_video():
        generate_synthetic_video()

