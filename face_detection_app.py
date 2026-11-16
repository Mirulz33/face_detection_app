

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

DETECTION_MAX_DIMENSION = 640

# Face detection is the expensive part of the pipeline.  Bounding boxes are
# refreshed at this rate while every source frame continues to be displayed
# and written to the output video.  This keeps live video fluid without a
# noticeable delay in the overlay.
DETECTION_INTERVAL_SECONDS = 1.0 / 15.0

# Tk only needs to repaint when a newer frame is available, but a short timer
# keeps the preview responsive on cameras that can deliver more than 30 FPS.
UI_REFRESH_INTERVAL_MS = 16

FACE_BOX_COLOR = (40, 220, 90)
# Keep boxes slim while ensuring they remain visible when a large source video
# is scaled down to fit the preview.
FACE_BOX_THICKNESS = 2

SUPPORTED_VIDEO_TYPES = (
    (
        "Video files",
        "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v "
        "*.mpeg *.mpg *.3gp *.ts *.mts *.m2ts *.ogv"
    ),
    ("All files", "*.*"),
)


# ============================================================
# FACE DETECTOR
# ============================================================

class FaceDetector:

    def __init__(self, status_callback=None):

        self.kind = ""
        self.yunet = None
        self.cascade = None

        self.status_callback = (
            status_callback or (lambda message: None)
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Download YuNet model if not available.
        if not MODEL_PATH.exists():

            self.status_callback(
                "Downloading YuNet face detection model..."
            )

            try:
                self._download_model()

            except Exception as exc:

                self.status_callback(
                    f"YuNet download unavailable: {exc}. "
                    "Using fallback detector."
                )

        # Load YuNet.
        if MODEL_PATH.exists():

            try:

                creator = getattr(
                    cv2,
                    "FaceDetectorYN_create",
                    None
                )

                if creator is not None:

                    self.yunet = creator(
                        str(MODEL_PATH),
                        "",
                        (320, 320),
                        0.60,
                        0.30,
                        5000
                    )

                else:

                    self.yunet = cv2.FaceDetectorYN.create(
                        str(MODEL_PATH),
                        "",
                        (320, 320),
                        0.60,
                        0.30,
                        5000
                    )

                self.kind = "YuNet"

                self.status_callback(
                    "YuNet detector ready."
                )

                return

            except Exception as exc:

                self.status_callback(
                    f"YuNet loading failed: {exc}. "
                    "Using fallback."
                )

        # Fallback: Haar Cascade.
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.cascade = cv2.CascadeClassifier(
            cascade_path
        )

        if self.cascade.empty():

            raise RuntimeError(
                "Could not load YuNet or Haar Cascade."
            )

        self.kind = "Haar Cascade (Fallback)"

        self.status_callback(
            "Haar Cascade detector ready."
        )

    @staticmethod
    def _download_model():

        request = urllib.request.Request(
            MODEL_URL,
            headers={
                "User-Agent": "FaceDetectionStudio/1.0"
            }
        )

        temporary_path = MODEL_PATH.with_suffix(
            ".download"
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=30
            ) as response:

                data = response.read()

            if (
                len(data) < 50000
                or b"version https://git-lfs" in data[:200]
            ):

                raise RuntimeError(
                    "Downloaded model is incomplete."
                )

            temporary_path.write_bytes(data)

            temporary_path.replace(MODEL_PATH)

        finally:

            if temporary_path.exists():

                temporary_path.unlink(
                    missing_ok=True
                )

    def detect(self, frame):

        height, width = frame.shape[:2]
        scale = min(
            1.0,
            DETECTION_MAX_DIMENSION / max(width, height)
        )

        if scale < 1.0:

            detection_frame = cv2.resize(
                frame,
                (
                    max(1, round(width * scale)),
                    max(1, round(height * scale))
                ),
                interpolation=cv2.INTER_AREA
            )

        else:

            detection_frame = frame

        detection_height, detection_width = (
            detection_frame.shape[:2]
        )
        scale_x = width / detection_width
        scale_y = height / detection_height

        results = []

        # YuNet detection.
        if self.kind == "YuNet":

            self.yunet.setInputSize(
                (detection_width, detection_height)
            )

            _retval, faces = self.yunet.detect(
                detection_frame
            )

            if faces is not None:

                for face in faces:

                    x = int(round(face[0] * scale_x))
                    y = int(round(face[1] * scale_y))
                    w = int(round(face[2] * scale_x))
                    h = int(round(face[3] * scale_y))

                    confidence = float(face[-1])

                    results.append(
                        (x, y, w, h, confidence)
                    )

        # Haar Cascade fallback.
        else:

            gray = cv2.cvtColor(
                detection_frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(28, 28),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for x, y, w, h in faces:

                results.append(
                    (
                        int(round(x * scale_x)),
                        int(round(y * scale_y)),
                        int(round(w * scale_x)),
                        int(round(h * scale_y)),
                        None
                    )
                )

        return results


# ============================================================
# VIDEO SUPPORT
# ============================================================

def find_ffmpeg():

    return shutil.which("ffmpeg")


def open_video_capture(source, is_camera=False):

    # Open webcam.
    if is_camera:

        if os.name == "nt":

            cap = cv2.VideoCapture(
                int(source),
                cv2.CAP_DSHOW
            )

            if cap.isOpened():

                # Do not let an internal capture queue add visible latency.
                # Some backends ignore this property, which is harmless.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                return cap, None

            cap.release()

        cap = cv2.VideoCapture(
            int(source)
        )

        if cap.isOpened():

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap, None

    # Open normal video.
    cap = cv2.VideoCapture(
        str(source)
    )

    if cap.isOpened():

        ok, first = cap.read()

        if ok and first is not None:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            return cap, None

        cap.release()

    # Try FFmpeg if OpenCV cannot read the video.
    ffmpeg = find_ffmpeg()

    if not ffmpeg:

        raise RuntimeError(
            "OpenCV cannot read this video. "
            "Install FFmpeg and try again."
        )

    temporary = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )

    temporary_path = Path(
        temporary.name
    )

    temporary.close()

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        str(temporary_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        temporary_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "FFmpeg could not decode this video.\n"
            + (result.stderr or "")[-1200:]
        )

    cap = cv2.VideoCapture(
        str(temporary_path)
    )

    if not cap.isOpened():

        cap.release()

        temporary_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Could not open converted video."
        )

    return cap, temporary_path


# ============================================================
# MAIN APPLICATION
# ============================================================

class FaceDetectionApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Face Detection Studio"
        )

        self.root.geometry(
            "1120x760"
        )

        self.root.minsize(
            850,
            600
        )

        self.detector = None

        self.detector_ready = False
        self.detector_error = None

        self.worker_thread = None

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self.frame_lock = threading.Lock()

        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_face_count = 0
        self.displayed_frame_id = -1

        self.message_queue = queue.Queue()

        self.source_label = tk.StringVar(
            value="Source: none selected"
        )

        self.status_var = tk.StringVar(
            value="Loading face detector..."
        )

        self.face_count_var = tk.StringVar(
            value="Faces detected: 0"
        )

        self.detector_var = tk.StringVar(
            value="Detector: loading"
        )

        self.camera_index_var = tk.StringVar(
            value="0"
        )

        self.pause_button_var = tk.StringVar(
            value="Pause"
        )

        self.output_path = None

        self.is_running = False

        self._build_ui()
        self._update_controls()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close
        )

        self.root.after(
            UI_REFRESH_INTERVAL_MS,
            self._refresh_ui
        )

        threading.Thread(
            target=self._load_detector,
            daemon=True
        ).start()

    # ========================================================
    # BUILD GUI
    # ========================================================

    def _build_ui(self):

        self.root.configure(
            bg="#edf1f5"
        )

        style = ttk.Style()

        try:

            style.theme_use("clam")

        except tk.TclError:

            pass

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 19, "bold"),
            background="#edf1f5"
        )

        style.configure(
            "Sub.TLabel",
            font=("Segoe UI", 9),
            background="#edf1f5",
            foreground="#526171"
        )

        style.configure(
            "Section.TLabel",
            font=("Segoe UI", 8, "bold"),
            background="#edf1f5",
            foreground="#526171"
        )

        style.configure(
            "Toolbar.TFrame",
            background="#edf1f5"
        )

        style.configure(
            "Action.TButton",
            padding=(12, 8),
            font=("Segoe UI", 9, "bold")
        )

        style.configure(
            "Primary.TButton",
            padding=(12, 8),
            font=("Segoe UI", 9, "bold")
        )

        # Header.
        header = ttk.Frame(
            self.root,
            padding=(18, 14, 18, 8),
            style="Toolbar.TFrame"
        )

        header.pack(fill="x")

        ttk.Label(
            header,
            text="Face Detection Studio",
            style="Title.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text=(
                "Choose a webcam or video, then pause, stop, or save "
                "the annotated result."
            ),
            style="Sub.TLabel"
        ).pack(
            anchor="w",
            pady=(3, 0)
        )

        # Source and playback controls.
        toolbar = ttk.Frame(
            self.root,
            padding=(18, 8, 18, 8),
            style="Toolbar.TFrame"
        )

        toolbar.pack(fill="x")

        ttk.Label(
            toolbar,
            text="SOURCE",
            style="Section.TLabel"
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Label(
            toolbar,
            text="Camera"
        ).pack(
            side="left",
            padx=(0, 5)
        )

        self.camera_input = ttk.Spinbox(
            toolbar,
            from_=0,
            to=9,
            textvariable=self.camera_index_var,
            width=3,
            justify="center"
        )

        self.camera_input.pack(
            side="left",
            padx=(0, 7)
        )

        self.webcam_button = ttk.Button(
            toolbar,
            text="Use Webcam",
            style="Primary.TButton",
            command=self.start_webcam
        )

        self.webcam_button.pack(
            side="left",
            padx=(0, 9)
        )

        ttk.Label(
            toolbar,
            text="or",
            style="Sub.TLabel"
        ).pack(
            side="left",
            padx=(0, 9)
        )

        self.open_video_button = ttk.Button(
            toolbar,
            text="Open Video...",
            style="Action.TButton",
            command=self.open_video
        )

        self.open_video_button.pack(
            side="left",
            padx=(0, 14)
        )

        ttk.Separator(
            toolbar,
            orient="vertical"
        ).pack(
            side="left",
            fill="y",
            padx=(0, 14)
        )

        ttk.Label(
            toolbar,
            text="PLAYBACK",
            style="Section.TLabel"
        ).pack(
            side="left",
            padx=(0, 8)
        )

        self.pause_button = ttk.Button(
            toolbar,
            textvariable=self.pause_button_var,
            command=self.toggle_pause
        )

        self.pause_button.pack(
            side="left",
            padx=(0, 7)
        )

        self.stop_button = ttk.Button(
            toolbar,
            text="Stop",
            command=self.stop_processing
        )

        self.stop_button.pack(
            side="left"
        )

        # Output controls.
        output_frame = ttk.Frame(
            self.root,
            padding=(18, 0, 18, 8),
            style="Toolbar.TFrame"
        )

        output_frame.pack(
            fill="x"
        )

        ttk.Label(
            output_frame,
            text="SAVE (OPTIONAL)",
            style="Section.TLabel"
        ).pack(
            side="left",
            padx=(0, 8)
        )

        self.choose_output_button = ttk.Button(
            output_frame,
            text="Choose save location...",
            command=self.choose_output
        )

        self.choose_output_button.pack(
            side="left",
            padx=(0, 7)
        )

        self.clear_output_button = ttk.Button(
            output_frame,
            text="Clear",
            command=self.clear_output
        )

        self.clear_output_button.pack(
            side="left",
            padx=(0, 8)
        )

        self.output_label = ttk.Label(
            output_frame,
            text="No output file selected — preview only",
            style="Sub.TLabel"
        )

        self.output_label.pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Label(
            output_frame,
            text="Ctrl+O: video   Space: pause/resume   Esc: stop",
            style="Sub.TLabel"
        ).pack(
            side="right"
        )

        # Video display.
        video_outer = tk.Frame(
            self.root,
            bg="#101820",
            padx=2,
            pady=2
        )

        video_outer.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(5, 10)
        )

        self.video_label = tk.Label(
            video_outer,
            text="Loading face detector...",
            bg="#101820",
            fg="#e7edf3",
            font=("Segoe UI", 15),
            compound="center"
        )

        self.video_label.pack(
            fill="both",
            expand=True
        )

        self.video_label.bind(
            "<Configure>",
            lambda _event: self._render_current_frame()
        )

        self.root.bind(
            "<Control-o>",
            lambda _event: self.open_video()
        )

        self.root.bind(
            "<space>",
            lambda _event: self.toggle_pause()
        )

        self.root.bind(
            "<Escape>",
            lambda _event: self.stop_processing()
        )

        # Footer.
        footer = ttk.Frame(
            self.root,
            padding=(18, 0, 18, 12),
            style="Toolbar.TFrame"
        )

        footer.pack(
            fill="x"
        )

        top_line = ttk.Frame(
            footer,
            style="Toolbar.TFrame"
        )

        top_line.pack(
            fill="x"
        )

        ttk.Label(
            top_line,
            textvariable=self.status_var
        ).pack(
            side="left"
        )

        ttk.Label(
            top_line,
            textvariable=self.face_count_var,
            font=("Segoe UI", 9, "bold")
        ).pack(
            side="right"
        )

        bottom_line = ttk.Frame(
            footer,
            style="Toolbar.TFrame"
        )

        bottom_line.pack(
            fill="x",
            pady=(5, 0)
        )

        ttk.Label(
            bottom_line,
            textvariable=self.detector_var,
            style="Sub.TLabel"
        ).pack(
            side="left"
        )

        ttk.Label(
            bottom_line,
            textvariable=self.source_label,
            style="Sub.TLabel"
        ).pack(
            side="right"
        )

    # ========================================================
    # INITIALISE MODEL
    # ========================================================

    def _load_detector(self):

        try:

            detector = FaceDetector(
                status_callback=lambda message:
                self.message_queue.put(
                    ("status", message)
                )
            )

            self.message_queue.put(
                ("detector_ready", detector)
            )

        except Exception as exc:

            self.message_queue.put(
                ("detector_error", str(exc))
            )

    def _ready(self):

        if self.detector_error:

            messagebox.showerror(
                "Face detector error",
                self.detector_error
            )

            return False

        if not self.detector_ready:

            messagebox.showinfo(
                "Please wait",
                "The face detector is still loading."
            )

            return False

        if self.is_running:

            messagebox.showinfo(
                "Already running",
                "Stop the current source first."
            )

            return False

        return True

    def _update_controls(self):

        can_start = (
            self.detector_ready
            and not self.detector_error
            and not self.is_running
        )

        source_state = "normal" if can_start else "disabled"
        playback_state = "normal" if self.is_running else "disabled"

        for control in (
            self.camera_input,
            self.webcam_button,
            self.open_video_button,
            self.choose_output_button,
            self.clear_output_button
        ):

            control.configure(
                state=source_state
            )

        self.pause_button.configure(
            state=playback_state
        )

        self.stop_button.configure(
            state=playback_state
        )

    # ========================================================
    # WEBCAM
    # ========================================================

    def start_webcam(self):

        if not self._ready():

            return

        try:

            index = int(
                self.camera_index_var.get().strip()
            )

            if index < 0:

                raise ValueError

        except ValueError:

            messagebox.showerror(
                "Invalid camera",
                "Camera index must be zero or a positive number."
            )

            return

        self._start_worker(
            index,
            is_camera=True,
            source_name=f"Camera {index}"
        )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    def open_video(self):

        if not self._ready():

            return

        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=SUPPORTED_VIDEO_TYPES
        )

        if not path:

            return

        self._start_worker(
            path,
            is_camera=False,
            source_name=Path(path).name
        )

    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    def choose_output(self):

        path = filedialog.asksaveasfilename(
            title="Save detected video as",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 video", "*.mp4")
            ]
        )

        if path:

            self.output_path = Path(path)

            self.output_label.configure(
                text=f"Output: {self.output_path.name}"
            )

    def clear_output(self):

        self.output_path = None

        self.output_label.configure(
            text="No output file selected — preview only"
        )

    # ========================================================
    # START PROCESSING
    # ========================================================

    def _start_worker(
        self,
        source,
        is_camera,
        source_name
    ):

        self.stop_event.clear()
        self.pause_event.clear()

        self.pause_button_var.set(
            "Pause"
        )

        self.is_running = True
        self._update_controls()

        with self.frame_lock:

            self.latest_frame = None
            self.latest_frame_id = 0
            self.latest_face_count = 0
            self.displayed_frame_id = -1

        self.face_count_var.set(
            "Faces detected: 0"
        )

        self.source_label.set(
            f"Source: {source_name}"
        )

        self.status_var.set(
            "Opening video source..."
        )

        self.video_label.configure(
            image="",
            text="Opening video source..."
        )

        self._photo = None

        self.worker_thread = threading.Thread(
            target=self._process_source,
            args=(source, is_camera),
            daemon=True
        )

        self.worker_thread.start()

    # ========================================================
    # PAUSE / RESUME
    # ========================================================

    def toggle_pause(self):

        if not self.is_running:

            return

        if self.pause_event.is_set():

            self.pause_event.clear()

            self.pause_button_var.set(
                "Pause"
            )

            self.status_var.set(
                "Resuming..."
            )

        else:

            self.pause_event.set()

            self.pause_button_var.set(
                "Resume"
            )

            self.status_var.set(
                "Paused"
            )

    # ========================================================
    # STOP
    # ========================================================

    def stop_processing(self):

        if self.is_running:

            self.stop_event.set()

            self.pause_event.clear()

            self.status_var.set(
                "Stopping..."
            )

    # ========================================================
    # MAIN VIDEO PROCESSING
    # ========================================================

    def _process_source(
        self,
        source,
        is_camera
    ):

        capture = None
        converted_temp = None
        writer = None
        silent_temp = None
        current_output = None
        writer_target = None

        frame_count = 0
        latest_count = 0

        # ================================================
        # FPS VARIABLES
        # ================================================

        fps_window_start = time.perf_counter()

        fps_window_frames = 0

        processing_fps = 0.0

        # Reuse the most recent boxes between detector passes.  The detector
        # remains responsive, while the video pipeline is free to keep up with
        # the camera or the source video's frame rate.
        latest_faces = []
        next_detection_time = 0.0

        source_path = (
            None if is_camera else Path(source)
        )

        try:

            # Open webcam or video.
            capture, converted_temp = open_video_capture(
                source,
                is_camera
            )

            if not capture.isOpened():

                raise RuntimeError(
                    "Could not open the camera or video."
                )

            # Get source FPS.
            fps = capture.get(
                cv2.CAP_PROP_FPS
            )

            if (
                not fps
                or fps < 1
                or fps > 120
            ):

                fps = 30.0

            frame_interval = 1.0 / fps

            self.message_queue.put(
                ("status", "Detecting faces...")
            )

            # Configure output file.
            if self.output_path:

                current_output = self.output_path

                if not is_camera:

                    silent_temp = current_output.with_name(
                        current_output.stem
                        + "_video_only_temp.mp4"
                    )

                    silent_temp.unlink(
                        missing_ok=True
                    )

                    writer_target = silent_temp

                else:

                    writer_target = current_output

            # ============================================
            # VIDEO FRAME LOOP
            # ============================================

            while not self.stop_event.is_set():

                # Pause.
                if self.pause_event.is_set():

                    time.sleep(0.05)

                    continue

                tick = time.monotonic()

                # Capture frame.
                ok, frame = capture.read()

                if not ok or frame is None:

                    if is_camera:

                        raise RuntimeError(
                            "Camera stopped sending frames."
                        )

                    break

                # ========================================
                # FACE DETECTION
                # ========================================

                if time.monotonic() >= next_detection_time:

                    latest_faces = self.detector.detect(
                        frame
                    )

                    next_detection_time = (
                        time.monotonic()
                        + DETECTION_INTERVAL_SECONDS
                    )

                faces = latest_faces

                latest_count = len(faces)

                # ========================================
                # DRAW FACE BOUNDING BOXES
                # ========================================

                box_thickness = max(
                    FACE_BOX_THICKNESS,
                    round(max(frame.shape[:2]) / 720)
                )

                label_font_scale = min(
                    1.0,
                    max(0.62, max(frame.shape[:2]) / 1080 * 0.72)
                )

                label_thickness = max(
                    1,
                    box_thickness - 1
                )

                for x, y, w, h, confidence in faces:

                    x1 = max(0, x)
                    y1 = max(0, y)

                    x2 = min(
                        frame.shape[1] - 1,
                        x + w
                    )

                    y2 = min(
                        frame.shape[0] - 1,
                        y + h
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        FACE_BOX_COLOR,
                        box_thickness
                    )

                    if confidence is None:

                        label = "Face"

                    else:

                        label = (
                            f"Face {confidence:.0%}"
                        )

                    (label_width, label_height), label_baseline = (
                        cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            label_font_scale,
                            label_thickness
                        )
                    )

                    label_x = min(
                        x1,
                        max(0, frame.shape[1] - label_width - 8)
                    )

                    label_y = y1 - 7

                    if label_y - label_height - label_baseline - 6 < 0:

                        label_y = min(
                            frame.shape[0] - 5,
                            y1 + label_height + label_baseline + 6
                        )

                    cv2.rectangle(
                        frame,
                        (
                            max(0, label_x - 3),
                            max(
                                0,
                                label_y - label_height
                                - label_baseline - 4
                            )
                        ),
                        (
                            min(frame.shape[1] - 1, label_x + label_width + 3),
                            min(frame.shape[0] - 1, label_y + 4)
                        ),
                        FACE_BOX_COLOR,
                        cv2.FILLED
                    )

                    cv2.putText(
                        frame,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale,
                        (15, 27, 20),
                        label_thickness,
                        cv2.LINE_AA
                    )

                # ========================================
                # FPS CALCULATION
                # ========================================

                fps_window_frames += 1

                fps_elapsed = (
                    time.perf_counter()
                    - fps_window_start
                )

                # Update FPS every second.
                if fps_elapsed >= 1.0:

                    processing_fps = (
                        fps_window_frames / fps_elapsed
                    )

                    fps_window_frames = 0

                    fps_window_start = (
                        time.perf_counter()
                    )

                # ========================================
                # DISPLAY FACE COUNT + FPS
                # ========================================

                hud_text = (
                    f"Faces: {latest_count}"
                    f"  |  FPS: {processing_fps:.1f}"
                )

                (hud_width, hud_height), hud_baseline = (
                    cv2.getTextSize(
                        hud_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        2
                    )
                )

                hud_x = max(
                    12,
                    frame.shape[1] - hud_width - 18
                )

                cv2.rectangle(
                    frame,
                    (hud_x - 6, 5),
                    (
                        min(frame.shape[1] - 6, hud_x + hud_width + 6),
                        17 + hud_height + hud_baseline
                    ),
                    (20, 35, 25),
                    cv2.FILLED
                )

                cv2.putText(
                    frame,
                    hud_text,
                    (hud_x, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # ========================================
                # SAVE OUTPUT VIDEO
                # ========================================

                if current_output and writer is None:

                    height, width = frame.shape[:2]

                    output_fps = (
                        fps
                        if not is_camera
                        else min(max(fps, 15.0), 30.0)
                    )

                    fourcc = cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )

                    writer = cv2.VideoWriter(
                        str(writer_target),
                        fourcc,
                        output_fps,
                        (width, height)
                    )

                    if not writer.isOpened():

                        raise RuntimeError(
                            "Could not create output MP4."
                        )

                if writer is not None:

                    writer.write(frame)

                # ========================================
                # UPDATE GUI FRAME
                # ========================================

                with self.frame_lock:

                    # OpenCV returns a new frame for each capture.read().
                    # The worker does not mutate this frame after publishing
                    # it, so a costly full-frame copy is unnecessary.
                    self.latest_frame = frame

                    self.latest_frame_id += 1
                    self.latest_face_count = latest_count

                frame_count += 1

                # Maintain original video playback speed.
                if not is_camera:

                    remaining = (
                        frame_interval
                        - (time.monotonic() - tick)
                    )

                    if remaining > 0:

                        time.sleep(remaining)

            # ============================================
            # FINISH PROCESSING
            # ============================================

            if (
                frame_count == 0
                and not self.stop_event.is_set()
            ):

                raise RuntimeError(
                    "No frames were read from the source."
                )

            if writer is not None:

                writer.release()

                writer = None

            # ============================================
            # RESTORE VIDEO AUDIO
            # ============================================

            if silent_temp and silent_temp.exists():

                ffmpeg = find_ffmpeg()

                muxed = False

                if (
                    ffmpeg
                    and source_path
                    and source_path.exists()
                ):

                    self.message_queue.put(
                        (
                            "status",
                            "Restoring original video audio..."
                        )
                    )

                    mux_temp = current_output.with_name(
                        current_output.stem
                        + "_mux_temp.mp4"
                    )

                    command = [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(silent_temp),
                        "-i",
                        str(source_path),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a?",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-shortest",
                        "-movflags",
                        "+faststart",
                        str(mux_temp)
                    ]

                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True
                    )

                    if (
                        result.returncode == 0
                        and mux_temp.exists()
                    ):

                        mux_temp.replace(
                            current_output
                        )

                        muxed = True

                    else:

                        mux_temp.unlink(
                            missing_ok=True
                        )

                if not muxed:

                    silent_temp.replace(
                        current_output
                    )

            message = (
                "Stopped."
                if self.stop_event.is_set()
                else "Video processing complete."
            )

            if (
                current_output
                and current_output.exists()
            ):

                message += (
                    f" Saved: {current_output.name}"
                )

            self.message_queue.put(
                ("finished", message)
            )

        except Exception as exc:

            self.message_queue.put(
                ("worker_error", str(exc))
            )

        finally:

            if writer is not None:

                writer.release()

            if capture is not None:

                capture.release()

            if converted_temp is not None:

                converted_temp.unlink(
                    missing_ok=True
                )

            if (
                silent_temp is not None
                and silent_temp.exists()
            ):

                if (
                    current_output
                    and not current_output.exists()
                ):

                    try:

                        silent_temp.replace(
                            current_output
                        )

                    except OSError:

                        silent_temp.unlink(
                            missing_ok=True
                        )

                else:

                    silent_temp.unlink(
                        missing_ok=True
                    )

            self.message_queue.put(
                ("worker_done", None)
            )

    # ========================================================
    # REFRESH GUI
    # ========================================================

    def _refresh_ui(self):

        try:

            while True:

                kind, value = (
                    self.message_queue.get_nowait()
                )

                if kind == "status":

                    self.status_var.set(
                        value
                    )

                elif kind == "detector_ready":

                    self.detector = value

                    self.detector_ready = True

                    self.detector_var.set(
                        f"Detector: {value.kind}"
                    )

                    self.status_var.set(
                        "Ready. Start webcam or open video."
                    )

                    self.video_label.configure(
                        text=(
                            "Choose Use Webcam or Open Video...\n"
                            "Tip: set an optional save location first."
                        ),
                        image=""
                    )

                    self._update_controls()

                elif kind == "detector_error":

                    self.detector_error = value

                    self.status_var.set(
                        "Face detector could not start."
                    )

                    self.video_label.configure(
                        text="Face detector error",
                        image=""
                    )

                    self._update_controls()

                elif kind == "finished":

                    self.status_var.set(
                        value
                    )

                elif kind == "worker_error":

                    self.status_var.set(
                        "Processing error."
                    )

                    messagebox.showerror(
                        "Processing error",
                        value
                    )

                elif kind == "worker_done":

                    self.is_running = False

                    self.pause_button_var.set(
                        "Pause"
                    )

                    self.pause_event.clear()

                    self._update_controls()

        except queue.Empty:

            pass

        # Get latest video frame.
        with self.frame_lock:

            frame = self.latest_frame

            frame_id = self.latest_frame_id
            face_count = self.latest_face_count

        if (
            frame is not None
            and frame_id != self.displayed_frame_id
        ):

            self._display_frame(
                frame
            )

            self.displayed_frame_id = frame_id

            self.face_count_var.set(
                f"Faces detected: {face_count}"
            )

        self.root.after(
            UI_REFRESH_INTERVAL_MS,
            self._refresh_ui
        )

    # ========================================================
    # DISPLAY VIDEO
    # ========================================================

    def _display_frame(self, frame):

        self._render_frame_array(
            frame
        )

    def _render_current_frame(self):

        with self.frame_lock:

            frame = self.latest_frame

        if frame is not None:

            self._render_frame_array(
                frame
            )

    def _render_frame_array(self, frame):

        label_width = max(
            self.video_label.winfo_width(),
            320
        )

        label_height = max(
            self.video_label.winfo_height(),
            220
        )

        frame_height, frame_width = frame.shape[:2]

        scale = min(
            1.0,
            (label_width - 10) / frame_width,
            (label_height - 10) / frame_height
        )

        if scale < 1.0:

            render_frame = cv2.resize(
                frame,
                (
                    max(1, round(frame_width * scale)),
                    max(1, round(frame_height * scale))
                ),
                interpolation=cv2.INTER_AREA
            )

        else:

            render_frame = frame

        rgb = cv2.cvtColor(
            render_frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        photo = ImageTk.PhotoImage(
            image=image
        )

        self.video_label.configure(
            image=photo,
            text=""
        )

        self.video_label.image = photo

        self._photo = photo

    # ========================================================
    # CLOSE APPLICATION
    # ========================================================

    def _on_close(self):

        self.stop_event.set()

        self.pause_event.clear()

        self.root.after(
            100,
            self._finish_close
        )

    def _finish_close(self):

        if (
            self.worker_thread
            and self.worker_thread.is_alive()
        ):

            self.root.after(
                100,
                self._finish_close
            )

        else:

            self.root.destroy()


# ============================================================
# RUN APPLICATION
# ============================================================

def main():

    root = tk.Tk()

    FaceDetectionApp(root)

    root.mainloop()


if __name__ == "__main__":

    main()
