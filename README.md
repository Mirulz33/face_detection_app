# Face Detection Studio

Face Detection Studio is a Python desktop application that detects human faces from live webcam feeds and existing video files. It uses OpenCV and the YuNet face detection model to locate faces and draw rectangular bounding boxes around detected faces.

The application displays the number of detected faces, detection confidence scores (when YuNet is used), and processing speed in Frames Per Second (FPS). Users can also save the processed video as an MP4 file.

## Features

- **Live Webcam Detection:** Detect faces directly from a live webcam.
- **Multiple Camera Support:** Select different cameras using the camera index.
- **Video File Detection:** Open and process existing video files.
- **Multiple Face Detection:** Detect multiple faces within the same frame.
- **YuNet AI Model:** Use YuNet to detect faces and display detection confidence scores.
- **Haar Cascade Fallback:** Use OpenCV's built-in Haar Cascade detector when YuNet cannot be downloaded or loaded.
- **Real-Time FPS Counter:** Display the number of frames processed per second.
- **Face Counter:** Display the number of detected faces.
- **Detection Optimization:** Perform face detection at a target interval of approximately 15 detection passes per second and reuse the latest detected face boxes between detection passes.
- **Image Resizing:** Reduce large video frames to a maximum dimension of 640 pixels for face detection to reduce processing workload.
- **Video Controls:** Pause, resume, and stop video processing.
- **Save Processed Video:** Export annotated videos in MP4 format.
- **FFmpeg Support:** Use FFmpeg for additional video decoding support and original audio preservation when possible.
- **Local Processing:** Process webcam footage and video files locally without uploading them to external servers.

## Face Detection Model

This application uses **YuNet**, a lightweight deep-learning face detection model distributed through the OpenCV Zoo project.

YuNet analyses video frames and detects human faces. It provides information such as:

- Face location (X and Y coordinates).
- Face width and height.
- Face detection confidence score.

OpenCV uses the detection results to draw rectangular bounding boxes around detected faces.

### YuNet Configuration

The application uses the following YuNet configuration:

| Parameter | Value | Description |
|---|---:|---|
| Input size | 320 × 320 initially | Initial model input size |
| Confidence threshold | 0.60 | Minimum confidence score for detection |
| NMS threshold | 0.30 | Controls overlapping detection removal |
| Top K | 5000 | Maximum candidate detections considered |

The detection input size is adjusted according to the resized video frame.

### Fallback Model: Haar Cascade

If the YuNet model cannot be downloaded or loaded, the application uses OpenCV's built-in Haar Cascade face detector.

Haar Cascade is a traditional computer vision face detection method that uses visual patterns to identify faces.

It provides basic face detection functionality. However, unlike YuNet, the fallback detector does not return the same confidence score information.

## Requirements

- Python 3.10 or newer.
- OpenCV Python package.
- Pillow library.
- Tkinter for the desktop graphical interface.
- A webcam for live face detection.
- Internet connection during the first launch to download the YuNet model, unless the model file is already available locally.
- FFmpeg is optional but recommended for additional video decoding support and audio preservation.

Use a Python version compatible with the OpenCV package available for your operating system.

## Installation on Windows

### 1. Install Python

Download Python from the official website:

https://www.python.org/downloads/

During installation, select:

**Add Python to PATH**

Make sure the standard Tkinter components are included in the Python installation.

### 2. Extract the project

Extract the Face Detection Studio ZIP file into a folder.

Example:

```text
C:\Users\YourName\Desktop\face_detection_studio
```

### 3. Run the application

Double-click:

`run_windows.bat`

The batch file creates a Python virtual environment, installs the required Python packages, and starts the application.

### 4. Manual installation

Alternatively, open PowerShell inside the project folder and execute these commands separately:

Create a virtual environment:

```powershell
python -m venv .venv
```

Upgrade pip:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the application:

```powershell
.\.venv\Scripts\python.exe face_detection_app.py
```

### 5. Verify OpenCV installation

To check whether OpenCV is installed correctly:

```powershell
python -c "import cv2; print(cv2.__version__)"
```

If the command displays an OpenCV version number, the installation is successful.

## FFmpeg Installation on Windows

FFmpeg is optional but recommended.

It provides additional video decoding support and helps preserve original audio when exporting processed video files.

Install FFmpeg using Windows Package Manager:

```powershell
winget install --id Gyan.FFmpeg --exact
```

After installation, open a new PowerShell window and verify:

```powershell
ffmpeg -version
```

If FFmpeg is installed correctly, its version information will be displayed.

## Installation on Linux

For Ubuntu, Linux Mint, Kali Linux, and similar distributions, install the required system packages first.

Update the package list:

```bash
sudo apt update
```

Install system dependencies:

```bash
sudo apt install -y python3 python3-venv python3-tk ffmpeg
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python face_detection_app.py
```

Make sure your user account has permission to access the webcam device, such as `/dev/video0`.

## Using the Application

### 1. Start the Application

Run `run_windows.bat` on Windows or execute the Python script manually.

Wait until the application displays that the face detector is ready.

The application automatically downloads the YuNet model during the first launch if the model file is missing.

### 2. Live Webcam Detection

To use the webcam:

1. Locate the Camera input in the application.
2. Enter the camera index.
3. Start with camera index `0` for the default webcam.
4. Click **Use Webcam**.
5. The webcam begins capturing video frames.
6. YuNet analyses the frames and detects human faces.
7. Green rectangular bounding boxes appear around detected faces.

For additional cameras, try index `1`, `2`, or another available camera number.

### 3. Video File Detection

To detect faces in an existing video:

1. Click **Open Video...**.
2. Select an existing video file.
3. The application reads the video frame by frame.
4. YuNet analyses the frames and detects human faces.
5. Green bounding boxes appear around detected faces.

The application attempts to open videos directly using OpenCV. If OpenCV cannot read the video, it attempts to use FFmpeg as a fallback.

### 4. Video Controls

The application provides the following controls:

| Button | Function |
|---|---|
| Use Webcam | Start live webcam face detection |
| Open Video... | Select an existing video file |
| Pause | Pause the current processing session |
| Resume | Continue processing |
| Stop | Stop the current processing session |
| Choose save location... | Select the output MP4 destination |
| Clear | Disable video saving |

### 5. Keyboard Shortcuts

The application also supports keyboard shortcuts:

| Shortcut | Function |
|---|---|
| Ctrl + O | Open a video file |
| Space | Pause or resume processing |
| Esc | Stop processing |

### 6. Save Processed Video

The application allows users to save annotated video files.

To save the processed video:

1. Click **Choose save location...**.
2. Select the destination folder.
3. Enter an output filename.
4. Start webcam detection or open a video file.
5. The application saves the processed frames in MP4 format.

For an existing video file, the application attempts to restore the original audio using FFmpeg.

If FFmpeg is unavailable or audio restoration fails, the application still attempts to save the processed video without audio.

For webcam recordings, only video is saved. Audio recording from the microphone is not implemented.

## FPS (Frames Per Second)

FPS stands for Frames Per Second. It measures how many video frames the application processes in one second.

The application calculates processing FPS using the number of processed frames divided by the elapsed time.

The formula is:

```text
FPS = Number of Processed Frames / Elapsed Time
```

For example, if the application processes 30 frames in approximately one second, its processing speed is approximately 30 FPS.

The FPS counter is displayed on the video together with the number of detected faces.

### Face Detection Frequency and FPS

It is important to distinguish between detection frequency and processing FPS.

**Detection Frequency:**

The application uses:

```python
DETECTION_INTERVAL_SECONDS = 1.0 / 15.0
```

This sets a target interval of approximately 0.067 seconds between detection passes, equivalent to up to approximately 15 detection passes per second under normal conditions.

The application reuses the latest detected face locations between detection passes. This helps reduce the computational workload while continuing to display and save the video frames.

**Processing FPS:**

The FPS counter measures the number of video frames processed by the application over approximately one-second intervals.

Therefore, detection frequency and processing FPS are not necessarily the same.

Actual performance depends on:

- CPU performance.
- Video resolution.
- Webcam capture rate.
- Number of detected faces.
- Face detection model.
- Video decoding performance.

For recorded videos, playback is paced according to the video's original frame rate. Therefore, the displayed FPS is not an unrestricted maximum-speed benchmark.

## Detection Optimization

The application includes two main optimizations to reduce the processing workload.

### 1. Image Resizing

The application uses:

```python
DETECTION_MAX_DIMENSION = 640
```

If the original frame is larger than 640 pixels in either dimension, it is resized before face detection.

For example:

- Original frame: 1920 × 1080 pixels.
- Detection frame: 640 × 360 pixels.

After face detection, the detected face coordinates are scaled back to the original frame dimensions.

This reduces the amount of image data that the detector needs to process.

### 2. Detection Interval

The application performs face detection at a target interval of approximately 15 detection passes per second.

Instead of running the face detector on every captured frame, it reuses the most recent face detection results between detection passes.

This helps reduce repeated AI inference and CPU usage.

However, the displayed face boxes may temporarily use the previous detection results between detection passes.

## Video Format Support

The application is designed to handle common video formats, including:

| Format | Extension |
|---|---|
| MP4 | .mp4 |
| AVI | .avi |
| MOV | .mov |
| MKV | .mkv |
| WMV | .wmv |
| FLV | .flv |
| WebM | .webm |
| M4V | .m4v |
| MPEG | .mpeg, .mpg |
| 3GP | .3gp |
| Transport Stream | .ts, .mts, .m2ts |
| OGG Video | .ogv |

Actual compatibility depends on the codecs supported by OpenCV and FFmpeg on the computer.

Installing FFmpeg may help open video files that OpenCV cannot decode directly.

However, support for every video format and codec cannot be guaranteed.

## Limitations

- Face detection may not identify every face in difficult lighting conditions, unusual angles, or partially obstructed faces.
- YuNet confidence scores are only available when the YuNet detector is successfully loaded.
- Haar Cascade fallback does not provide the same confidence scores.
- Detection boxes are reused between detection passes, so their positions may not update on every video frame.
- Processing FPS depends on computer hardware, video resolution, and video decoding performance.
- All video formats and codecs are not guaranteed to work.
- Audio restoration requires FFmpeg and may fail for some source files.
- Webcam recording saves video only and does not capture microphone audio.
- The application performs face detection, not face recognition. It does not identify the names or identities of people.
- An internet connection is required for the initial YuNet model download unless the model file is already available locally.

## Privacy

The application processes webcam footage and video files locally on the computer.

Video frames are not uploaded to an external server.

An internet connection is used to download the YuNet model when it is not already available in the project.

## Project Structure

The project folder is organized as follows:

```text
face_detection_studio/
|
|-- face_detection_app.py
|-- requirements.txt
|-- run_windows.bat
|-- run_linux.sh
|-- README.md
|
`-- face_detection_app/
    `-- models/
        `-- face_detection_yunet_2023mar.onnx
            # Downloaded automatically if missing
```

The exact folder arrangement may vary depending on where the project is extracted. The Python script stores the downloaded model in a `models` folder beside itself.

## Python Dependencies

The application uses the following libraries and tools:

| Library / Tool | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV (opencv-python) | Webcam access, video processing, and face detection |
| YuNet | AI face detection model |
| Haar Cascade | Fallback face detection method |
| Pillow | Convert and display video frames in the desktop interface |
| Tkinter | Create the desktop graphical user interface |
| NumPy | Image-processing dependency used by OpenCV |
| FFmpeg | Additional video decoding and audio restoration |

The Python dependencies can be installed using:

```bash
python -m pip install -r requirements.txt
```

FFmpeg is an optional external application and is not installed through `requirements.txt`.

## Model Credit and License

This application uses the **YuNet Face Detection Model** from the OpenCV Zoo project.

Model information:

- Model name: YuNet
- Model file: `face_detection_yunet_2023mar.onnx`
- Copyright: Copyright (c) 2020 Shiqi Yu
- Model license: MIT License
- Distribution: OpenCV Zoo

Official repository:

https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Official license:

https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE

The YuNet model is distributed under the MIT License. The license permits use, modification, and redistribution, provided that the original copyright and permission notice are included in copies or substantial portions of the software.

The MIT license notice for YuNet applies to the model and does not automatically determine the license of the Face Detection Studio application code.

## Conclusion

Face Detection Studio demonstrates the use of computer vision and AI-based face detection for live webcam feeds and recorded video files.

By integrating Python, OpenCV, and YuNet, the application can detect multiple faces, display bounding boxes and confidence scores, measure processing FPS, and save annotated videos.

The application also uses detection interval control and image resizing to reduce the processing workload while maintaining continuous video playback.

Overall, Face Detection Studio provides a simple desktop application for demonstrating real-time face detection on Windows and Linux systems.