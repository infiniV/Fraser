# Fraser - Face Anonymization Desktop App

**Date:** 2025-11-26
**Status:** Design Complete
**Version:** 1.0

## Overview

Fraser (Face + Eraser) is a desktop application for batch video face anonymization, designed for hospital data collection compliance (HIPAA/GDPR). It processes weeks of footage with GPU acceleration, anonymizing all detected faces.

### Key Requirements

- **Use Case:** Hospital data anonymization - batch processing SD card footage
- **Secondary:** RTSP stream support for live camera feeds
- **Detection:** High recall (miss no faces) - compliance critical
- **Speed:** GPU-accelerated, process weeks of footage efficiently
- **Platform:** Windows, macOS, Linux with GPU/CPU flexibility
- **Output:** Same format as input, with summary reports

## Architecture

### High-Level Architecture

```
+-------------------------------------------------------------+
|                    Electron Shell                            |
|  +-------------------------------------------------------+  |
|  |              Web Frontend (HTML/CSS/JS)               |  |
|  |   - File/folder picker                                |  |
|  |   - Model selector (YOLOv8n/m/l, YOLO11)              |  |
|  |   - Anonymization mode (blur/black/color)             |  |
|  |   - Queue display with progress                       |  |
|  |   - Summary reports viewer                            |  |
|  +-------------------------------------------------------+  |
|                         | IPC                                |
|  +-------------------------------------------------------+  |
|  |           Electron Main Process                       |  |
|  |   - Python process lifecycle (spawn/kill)             |  |
|  |   - File system access                                |  |
|  |   - Queue persistence (resume on crash)               |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
                          | HTTP/WebSocket
+-------------------------------------------------------------+
|                  Python Backend                              |
|   - Ultralytics YOLO (face detection)                       |
|   - PyAV (video encoding with anonymization)                |
|   - FastAPI server for IPC                                  |
|   - GPU/CPU auto-detection                                  |
+-------------------------------------------------------------+
```

### Key Components

1. **Electron** - Desktop shell, file access, process management
2. **Python backend** - All ML/video processing
3. **uv** - Bundled for fast Python environment setup
4. **Queue file** - JSON persistence for crash recovery

## Installation & First Run

### Installer Contents (~60-80MB)

```
fraser-setup.exe / fraser.dmg / fraser.AppImage
+-- Electron app (~50MB)
+-- uv binaries (~15MB per platform)
|   +-- win/uv.exe
|   +-- macos/uv
|   +-- linux/uv
+-- UI assets (icons, HTML/CSS/JS)
+-- requirements.compiled (per platform)
    +-- windows_nvidia.compiled
    +-- windows_cpu.compiled
    +-- macos.compiled
    +-- linux.compiled
```

### First Run Flow

1. User launches Fraser
2. Detect GPU: NVIDIA CUDA / AMD ROCm / Apple MPS / CPU
3. Show "Setting up Fraser..." with progress
4. uv creates virtualenv (~30 seconds)
5. uv installs from requirements.compiled:
   - PyTorch (CUDA/MPS/CPU version based on detection)
   - Ultralytics
   - PyAV
   - FastAPI + uvicorn
   - Face detection models (~150MB total for all variants)
6. Verify installation (test import torch, ultralytics)
7. Launch main UI

### Data Paths

- **Windows:** `%APPDATA%\Fraser` (config), `%LOCALAPPDATA%\Fraser` (venv, cache)
- **macOS:** `~/Library/Application Support/Fraser`
- **Linux:** `~/.config/fraser`

## Video Processing Pipeline

### Processing Flow

```
Input Video (SD card / RTSP)
         |
         v
+------------------------------------------+
|  Ultralytics YOLO (stream=True)          |
|  - Memory-efficient frame generator      |
|  - GPU inference (batched if possible)   |
|  - Returns bounding boxes per frame      |
+------------------------------------------+
         |
         v
+------------------------------------------+
|  Anonymization Layer                     |
|  - Black rectangle: cv2.rectangle()      |
|  - Blur: cv2.GaussianBlur() on ROI       |
|  - Solid color: cv2.rectangle(fill)      |
|  - Padding option (expand box 10-20%)    |
+------------------------------------------+
         |
         v
+------------------------------------------+
|  PyAV Encoder                            |
|  - Match input codec/resolution/fps      |
|  - Hardware encoding if available        |
|  - Write to output path                  |
+------------------------------------------+
         |
         v
Output Video + Summary Report (JSON)
```

### Core Processing Code

```python
from ultralytics import YOLO
import av
import cv2

def process_video(input_path, output_path, model_name, anon_mode, confidence=0.3):
    model = YOLO(f"models/{model_name}.pt")

    input_container = av.open(input_path)
    output_container = av.open(output_path, mode='w')

    stream = input_container.streams.video[0]
    output_stream = output_container.add_stream(stream.codec.name, rate=stream.rate)

    face_count = 0

    for frame in input_container.decode(video=0):
        img = frame.to_ndarray(format='bgr24')

        results = model.predict(img, conf=confidence, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_count += 1
            img = apply_anonymization(img, x1, y1, x2, y2, anon_mode)

        out_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = output_stream.encode(out_frame)
        output_container.mux(packet)

    return {"faces_detected": face_count, "frames": stream.frames}
```

### Performance Optimizations

- **Batch inference:** Process multiple frames at once if GPU memory allows
- **Low confidence threshold:** 0.3 for high recall (catch all faces)
- **Stream mode:** `stream=True` for memory-efficient processing

### Available Models

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| YOLOv8n-face | ~6MB | Fastest | Quick previews |
| YOLOv8m-face | ~50MB | Fast | Recommended default |
| YOLOv8l-face | ~85MB | Moderate | Maximum accuracy |
| YOLO11n-face | ~12MB | Fast | Latest architecture |

## Frontend UI Design

### Theme (Dark Mode)

```css
:root {
  --background: #0a0a0a;
  --foreground: #f5f5f5;
  --card: #121212;
  --primary: #7c9082;        /* Sage green */
  --primary-foreground: #000000;
  --muted: #1a1a1a;
  --muted-foreground: #a0a0a0;
  --border: #2a2a2a;
  --destructive: #ef4444;
  --radius: 0.35rem;
  --font-sans: Antic, ui-sans-serif, sans-serif;
  --font-mono: JetBrains Mono, monospace;
}
```

### UI Layout

```
+-------------------------------------------------------------+
| Fraser                                         [-] [=] [x]  |
+-------------------------------------------------------------+
|                                                             |
|  +-------------------------------------------------------+  |
|  |  + Add Files    + Add Folder    + RTSP Stream         |  |
|  +-------------------------------------------------------+  |
|                                                             |
|  +----------------+ +----------------+ +----------------+   |
|  | Model          | | Mode           | | Confidence     |   |
|  | YOLOv8m      v | | Blur         v | | 0.3          v |   |
|  +----------------+ +----------------+ +----------------+   |
|                                                             |
|  Queue                                                      |
|  +-------------------------------------------------------+  |
|  | [check] video_001.mp4    45:00    1,247 faces   Done  |  |
|  | [>]     video_002.mp4    1:23:00  [========  ] 78%    |  |
|  | [o]     video_003.mp4    58:00    Pending             |  |
|  | [o]     video_004.mp4    2:10:00  Pending             |  |
|  +-------------------------------------------------------+  |
|                                                             |
|  Output: /Users/data/anonymized              [Browse]       |
|                                                             |
|  +-------------------+  +-------------------+               |
|  |      > Start      |  |      || Pause     |               |
|  +-------------------+  +-------------------+               |
|                                                             |
|  -----------------------------------------------------------+
|  Processing video_002.mp4 | RTX 3080 | 45 fps              |
+-------------------------------------------------------------+
```

### Component Styling

- **Buttons:** `#7c9082` background, `#000000` text, `0.35rem` radius
- **Cards:** `#121212` background, `#2a2a2a` border
- **Progress bars:** `#7c9082` fill on `#1a1a1a` track
- **Queue items:** Hover `#1a1a1a`, selected `#36443a`
- **Status icons:** `#7c9082` (success), `#ef4444` (error), `#a0a0a0` (pending)

## Project Structure

```
fraser/
+-- package.json
+-- tsconfig.json
+-- vite.config.ts
+-- builder.config.ts
|
+-- assets/
|   +-- UI/
|   |   +-- fraser-icon.ico
|   |   +-- fraser-icon.icns
|   |   +-- fraser-icon.png
|   +-- uv/
|   |   +-- win/uv.exe
|   |   +-- macos/uv
|   |   +-- linux/uv
|   +-- requirements/
|       +-- windows_nvidia.compiled
|       +-- windows_cpu.compiled
|       +-- macos.compiled
|       +-- linux.compiled
|
+-- src/
|   +-- main.ts
|   +-- preload.ts
|   |
|   +-- main-process/
|   |   +-- app.ts
|   |   +-- pythonServer.ts
|   |   +-- virtualEnvironment.ts
|   |   +-- queue.ts
|   |
|   +-- handlers/
|   |   +-- fileHandlers.ts
|   |   +-- queueHandlers.ts
|   |   +-- settingsHandlers.ts
|   |
|   +-- store/
|   |   +-- appConfig.ts
|   |   +-- queueState.ts
|   |
|   +-- constants.ts
|
+-- renderer/
|   +-- index.html
|   +-- styles/
|   |   +-- theme.css
|   |   +-- components.css
|   |   +-- main.css
|   |
|   +-- components/
|   |   +-- Header.js
|   |   +-- AddButtons.js
|   |   +-- Settings.js
|   |   +-- Queue.js
|   |   +-- QueueItem.js
|   |   +-- ProgressBar.js
|   |   +-- OutputSelector.js
|   |   +-- ActionButtons.js
|   |   +-- StatusBar.js
|   |
|   +-- services/
|   |   +-- api.js
|   |   +-- ipc.js
|   |
|   +-- app.js
|
+-- python/
|   +-- server.py
|   +-- processing/
|   |   +-- detector.py
|   |   +-- anonymizer.py
|   |   +-- pipeline.py
|   +-- models/
|   |   +-- yolov8n-face.pt
|   |   +-- yolov8m-face.pt
|   |   +-- yolov8l-face.pt
|   |   +-- yolo11n-face.pt
|   +-- utils/
|       +-- device.py
|       +-- report.py
|
+-- scripts/
|   +-- downloadUV.js
|   +-- makeAssets.js
|   +-- installer.nsh
|
+-- tests/
    +-- unit/
    +-- integration/
```

## Python Backend API

### Endpoints

```python
# GET /health - Server ready check
{"status": "ok", "gpu": "NVIDIA RTX 3080"}

# GET /models - Available models
{"models": [{"id": "yolov8m-face", "name": "YOLOv8 Medium", "size": "50MB"}]}

# POST /process - Start processing
{"file": "/path/to/video.mp4", "output_dir": "/output", "model": "yolov8m-face", "mode": "blur", "confidence": 0.3}

# POST /cancel/{job_id} - Cancel job
{"status": "cancelled"}

# WebSocket /ws - Real-time progress
```

### WebSocket Messages

```python
# Progress update
{"type": "progress", "job_id": "abc123", "percent": 28, "faces_detected": 342, "fps": 45.2}

# Job complete
{"type": "complete", "job_id": "abc123", "report": {"total_frames": 45000, "faces_detected": 1247}}

# Error
{"type": "error", "job_id": "abc123", "message": "Failed to open video file"}
```

## IPC Communication

### Preload Script

```typescript
contextBridge.exposeInMainWorld('electronAPI', {
  selectFiles: () => ipcRenderer.invoke('dialog:selectFiles'),
  selectFolder: () => ipcRenderer.invoke('dialog:selectFolder'),
  selectOutputDir: () => ipcRenderer.invoke('dialog:selectOutputDir'),
  getAppInfo: () => ipcRenderer.invoke('app:info'),
  getPythonStatus: () => ipcRenderer.invoke('python:status'),
  getSettings: () => ipcRenderer.invoke('settings:get'),
  saveSettings: (settings) => ipcRenderer.invoke('settings:save', settings),
  onPythonReady: (callback) => ipcRenderer.on('python:ready', callback),
  onPythonError: (callback) => ipcRenderer.on('python:error', callback),
});
```

## Error Handling & Crash Recovery

### Error Categories

| Error Type | Handling Strategy |
|------------|-------------------|
| Install failure | Show error, offer retry/reinstall |
| Python crash | Auto-restart, resume from last frame |
| Video read error | Skip file, log error, continue queue |
| GPU OOM | Reduce batch size, retry |
| Disk full | Pause, alert user, wait for space |
| Corrupt video | Skip file, mark failed in report |

### Queue State Persistence

```typescript
interface QueueState {
  version: number;
  lastUpdated: string;
  currentJob: {
    id: string;
    file: string;
    lastFrame: number;       // Resume point
    facesDetected: number;
  } | null;
  pendingJobs: Job[];
  completedJobs: Job[];
  failedJobs: FailedJob[];
}
// Saved to: {userData}/queue-state.json
// Updated every 100 frames during processing
```

### Summary Report

```json
{
  "file": "video_002.mp4",
  "output": "video_002_anonymized.mp4",
  "status": "completed",
  "stats": {
    "total_frames": 45000,
    "faces_detected": 1247,
    "processing_time_seconds": 892.5,
    "average_fps": 50.4
  },
  "settings": {
    "model": "yolov8m-face",
    "mode": "blur",
    "confidence": 0.3
  }
}
```

## Build & Packaging

### Build Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build && vite build --config vite.preload.config.ts",
    "download:uv": "node scripts/downloadUV.js",
    "make": "yarn build && electron-builder --config builder.config.ts",
    "make:win": "yarn make --win nsis",
    "make:mac": "yarn make --mac dmg",
    "make:linux": "yarn make --linux AppImage"
  }
}
```

### Package Sizes

| Stage | Size |
|-------|------|
| Installer (Windows) | ~80MB |
| Installer (macOS) | ~75MB |
| Installer (Linux) | ~70MB |
| After setup (CUDA) | ~3.5GB |
| After setup (CPU) | ~1.2GB |
| After setup (macOS MPS) | ~1.5GB |

## References

- [ComfyUI Desktop](https://github.com/Comfy-Org/desktop) - Architecture reference
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/modes/predict/) - Video processing
- [YOLOv8-Face](https://github.com/Yusepp/YOLOv8-Face) - Face detection model
- [PyAV Documentation](https://pyav.org/) - Video encoding
- [Electron Builder](https://www.electron.build/) - Packaging
