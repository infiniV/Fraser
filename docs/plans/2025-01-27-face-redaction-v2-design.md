# Fraser Video Face Redaction v2.0 - Design Document

**Date:** 2025-01-27
**Status:** Approved
**Author:** Design Session

## Executive Summary

Complete redesign of the Fraser video face redaction pipeline to achieve HIPAA-compliant, high-performance video anonymization for day-long surveillance footage from Tapo cameras (720p, 15fps).

### Key Improvements Over v1.0

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| Detection Model | YOLOv8 person fallback | YOLOv11n-face (dedicated) |
| Tracking | None | ByteTrack + Kalman filter |
| Temporal Consistency | None (flicker possible) | 5-frame conservative buffer |
| Resolution | Native only | Configurable (144p-720p) |
| Progress Updates | HTTP polling 500ms | WebSocket real-time |
| Redaction | Blur, Black, Color | Black, Color only (irreversible) |
| Audit Trail | Basic JSON | JSON + thumbnail montage |
| Expected Speed | ~30 FPS | ~220 FPS (at 360p) |

## Requirements

### Source Material
- **Camera:** TP-Link Tapo
- **Resolution:** 720p (1280x720)
- **Frame Rate:** 15 FPS
- **Duration:** Up to 24+ hours per file
- **Source:** SD card file selection

### Compliance
- HIPAA-compliant (zero-miss philosophy)
- Irreversible redaction (no blur)
- Full audit trail for compliance verification

### Performance Targets
- Process 24-hour video in <2 hours
- Real-time progress feedback
- Crash recovery for long videos

## Architecture

```
INPUT: 720p 15fps video file

PHASE 1: DECODE + BUFFER
    Frame Buffer (60 frames = 4 seconds)
    Maintains sliding window for temporal context

PHASE 2: DETECTION (GPU)
    Downscale to detection resolution (default 360p)
    Batch inference (32 frames per batch)
    YOLOv11n-face, FP16, confidence 0.25
    Scale boxes back to native resolution

PHASE 3: TRACKING (CPU)
    ByteTrack association
      - High confidence matches (>0.5)
      - Low confidence recovery (0.1-0.5)
    Kalman filter smoothing
    Track lifecycle management

PHASE 4: TEMPORAL EXTENSION
    Extend all tracks by 5 frames each direction
    Interpolate positions for extended frames
    Merge overlapping boxes

PHASE 5: REDACTION (GPU)
    Apply 20% padding to bounding boxes
    Draw solid rectangle (black or user color)
    Clamp to frame boundaries

PHASE 6: ENCODE
    NVENC H.264 if available, else libx264
    CRF 23, preset "fast"
    Maintain original audio stream

PHASE 7: AUDIT (parallel)
    Sample thumbnail every 30 seconds
    Generate JSON metadata
    Create thumbnail montage on completion

OUTPUT:
    {input}_redacted.mp4
    {input}_redacted_audit.json
    {input}_redacted_audit.png
```

## Model Selection

### Primary: YOLOv11n-face

| Metric | Value |
|--------|-------|
| Source | [YapaLab/yolo-face](https://github.com/YapaLab/yolo-face) |
| Size | 5.2 MB |
| WIDERFace Easy | 94.2% mAP |
| WIDERFace Medium | 92.1% mAP |
| WIDERFace Hard | 81.0% mAP |

**Rationale:** Best accuracy-to-speed ratio, proven WIDERFace performance.

### Alternative: YOLOv12n-face

For "Maximum Recall" mode when zero-miss is critical.
- Attention-based architecture
- Slightly better recall
- ~9% slower than YOLOv11n

### Model Download URLs

```
yolov11n-face.pt: https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11n-face.pt
yolov11s-face.pt: https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11s-face.pt
yolov11m-face.pt: https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11m-face.pt
yolov11l-face.pt: https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11l-face.pt
```

## Resolution Downscaling

Detection runs at reduced resolution for speed, boxes scaled back to native for redaction.

| Preset | Dimensions | Speed vs Native | Use Case |
|--------|------------|-----------------|----------|
| 144p | 256x144 | ~40x | Maximum speed, clear footage |
| 240p | 416x240 | ~27x | Fast processing |
| **360p** | **640x360** | **~17x** | **Default - matches YOLO training** |
| Native | 1280x720 | 1x | Maximum accuracy |

**Default:** 360p (640x360) - optimal because YOLO models train on 640px input.

## ByteTrack Integration

### Why ByteTrack

Standard trackers discard low-confidence detections (< 0.5). For HIPAA compliance, faces that turn sideways or become blurry drop to 0.3-0.4 confidence. ByteTrack's two-pass association recovers these "ghost" detections.

### Algorithm

```python
# Pass 1: Match high-confidence detections to tracks
matched_tracks, unmatched_tracks, unmatched_dets = associate(
    tracks, high_conf_dets, iou_threshold=0.3
)

# Pass 2: Match remaining tracks to LOW-confidence detections
matched_tracks_2, _, _ = associate(
    unmatched_tracks, low_conf_dets, iou_threshold=0.5
)
```

### Kalman Filter

- Predicts face position when detection fails
- Smooths bounding box jitter
- Coasts through brief occlusions (up to 30 frames)

## Temporal Buffer

### Conservative Coverage Strategy

For each track, extend redaction by N frames in each direction:

```
Detection: frames [100, 200]
Buffer: 5 frames
Redaction: frames [95, 205]
```

**Rationale:**
- Pre-buffer catches faces entering frame before detection fires
- Post-buffer ensures faces leaving frame stay covered
- 5 frames at 15fps = 0.33 seconds - sufficient for motion

### Interpolation

When extending into frames without direct detection:
- Use Kalman filter predicted position
- Or linear interpolation between nearest detections

## Redaction Modes

### Removed: Blur

Blur/pixelation is **removed** because:
- Reversible via AI reconstruction (GANs)
- Fails HIPAA "Safe Harbor" standard
- Creates liability

### Kept: Black Rectangle

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
```

### Kept: Solid Color

```python
color_bgr = hex_to_bgr(user_color)  # e.g., "#FF0000" -> (0, 0, 255)
cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, -1)
```

### Padding

All bounding boxes expanded by 20% to cover ears, hair, chin:

```python
def apply_padding(x1, y1, x2, y2, padding=0.20, img_h, img_w):
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * padding), int(h * padding)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y)
    )
```

## WebSocket Progress

### Replace HTTP Polling

Current: Poll `/job/{id}` every 500ms
New: WebSocket stream at `/ws/progress/{job_id}`

### Message Format

```json
{
  "type": "progress",
  "frame": 12345,
  "total_frames": 1296000,
  "percent": 0.95,
  "fps": 220.5,
  "active_tracks": 3,
  "total_faces_detected": 45678,
  "estimated_remaining_sec": 360,
  "detection_resolution": "360p"
}
```

## Audit Trail

### JSON Metadata

```json
{
  "version": "2.0",
  "input_file": "camera1_2025-01-15.mp4",
  "output_file": "camera1_2025-01-15_redacted.mp4",
  "processing_date": "2025-01-16T14:30:00Z",
  "duration_seconds": 86400,
  "total_frames": 1296000,
  "processing_time_seconds": 5880,
  "config": {
    "model": "yolov11n-face.pt",
    "detection_resolution": "360p",
    "confidence": 0.25,
    "temporal_buffer": 5,
    "redaction_mode": "black",
    "padding": 0.20
  },
  "statistics": {
    "total_detections": 45678,
    "unique_tracks": 234,
    "frames_with_faces": 89012,
    "average_faces_per_frame": 1.2
  },
  "tracks": [
    {
      "track_id": 1,
      "first_frame": 100,
      "last_frame": 450,
      "detection_count": 320,
      "avg_confidence": 0.87
    }
  ],
  "checksums": {
    "input_sha256": "abc123...",
    "output_sha256": "def456..."
  }
}
```

### Thumbnail Montage

- One thumbnail per 30 seconds of video
- Size: 64x36 pixels each
- Arranged in grid layout
- Shows redacted frames (proof of redaction)
- ~3MB PNG for 24-hour video

## Configuration Defaults

```python
CONFIG = {
    # Model
    "model": "yolov11n-face.pt",
    "confidence": 0.25,

    # Resolution
    "detection_resolution": "360p",

    # Tracking
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "track_buffer": 30,

    # Temporal
    "temporal_buffer_frames": 5,

    # Redaction
    "redaction_mode": "black",
    "redaction_color": "#000000",
    "bbox_padding": 0.20,

    # Processing
    "batch_size": 32,
    "use_fp16": True,

    # Audit
    "thumbnail_interval_sec": 30,

    # Checkpoints
    "checkpoint_interval_frames": 4500,
}
```

## Performance Estimates

### RTX 4070, 720p 15fps Input

| Resolution | Detection FPS | 24hr Video | Realtime Multiple |
|------------|---------------|------------|-------------------|
| 144p | ~500 | ~43 min | 40x |
| 240p | ~350 | ~62 min | 27x |
| **360p** | **~220** | **~98 min** | **17x** |
| Native | ~80 | ~4.5 hr | 5x |

### Memory Usage

| Component | VRAM |
|-----------|------|
| YOLOv11n-face model | ~200 MB |
| Frame batch (32 @ 360p) | ~150 MB |
| Working buffers | ~200 MB |
| **Total** | **~550 MB** |

## Implementation Plan

### Files to Modify

| File | Change | Est. Lines |
|------|--------|------------|
| `python/processing/pipeline.py` | Major rewrite | ~450 |
| `python/processing/tracker.py` | **NEW** | ~250 |
| `python/processing/anonymizer.py` | Remove blur | -30 |
| `python/server.py` | WebSocket | ~80 |
| `python/requirements.txt` | Add deps | +3 |
| `renderer/app.js` | UI updates | ~100 |
| `renderer/index.html` | New elements | ~30 |
| `src/handlers/processingHandlers.ts` | WebSocket | ~60 |

### New Dependencies

```txt
scipy>=1.11.0    # Linear assignment for ByteTrack
lap>=0.4.0       # Fast Hungarian algorithm
```

### UI Changes

**Remove:**
- Blur redaction option

**Add:**
- Detection resolution dropdown (144p, 240p, 360p, Native)
- Model selector (Standard / Maximum Recall)
- Real-time FPS display
- Active tracks count
- Estimated time remaining

## References

- [YapaLab/yolo-face](https://github.com/YapaLab/yolo-face) - Face detection models
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864) - Tracking algorithm
- [WIDERFace Benchmark](http://shuoyang1213.me/WIDERFACE/) - Face detection benchmark
- [YOLOv12 Paper](https://github.com/sunsmarterjie/yolov12) - Latest YOLO architecture
