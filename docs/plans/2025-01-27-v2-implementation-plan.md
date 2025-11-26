# Fraser v2.0 Face Redaction - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the video face redaction pipeline with ByteTrack tracking, configurable resolution downscaling, WebSocket progress, and HIPAA-compliant solid masking.

**Architecture:** Single-pass processing with 60-frame sliding buffer. Detection at configurable resolution, boxes scaled to native for redaction. ByteTrack maintains temporal consistency with 5-frame conservative buffer.

**Tech Stack:** Python 3.11+, FastAPI, YOLOv11n-face, ByteTrack, PyAV, OpenCV, WebSocket

---

## Parallel Execution Strategy

This plan is organized into **5 parallel tracks** that can be executed simultaneously by different agents:

```
TRACK A: ByteTrack Tracker     ──┐
TRACK B: WebSocket Server      ──┼──▶ TRACK D: Pipeline Core ──▶ TRACK F: Integration
TRACK C: Frontend UI           ──┤
TRACK E: Anonymizer Cleanup    ──┘
```

**Dependencies:**
- Track D depends on Track A (tracker module)
- Track F depends on all tracks
- Tracks A, B, C, E are fully independent

---

## TRACK A: ByteTrack Tracker Module

**Parallelizable:** YES - No dependencies
**Estimated Time:** 25 minutes
**Files:**
- Create: `python/processing/tracker.py`

### Task A1: Create ByteTrack Core Classes

**Files:**
- Create: `python/processing/tracker.py`

**Step 1: Create the tracker module with STrack class**

```python
"""
ByteTrack implementation for face tracking.
Maintains temporal consistency across video frames.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class STrack:
    """Single object track with Kalman filter state."""

    track_id: int
    tlbr: np.ndarray  # top-left, bottom-right [x1, y1, x2, y2]
    score: float
    state: TrackState = TrackState.NEW
    frame_id: int = 0
    start_frame: int = 0
    tracklet_len: int = 0

    # Kalman filter state [x, y, w, h, vx, vy, vw, vh]
    mean: np.ndarray = field(default_factory=lambda: np.zeros(8))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8))

    _count: int = field(default=0, repr=False)

    @staticmethod
    def next_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0

    @property
    def tlwh(self) -> np.ndarray:
        """Convert to top-left width height format."""
        ret = self.tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Convert to center x, center y, width, height."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret

    def predict(self):
        """Predict next state using Kalman filter motion model."""
        # Simple constant velocity model
        # State: [x, y, w, h, vx, vy, vw, vh]
        F = np.eye(8)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = 1  # Position += velocity

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + np.eye(8) * 0.1

        # Update tlbr from predicted state
        x, y, w, h = self.mean[:4]
        self.tlbr = np.array([x - w/2, y - h/2, x + w/2, y + h/2])

    def update(self, new_tlbr: np.ndarray, new_score: float, frame_id: int):
        """Update track with new detection."""
        self.tlbr = new_tlbr
        self.score = new_score
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.TRACKED

        # Update Kalman state
        x, y = (new_tlbr[0] + new_tlbr[2]) / 2, (new_tlbr[1] + new_tlbr[3]) / 2
        w, h = new_tlbr[2] - new_tlbr[0], new_tlbr[3] - new_tlbr[1]

        # Simple update: blend prediction with measurement
        alpha = 0.7
        self.mean[:4] = alpha * np.array([x, y, w, h]) + (1 - alpha) * self.mean[:4]
        # Estimate velocity from change
        if self.tracklet_len > 1:
            self.mean[4:] = np.array([x, y, w, h]) - self.mean[:4]

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED

    @classmethod
    def from_detection(cls, tlbr: np.ndarray, score: float, frame_id: int) -> 'STrack':
        """Create new track from detection."""
        track = cls(
            track_id=cls.next_id(),
            tlbr=tlbr.copy(),
            score=score,
            frame_id=frame_id,
            start_frame=frame_id,
            tracklet_len=1,
        )
        # Initialize Kalman state
        x, y = (tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2
        w, h = tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
        track.mean = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        return track
```

**Step 2: Verify syntax**

Run: `cd /home/raw/fraser/.worktrees/v2-face-redaction && python -c "from python.processing.tracker import STrack, TrackState; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add python/processing/tracker.py
git commit -m "feat(tracker): add STrack class with Kalman filter"
```

---

### Task A2: Add IoU and Linear Assignment Functions

**Files:**
- Modify: `python/processing/tracker.py`

**Step 1: Add IoU calculation and assignment functions**

Append to `python/processing/tracker.py`:

```python
def iou_batch(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.

    Args:
        atlbrs: (N, 4) array of boxes [x1, y1, x2, y2]
        btlbrs: (M, 4) array of boxes [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)))

    # Intersection
    tl = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])  # (N, M, 2)
    br = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])  # (N, M, 2)

    wh = np.maximum(br - tl, 0)  # (N, M, 2)
    intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Union
    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])  # (N,)
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])  # (M,)
    union = area_a[:, None] + area_b[None, :] - intersection

    return intersection / (union + 1e-6)


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List, List, List]:
    """
    Solve linear assignment problem using scipy.

    Args:
        cost_matrix: (N, M) cost matrix (lower is better)
        thresh: Maximum cost threshold for valid assignment

    Returns:
        matches: List of (row, col) matches
        unmatched_a: List of unmatched row indices
        unmatched_b: List of unmatched col indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    except ImportError:
        # Fallback to greedy assignment
        row_indices, col_indices = [], []
        cost_copy = cost_matrix.copy()
        while True:
            if cost_copy.size == 0:
                break
            min_idx = np.unravel_index(np.argmin(cost_copy), cost_copy.shape)
            if cost_copy[min_idx] > thresh:
                break
            row_indices.append(min_idx[0])
            col_indices.append(min_idx[1])
            cost_copy[min_idx[0], :] = np.inf
            cost_copy[:, min_idx[1]] = np.inf

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return matches, unmatched_a, unmatched_b
```

**Step 2: Verify**

Run: `cd /home/raw/fraser/.worktrees/v2-face-redaction && python -c "from python.processing.tracker import iou_batch, linear_assignment; import numpy as np; a = np.array([[0,0,10,10]]); b = np.array([[5,5,15,15]]); print(f'IoU: {iou_batch(a,b)[0,0]:.3f}')"`

Expected: `IoU: 0.143` (approximately)

**Step 3: Commit**

```bash
git add python/processing/tracker.py
git commit -m "feat(tracker): add IoU batch and linear assignment"
```

---

### Task A3: Add ByteTrack Main Class

**Files:**
- Modify: `python/processing/tracker.py`

**Step 1: Add the ByteTracker class**

Append to `python/processing/tracker.py`:

```python
@dataclass
class ByteTrackerConfig:
    """Configuration for ByteTrack."""
    track_high_thresh: float = 0.5      # High confidence detection threshold
    track_low_thresh: float = 0.1       # Low confidence threshold for second association
    new_track_thresh: float = 0.6       # Threshold to create new track
    match_thresh: float = 0.8           # IoU threshold for matching
    track_buffer: int = 30              # Frames to keep lost track
    temporal_buffer: int = 5            # Frames to extend redaction before/after


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Key innovation: Uses low-confidence detections for second-round association
    to recover tracks during occlusion/blur.
    """

    def __init__(self, config: Optional[ByteTrackerConfig] = None):
        self.config = config or ByteTrackerConfig()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

        # For audit trail
        self.all_tracks: List[STrack] = []

        STrack.reset_id()

    def update(self, detections: np.ndarray, scores: np.ndarray) -> List[STrack]:
        """
        Update tracker with new detections.

        Args:
            detections: (N, 4) array of [x1, y1, x2, y2]
            scores: (N,) array of confidence scores

        Returns:
            List of active tracks
        """
        self.frame_id += 1

        # Split detections by confidence
        high_mask = scores >= self.config.track_high_thresh
        low_mask = (scores >= self.config.track_low_thresh) & ~high_mask

        high_dets = detections[high_mask]
        high_scores = scores[high_mask]
        low_dets = detections[low_mask]
        low_scores = scores[low_mask]

        # Predict new locations for all tracked tracks
        for track in self.tracked_stracks:
            track.predict()

        # === First association: high-confidence detections ===
        track_boxes = np.array([t.tlbr for t in self.tracked_stracks]) if self.tracked_stracks else np.empty((0, 4))

        if len(track_boxes) > 0 and len(high_dets) > 0:
            iou_matrix = iou_batch(track_boxes, high_dets)
            cost_matrix = 1 - iou_matrix
            matches, unmatched_tracks, unmatched_dets = linear_assignment(
                cost_matrix, 1 - self.config.match_thresh
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracked_stracks)))
            unmatched_dets = list(range(len(high_dets)))

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracked_stracks[track_idx].update(
                high_dets[det_idx], high_scores[det_idx], self.frame_id
            )

        # === Second association: low-confidence detections with unmatched tracks ===
        unmatched_track_objs = [self.tracked_stracks[i] for i in unmatched_tracks]

        if len(unmatched_track_objs) > 0 and len(low_dets) > 0:
            track_boxes = np.array([t.tlbr for t in unmatched_track_objs])
            iou_matrix = iou_batch(track_boxes, low_dets)
            cost_matrix = 1 - iou_matrix
            matches2, still_unmatched_tracks, _ = linear_assignment(
                cost_matrix, 1 - self.config.match_thresh
            )

            # Update matched tracks with low-confidence detections
            for rel_idx, det_idx in matches2:
                unmatched_track_objs[rel_idx].update(
                    low_dets[det_idx], low_scores[det_idx], self.frame_id
                )

            # Update unmatched_tracks to use actual indices
            unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched_tracks]

        # === Handle unmatched tracks ===
        for track_idx in unmatched_tracks:
            track = self.tracked_stracks[track_idx]
            if self.frame_id - track.frame_id > self.config.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
            else:
                track.mark_lost()
                self.lost_stracks.append(track)

        # === Try to recover lost tracks ===
        lost_track_boxes = np.array([t.tlbr for t in self.lost_stracks]) if self.lost_stracks else np.empty((0, 4))
        remaining_high_dets = high_dets[unmatched_dets] if unmatched_dets else np.empty((0, 4))
        remaining_high_scores = high_scores[unmatched_dets] if unmatched_dets else np.array([])

        if len(lost_track_boxes) > 0 and len(remaining_high_dets) > 0:
            iou_matrix = iou_batch(lost_track_boxes, remaining_high_dets)
            cost_matrix = 1 - iou_matrix
            matches3, unmatched_lost, unmatched_dets = linear_assignment(
                cost_matrix, 1 - self.config.match_thresh
            )

            for lost_idx, det_idx in matches3:
                self.lost_stracks[lost_idx].update(
                    remaining_high_dets[det_idx], remaining_high_scores[det_idx], self.frame_id
                )
                self.lost_stracks[lost_idx].state = TrackState.TRACKED

            # Move recovered tracks back to tracked
            recovered = [self.lost_stracks[i] for i, _ in matches3]
            self.lost_stracks = [t for t in self.lost_stracks if t not in recovered]
        else:
            unmatched_dets = list(range(len(remaining_high_dets)))
            recovered = []

        # === Create new tracks from unmatched high-confidence detections ===
        new_tracks = []
        for det_idx in unmatched_dets:
            if remaining_high_scores[det_idx] >= self.config.new_track_thresh:
                new_track = STrack.from_detection(
                    remaining_high_dets[det_idx],
                    remaining_high_scores[det_idx],
                    self.frame_id
                )
                new_tracks.append(new_track)
                self.all_tracks.append(new_track)

        # Update tracked_stracks list
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]
        self.tracked_stracks.extend(recovered)
        self.tracked_stracks.extend(new_tracks)

        # Clean up old lost tracks
        self.lost_stracks = [
            t for t in self.lost_stracks
            if self.frame_id - t.frame_id <= self.config.track_buffer
        ]

        return self.tracked_stracks

    def get_active_boxes(self, include_lost: bool = True) -> List[Tuple[int, np.ndarray, float]]:
        """
        Get all boxes that should be redacted.

        Returns:
            List of (track_id, tlbr, score) tuples
        """
        result = [(t.track_id, t.tlbr.copy(), t.score) for t in self.tracked_stracks]

        if include_lost:
            # Include recently lost tracks (within temporal buffer)
            for t in self.lost_stracks:
                if self.frame_id - t.frame_id <= self.config.temporal_buffer:
                    result.append((t.track_id, t.tlbr.copy(), t.score))

        return result

    def get_statistics(self) -> dict:
        """Get tracking statistics for audit."""
        return {
            "total_tracks": len(self.all_tracks),
            "active_tracks": len(self.tracked_stracks),
            "lost_tracks": len(self.lost_stracks),
            "removed_tracks": len(self.removed_stracks),
            "frame_id": self.frame_id,
        }
```

**Step 2: Verify ByteTracker**

Run: `cd /home/raw/fraser/.worktrees/v2-face-redaction && python -c "
from python.processing.tracker import ByteTracker
import numpy as np

tracker = ByteTracker()

# Frame 1: Two faces
dets1 = np.array([[100, 100, 200, 200], [300, 100, 400, 200]], dtype=np.float32)
scores1 = np.array([0.9, 0.85])
tracks = tracker.update(dets1, scores1)
print(f'Frame 1: {len(tracks)} tracks')

# Frame 2: Same faces, slightly moved
dets2 = np.array([[105, 102, 205, 202], [295, 98, 395, 198]], dtype=np.float32)
scores2 = np.array([0.88, 0.82])
tracks = tracker.update(dets2, scores2)
print(f'Frame 2: {len(tracks)} tracks (same IDs: {[t.track_id for t in tracks]})')

# Frame 3: One face drops to low confidence
dets3 = np.array([[110, 105, 210, 205], [290, 95, 390, 195]], dtype=np.float32)
scores3 = np.array([0.85, 0.35])  # Second face now low confidence
tracks = tracker.update(dets3, scores3)
print(f'Frame 3: {len(tracks)} tracks (low-conf recovery: {[t.track_id for t in tracks]})')

print('ByteTracker OK')
"`

Expected:
```
Frame 1: 2 tracks
Frame 2: 2 tracks (same IDs: [1, 2])
Frame 3: 2 tracks (low-conf recovery: [1, 2])
ByteTracker OK
```

**Step 3: Commit**

```bash
git add python/processing/tracker.py
git commit -m "feat(tracker): add ByteTracker with two-pass association"
```

---

### Task A4: Add Temporal Buffer Extension

**Files:**
- Modify: `python/processing/tracker.py`

**Step 1: Add TemporalBuffer class for conservative redaction**

Append to `python/processing/tracker.py`:

```python
@dataclass
class TrackHistory:
    """Stores historical positions for a track."""
    track_id: int
    positions: List[Tuple[int, np.ndarray, float]] = field(default_factory=list)  # (frame, tlbr, score)

    def add(self, frame_id: int, tlbr: np.ndarray, score: float):
        self.positions.append((frame_id, tlbr.copy(), score))

    def get_box_at_frame(self, frame_id: int, buffer_frames: int) -> Optional[np.ndarray]:
        """
        Get box for frame, including temporal buffer extension.
        Returns interpolated/extended box if within buffer range.
        """
        # Direct hit
        for fid, tlbr, _ in self.positions:
            if fid == frame_id:
                return tlbr

        # Check if within buffer range
        frame_ids = [p[0] for p in self.positions]
        if not frame_ids:
            return None

        min_frame = min(frame_ids) - buffer_frames
        max_frame = max(frame_ids) + buffer_frames

        if frame_id < min_frame or frame_id > max_frame:
            return None

        # Pre-buffer: use first known position
        if frame_id < min(frame_ids):
            first_pos = min(self.positions, key=lambda p: p[0])
            return first_pos[1]

        # Post-buffer: use last known position
        if frame_id > max(frame_ids):
            last_pos = max(self.positions, key=lambda p: p[0])
            return last_pos[1]

        # Interpolate between two nearest positions
        before = [(fid, tlbr) for fid, tlbr, _ in self.positions if fid < frame_id]
        after = [(fid, tlbr) for fid, tlbr, _ in self.positions if fid > frame_id]

        if before and after:
            prev_frame, prev_box = max(before, key=lambda x: x[0])
            next_frame, next_box = min(after, key=lambda x: x[0])

            # Linear interpolation
            t = (frame_id - prev_frame) / (next_frame - prev_frame)
            return prev_box * (1 - t) + next_box * t

        return None


class TemporalBuffer:
    """
    Manages temporal buffering for conservative redaction.
    Extends redaction N frames before first detection and N frames after last.
    """

    def __init__(self, buffer_frames: int = 5):
        self.buffer_frames = buffer_frames
        self.histories: dict[int, TrackHistory] = {}

    def record(self, frame_id: int, tracks: List[STrack]):
        """Record track positions for a frame."""
        for track in tracks:
            if track.track_id not in self.histories:
                self.histories[track.track_id] = TrackHistory(track.track_id)
            self.histories[track.track_id].add(frame_id, track.tlbr, track.score)

    def get_boxes_for_frame(self, frame_id: int) -> List[Tuple[int, np.ndarray]]:
        """
        Get all boxes that should be redacted at given frame,
        including temporal buffer extensions.

        Returns:
            List of (track_id, tlbr) tuples
        """
        result = []
        for track_id, history in self.histories.items():
            box = history.get_box_at_frame(frame_id, self.buffer_frames)
            if box is not None:
                result.append((track_id, box))
        return result

    def get_all_track_summaries(self) -> List[dict]:
        """Get summary of all tracks for audit."""
        summaries = []
        for track_id, history in self.histories.items():
            if history.positions:
                frames = [p[0] for p in history.positions]
                scores = [p[2] for p in history.positions]
                summaries.append({
                    "track_id": track_id,
                    "first_frame": min(frames),
                    "last_frame": max(frames),
                    "detection_count": len(history.positions),
                    "avg_confidence": sum(scores) / len(scores),
                })
        return summaries
```

**Step 2: Verify TemporalBuffer**

Run: `cd /home/raw/fraser/.worktrees/v2-face-redaction && python -c "
from python.processing.tracker import TemporalBuffer, STrack, TrackState
import numpy as np

buffer = TemporalBuffer(buffer_frames=5)

# Create mock track
class MockTrack:
    def __init__(self, tid, tlbr, score):
        self.track_id = tid
        self.tlbr = np.array(tlbr)
        self.score = score

# Record detections at frames 10, 11, 12
buffer.record(10, [MockTrack(1, [100, 100, 200, 200], 0.9)])
buffer.record(11, [MockTrack(1, [105, 100, 205, 200], 0.88)])
buffer.record(12, [MockTrack(1, [110, 100, 210, 200], 0.85)])

# Test pre-buffer (frame 6 = 10 - 4, should get first position)
boxes = buffer.get_boxes_for_frame(6)
print(f'Frame 6 (pre-buffer): {len(boxes)} boxes')

# Test post-buffer (frame 16 = 12 + 4, should get last position)
boxes = buffer.get_boxes_for_frame(16)
print(f'Frame 16 (post-buffer): {len(boxes)} boxes')

# Test outside buffer (frame 4 = 10 - 6, should be empty)
boxes = buffer.get_boxes_for_frame(4)
print(f'Frame 4 (outside buffer): {len(boxes)} boxes')

print('TemporalBuffer OK')
"`

Expected:
```
Frame 6 (pre-buffer): 1 boxes
Frame 16 (post-buffer): 1 boxes
Frame 4 (outside buffer): 0 boxes
TemporalBuffer OK
```

**Step 3: Commit**

```bash
git add python/processing/tracker.py
git commit -m "feat(tracker): add TemporalBuffer for conservative redaction"
```

---

## TRACK B: WebSocket Server

**Parallelizable:** YES - No dependencies
**Estimated Time:** 15 minutes
**Files:**
- Modify: `python/server.py`

### Task B1: Add WebSocket Endpoint

**Files:**
- Modify: `python/server.py`

**Step 1: Read current server.py**

Read file first to understand current structure.

**Step 2: Add WebSocket imports and endpoint**

Add after existing imports in `python/server.py`:

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio

# WebSocket connections store
active_connections: Dict[str, WebSocket] = {}
```

Add new endpoint after existing routes:

```python
@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()
    active_connections[job_id] = websocket

    try:
        while True:
            # Keep connection alive, actual updates sent from processing
            await asyncio.sleep(1)

            # Check if job still exists
            if job_id not in jobs:
                await websocket.send_json({
                    "type": "error",
                    "message": "Job not found"
                })
                break

            job = jobs[job_id]
            if job.get("type") == "completed":
                await websocket.send_json(job)
                break
            elif job.get("type") == "error":
                await websocket.send_json(job)
                break

    except WebSocketDisconnect:
        pass
    finally:
        if job_id in active_connections:
            del active_connections[job_id]


async def broadcast_progress(job_id: str, data: dict):
    """Send progress update to connected WebSocket client."""
    if job_id in active_connections:
        try:
            await active_connections[job_id].send_json(data)
        except Exception:
            # Connection closed
            if job_id in active_connections:
                del active_connections[job_id]
```

**Step 3: Verify syntax**

Run: `cd /home/raw/fraser/.worktrees/v2-face-redaction && python -c "from python.server import app; print('Server imports OK')"`

Expected: `Server imports OK`

**Step 4: Commit**

```bash
git add python/server.py
git commit -m "feat(server): add WebSocket endpoint for real-time progress"
```

---

### Task B2: Update ProcessRequest Model

**Files:**
- Modify: `python/server.py`

**Step 1: Update the ProcessRequest model with new fields**

Find and replace the `ProcessRequest` class:

```python
class ProcessRequest(BaseModel):
    input_path: str
    output_path: str
    model: str = "yolov11n-face.pt"
    confidence: float = 0.25
    padding: float = 0.20
    detection_resolution: str = "360p"  # 144p, 240p, 360p, native
    redaction_mode: str = "black"  # black, color
    redaction_color: str = "#000000"
    temporal_buffer: int = 5
    generate_audit: bool = True
    thumbnail_interval: int = 30
```

**Step 2: Add resolution mapping**

Add after ProcessRequest class:

```python
RESOLUTION_MAP = {
    "144p": (256, 144),
    "240p": (416, 240),
    "360p": (640, 360),
    "native": None,  # Use original resolution
}
```

**Step 3: Commit**

```bash
git add python/server.py
git commit -m "feat(server): update ProcessRequest with new config options"
```

---

## TRACK C: Frontend UI Updates

**Parallelizable:** YES - No dependencies
**Estimated Time:** 20 minutes
**Files:**
- Modify: `renderer/app.js`
- Modify: `renderer/index.html`

### Task C1: Update Settings Object and Add Resolution Dropdown

**Files:**
- Modify: `renderer/app.js`

**Step 1: Read current app.js**

Read file first to understand current structure.

**Step 2: Update settings object**

Find the settings initialization and replace with:

```javascript
let settings = {
    model: 'yolov11n-face',
    mode: 'black',  // removed 'blur' option
    confidence: 0.25,
    detectionResolution: '360p',
    redactionColor: '#000000',
    temporalBuffer: 5,
};
```

**Step 3: Commit**

```bash
git add renderer/app.js
git commit -m "feat(ui): update settings with new config options"
```

---

### Task C2: Add Resolution Dropdown to HTML

**Files:**
- Modify: `renderer/index.html`

**Step 1: Read current index.html**

Read file first.

**Step 2: Add resolution dropdown**

Find the settings section and add resolution dropdown after model selection:

```html
<div class="setting-group">
    <label for="resolution-select">Detection Resolution</label>
    <select id="resolution-select" class="setting-select">
        <option value="144p">144p (Fastest)</option>
        <option value="240p">240p (Fast)</option>
        <option value="360p" selected>360p (Balanced)</option>
        <option value="native">Native (Max Quality)</option>
    </select>
</div>
```

**Step 3: Remove blur option from mode selector**

Find the mode selector and remove the blur option, keeping only black and color.

**Step 4: Commit**

```bash
git add renderer/index.html
git commit -m "feat(ui): add resolution dropdown, remove blur option"
```

---

### Task C3: Add WebSocket Client

**Files:**
- Modify: `renderer/app.js`

**Step 1: Add WebSocket connection handling**

Add WebSocket client code:

```javascript
let progressSocket = null;

function connectProgressSocket(jobId) {
    const wsUrl = `ws://localhost:8420/ws/progress/${jobId}`;
    progressSocket = new WebSocket(wsUrl);

    progressSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleProgressUpdate(data);
    };

    progressSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Fall back to polling
        startPolling(jobId);
    };

    progressSocket.onclose = () => {
        progressSocket = null;
    };
}

function handleProgressUpdate(data) {
    if (data.type === 'progress') {
        updateProgressUI({
            percent: data.percent,
            fps: data.fps,
            activeTracks: data.active_tracks,
            totalFaces: data.total_faces_detected,
            estimatedRemaining: data.estimated_remaining_sec,
        });
    } else if (data.type === 'completed') {
        handleJobComplete(data);
    } else if (data.type === 'error') {
        handleJobError(data);
    }
}

function disconnectProgressSocket() {
    if (progressSocket) {
        progressSocket.close();
        progressSocket = null;
    }
}
```

**Step 2: Commit**

```bash
git add renderer/app.js
git commit -m "feat(ui): add WebSocket client for real-time progress"
```

---

### Task C4: Update Progress Display

**Files:**
- Modify: `renderer/app.js`
- Modify: `renderer/index.html`

**Step 1: Add new progress info elements to HTML**

```html
<div class="progress-stats">
    <span id="fps-display">-- FPS</span>
    <span id="tracks-display">-- tracks</span>
    <span id="eta-display">ETA: --:--</span>
</div>
```

**Step 2: Add updateProgressUI function**

```javascript
function updateProgressUI({ percent, fps, activeTracks, totalFaces, estimatedRemaining }) {
    // Update progress bar
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
    }

    // Update stats
    const fpsDisplay = document.getElementById('fps-display');
    if (fpsDisplay && fps) {
        fpsDisplay.textContent = `${fps.toFixed(1)} FPS`;
    }

    const tracksDisplay = document.getElementById('tracks-display');
    if (tracksDisplay && activeTracks !== undefined) {
        tracksDisplay.textContent = `${activeTracks} tracks`;
    }

    const etaDisplay = document.getElementById('eta-display');
    if (etaDisplay && estimatedRemaining) {
        const mins = Math.floor(estimatedRemaining / 60);
        const secs = Math.floor(estimatedRemaining % 60);
        etaDisplay.textContent = `ETA: ${mins}:${secs.toString().padStart(2, '0')}`;
    }
}
```

**Step 3: Commit**

```bash
git add renderer/app.js renderer/index.html
git commit -m "feat(ui): add FPS, tracks, and ETA display"
```

---

## TRACK D: Pipeline Core Rewrite

**Parallelizable:** NO - Depends on Track A (tracker)
**Estimated Time:** 40 minutes
**Files:**
- Rewrite: `python/processing/pipeline.py`

### Task D1: Add Resolution Scaling Utilities

**Files:**
- Modify: `python/processing/pipeline.py`

**Step 1: Read current pipeline.py**

Read file first.

**Step 2: Add resolution utilities at top of file**

```python
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import numpy as np
import cv2
import torch
import av

RESOLUTION_MAP = {
    "144p": (256, 144),
    "240p": (416, 240),
    "360p": (640, 360),
    "native": None,
}


def scale_detections(
    boxes: np.ndarray,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int]
) -> np.ndarray:
    """
    Scale bounding boxes from one resolution to another.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2]
        from_size: (width, height) of source
        to_size: (width, height) of target

    Returns:
        Scaled boxes
    """
    if boxes.size == 0:
        return boxes

    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]

    scaled = boxes.copy()
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y

    return scaled


def resize_for_detection(
    frame: np.ndarray,
    target_resolution: str
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Resize frame for detection.

    Returns:
        (resized_frame, original_size, detection_size)
    """
    original_size = (frame.shape[1], frame.shape[0])  # (width, height)

    if target_resolution == "native" or target_resolution not in RESOLUTION_MAP:
        return frame, original_size, original_size

    target_size = RESOLUTION_MAP[target_resolution]
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    return resized, original_size, target_size
```

**Step 3: Commit**

```bash
git add python/processing/pipeline.py
git commit -m "feat(pipeline): add resolution scaling utilities"
```

---

### Task D2: Create ProcessingConfig Dataclass

**Files:**
- Modify: `python/processing/pipeline.py`

**Step 1: Add configuration dataclass**

```python
@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    input_path: Path
    output_path: Path
    model_name: str = "yolov11n-face.pt"
    confidence: float = 0.25
    padding: float = 0.20
    detection_resolution: str = "360p"
    redaction_mode: str = "black"
    redaction_color: str = "#000000"
    temporal_buffer: int = 5
    batch_size: int = 32
    generate_audit: bool = True
    thumbnail_interval: int = 30
    checkpoint_interval: int = 4500

    def __post_init__(self):
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)


@dataclass
class ProcessingStats:
    """Statistics from processing."""
    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    unique_tracks: int = 0
    frames_with_faces: int = 0
    processing_fps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "total_detections": self.total_detections,
            "unique_tracks": self.unique_tracks,
            "frames_with_faces": self.frames_with_faces,
            "processing_fps": self.processing_fps,
        }
```

**Step 2: Commit**

```bash
git add python/processing/pipeline.py
git commit -m "feat(pipeline): add ProcessingConfig and ProcessingStats"
```

---

### Task D3: Create VideoProcessor Class

**Files:**
- Modify: `python/processing/pipeline.py`

**Step 1: Add VideoProcessor class**

```python
from python.processing.tracker import ByteTracker, ByteTrackerConfig, TemporalBuffer
from python.processing.anonymizer import Anonymizer


class VideoProcessor:
    """
    Main video processing pipeline with tracking and temporal buffering.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model = None
        self.device = None
        self.tracker = None
        self.temporal_buffer = None
        self.stats = ProcessingStats()

        # Callbacks
        self.on_progress: Optional[Callable] = None

        # Audit data
        self.thumbnails: List[np.ndarray] = []
        self.audit_data: dict = {}

    def _load_model(self):
        """Load YOLO model."""
        from ultralytics import YOLO

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load model
        model_path = Path(__file__).parent.parent / "models" / self.config.model_name
        if not model_path.exists():
            # Try downloading from ultralytics
            self.model = YOLO(self.config.model_name)
        else:
            self.model = YOLO(str(model_path))

        self.model.to(self.device)

    def _init_tracker(self):
        """Initialize ByteTrack tracker."""
        tracker_config = ByteTrackerConfig(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            match_thresh=0.8,
            track_buffer=30,
            temporal_buffer=self.config.temporal_buffer,
        )
        self.tracker = ByteTracker(tracker_config)
        self.temporal_buffer = TemporalBuffer(self.config.temporal_buffer)

    def _detect_faces(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run face detection on batch of frames.

        Returns:
            List of (boxes, scores) for each frame
        """
        if not frames:
            return []

        # Resize frames for detection
        detection_frames = []
        original_sizes = []
        detection_sizes = []

        for frame in frames:
            resized, orig_size, det_size = resize_for_detection(
                frame, self.config.detection_resolution
            )
            detection_frames.append(resized)
            original_sizes.append(orig_size)
            detection_sizes.append(det_size)

        # Batch inference
        results = self.model.predict(
            detection_frames,
            conf=self.config.confidence,
            half=True if self.device == "cuda" else False,
            verbose=False,
        )

        # Extract and scale boxes
        outputs = []
        for i, result in enumerate(results):
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                # Scale boxes back to original resolution
                if detection_sizes[i] != original_sizes[i]:
                    boxes = scale_detections(boxes, detection_sizes[i], original_sizes[i])

                outputs.append((boxes, scores))
            else:
                outputs.append((np.array([]), np.array([])))

        return outputs

    def _apply_redaction(self, frame: np.ndarray, boxes: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Apply redaction to frame."""
        result = frame.copy()

        for track_id, tlbr in boxes:
            x1, y1, x2, y2 = map(int, tlbr)

            # Apply padding
            w, h = x2 - x1, y2 - y1
            pad_x = int(w * self.config.padding)
            pad_y = int(h * self.config.padding)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)

            # Apply redaction
            if self.config.redaction_mode == "color":
                color = Anonymizer.hex_to_bgr(self.config.redaction_color)
            else:
                color = (0, 0, 0)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)

        return result

    def process(self) -> ProcessingStats:
        """
        Process video file.

        Returns:
            Processing statistics
        """
        import time

        self._load_model()
        self._init_tracker()

        # Open input video
        input_container = av.open(str(self.config.input_path))
        video_stream = input_container.streams.video[0]

        total_frames = video_stream.frames or 0
        fps = float(video_stream.average_rate)
        width = video_stream.width
        height = video_stream.height

        self.stats.total_frames = total_frames

        # Open output video
        output_container = av.open(str(self.config.output_path), "w")
        output_stream = output_container.add_stream("h264", rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": "23", "preset": "fast"}

        # Processing loop
        frame_buffer = []
        frame_indices = []
        start_time = time.time()
        processed = 0
        thumbnail_frame_interval = int(fps * self.config.thumbnail_interval)

        try:
            for frame_idx, frame in enumerate(input_container.decode(video=0)):
                # Convert to numpy
                img = frame.to_ndarray(format="bgr24")

                frame_buffer.append(img)
                frame_indices.append(frame_idx)

                # Process batch
                if len(frame_buffer) >= self.config.batch_size:
                    self._process_batch(
                        frame_buffer, frame_indices,
                        output_container, output_stream,
                        thumbnail_frame_interval
                    )
                    processed += len(frame_buffer)
                    frame_buffer = []
                    frame_indices = []

                    # Progress callback
                    elapsed = time.time() - start_time
                    current_fps = processed / elapsed if elapsed > 0 else 0

                    if self.on_progress:
                        self.on_progress({
                            "type": "progress",
                            "frame": processed,
                            "total_frames": total_frames,
                            "percent": (processed / total_frames * 100) if total_frames else 0,
                            "fps": current_fps,
                            "active_tracks": len(self.tracker.tracked_stracks),
                            "total_faces_detected": self.stats.total_detections,
                            "estimated_remaining_sec": (total_frames - processed) / current_fps if current_fps > 0 else 0,
                        })

            # Process remaining frames
            if frame_buffer:
                self._process_batch(
                    frame_buffer, frame_indices,
                    output_container, output_stream,
                    thumbnail_frame_interval
                )
                processed += len(frame_buffer)

        finally:
            output_container.close()
            input_container.close()

        # Finalize stats
        elapsed = time.time() - start_time
        self.stats.processed_frames = processed
        self.stats.processing_fps = processed / elapsed if elapsed > 0 else 0
        self.stats.unique_tracks = len(self.tracker.all_tracks)

        # Generate audit if requested
        if self.config.generate_audit:
            self._generate_audit()

        return self.stats

    def _process_batch(
        self,
        frames: List[np.ndarray],
        indices: List[int],
        output_container,
        output_stream,
        thumbnail_interval: int
    ):
        """Process a batch of frames."""
        # Detect faces
        detections = self._detect_faces(frames)

        # Track and redact each frame
        for i, (frame, frame_idx) in enumerate(zip(frames, indices)):
            boxes, scores = detections[i]

            # Update tracker
            if len(boxes) > 0:
                tracks = self.tracker.update(boxes, scores)
                self.temporal_buffer.record(frame_idx, tracks)
                self.stats.total_detections += len(boxes)
                self.stats.frames_with_faces += 1
            else:
                self.tracker.update(np.array([]), np.array([]))

            # Get boxes to redact (including temporal buffer)
            redact_boxes = self.temporal_buffer.get_boxes_for_frame(frame_idx)

            # Apply redaction
            redacted = self._apply_redaction(frame, redact_boxes)

            # Capture thumbnail
            if frame_idx % thumbnail_interval == 0 and self.config.generate_audit:
                thumb = cv2.resize(redacted, (64, 36))
                self.thumbnails.append(thumb)

            # Encode and write
            out_frame = av.VideoFrame.from_ndarray(redacted, format="bgr24")
            for packet in output_stream.encode(out_frame):
                output_container.mux(packet)

    def _generate_audit(self):
        """Generate audit trail files."""
        import json
        import hashlib

        # Generate JSON audit
        self.audit_data = {
            "version": "2.0",
            "input_file": str(self.config.input_path),
            "output_file": str(self.config.output_path),
            "config": {
                "model": self.config.model_name,
                "detection_resolution": self.config.detection_resolution,
                "confidence": self.config.confidence,
                "temporal_buffer": self.config.temporal_buffer,
                "redaction_mode": self.config.redaction_mode,
                "padding": self.config.padding,
            },
            "statistics": self.stats.to_dict(),
            "tracks": self.temporal_buffer.get_all_track_summaries(),
        }

        # Save JSON
        audit_json_path = self.config.output_path.with_suffix(".audit.json")
        with open(audit_json_path, "w") as f:
            json.dump(self.audit_data, f, indent=2)

        # Generate thumbnail montage
        if self.thumbnails:
            self._save_thumbnail_montage()

    def _save_thumbnail_montage(self):
        """Save thumbnail grid image."""
        if not self.thumbnails:
            return

        thumb_h, thumb_w = self.thumbnails[0].shape[:2]
        n_thumbs = len(self.thumbnails)

        # Calculate grid size
        cols = min(48, n_thumbs)
        rows = (n_thumbs + cols - 1) // cols

        # Create montage
        montage = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

        for i, thumb in enumerate(self.thumbnails):
            row = i // cols
            col = i % cols
            y1 = row * thumb_h
            x1 = col * thumb_w
            montage[y1:y1+thumb_h, x1:x1+thumb_w] = thumb

        # Save
        montage_path = self.config.output_path.with_suffix(".audit.png")
        cv2.imwrite(str(montage_path), montage)
```

**Step 2: Commit**

```bash
git add python/processing/pipeline.py
git commit -m "feat(pipeline): add VideoProcessor with tracking and batching"
```

---

## TRACK E: Anonymizer Cleanup

**Parallelizable:** YES - No dependencies
**Estimated Time:** 5 minutes
**Files:**
- Modify: `python/processing/anonymizer.py`

### Task E1: Remove Blur Method

**Files:**
- Modify: `python/processing/anonymizer.py`

**Step 1: Read current anonymizer.py**

Read file first.

**Step 2: Remove blur method, keep black_rectangle and color_fill**

Remove the `blur` method entirely. Keep only:
- `black_rectangle`
- `color_fill`
- Helper methods (`_apply_padding`, `hex_to_bgr`)

**Step 3: Add hex_to_bgr helper if not exists**

```python
@staticmethod
def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)
```

**Step 4: Commit**

```bash
git add python/processing/anonymizer.py
git commit -m "feat(anonymizer): remove blur, keep only solid redaction"
```

---

## TRACK F: Integration and Testing

**Parallelizable:** NO - Depends on all tracks
**Estimated Time:** 20 minutes

### Task F1: Update Server to Use New Pipeline

**Files:**
- Modify: `python/server.py`

**Step 1: Update process endpoint**

Replace the processing logic to use new VideoProcessor:

```python
from python.processing.pipeline import VideoProcessor, ProcessingConfig

@app.post("/process")
async def process_video(request: ProcessRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"type": "queued", "job_id": job_id}

    background_tasks.add_task(run_processing, job_id, request)

    return {"job_id": job_id}


async def run_processing(job_id: str, request: ProcessRequest):
    try:
        config = ProcessingConfig(
            input_path=request.input_path,
            output_path=request.output_path,
            model_name=request.model,
            confidence=request.confidence,
            padding=request.padding,
            detection_resolution=request.detection_resolution,
            redaction_mode=request.redaction_mode,
            redaction_color=request.redaction_color,
            temporal_buffer=request.temporal_buffer,
            generate_audit=request.generate_audit,
            thumbnail_interval=request.thumbnail_interval,
        )

        processor = VideoProcessor(config)

        # Set up progress callback
        def on_progress(data):
            jobs[job_id] = data
            # Also broadcast via WebSocket
            import asyncio
            try:
                asyncio.create_task(broadcast_progress(job_id, data))
            except RuntimeError:
                pass  # No event loop

        processor.on_progress = on_progress

        stats = processor.process()

        jobs[job_id] = {
            "type": "completed",
            "job_id": job_id,
            "statistics": stats.to_dict(),
        }

    except Exception as e:
        jobs[job_id] = {
            "type": "error",
            "job_id": job_id,
            "error": str(e),
        }
```

**Step 2: Commit**

```bash
git add python/server.py
git commit -m "feat(server): integrate new VideoProcessor pipeline"
```

---

### Task F2: Update Python Requirements

**Files:**
- Modify: `python/requirements.txt`

**Step 1: Add new dependencies**

```txt
fastapi>=0.109.0
uvicorn>=0.27.0
ultralytics>=8.1.0
av>=12.0.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
torch>=2.1.0
torchvision>=0.16.0
scipy>=1.11.0
websockets>=12.0
```

**Step 2: Commit**

```bash
git add python/requirements.txt
git commit -m "chore: add scipy and websockets to requirements"
```

---

### Task F3: Update Frontend Processing Handler

**Files:**
- Modify: `src/handlers/processingHandlers.ts`

**Step 1: Read current file**

**Step 2: Update to use WebSocket instead of polling**

Update the handler to try WebSocket first, fall back to polling:

```typescript
// In startProcessing function, after getting job_id:
// Try WebSocket connection
window.api.send('CONNECT_WEBSOCKET', { jobId });

// Keep polling as fallback
```

**Step 3: Commit**

```bash
git add src/handlers/processingHandlers.ts
git commit -m "feat(handlers): prefer WebSocket for progress updates"
```

---

### Task F4: End-to-End Test

**Step 1: Build the application**

```bash
cd /home/raw/fraser/.worktrees/v2-face-redaction
npm run build
```

**Step 2: Test Python pipeline directly**

```bash
cd /home/raw/fraser/.worktrees/v2-face-redaction
python -c "
from python.processing.pipeline import VideoProcessor, ProcessingConfig
from pathlib import Path

# Create test config (use a short test video if available)
config = ProcessingConfig(
    input_path=Path('test_video.mp4'),
    output_path=Path('test_output.mp4'),
    detection_resolution='360p',
)

print('Pipeline imports OK')
print(f'Config: {config}')
"
```

**Step 3: Commit all integration work**

```bash
git add -A
git commit -m "feat: complete v2 pipeline integration"
```

---

## Summary

| Track | Tasks | Time | Parallelizable |
|-------|-------|------|----------------|
| A: ByteTrack | A1-A4 | 25 min | YES |
| B: WebSocket | B1-B2 | 15 min | YES |
| C: Frontend | C1-C4 | 20 min | YES |
| D: Pipeline | D1-D3 | 40 min | NO (needs A) |
| E: Anonymizer | E1 | 5 min | YES |
| F: Integration | F1-F4 | 20 min | NO (needs all) |

**Optimal Parallel Execution:**

```
Time 0    ───▶ Agents 1-4 start Tracks A, B, C, E in parallel
Time 25   ───▶ Track A complete, Agent 1 starts Track D
Time 40   ───▶ All tracks complete, start Track F
Time 60   ───▶ Done
```

**Total Time (Sequential):** ~125 minutes
**Total Time (Parallel):** ~60 minutes
