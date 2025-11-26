"""ByteTrack tracker implementation for face tracking.

ByteTrack uses two-pass association with low-confidence detections
to recover tracks during occlusion/blur - critical for HIPAA compliance.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter


class TrackState(Enum):
    """Track state enumeration."""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class ByteTrackerConfig:
    """Configuration for ByteTracker."""
    track_thresh: float = 0.5      # High confidence threshold for first pass
    track_buffer: int = 30          # Number of frames to keep lost tracks
    match_thresh: float = 0.8       # IoU threshold for matching
    min_box_area: int = 10          # Minimum box area to consider

    # ByteTrack innovation: use low-confidence detections for recovery
    low_thresh: float = 0.1         # Low confidence threshold for second pass
    second_match_thresh: float = 0.5  # IoU threshold for second pass matching


@dataclass
class TrackHistory:
    """Store track positions over time."""
    track_id: int
    boxes: List[np.ndarray] = field(default_factory=list)  # [N, 4] tlbr format
    scores: List[float] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)

    def add(self, box: np.ndarray, score: float, frame_id: int):
        """Add new detection to history."""
        self.boxes.append(box.copy())
        self.scores.append(score)
        self.frame_ids.append(frame_id)

    def get_boxes_array(self) -> np.ndarray:
        """Get all boxes as numpy array."""
        if not self.boxes:
            return np.empty((0, 4))
        return np.array(self.boxes)


class STrack:
    """Single target track with Kalman filter state.

    Uses Kalman filter to predict box position and smooth tracking.
    State vector: [x, y, w, h, vx, vy, vw, vh]
    Measurement: [x, y, w, h]
    """

    _count = 0  # Global track ID counter

    def __init__(self, tlbr: np.ndarray, score: float):
        """Initialize new track.

        Args:
            tlbr: Bounding box in [x1, y1, x2, y2] format
            score: Detection confidence score
        """
        # Convert tlbr to center format for Kalman filter
        self._tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = score

        # Assign unique track ID
        STrack._count += 1
        self.track_id = STrack._count

        # Track state
        self.state = TrackState.NEW
        self.frame_id = 0
        self.tracklet_len = 0
        self.time_since_update = 0

        # Initialize Kalman filter
        self.kf = self._init_kalman_filter()

        # Set initial state
        x, y, w, h = self._tlbr_to_xywh(self._tlbr)
        self.mean = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32)
        self.covariance[4:, 4:] *= 1000.0  # High uncertainty in velocity

    def _init_kalman_filter(self) -> KalmanFilter:
        """Initialize Kalman filter for box tracking."""
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.F[i, i+4] = 1.0  # position += velocity

        # Measurement matrix (observe position only)
        kf.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            kf.H[i, i] = 1.0

        # Measurement noise
        kf.R *= 10.0

        # Process noise
        kf.Q[-4:, -4:] *= 0.01  # Low process noise for velocity

        return kf

    @staticmethod
    def _tlbr_to_xywh(tlbr: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert top-left-bottom-right to center-x-y-width-height."""
        x1, y1, x2, y2 = tlbr
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        return x, y, w, h

    @staticmethod
    def _xywh_to_tlbr(x: float, y: float, w: float, h: float) -> np.ndarray:
        """Convert center-x-y-width-height to top-left-bottom-right."""
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self):
        """Predict next state using Kalman filter."""
        if self.state != TrackState.TRACKED:
            # Reset velocity for non-tracked states
            self.mean[4:] = 0

        # Kalman predict
        self.kf.x = self.mean
        self.kf.P = self.covariance
        self.kf.predict()
        self.mean = self.kf.x
        self.covariance = self.kf.P

        # Update tlbr from predicted state
        x, y, w, h = self.mean[:4]
        self._tlbr = self._xywh_to_tlbr(x, y, w, h)

    def update(self, tlbr: np.ndarray, score: float, frame_id: int):
        """Update track with new detection.

        Args:
            tlbr: New detection box in [x1, y1, x2, y2] format
            score: Detection score
            frame_id: Current frame number
        """
        self._tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = score
        self.frame_id = frame_id

        # Convert to measurement
        x, y, w, h = self._tlbr_to_xywh(tlbr)
        measurement = np.array([x, y, w, h], dtype=np.float32)

        # Kalman update
        self.kf.x = self.mean
        self.kf.P = self.covariance
        self.kf.update(measurement)
        self.mean = self.kf.x
        self.covariance = self.kf.P

        # Update state
        self.tracklet_len += 1
        self.state = TrackState.TRACKED
        self.time_since_update = 0

        # Update tlbr from corrected state
        x, y, w, h = self.mean[:4]
        self._tlbr = self._xywh_to_tlbr(x, y, w, h)

    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.LOST

    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.REMOVED

    @property
    def tlbr(self) -> np.ndarray:
        """Get current bounding box in tlbr format."""
        return self._tlbr.copy()

    @property
    def is_activated(self) -> bool:
        """Check if track is activated (tracked state)."""
        return self.state == TrackState.TRACKED


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bounding boxes.

    Args:
        boxes_a: Array of boxes [N, 4] in tlbr format
        boxes_b: Array of boxes [M, 4] in tlbr format

    Returns:
        IoU matrix [N, M]
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Expand dimensions for broadcasting
    boxes_a = np.asarray(boxes_a)
    boxes_b = np.asarray(boxes_b)

    # Compute intersection
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute union
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # [N]
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # [M]
    union = area_a[:, None] + area_b[None, :] - inter  # [N, M]

    # Compute IoU
    iou = inter / np.clip(union, 1e-6, None)
    return iou


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, List[int], List[int]]:
    """Solve linear assignment problem using scipy.

    Uses Hungarian algorithm (Kuhn-Munkres) for optimal assignment.

    Args:
        cost_matrix: Cost matrix [N, M] where cost_matrix[i, j] is cost of assigning i to j
        thresh: Threshold for valid assignments

    Returns:
        Tuple of (matches, unmatched_a, unmatched_b)
        - matches: Array of [K, 2] with matched pairs
        - unmatched_a: List of unmatched indices from first set
        - unmatched_b: List of unmatched indices from second set
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    # Convert to cost (1 - IoU for IoU matrices)
    # For IoU input, we want to maximize, so we minimize (1 - IoU)

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter by threshold
    matches = []
    unmatched_a = []
    unmatched_b = []

    matched_rows = set()
    matched_cols = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            matched_rows.add(r)
            matched_cols.add(c)

    # Find unmatched
    for i in range(cost_matrix.shape[0]):
        if i not in matched_rows:
            unmatched_a.append(i)

    for j in range(cost_matrix.shape[1]):
        if j not in matched_cols:
            unmatched_b.append(j)

    matches = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)

    return matches, unmatched_a, unmatched_b


class ByteTracker:
    """ByteTrack multi-object tracker with two-pass association.

    Key innovation: Uses low-confidence detections (0.1-0.5) in second pass
    to recover tracks during occlusion/blur. Critical for HIPAA compliance.
    """

    def __init__(self, config: Optional[ByteTrackerConfig] = None):
        """Initialize ByteTracker.

        Args:
            config: Tracker configuration
        """
        self.config = config or ByteTrackerConfig()
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        self.frame_id = 0

        # Statistics
        self.total_tracks = 0
        self.active_tracks = 0

    def update(self, detections: np.ndarray, scores: np.ndarray) -> List[STrack]:
        """Update tracker with new detections using two-pass association.

        ByteTrack two-pass algorithm:
        1. First pass: Associate high-confidence detections with existing tracks
        2. Second pass: Use low-confidence detections to recover lost tracks

        Args:
            detections: Array of bounding boxes [N, 4] in tlbr format
            scores: Array of detection scores [N]

        Returns:
            List of active tracks
        """
        self.frame_id += 1

        # Separate high and low confidence detections
        high_det_mask = scores >= self.config.track_thresh
        low_det_mask = (scores >= self.config.low_thresh) & (scores < self.config.track_thresh)

        high_dets = detections[high_det_mask]
        high_scores = scores[high_det_mask]

        low_dets = detections[low_det_mask]
        low_scores = scores[low_det_mask]

        # Filter by minimum box area
        if self.config.min_box_area > 0:
            high_areas = (high_dets[:, 2] - high_dets[:, 0]) * (high_dets[:, 3] - high_dets[:, 1])
            valid_high = high_areas > self.config.min_box_area
            high_dets = high_dets[valid_high]
            high_scores = high_scores[valid_high]

            low_areas = (low_dets[:, 2] - low_dets[:, 0]) * (low_dets[:, 3] - low_dets[:, 1])
            valid_low = low_areas > self.config.min_box_area
            low_dets = low_dets[valid_low]
            low_scores = low_scores[valid_low]

        # Initialize containers for track management
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        # Predict all existing tracks
        for track in self.tracked_tracks:
            track.predict()

        for track in self.lost_tracks:
            track.predict()

        # FIRST PASS: Associate high-confidence detections with tracked tracks
        track_pool = self.tracked_tracks

        if len(track_pool) > 0 and len(high_dets) > 0:
            # Compute IoU between tracks and high-confidence detections
            track_boxes = np.array([t.tlbr for t in track_pool])
            iou_matrix = iou_batch(track_boxes, high_dets)

            # Convert to cost matrix (1 - IoU)
            cost_matrix = 1 - iou_matrix

            # Solve assignment
            matches, unmatched_tracks, unmatched_dets = linear_assignment(
                cost_matrix,
                thresh=1 - self.config.match_thresh
            )

            # Update matched tracks
            for track_idx, det_idx in matches:
                track = track_pool[track_idx]
                track.update(high_dets[det_idx], high_scores[det_idx], self.frame_id)
                activated_tracks.append(track)

            # Handle unmatched tracks
            for track_idx in unmatched_tracks:
                track = track_pool[track_idx]
                track.mark_lost()
                lost_tracks.append(track)

            # Remaining high-confidence detections
            remaining_dets = high_dets[unmatched_dets]
            remaining_scores = high_scores[unmatched_dets]
        else:
            # No tracks or no detections
            for track in track_pool:
                track.mark_lost()
                lost_tracks.append(track)
            remaining_dets = high_dets
            remaining_scores = high_scores

        # SECOND PASS: Associate low-confidence detections with lost tracks
        # This is the ByteTrack innovation for recovering tracks during occlusion
        if len(self.lost_tracks) > 0 and len(low_dets) > 0:
            lost_boxes = np.array([t.tlbr for t in self.lost_tracks])
            iou_matrix = iou_batch(lost_boxes, low_dets)
            cost_matrix = 1 - iou_matrix

            matches, unmatched_lost, _ = linear_assignment(
                cost_matrix,
                thresh=1 - self.config.second_match_thresh
            )

            # Recover matched lost tracks
            for lost_idx, det_idx in matches:
                track = self.lost_tracks[lost_idx]
                track.update(low_dets[det_idx], low_scores[det_idx], self.frame_id)
                refind_tracks.append(track)

            # Update lost tracks list
            self.lost_tracks = [self.lost_tracks[i] for i in unmatched_lost]

        # Create new tracks from remaining high-confidence detections
        for det, score in zip(remaining_dets, remaining_scores):
            new_track = STrack(det, score)
            new_track.frame_id = self.frame_id
            new_track.state = TrackState.TRACKED
            activated_tracks.append(new_track)
            self.total_tracks += 1

        # Remove tracks that have been lost for too long
        for track in self.lost_tracks:
            track.time_since_update += 1
            if track.time_since_update > self.config.track_buffer:
                track.mark_removed()
                removed_tracks.append(track)

        # Update lost tracks
        self.lost_tracks = [t for t in lost_tracks if t.time_since_update <= self.config.track_buffer]

        # Update tracked tracks
        self.tracked_tracks = activated_tracks + refind_tracks
        self.removed_tracks.extend(removed_tracks)

        # Update statistics
        self.active_tracks = len(self.tracked_tracks)

        return self.tracked_tracks

    def get_active_boxes(self) -> np.ndarray:
        """Get bounding boxes of all active tracks.

        Returns:
            Array of boxes [N, 4] in tlbr format
        """
        if not self.tracked_tracks:
            return np.empty((0, 4), dtype=np.float32)
        return np.array([t.tlbr for t in self.tracked_tracks], dtype=np.float32)

    def get_statistics(self) -> Dict[str, int]:
        """Get tracker statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'frame_id': self.frame_id,
            'active_tracks': self.active_tracks,
            'lost_tracks': len(self.lost_tracks),
            'removed_tracks': len(self.removed_tracks),
            'total_tracks': self.total_tracks,
        }


class TemporalBuffer:
    """Conservative temporal buffer for extending redaction boxes.

    Extends boxes N frames before and after detection for conservative
    redaction to ensure no faces are missed during transitions.
    """

    def __init__(self, buffer_frames: int = 5):
        """Initialize temporal buffer.

        Args:
            buffer_frames: Number of frames to extend before/after detection
        """
        self.buffer_frames = buffer_frames
        self.track_histories: Dict[int, TrackHistory] = {}
        self.frame_id = 0

    def add_tracks(self, tracks: List[STrack], frame_id: int):
        """Add tracks to buffer.

        Args:
            tracks: List of active tracks
            frame_id: Current frame number
        """
        self.frame_id = frame_id

        for track in tracks:
            if track.track_id not in self.track_histories:
                self.track_histories[track.track_id] = TrackHistory(track_id=track.track_id)

            self.track_histories[track.track_id].add(
                track.tlbr,
                track.score,
                frame_id
            )

    def get_boxes_for_frame(self, frame_id: int) -> np.ndarray:
        """Get all boxes that should be redacted at given frame.

        Includes boxes from buffer_frames before and after.

        Args:
            frame_id: Frame number to get boxes for

        Returns:
            Array of boxes [N, 4] in tlbr format
        """
        boxes = []

        min_frame = frame_id - self.buffer_frames
        max_frame = frame_id + self.buffer_frames

        for history in self.track_histories.values():
            for box, fid in zip(history.boxes, history.frame_ids):
                if min_frame <= fid <= max_frame:
                    boxes.append(box)

        if not boxes:
            return np.empty((0, 4), dtype=np.float32)

        return np.array(boxes, dtype=np.float32)

    def cleanup_old_tracks(self, max_age_frames: int = 300):
        """Remove old track histories to prevent memory growth.

        Args:
            max_age_frames: Maximum age in frames to keep
        """
        cutoff_frame = self.frame_id - max_age_frames

        tracks_to_remove = []
        for track_id, history in self.track_histories.items():
            if history.frame_ids and max(history.frame_ids) < cutoff_frame:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.track_histories[track_id]

    def get_statistics(self) -> Dict[str, int]:
        """Get buffer statistics.

        Returns:
            Dictionary with statistics
        """
        total_detections = sum(len(h.boxes) for h in self.track_histories.values())

        return {
            'buffer_frames': self.buffer_frames,
            'tracked_objects': len(self.track_histories),
            'total_detections': total_detections,
            'current_frame': self.frame_id,
        }
