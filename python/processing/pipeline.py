"""Video processing pipeline with resolution downscaling, batched inference, and ByteTrack integration.

This module implements a high-performance video processing pipeline with:
- Resolution downscaling for faster detection
- Batched inference for improved throughput
- ByteTrack integration for temporal consistency
- Conservative temporal buffering for HIPAA compliance
- Audit trail generation with thumbnails
"""
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
import threading

import av
import cv2
import numpy as np
from ultralytics import YOLO

from processing.tracker import ByteTracker, ByteTrackerConfig, TemporalBuffer
from processing.anonymizer import Anonymizer


# Resolution presets for detection
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
    """Scale bounding boxes between resolutions.

    Args:
        boxes: Bounding boxes in tlbr format [N, 4]
        from_size: Original size (width, height)
        to_size: Target size (width, height)

    Returns:
        Scaled bounding boxes in tlbr format [N, 4]
    """
    if len(boxes) == 0:
        return boxes

    from_w, from_h = from_size
    to_w, to_h = to_size

    scale_x = to_w / from_w
    scale_y = to_h / from_h

    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x  # x1, x2
    scaled_boxes[:, [1, 3]] *= scale_y  # y1, y2

    return scaled_boxes.astype(np.int32)


def resize_for_detection(
    frame: np.ndarray,
    target_resolution: str
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Resize frame for detection while preserving aspect ratio.

    Args:
        frame: Input frame in BGR format
        target_resolution: Target resolution from RESOLUTION_MAP

    Returns:
        Tuple of (resized_frame, original_size, detection_size)
        - resized_frame: Resized frame for detection
        - original_size: (width, height) of original frame
        - detection_size: (width, height) of detection frame
    """
    original_h, original_w = frame.shape[:2]
    original_size = (original_w, original_h)

    # If native resolution, no resizing needed
    if target_resolution == "native" or target_resolution not in RESOLUTION_MAP:
        return frame, original_size, original_size

    target_w, target_h = RESOLUTION_MAP[target_resolution]

    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_w / original_w, target_h / original_h)

    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    detection_size = (new_w, new_h)

    return resized, original_size, detection_size


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline."""
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


@dataclass
class ProcessingStats:
    """Statistics from video processing."""
    total_frames: int = 0
    processed_frames: int = 0
    faces_detected: int = 0
    unique_tracks: int = 0
    processing_time: float = 0.0
    average_fps: float = 0.0
    detection_fps: float = 0.0
    video_info: Dict[str, any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class VideoProcessor:
    """High-performance video processor with tracking and batched inference.

    Features:
    - Resolution downscaling for faster detection
    - Batched inference with FP16 on CUDA
    - ByteTrack for temporal consistency
    - Conservative temporal buffering
    - Audit trail generation
    """

    def __init__(
        self,
        config: ProcessingConfig,
        on_progress: Optional[Callable[[int, int, int, float], None]] = None
    ):
        """Initialize video processor.

        Args:
            config: Processing configuration
            on_progress: Optional callback(frame, total_frames, faces, fps)
        """
        self.config = config
        self.on_progress = on_progress
        self._cancel_flag = threading.Event()

        # Will be initialized in process()
        self.model: Optional[YOLO] = None
        self.tracker: Optional[ByteTracker] = None
        self.temporal_buffer: Optional[TemporalBuffer] = None

        # Statistics
        self.stats = ProcessingStats()

        # Audit data - convert Path objects to strings for JSON serialization
        self.audit_thumbnails: List[np.ndarray] = []
        config_dict = asdict(config)
        # Convert any Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        self.audit_data: Dict = {
            "config": config_dict,
            "frames": [],
        }

    def _load_model(self) -> YOLO:
        """Load YOLO model with device detection.

        Returns:
            Loaded YOLO model on appropriate device
        """
        import torch

        model_name = self.config.model_name

        # Add .pt extension if not present
        if not model_name.endswith('.pt'):
            model_name = f"{model_name}.pt"

        # Check multiple locations for the model
        possible_paths = [
            Path(model_name),  # Absolute or relative path as-is
            Path(__file__).parent.parent / "models" / model_name,  # python/models/
            Path(__file__).parent.parent / model_name,  # python/
        ]

        model_path = None
        for p in possible_paths:
            if p.exists():
                model_path = p
                break

        # Load model
        if model_path:
            print(f"Loading model from: {model_path}")
            model = YOLO(str(model_path))
        else:
            # Try ultralytics auto-download as fallback
            print(f"Loading model: {model_name}")
            model = YOLO(model_name)

        # Detect and use best available device
        if torch.cuda.is_available():
            device = 'cuda'
            print("Using CUDA for inference")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("Using MPS (Apple Silicon) for inference")
        else:
            device = 'cpu'
            print("Using CPU for inference")

        model.to(device)

        return model

    def _init_tracker(self):
        """Initialize ByteTracker and TemporalBuffer."""
        tracker_config = ByteTrackerConfig(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            low_thresh=0.1,
            second_match_thresh=0.5,
        )

        self.tracker = ByteTracker(tracker_config)
        self.temporal_buffer = TemporalBuffer(buffer_frames=self.config.temporal_buffer)

    def _detect_faces(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Batch detection with resolution scaling.

        Args:
            frames: List of frames in BGR format

        Returns:
            Tuple of (all_boxes, all_scores) - one per frame
            Each element is an array of boxes/scores for that frame
        """
        if not frames:
            return [], []

        # Resize frames for detection
        resized_frames = []
        original_sizes = []
        detection_sizes = []

        for frame in frames:
            resized, orig_size, det_size = resize_for_detection(
                frame,
                self.config.detection_resolution
            )
            resized_frames.append(resized)
            original_sizes.append(orig_size)
            detection_sizes.append(det_size)

        # Batch inference
        import torch
        use_half = torch.cuda.is_available()  # FP16 only on CUDA

        results = self.model.predict(
            resized_frames,
            conf=self.config.confidence,
            half=use_half,
            verbose=False,
        )

        # Process results and scale back to original resolution
        all_boxes = []
        all_scores = []

        for i, result in enumerate(results):
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                # Get boxes and scores
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()

                # Scale boxes back to original resolution
                scaled_boxes = scale_detections(
                    xyxy,
                    from_size=detection_sizes[i],
                    to_size=original_sizes[i]
                )

                all_boxes.append(scaled_boxes)
                all_scores.append(scores)
            else:
                all_boxes.append(np.empty((0, 4), dtype=np.int32))
                all_scores.append(np.empty((0,), dtype=np.float32))

        return all_boxes, all_scores

    def _apply_redaction(
        self,
        frame: np.ndarray,
        boxes: np.ndarray
    ) -> np.ndarray:
        """Apply solid redaction with padding to frame.

        Args:
            frame: Frame in BGR format
            boxes: Bounding boxes in tlbr format [N, 4]

        Returns:
            Redacted frame
        """
        result = frame.copy()

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            result = Anonymizer.apply(
                result,
                x1, y1, x2, y2,
                mode=self.config.redaction_mode,
                color=self.config.redaction_color,
                padding=self.config.padding
            )

        return result

    def _process_batch(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int]
    ) -> List[np.ndarray]:
        """Process batch of frames with detection and tracking.

        Args:
            frames: List of frames in BGR format
            frame_numbers: Corresponding frame numbers

        Returns:
            List of redacted frames
        """
        # Detect faces in batch
        batch_boxes, batch_scores = self._detect_faces(frames)

        # Process each frame with tracking
        redacted_frames = []

        for i, (frame, boxes, scores, frame_num) in enumerate(
            zip(frames, batch_boxes, batch_scores, frame_numbers)
        ):
            # Update tracker with detections
            if len(boxes) > 0:
                tracks = self.tracker.update(boxes, scores)
                self.temporal_buffer.add_tracks(tracks, frame_num)

                # Update statistics
                self.stats.faces_detected += len(boxes)
                tracker_stats = self.tracker.get_statistics()
                self.stats.unique_tracks = tracker_stats['total_tracks']
            else:
                self.tracker.update(
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32)
                )

            # Get boxes from temporal buffer (includes extended frames)
            redaction_boxes = self.temporal_buffer.get_boxes_for_frame(frame_num)

            # Apply redaction
            redacted = self._apply_redaction(frame, redaction_boxes)
            redacted_frames.append(redacted)

            # Capture thumbnail for audit
            if (self.config.generate_audit and
                frame_num % self.config.thumbnail_interval == 0):
                # Store small thumbnail
                thumb = cv2.resize(redacted, (160, 90))
                self.audit_thumbnails.append(thumb)

                # Store frame metadata
                self.audit_data["frames"].append({
                    "frame_number": frame_num,
                    "faces_detected": len(boxes),
                    "redaction_boxes": len(redaction_boxes),
                    "thumbnail_index": len(self.audit_thumbnails) - 1,
                })

        return redacted_frames

    def _generate_audit(self, output_dir: Path):
        """Generate audit.json and audit.png thumbnail montage.

        Args:
            output_dir: Directory to save audit files
        """
        if not self.config.generate_audit or not self.audit_thumbnails:
            return

        # Save audit JSON
        audit_json_path = output_dir / "audit.json"

        audit_output = {
            "config": self.audit_data["config"],
            "stats": asdict(self.stats),
            "frames": self.audit_data["frames"],
            "timestamp": time.time(),
        }

        with open(audit_json_path, 'w') as f:
            json.dump(audit_output, f, indent=2)

        print(f"Audit JSON saved to: {audit_json_path}")

        # Create thumbnail montage
        if self.audit_thumbnails:
            # Calculate grid dimensions
            num_thumbs = len(self.audit_thumbnails)
            cols = min(8, num_thumbs)
            rows = (num_thumbs + cols - 1) // cols

            # Create montage
            thumb_h, thumb_w = self.audit_thumbnails[0].shape[:2]
            montage = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

            for i, thumb in enumerate(self.audit_thumbnails):
                row = i // cols
                col = i % cols
                y = row * thumb_h
                x = col * thumb_w
                montage[y:y+thumb_h, x:x+thumb_w] = thumb

            # Save montage
            audit_png_path = output_dir / "audit.png"
            cv2.imwrite(str(audit_png_path), montage)
            print(f"Audit thumbnail montage saved to: {audit_png_path}")

    def cancel(self):
        """Set cancellation flag to stop processing."""
        self._cancel_flag.set()

    def process(self) -> ProcessingStats:
        """Process video with face detection, tracking, and anonymization.

        Returns:
            ProcessingStats with results

        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If processing fails
        """
        self._cancel_flag.clear()

        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.config.input_path}")

        # Ensure output directory exists
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load model and initialize tracker
        print("Loading model...")
        self.model = self._load_model()

        print("Initializing tracker...")
        self._init_tracker()

        start_time = time.time()
        detection_time = 0.0

        try:
            # Open input video
            print(f"Opening input video: {self.config.input_path}")
            input_container = av.open(str(self.config.input_path))
            input_stream = input_container.streams.video[0]

            # Get video properties
            self.stats.total_frames = input_stream.frames
            if self.stats.total_frames == 0:
                # Fallback: estimate from duration and fps
                self.stats.total_frames = int(
                    input_stream.duration * input_stream.time_base * input_stream.average_rate
                )

            self.stats.video_info = {
                "width": input_stream.width,
                "height": input_stream.height,
                "fps": float(input_stream.average_rate),
                "codec": input_stream.codec_context.name,
                "total_frames": self.stats.total_frames,
            }

            print(f"Video: {input_stream.width}x{input_stream.height} @ {input_stream.average_rate} fps")
            print(f"Total frames: {self.stats.total_frames}")
            print(f"Detection resolution: {self.config.detection_resolution}")
            print(f"Batch size: {self.config.batch_size}")

            # Open output video
            output_container = av.open(str(self.config.output_path), 'w')
            output_stream = output_container.add_stream(
                'h264',
                rate=input_stream.average_rate
            )
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = 'yuv420p'
            output_stream.options = {'crf': '23', 'preset': 'veryfast'}

            # Checkpoint file for crash recovery
            checkpoint_file = self.config.output_path.parent / f".{self.config.output_path.stem}_checkpoint.json"

            # Batch processing
            frame_batch = []
            frame_numbers = []
            frame_objects = []  # Store av.VideoFrame objects for pts/time_base
            frame_count = 0

            print("Processing video...")

            for frame in input_container.decode(video=0):
                # Check cancellation
                if self._cancel_flag.is_set():
                    self.stats.warnings.append("Processing cancelled by user")
                    break

                # Convert frame to numpy array
                img = frame.to_ndarray(format='bgr24')

                # Add to batch
                frame_batch.append(img)
                frame_numbers.append(frame_count)
                frame_objects.append(frame)

                # Process batch when full
                if len(frame_batch) >= self.config.batch_size:
                    # Process batch
                    batch_start = time.time()
                    redacted_batch = self._process_batch(frame_batch, frame_numbers)
                    detection_time += time.time() - batch_start

                    # Write frames
                    for redacted, orig_frame in zip(redacted_batch, frame_objects):
                        new_frame = av.VideoFrame.from_ndarray(redacted, format='bgr24')
                        new_frame.pts = orig_frame.pts
                        new_frame.time_base = orig_frame.time_base

                        for packet in output_stream.encode(new_frame):
                            output_container.mux(packet)

                    # Update progress
                    frame_count += len(frame_batch)
                    self.stats.processed_frames = frame_count

                    if self.on_progress:
                        elapsed = time.time() - start_time
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        self.on_progress(
                            frame_count,
                            self.stats.total_frames,
                            len(self.tracker.tracked_tracks) if self.tracker else 0,
                            current_fps
                        )

                    # Checkpoint
                    if frame_count % self.config.checkpoint_interval == 0:
                        checkpoint_data = {
                            'frame': frame_count,
                            'faces_detected': self.stats.faces_detected,
                            'timestamp': time.time()
                        }
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint_data, f)

                        # Cleanup old track histories
                        self.temporal_buffer.cleanup_old_tracks()

                    # Clear batch
                    frame_batch = []
                    frame_numbers = []
                    frame_objects = []
                else:
                    frame_count += 1

            # Process remaining frames in batch
            if frame_batch:
                batch_start = time.time()
                redacted_batch = self._process_batch(frame_batch, frame_numbers)
                detection_time += time.time() - batch_start

                for redacted, orig_frame in zip(redacted_batch, frame_objects):
                    new_frame = av.VideoFrame.from_ndarray(redacted, format='bgr24')
                    new_frame.pts = orig_frame.pts
                    new_frame.time_base = orig_frame.time_base

                    for packet in output_stream.encode(new_frame):
                        output_container.mux(packet)

                self.stats.processed_frames = frame_count

            # Flush encoder
            for packet in output_stream.encode():
                output_container.mux(packet)

            # Close containers
            input_container.close()
            output_container.close()

            # Clean up checkpoint file
            if checkpoint_file.exists():
                checkpoint_file.unlink()

            # Calculate final stats
            self.stats.processing_time = time.time() - start_time
            self.stats.average_fps = (
                self.stats.processed_frames / self.stats.processing_time
                if self.stats.processing_time > 0 else 0
            )
            self.stats.detection_fps = (
                self.stats.processed_frames / detection_time
                if detection_time > 0 else 0
            )

            # Generate audit trail
            if self.config.generate_audit:
                print("Generating audit trail...")
                self._generate_audit(self.config.output_path.parent)

            print(f"\nProcessing complete!")
            print(f"Processed {self.stats.processed_frames} frames in {self.stats.processing_time:.2f}s")
            print(f"Average FPS: {self.stats.average_fps:.2f}")
            print(f"Detection FPS: {self.stats.detection_fps:.2f}")
            print(f"Faces detected: {self.stats.faces_detected}")
            print(f"Unique tracks: {self.stats.unique_tracks}")

        except Exception as e:
            self.stats.errors.append(str(e))
            raise RuntimeError(f"Video processing failed: {e}") from e

        return self.stats
