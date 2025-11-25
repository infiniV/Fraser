"""Video processing pipeline with YOLO face detection."""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Dict, List
import threading

import av
import numpy as np
from ultralytics import YOLO

from .anonymizer import Anonymizer


@dataclass
class ProcessingStats:
    """Statistics from video processing."""
    total_frames: int
    processed_frames: int
    faces_detected: int
    processing_time: float
    average_fps: float
    warnings: List[str]
    errors: List[str]


@dataclass
class ProcessingJob:
    """Video processing job configuration."""
    id: str
    input_path: str
    output_path: str
    model: str
    mode: str
    color: str
    confidence: float
    padding: float


class VideoProcessor:
    """Process videos with YOLO face detection and anonymization."""

    def __init__(self, models_dir: str):
        """Initialize video processor.

        Args:
            models_dir: Directory containing YOLO model files
        """
        self.models_dir = Path(models_dir)
        self.models_cache: Dict[str, YOLO] = {}
        self._cancel_flag = threading.Event()

    def load_model(self, model_name: str) -> YOLO:
        """Load YOLO model with caching.

        Args:
            model_name: Name of the model file (e.g., "yolov8n-face.pt")

        Returns:
            Loaded YOLO model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if model_name in self.models_cache:
            return self.models_cache[model_name]

        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = YOLO(str(model_path))
        self.models_cache[model_name] = model

        return model

    def cancel(self):
        """Set cancellation flag to stop processing."""
        self._cancel_flag.set()

    def process(
        self,
        job: ProcessingJob,
        progress_callback: Optional[Callable[[int, int, int, float], None]] = None,
        resume_frame: int = 0
    ) -> ProcessingStats:
        """Process video with face detection and anonymization.

        Args:
            job: Processing job configuration
            progress_callback: Optional callback(frame, total_frames, faces, fps)
            resume_frame: Frame number to resume from (for crash recovery)

        Returns:
            ProcessingStats with results

        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If processing fails
        """
        self._cancel_flag.clear()
        input_path = Path(job.input_path)
        output_path = Path(job.output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load model
        model = self.load_model(job.model)

        # Initialize stats
        stats = ProcessingStats(
            total_frames=0,
            processed_frames=0,
            faces_detected=0,
            processing_time=0.0,
            average_fps=0.0,
            warnings=[],
            errors=[]
        )

        start_time = time.time()

        try:
            # Open input video
            input_container = av.open(str(input_path))
            input_stream = input_container.streams.video[0]

            # Get video properties
            stats.total_frames = input_stream.frames
            if stats.total_frames == 0:
                # Fallback: estimate from duration and fps
                stats.total_frames = int(
                    input_stream.duration * input_stream.time_base * input_stream.average_rate
                )

            # Open output video
            output_container = av.open(str(output_path), 'w')
            output_stream = output_container.add_stream(
                'h264',
                rate=input_stream.average_rate
            )
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = 'yuv420p'
            output_stream.options = {'crf': '23', 'preset': 'medium'}

            # Checkpoint file for crash recovery
            checkpoint_file = output_path.parent / f".{output_path.stem}_checkpoint.json"

            frame_count = 0
            checkpoint_interval = 100

            # Process frames
            for frame in input_container.decode(video=0):
                # Check cancellation
                if self._cancel_flag.is_set():
                    stats.warnings.append("Processing cancelled by user")
                    break

                # Skip frames if resuming
                if frame_count < resume_frame:
                    frame_count += 1
                    continue

                # Convert frame to numpy array
                img = frame.to_ndarray(format='bgr24')

                # Run YOLO detection with stream=True for memory efficiency
                results = model.predict(
                    img,
                    conf=job.confidence,
                    stream=True,
                    verbose=False
                )

                # Process detections
                faces_in_frame = 0
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # Apply anonymization
                            img = Anonymizer.apply(
                                img,
                                x1, y1, x2, y2,
                                mode=job.mode,
                                color=job.color,
                                padding=job.padding
                            )

                            faces_in_frame += 1
                            stats.faces_detected += 1

                # Convert back to video frame
                new_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base

                # Encode and write frame
                for packet in output_stream.encode(new_frame):
                    output_container.mux(packet)

                frame_count += 1
                stats.processed_frames = frame_count

                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0

                # Progress callback
                if progress_callback:
                    progress_callback(frame_count, stats.total_frames, faces_in_frame, current_fps)

                # Checkpoint every N frames
                if frame_count % checkpoint_interval == 0:
                    checkpoint_data = {
                        'frame': frame_count,
                        'faces_detected': stats.faces_detected,
                        'timestamp': time.time()
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f)

            # Flush encoder
            for packet in output_stream.encode():
                output_container.mux(packet)

            # Close containers
            input_container.close()
            output_container.close()

            # Clean up checkpoint file
            if checkpoint_file.exists():
                checkpoint_file.unlink()

            # Final stats
            stats.processing_time = time.time() - start_time
            stats.average_fps = stats.processed_frames / stats.processing_time if stats.processing_time > 0 else 0

        except Exception as e:
            stats.errors.append(str(e))
            raise RuntimeError(f"Video processing failed: {e}") from e

        return stats

    def save_report(
        self,
        job: ProcessingJob,
        stats: ProcessingStats,
        output_dir: str
    ):
        """Save processing report as JSON.

        Args:
            job: Processing job configuration
            stats: Processing statistics
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / f"{job.id}_report.json"

        report = {
            'job': asdict(job),
            'stats': asdict(stats),
            'timestamp': time.time()
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
