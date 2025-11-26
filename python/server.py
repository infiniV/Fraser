from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List, Optional, Literal, Dict
import concurrent.futures
from pathlib import Path
import torch
import asyncio
import uuid

from processing import VideoProcessor
from processing.pipeline import ProcessingJob


class ProcessRequest(BaseModel):
    input_path: str = Field(..., description="Path to the video file to process")
    output_path: str = Field(..., description="Path to save processed output")
    model: str = Field(default="yolov11n-face.pt", description="Face detection model to use")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0, description="Detection confidence threshold")
    padding: float = Field(default=0.20, ge=0.0, le=1.0, description="Padding as fraction of face size")
    detection_resolution: str = Field(default="360p", description="Detection resolution: 144p, 240p, 360p, or native")
    redaction_mode: str = Field(default="black", description="Redaction mode: blur, black, or color")
    redaction_color: str = Field(default="#000000", description="Color for color mode (hex format)")
    temporal_buffer: int = Field(default=5, ge=0, description="Temporal smoothing buffer size")
    generate_audit: bool = Field(default=True, description="Generate audit trail")
    thumbnail_interval: int = Field(default=30, ge=1, description="Thumbnail generation interval in seconds")


class DeviceInfo(BaseModel):
    type: str = Field(..., description="Device type: cuda, mps, or cpu")
    name: str = Field(..., description="Device name or description")
    cuda_available: bool = Field(..., description="Whether CUDA is available")
    mps_available: bool = Field(..., description="Whether MPS (Apple Metal) is available")


class HealthResponse(BaseModel):
    status: str
    device_info: DeviceInfo


class ModelsResponse(BaseModel):
    models: List[str]


class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str


class CancelResponse(BaseModel):
    job_id: str
    status: str
    message: str


def get_device_info() -> DeviceInfo:
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if cuda_available:
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif mps_available:
        device_type = "mps"
        device_name = "Apple Metal Performance Shaders"
    else:
        device_type = "cpu"
        device_name = "CPU"

    return DeviceInfo(
        type=device_type,
        name=device_name,
        cuda_available=cuda_available,
        mps_available=mps_available
    )


RESOLUTION_MAP = {
    "144p": (256, 144),
    "240p": (416, 240),
    "360p": (640, 360),
    "native": None,
}

websocket_connections: List[WebSocket] = []
active_connections: Dict[str, WebSocket] = {}
active_jobs: Dict[str, dict] = {}
video_processor: Optional[VideoProcessor] = None


async def broadcast_progress(job_id: str, data: dict):
    """Send progress update to specific job's WebSocket connection."""
    if job_id in active_connections:
        try:
            await active_connections[job_id].send_json(data)
        except Exception:
            # Connection closed, remove it
            active_connections.pop(job_id, None)


async def broadcast_to_all(message: dict):
    """Broadcast message to all general WebSocket connections."""
    disconnected = []
    for websocket in websocket_connections:
        try:
            await websocket.send_json(message)
        except Exception:
            disconnected.append(websocket)

    for websocket in disconnected:
        websocket_connections.remove(websocket)


def sync_progress_callback(job_id: str):
    last_logged = [0]  # Use list to allow modification in closure

    def callback(frame: int, total_frames: int, faces: int, fps: float):
        progress = {
            "type": "progress",
            "job_id": job_id,
            "frame": frame,
            "total_frames": total_frames,
            "faces_in_frame": faces,
            "fps": round(fps, 2),
            "percent": round((frame / total_frames) * 100, 1) if total_frames > 0 else 0
        }
        active_jobs[job_id] = progress

        # Only log every 100 frames to reduce overhead
        if frame - last_logged[0] >= 100 or frame == total_frames:
            last_logged[0] = frame
            print(f"[{job_id[:8]}] Frame {frame}/{total_frames} ({progress['percent']}%) - {faces} faces - {fps:.1f} FPS")
    return callback


@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_processor
    print("Starting Fraser FastAPI server...")
    device_info = get_device_info()
    print(f"Device: {device_info.type} - {device_info.name}")

    # Initialize video processor
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    video_processor = VideoProcessor(str(models_dir))

    yield
    print("Shutting down Fraser FastAPI server...")
    websocket_connections.clear()
    active_jobs.clear()


app = FastAPI(
    title="Fraser - Face Detection & Obfuscation API",
    description="API for processing videos with face detection and obfuscation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    device_info = get_device_info()
    return HealthResponse(
        status="healthy",
        device_info=device_info
    )


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    available_models = [
        "yolov8n-face.pt",
        "yolov8s-face.pt",
        "yolov8m-face.pt",
        "yolov8l-face.pt",
        "yolov8x-face.pt"
    ]
    return ModelsResponse(models=available_models)


def run_processing(job_id: str, request: ProcessRequest):
    """Run video processing synchronously in background thread."""
    global video_processor

    input_path = Path(request.input_path)
    output_path = Path(request.output_path)

    job = ProcessingJob(
        id=job_id,
        input_path=str(input_path),
        output_path=str(output_path),
        model=request.model,
        mode=request.redaction_mode,
        color=request.redaction_color,
        confidence=request.confidence,
        padding=request.padding
    )

    try:
        print(f"[{job_id[:8]}] Starting processing: {input_path.name}")
        stats = video_processor.process(
            job,
            progress_callback=sync_progress_callback(job_id)
        )
        print(f"[{job_id[:8]}] Completed: {stats.processed_frames} frames, {stats.faces_detected} faces, {stats.average_fps:.1f} FPS")

        # Save summary report
        output_dir = Path(output_path).parent
        report_path = video_processor.save_report(job, stats, str(output_dir))
        print(f"[{job_id[:8]}] Report saved: {report_path}")

        active_jobs[job_id] = {
            "type": "completed",
            "job_id": job_id,
            "stats": {
                "total_frames": stats.total_frames,
                "processed_frames": stats.processed_frames,
                "faces_detected": stats.faces_detected,
                "processing_time": round(stats.processing_time, 2),
                "average_fps": round(stats.average_fps, 2)
            },
            "output_path": str(output_path),
            "report_path": report_path
        }
    except Exception as e:
        print(f"[{job_id[:8]}] Error: {e}")
        import traceback
        traceback.print_exc()
        active_jobs[job_id] = {
            "type": "error",
            "job_id": job_id,
            "error": str(e)
        }


@app.post("/process", response_model=ProcessResponse)
async def process_video(request: ProcessRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    # Validate input file exists
    if not Path(request.input_path).exists():
        raise HTTPException(status_code=400, detail=f"Input file not found: {request.input_path}")

    # Validate output directory
    output_dir = Path(request.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start processing in background
    background_tasks.add_task(run_processing, job_id, request)

    return ProcessResponse(
        job_id=job_id,
        status="processing",
        message=f"Video processing started for {Path(request.input_path).name}"
    )


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return active_jobs[job_id]


@app.post("/cancel/{job_id}", response_model=CancelResponse)
async def cancel_job(job_id: str):
    if video_processor:
        video_processor.cancel()
    return CancelResponse(
        job_id=job_id,
        status="cancelled",
        message=f"Job {job_id} cancellation requested"
    )


@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for receiving real-time progress updates for a specific job."""
    await websocket.accept()
    active_connections[job_id] = websocket

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "message": "Connected to progress stream"
        })

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back pong for heartbeat
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "job_id": job_id
                })
    except WebSocketDisconnect:
        # Clean up connection on disconnect
        active_connections.pop(job_id, None)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """General WebSocket endpoint for system-wide updates."""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "pong",
                "message": f"Received: {data}"
            })
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8420,
        reload=True
    )
