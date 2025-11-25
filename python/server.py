from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List, Optional, Literal
import torch
import asyncio
import uuid


class ProcessRequest(BaseModel):
    file: str = Field(..., description="Path to the video file to process")
    output_dir: str = Field(..., description="Directory to save processed output")
    model: str = Field(default="yolov8n-face.pt", description="Face detection model to use")
    mode: Literal["blur", "pixelate", "mask"] = Field(default="blur", description="Processing mode")
    color: Optional[str] = Field(default=None, description="Color for mask mode (hex format)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    padding: int = Field(default=10, ge=0, description="Padding around detected faces in pixels")


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


websocket_connections: List[WebSocket] = []


async def broadcast_progress(message: dict):
    disconnected = []
    for websocket in websocket_connections:
        try:
            await websocket.send_json(message)
        except Exception:
            disconnected.append(websocket)

    for websocket in disconnected:
        websocket_connections.remove(websocket)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Fraser FastAPI server...")
    device_info = get_device_info()
    print(f"Device: {device_info.type} - {device_info.name}")
    yield
    print("Shutting down Fraser FastAPI server...")
    websocket_connections.clear()


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


@app.post("/process", response_model=ProcessResponse)
async def process_video(request: ProcessRequest):
    job_id = str(uuid.uuid4())

    return ProcessResponse(
        job_id=job_id,
        status="queued",
        message=f"Video processing job {job_id} has been queued"
    )


@app.post("/cancel/{job_id}", response_model=CancelResponse)
async def cancel_job(job_id: str):
    return CancelResponse(
        job_id=job_id,
        status="cancelled",
        message=f"Job {job_id} cancellation requested"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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
