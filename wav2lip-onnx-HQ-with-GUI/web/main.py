import os
import shutil
import subprocess
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

UPLOAD_DIR = "web/uploads"
RESULT_DIR = "web/results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.post("/generate")
def generate(video: UploadFile = File(...), audio: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    audio_path = os.path.join(UPLOAD_DIR, audio.filename)
    output_path = os.path.join(RESULT_DIR, f"output_{video.filename}_{audio.filename}.mp4")

    # Save uploaded files
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Run the ONNX inference script
    script = "inference_onnxModel_V2.py"
    checkpoint = "checkpoints/wav2lip_256.onnx"
    command = [
        "python", "-W", "ignore", script,
        "--checkpoint_path", checkpoint,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--pads", "10",
        "--fps", "29.97"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return Response(f"Error: {e}", status_code=500)

    # Return the output video
    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="video/mp4", filename="output.mp4")
    else:
        return Response("Output file not found", status_code=500)

@app.get("/")
def root():
    return FileResponse("web/static/index.html")
