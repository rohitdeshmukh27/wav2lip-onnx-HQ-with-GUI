

IMPORTANT: ONNX model files removed from this repository

All large model files (ONNX, .pt, .pth, etc.) have been removed from this git repository to keep the repo small and avoid GitHub's file-size limits. You must download the model files from the Google Drive folder maintained by the original project persona and place them into the local folders used by the project.

Google Drive (models):
https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ

Put downloaded files into one of these folders in the repo (examples):
- `checkpoints/`
- `wav2lip_onnx_models/`

Placeholders such as `place wav2lip onnx models here.txt` are kept in the repository so contributors know where to put the models locally.

Quick setup and run (Windows PowerShell)

1) Create a Python virtual environment (if you don't already have one):

```powershell
python -m venv venv
```

2) Activate the venv (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
# or for cmd.exe: .\venv\Scripts\activate.bat
```

3) Install requirements:

```powershell
pip install -r requirements.txt
```

4) Download the ONNX model files from the Google Drive link above and place them into `checkpoints/` or `wav2lip_onnx_models/`.

5) Start the web UI using the venv Python to run uvicorn (example uses the project's venv):

```powershell
"$(Resolve-Path .\venv\Scripts\python.exe)" -m uvicorn web.main:app --host 0.0.0.0 --port 8000
```

Open a browser to http://localhost:8000 and you can upload/select audio and video to run inference.

Notes
- If you prefer, use the included `scripts/download_models.py` to download model files from an externally hosted URL (edit with the Drive direct-download link or alternative host).
- We intentionally keep models out of git. If you want models tracked inside git, use Git LFS or host them on the Hugging Face Hub and update the README accordingly.

Follow the original repo + tutorial (recommended)

This repository includes a `web/` folder that provides a browser-based UI to run inference locally. To get the full experience and install checkpoints correctly, follow this recommended flow:

1. Watch the quick setup video that explains installing dependencies and checkpoints used by the original project:
  https://youtu.be/KZ8OtMqcFWM

2. Consult the original project repository for exact checkpoint names and installation steps:
  https://github.com/instant-high/wav2lip-onnx-HQ

  - Follow the "Model download" or "Checkpoints" section in the original repo and download the required ONNX models (the Google Drive link is already listed above).
  - Place the downloaded models into the same folders expected by this repo (for example `checkpoints/` and `wav2lip_onnx_models/`).

3. Use this repo for the web UI and local inference.

4. Install Python dependencies (from this repo's `requirements.txt`) and create/activate a virtualenv. Example (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

5. Start the web UI (run with the venv Python to ensure correct environment):

```powershell
"$(Resolve-Path .\venv\Scripts\python.exe)" -m uvicorn web.main:app --host 0.0.0.0 --port 8000
```

6. Open http://localhost:8000 in a browser. You should be able to select audio and video and run inference using the models you installed from the original repo.

Acknowledgements

Contributors

- https://github.com/lakxhit and https://github.com/cchaitanya96 improved the user interaction process for the existing ONNX model. In the original workflow, users had to manually select the human face in the video. Instead of retraining or updating the model, they modified the pipeline to remove this manual selection step and replaced it with an automated flow. They also developed a simple GUI that allows users to select video and audio inputs, execute the process, and view results more conveniently.

- https://github.com/rohitdeshmukh27 redesigned the web interface for this repository, improving usability and integrating the browser-based UI with the inference pipeline.


-- MAIN REPO README FILE
# wav2lip-onnx-HQ
Update 29.04.2025 (inference_onnxModel_V2.py)

  - replaced occlusion mask with xseg occlusion
  - added option frame enhancer realEsrgan (clear_reality_x4 model included)
  - added option short fade-in/fade-out
  - added option for facemode 0 or 1 for better result on different face shapes  
    (0=portrait like orig. wav2lip, 1=square for less mouth opening)
  - bugfix crashing when using xseg and specific face is not detected  

Update 08.02.2025

  - optmized occlusion mask
  - Replaced insightface with retinaface detection/alignment for easier installation
  - Replaced seg-mask with faster blendmasker
  - Added free cropping of final result video
  - Added specific target face selection from first frame

.

Just another Wav2Lip HQ local installation, fully running on Torch to ONNX converted models for:
- face-detection
- face-recognition
- face-alignment
- face-parsing
- face-enhancement
- wav2lip inference.

.

Can be run on CPU or Nvidia GPU

I've made some modifications such as:
* New face-detection and face-alignment code. (working for ~ +- 60º head tilt)
* Four different face enhancers available, adjustable enhancement level .
* Choose pingpong loop instead of original loop function.
* Set cut-in/cut-out position to create the loop or cut longer video.
* Cut-in position = used frame if static is selected.
* Select the target face.
* Use two audio files, eg. vocal for driving and full music mix for final output.
* This version does not crash if no face is detected, it just continues ...

Type --help for all commandline parameters

.
 
Model download - https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ?usp=sharing  

Original wav2lip - https://github.com/Rudrabha/Wav2Lip

Face enhancers taken from -  https://github.com/harisreedhar/Face-Upscalers-ONNX

Face detection taken from - https://github.com/neuralchen/SimSwap

Face occluder taken from - https://github.com/facefusion/facefusion-assets/releases

Blendmasker extracted from - https://github.com/mapooon/BlendFace during onnx conversion

Face recognition for specifc face taken from - https://github.com/jahongir7174/FaceID

.

.


