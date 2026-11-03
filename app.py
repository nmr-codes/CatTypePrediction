"""
MushukLens — cat breed classifier web app.

Run:   python app.py
Open:  http://127.0.0.1:5000

The app needs the trained weights file produced by the notebook:
    model/mushuklens_effnet_b0.pt
(or set MODEL_PATH to another location).
"""
import base64
import io
import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model" / "mushuklens_effnet_b0.pt"))
MAX_UPLOAD_MB = 8
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

model = None
class_names = []
load_error = None
lock = threading.Lock()  # Grad-CAM uses hooks, so run one prediction at a time


def load_model():
    global model, class_names, load_error
    if not MODEL_PATH.exists():
        load_error = f"Model file not found: {MODEL_PATH}"
        return
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        class_names = list(ckpt["classes"])
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, len(class_names))
        net.load_state_dict(ckpt["state_dict"])
        model = net.to(device).eval()
        load_error = None
        print(f"Model loaded: {len(class_names)} classes on {device}")
    except Exception as e:  # noqa: BLE001
        load_error = f"Could not load model: {e}"


def jet(x):
    """Tiny numpy version of the 'jet' colormap. x in [0,1] -> RGB uint8."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def predict_with_cam(img):
    x = preprocess(img).unsqueeze(0).to(device).requires_grad_(True)
    store = {}

    def fwd_hook(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = model.features[-1].register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            probs = F.softmax(logits, 1)[0].detach().cpu().numpy()
            idx = int(logits.argmax(1))
            model.zero_grad()
            logits[0, idx].backward()
    finally:
        handle.remove()

    a, g = store["a"].detach(), store["g"].detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).cpu().numpy()
    return probs, cam


def heatmap_overlay(img, cam):
    """Return a data-URL PNG of the cropped 224x224 view with the heatmap on top."""
    base = ImageOps.fit(img, (224, 224), Image.BICUBIC)  # same area the model saw (approx.)
    base_arr = np.asarray(base).astype(np.float32)
    heat = jet(cam).astype(np.float32)
    alpha = (0.25 + 0.5 * cam)[..., None]
    out = (base_arr * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).resize((448, 448), Image.BICUBIC).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    return jsonify(model_loaded=model is not None, error=load_error,
                   classes=class_names, device=str(device))


@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify(error=load_error or "Model is not loaded."), 503
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image received."), 400
    try:
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a readable image."), 400

    with lock:
        probs, cam = predict_with_cam(img)

    top = np.argsort(-probs)[:3]
    return jsonify(
        predictions=[{"breed": class_names[i], "probability": float(probs[i])} for i in top],
        heatmap=heatmap_overlay(img, cam),
    )


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error=f"Image is too large (max {MAX_UPLOAD_MB} MB)."), 413


load_model()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
