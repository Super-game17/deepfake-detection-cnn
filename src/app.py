import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.augmentations import get_eval_transform
from src.train_model import build_model

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT_DIR / "best_model.pth"

FRAME_INTERVAL = 15      # analyse 1 frame out of 15 (same as video_inference.py)
MAX_ANALYZED_FRAMES = 60 # safety limit so long videos do not run forever
MIN_CONFIDENCE = 0.50
MIN_FACE_SIZE = 60
MAX_FACES_IMAGE = 4
VIDEO_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
}
SUPPORTED_VIDEO_SUFFIXES = {".avi", ".mov", ".mp4"}

st.set_page_config(page_title="Deepfake Detector", layout="wide")


def get_device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_classifier(checkpoint_path, device_name):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_name = checkpoint.get("model_name", "resnet50")
    else:
        state_dict = checkpoint
        model_name = "resnet50"
    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device_name)
    model.eval()
    return model


@st.cache_resource
def load_detector(device_name):
    from insightface.app import FaceAnalysis

    if device_name == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    detector = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=providers,
    )
    detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return detector


# ------------------------------------------------------------------ helpers
def crop_faces_bgr(frame_bgr, detector, max_faces=None):
    """Detect faces with SCRFD and return RGB crops (largest first)."""
    faces = detector.get(frame_bgr)
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    height, width = frame_bgr.shape[:2]
    crops = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        box_w, box_h = x2 - x1, y2 - y1
        if face.det_score < MIN_CONFIDENCE or min(box_w, box_h) < MIN_FACE_SIZE:
            continue
        margin_x, margin_y = int(box_w * 0.1), int(box_h * 0.1)
        crop = frame_bgr[
            max(0, y1 - margin_y):min(height, y2 + margin_y),
            max(0, x1 - margin_x):min(width, x2 + margin_x),
        ]
        if crop.size:
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if max_faces and len(crops) >= max_faces:
            break
    return crops, len(faces)


def to_tensor(face_rgb):
    return get_eval_transform()(image=face_rgb)["image"]


def predict_faces(model, crops, device_name):
    """Return the fake probability of each crop (model output is 1 logit)."""
    batch = torch.stack([to_tensor(c) for c in crops]).to(device_name)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).view(-1).cpu().numpy()
    return probs


def gradcam_overlay(model, face_rgb, device_name):
    """Grad-CAM library on the last block of layer4."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    tensor = to_tensor(face_rgb).unsqueeze(0).to(device_name)
    with torch.no_grad():
        logit = model(tensor)
        fake_prob = torch.sigmoid(logit).item()

    target = [ClassifierOutputTarget(0)] if fake_prob >= 0.5 else [_RealClassTarget()]
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam_generator:
        cam = cam_generator(input_tensor=tensor, targets=target)[0]

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    return fake_prob, base, overlay


class _RealClassTarget:
    def __call__(self, model_output):
        if model_output.ndim == 1:
            return -model_output[0]
        return -model_output[:, 0]


def show_verdict(fake_prob):
    real_prob = 1.0 - fake_prob
    col1, col2 = st.columns(2)
    col1.metric("Fake probability", f"{fake_prob:.1%}")
    col2.metric("Real probability", f"{real_prob:.1%}")
    if abs(fake_prob - 0.5) < 0.10:
        st.warning("Score close to 50%: the model is uncertain, treat this result with caution.")
    if fake_prob >= 0.5:
        st.error("Result: FAKE")
    else:
        st.success("Result: REAL")


def get_video_mime_type(filename):
    """Return the MIME type Streamlit should send to the browser."""
    return VIDEO_MIME_TYPES.get(Path(filename).suffix.lower(), "video/mp4")


@st.cache_data(show_spinner=False)
def convert_video_for_preview(video_bytes, suffix):
    """Convert an uploaded video to a browser-compatible MP4."""
    input_path = output_path = None
    try:
        import imageio_ffmpeg

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_file:
            input_file.write(video_bytes)
            input_path = Path(input_file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
            output_path = Path(output_file.name)

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return output_path.read_bytes()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        details = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise RuntimeError(f"Impossible de préparer l'aperçu vidéo: {details}") from error
    finally:
        if input_path:
            input_path.unlink(missing_ok=True)
        if output_path:
            output_path.unlink(missing_ok=True)


def analyze_video_streamlit(video_path, model, detector, device_name, progress):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The video file could not be opened.")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    scores = []
    best_crop, best_score_gap = None, -1.0
    frame_index = 0
    analyzed = 0

    while analyzed < MAX_ANALYZED_FRAMES:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % FRAME_INTERVAL == 0:
            analyzed += 1
            crops, _ = crop_faces_bgr(frame, detector)
            if crops:
                probs = predict_faces(model, crops, device_name)
                scores.append(float(np.mean(probs)))
                gap = abs(float(probs[0]) - 0.5)
                if gap > best_score_gap:
                    best_score_gap = gap
                    best_crop = crops[0]
        frame_index += 1
        progress.progress(min(frame_index / total_frames, 1.0))
    capture.release()
    progress.progress(1.0)

    if not scores:
        return None
    fake_prob = float(np.mean(scores))
    return {"fake_probability": fake_prob, "frames": len(scores), "best_crop": best_crop}


# ------------------------------------------------------------------ UI
st.title("Deepfake Detector")
st.caption("SCRFD face detection + ResNet-50 classification + Grad-CAM explanation")

device_name = get_device_name()
st.sidebar.write(f"Device: **{device_name.upper()}**")
st.sidebar.write("Label 0 = Real, label 1 = Fake")

if not CHECKPOINT_PATH.exists():
    st.error(
        f"'{CHECKPOINT_PATH.name}' was not found next to app.py. "
        "Ask your teammate to send best_model.pth and put it in the project root."
    )
    st.stop()

mode = st.radio("Input type", ["Video", "Image"], horizontal=True)

# ---------------------------------------------------------------- VIDEO
if mode == "Video":
    uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    if uploaded is not None:
        video_bytes = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix.lower() or ".mp4"
        if suffix not in SUPPORTED_VIDEO_SUFFIXES:
            st.error("Format vidéo non pris en charge.")
            st.stop()
        try:
            preview_bytes = convert_video_for_preview(video_bytes, suffix)
            st.video(preview_bytes, format="video/mp4")
        except RuntimeError as error:
            st.error(str(error))
            st.info("Installez les dépendances avec `pip install -r requirements.txt`, puis relancez l'application.")
        if st.button("Analyze video"):
            video_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_bytes)
                    video_path = tmp.name

                with st.spinner("Loading model and face detector..."):
                    model = load_classifier(CHECKPOINT_PATH, device_name)
                    detector = load_detector(device_name)

                st.write("Analyzing video...")
                progress = st.progress(0.0)
                result = analyze_video_streamlit(video_path, model, detector, device_name, progress)

                if result is None:
                    st.warning(
                        "No usable face detected in this video "
                        "(faces may be too small, blurry or absent)."
                    )
                else:
                    show_verdict(result["fake_probability"])
                    st.write(f"Frames analyzed (with a face): {result['frames']}")
                    st.write(
                        f"Only the first {MAX_ANALYZED_FRAMES} sampled frames are analyzed "
                        "to keep the app fast."
                    )
                    with st.spinner("Generating Grad-CAM..."):
                        _, base, overlay = gradcam_overlay(model, result["best_crop"], device_name)
                    st.subheader("Grad-CAM on the most decisive frame")
                    left, right = st.columns(2)
                    left.image(base, caption="Face crop", width=300)
                    right.image(overlay, caption="Grad-CAM heatmap", width=300)
            except Exception as error:
                st.error(f"Could not analyze this video: {error}")
            finally:
                if video_path:
                    Path(video_path).unlink(missing_ok=True)

# ---------------------------------------------------------------- IMAGE
else:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    already_cropped = st.checkbox("This image is already a cropped face (skip face detection)")
    if uploaded is not None:
        try:
            image_rgb = np.array(Image.open(uploaded).convert("RGB"))
        except Exception as error:
            st.error(f"Unreadable image: {error}")
            st.stop()

        st.image(image_rgb, caption="Original image", width=400)

        if st.button("Analyze image"):
            try:
                model = load_classifier(CHECKPOINT_PATH, device_name)
                if already_cropped:
                    crops, total = [image_rgb], 1
                else:
                    with st.spinner("Detecting faces..."):
                        detector = load_detector(device_name)
                        frame_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
                        crops, total = crop_faces_bgr(frame_bgr, detector, MAX_FACES_IMAGE)

                if not crops:
                    st.warning(
                        "No face detected. If the image is already a face crop, "
                        "tick the checkbox above."
                    )
                else:
                    if total > 1:
                        st.info(f"{total} faces detected. Showing up to {MAX_FACES_IMAGE}, largest first.")
                    for i, face in enumerate(crops, start=1):
                        st.subheader(f"Face {i}")
                        fake_prob, base, overlay = gradcam_overlay(model, face, device_name)
                        left, right = st.columns(2)
                        left.image(base, caption="Face crop", width=300)
                        right.image(overlay, caption="Grad-CAM heatmap", width=300)
                        show_verdict(fake_prob)
            except Exception as error:
                st.error(f"Could not analyze this image: {error}")

st.divider()
st.caption(
    "This result is an experimental prediction. Performance may drop on compressed videos, "
    "very small faces, unusual angles, or generation methods absent from the training datasets."
)