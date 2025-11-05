import os
import logging
from pathlib import Path
from typing import NamedTuple
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

from sample_utils.download import download_file

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent
logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "models" / "YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# -----------------------------------------------------
# LOAD MODEL (cached)
# -----------------------------------------------------
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(str(MODEL_LOCAL_PATH))
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# -----------------------------------------------------
# UI HEADER
# -----------------------------------------------------
st.title("📸 Road Damage Detection - Image Mode")
st.write("Upload a road image to detect damages using the trained YOLOv8 model.")

# File upload
image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Confidence slider
score_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)
st.caption("Tip: lower the threshold if nothing is detected, raise it if there are false positives.")

# -----------------------------------------------------
# PROCESS IMAGE
# -----------------------------------------------------
if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    # Convert to numpy
    np_img = np.array(image)
    h_ori, w_ori = np_img.shape[:2]

    st.info("Running detection... 🚀")

    # Run YOLOv8 inference
    resized = cv2.resize(np_img, (640, 640))
    results = net.predict(resized, conf=score_threshold)

    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for _box in boxes:
            detections.append(
                Detection(
                    class_id=int(_box.cls),
                    label=CLASSES[int(_box.cls)],
                    score=float(_box.conf),
                    box=_box.xyxy[0].astype(int)
                )
            )

    annotated = results[0].plot()
    annotated = cv2.resize(annotated, (w_ori, h_ori))

    # -----------------------------------------------------
    # SHOW RESULTS
    # -----------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.image(np_img, caption="Original Image", width=500)
    with col2:
        st.image(annotated, caption="Detected Damages", width=500)

        # Download annotated image
        buf = BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.download_button(
            label="📥 Download Annotated Image",
            data=buf.getvalue(),
            file_name="RDD_Prediction.png",
            mime="image/png"
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------
    if detections:
        st.subheader("📊 Detection Summary")
        st.table([
            {"Damage Type": d.label, "Confidence": f"{d.score:.2f}", "Box": d.box.tolist()}
            for d in detections
        ])
    else:
        st.warning("No damages detected at current confidence threshold.")
