import streamlit as st
import time
import tempfile
import numpy as np
from gtts import gTTS
import os

# ---------------- SAFE IMPORTS ----------------
try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    st.error("❌ YOLO failed to load. Check requirements.txt")
    st.stop()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AR Navigation", layout="wide")

# -------- SESSION STATE INIT --------
if "last_spoken" not in st.session_state:
    st.session_state["last_spoken"] = ""

st.title("🧭 Augmented Reality Navigation for Visually Impaired")
st.markdown("### AI Powered Real-Time Navigation Assistance")

model = YOLO("yolov8n.pt")

# ---------------- AUDIO ----------------
def play_audio(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name)

# ---------------- SIDEBAR ----------------
mode = st.sidebar.radio("Select Mode", ["Live Camera (Local)", "Upload Video (Cloud)"])

IS_CLOUD = "STREAMLIT_SERVER_RUNNING" in os.environ

if IS_CLOUD:
    st.warning("⚠️ Live camera is not supported on Streamlit Cloud. Please use Upload Video mode.")

FRAME_WINDOW = st.image([])

# ---------------- OBJECT PRIORITY ----------------
priority = {
    "car": 1, "bus": 1, "truck": 1, "motorcycle": 1,
    "wall": 2, "door": 2, "stairs": 2,
    "chair": 3, "table": 3, "phone": 3,
    "person": 4,
    "bottle": 5, "backpack": 5, "laptop": 5
}

# ---------------- DETECTION FUNCTION ----------------
def detect_objects(frame):
    results = model(frame, stream=True)
    detected = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in priority:
                detected.append(label)

    return detected

# ---------------- ALERT LOGIC ----------------
def generate_alert(objects):
    if not objects:
        return ""

    objects = list(set(objects))
    objects.sort(key=lambda x: priority.get(x, 999))

    messages = []
    for obj in objects[:3]:
        messages.append(f"{obj} ahead, move carefully!")

    return ". ".join(messages)

# ---------------- LIVE CAMERA MODE ----------------
if mode == "Live Camera (Local)" and cv2 is not None and not IS_CLOUD:

    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Camera"):
            st.session_state.cam_on = True

    with col2:
        if st.button("⏹ Stop Camera"):
            st.session_state.cam_on = False

    if st.session_state.cam_on:
        cap = cv2.VideoCapture(0)

        while st.session_state.cam_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            objects = detect_objects(frame)
            alert = generate_alert(objects)

            if alert and alert != st.session_state["last_spoken"]:
                play_audio(alert)
                st.session_state["last_spoken"] = alert

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()

    else:
        st.info("Click ▶ Start Camera to begin navigation")

# ---------------- VIDEO UPLOAD MODE ----------------
else:
    video_file = st.file_uploader("Upload a walking video", type=["mp4", "avi", "mov"])

    if video_file is not None and cv2 is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            objects = detect_objects(frame)
            alert = generate_alert(objects)

            if alert and alert != st.session_state["last_spoken"]:
                play_audio(alert)
                st.session_state["last_spoken"] = alert

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()

    else:
        st.info("Upload a video for navigation analysis")
