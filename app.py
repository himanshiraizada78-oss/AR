import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import tempfile

st.set_page_config(page_title="AI Navigation Assistant")

st.title("AI Navigation Assistant for Visually Impaired")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Voice function
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# ---------------------------------
# Webcam Detection
# ---------------------------------

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img)

        annotated = results[0].plot()

        for box in results[0].boxes:

            cls = int(box.cls[0])

            label = model.names[cls]

            speak(f"{label} detected")

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

st.header("Live Webcam Detection")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="webcam",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION
)

# ---------------------------------
# Video Upload Detection
# ---------------------------------

st.header("Upload Video")

video_file = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

if video_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)

    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        annotated = results[0].plot()

        stframe.image(annotated, channels="BGR")

    cap.release()
