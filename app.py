import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import os
from PIL import Image

st.title("AR Navigation Assistant for Visually Impaired")

st.write("This system detects nearby objects and gives voice feedback.")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # lightweight model
    return model

model = load_model()

# Function for speech
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# Start camera
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

detected_objects = set()

while run:
    
    ret, frame = camera.read()

    if not ret:
        st.write("Camera not working")
        break

    results = model(frame)

    labels = []

    for r in results:
        boxes = r.boxes
        for box in boxes:

            cls = int(box.cls[0])
            label = model.names[cls]
            labels.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    FRAME_WINDOW.image(frame, channels="BGR")

    # Speak newly detected objects
    for obj in labels:
        if obj not in detected_objects:
            speak(obj + " detected")
            detected_objects.add(obj)

camera.release()
