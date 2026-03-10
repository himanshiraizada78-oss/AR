import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import time

st.set_page_config(page_title="AI Navigation Assistant", layout="wide")

st.title("AI Augmented Reality Navigation for Visually Impaired")

st.write("This system detects objects, estimates distance and gives voice guidance.")

# Load AI model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# Voice function
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# Distance estimation
def estimate_distance(box_height):

    if box_height > 400:
        return "Very Close"

    elif box_height > 250:
        return "Close"

    elif box_height > 120:
        return "Medium Distance"

    else:
        return "Far"

# Direction estimation
def estimate_direction(x_center, frame_width):

    if x_center < frame_width/3:
        return "Left"

    elif x_center > 2*frame_width/3:
        return "Right"

    else:
        return "Center"


# UI Buttons
start = st.button("Start Camera")
stop = st.button("Stop Camera")

FRAME = st.image([])

camera = None

if start:
    camera = cv2.VideoCapture(0)

detected_cache = {}

while start and camera is not None:

    ret, frame = camera.read()

    if not ret:
        st.error("Camera not detected")
        break

    frame_height, frame_width, _ = frame.shape

    results = model(frame)

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            box_height = y2 - y1

            distance = estimate_distance(box_height)

            x_center = (x1 + x2) / 2

            direction = estimate_direction(x_center, frame_width)

            text = f"{label} | {distance} | {direction}"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,text,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            key = label + direction

            # Speak only if new object
            if key not in detected_cache:

                message = f"{label} detected {direction} side {distance}"
                speak(message)

                detected_cache[key] = time.time()

    FRAME.image(frame, channels="BGR")

    if stop:
        break

if camera is not None:
    camera.release()
