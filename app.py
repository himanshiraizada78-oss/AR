import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import tempfile

st.set_page_config(page_title="AI Navigation Assistant", layout="wide")

st.title("AI Navigation Assistant for Visually Impaired")

st.write("Upload an image and the system will detect objects and give navigation guidance.")

# -----------------------------
# Load YOLO Model
# -----------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# Voice Function
# -----------------------------

def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# -----------------------------
# Distance Estimation
# -----------------------------

def estimate_distance(box_height):

    if box_height > 350:
        return "Very Close"

    elif box_height > 200:
        return "Close"

    elif box_height > 120:
        return "Medium"

    else:
        return "Far"

# -----------------------------
# Direction Estimation
# -----------------------------

def estimate_direction(x_center, width):

    if x_center < width/3:
        return "Left"

    elif x_center > 2*width/3:
        return "Right"

    else:
        return "Center"

# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)

    img = np.array(image)

    height, width, _ = img.shape

    results = model(img)

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])

            label = model.names[cls]

            box_height = y2 - y1

            distance = estimate_distance(box_height)

            x_center = (x1 + x2) / 2

            direction = estimate_direction(x_center, width)

            text = f"{label} | {distance} | {direction}"

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                img,
                text,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            message = f"{label} detected {direction} side {distance}"

            speak(message)

    st.image(img, channels="BGR")

# -----------------------------
# Emergency Button
# -----------------------------

if st.button("🚨 Emergency Help"):

    st.error("Emergency alert triggered!")

    speak("Emergency alert activated")
    
