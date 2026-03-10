import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import av
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import tempfile

st.set_page_config(page_title="AI Navigation Assistant")

st.title("Augmented Reality Navigation for Visually Impaired")

st.write("Detects objects and gives navigation guidance.")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()


def estimate_distance(box_height):

    if box_height > 350:
        return "Very Close"

    elif box_height > 200:
        return "Close"

    elif box_height > 120:
        return "Medium"

    else:
        return "Far"


def estimate_direction(x_center, width):

    if x_center < width/3:
        return "Left"

    elif x_center > 2*width/3:
        return "Right"

    else:
        return "Center"


def speak(text):

    tts = gTTS(text)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    tts.save(tmp.name)

    st.audio(tmp.name)


class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.detected = set()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

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

                cv2.putText(img,text,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)

                key = label + direction

                if key not in self.detected:

                    message = f"{label} detected {direction} side {distance}"

                    speak(message)

                    self.detected.add(key)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="ai-navigation",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
