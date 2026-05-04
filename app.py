import streamlit as st
import cv2
import torch
import threading
import time
from playsound import playsound

# Load YOLO model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

model = load_model()

st.title("🚗 Driver Drowsiness Detection System")

# Session state
if "running" not in st.session_state:
    st.session_state.running = False

if "alarm_playing" not in st.session_state:
    st.session_state.alarm_playing = False

if "drowsy_start" not in st.session_state:
    st.session_state.drowsy_start = None

# UI Buttons
col1, col2 = st.columns(2)

if col1.button("▶ Start"):
    st.session_state.running = True

if col2.button("⏹ Stop"):
    st.session_state.running = False

status = st.empty()
frame_window = st.image([])

def play_alarm():
    st.session_state.alarm_playing = True
    playsound("alarm.mpeg")
    st.session_state.alarm_playing = False

# Webcam
cap = cv2.VideoCapture(0)

while True:
    if st.session_state.running:
        status.success("Status: Running")

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        # YOLO detection
        results = model(frame)
        labels = results.names
        detections = results.xyxy[0]

        drowsy_detected = False

        for *box, conf, cls in detections:
            label = labels[int(cls)]
            if label.lower() == "drowsy":
                drowsy_detected = True

        # Alarm logic
        if drowsy_detected:
            if st.session_state.drowsy_start is None:
                st.session_state.drowsy_start = time.time()
            elif time.time() - st.session_state.drowsy_start > 2:
                if not st.session_state.alarm_playing:
                    threading.Thread(target=play_alarm).start()
        else:
            st.session_state.drowsy_start = None

        # Render frame
        frame = results.render()[0]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_window.image(frame)

    else:
        status.warning("Status: Stopped")
        time.sleep(0.5)