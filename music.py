import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

# --------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------
st.set_page_config(page_title="Emotion-Based Music Recommender")
st.header("🎵 Emotion Based Music Recommender")

# --------------------------------------------------------
# Load Model and Labels
# --------------------------------------------------------
try:
    model = load_model("model.h5")
    label = np.load("labels.npy", allow_pickle=True)
    st.success("✅ Model and labels loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or labels: {e}")
    st.stop()

# --------------------------------------------------------
# Mediapipe Setup
# --------------------------------------------------------
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
drawing = mp.solutions.drawing_utils

# --------------------------------------------------------
# Session Initialization
# --------------------------------------------------------
if "run" not in st.session_state:
    st.session_state["run"] = True

emotion_path = "emotion.npy"

if os.path.exists(emotion_path):
    try:
        emotion = np.load(emotion_path, allow_pickle=True)[0]
    except:
        emotion = ""
else:
    emotion = ""

st.session_state["run"] = not bool(emotion)


# --------------------------------------------------------
# Emotion Processor Class
# --------------------------------------------------------
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            # Extract facial landmark coordinates relative to landmark[1]
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Left hand landmarks
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            # Right hand landmarks
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            print(f"[INFO] Feature vector shape: {lst.shape}")

            # Check input size matches model
            expected_shape = model.input_shape[1]
            if lst.shape[1] == expected_shape:
                pred = label[np.argmax(model.predict(lst))]
                print(f"[PRED] Emotion Detected: {pred}")

                cv2.putText(frm, str(pred), (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                np.save(emotion_path, np.array([pred]))
            else:
                print(f"[WARN] Shape mismatch! Model expects {expected_shape}, got {lst.shape[1]}")

        else:
            print("[INFO] No face detected.")

        # Draw landmarks
        drawing.draw_landmarks(
            frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255),
                                                      thickness=-1, circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(thickness=1)
        )
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# --------------------------------------------------------
# User Input Section
# --------------------------------------------------------
lang = st.text_input("🎧 Enter Language (Chose any Language):")
singer = st.text_input("🎤 Enter Singer (optional):")

# --------------------------------------------------------
# Webcam Stream
# --------------------------------------------------------
if lang and st.session_state["run"]:
    webrtc_streamer(
        key="emotion-detector",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# --------------------------------------------------------
# Recommend Songs Button
# --------------------------------------------------------
btn = st.button("🎶 Recommend Songs")

if btn:
    if not os.path.exists(emotion_path):
        st.warning("⚠️ Please let me capture your emotion first.")
        st.session_state["run"] = True
    else:
        try:
            emotion = np.load(emotion_path, allow_pickle=True)[0]
        except:
            emotion = ""

        if not emotion:
            st.warning("⚠️ Emotion not detected. Try again.")
            st.session_state["run"] = True
        else:
            query = f"{lang}+{emotion}+song+{singer}".replace(" ", "+")
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            np.save(emotion_path, np.array([""]))
            st.session_state["run"] = False
            st.success(f"🎵 Showing {emotion} songs in {lang} on YouTube!")
