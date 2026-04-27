import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time

# Title
st.set_page_config(page_title="Point & Detection", layout="wide")
st.title("Point & Detection")
st.markdown("**Chỉ tay vào object → phát hiện**")

# Load models (chạy 1 lần)
@st.cache_resource
def load_models():
    st.info("Loading models... (1st time only)")
    
    # MediaPipe Hands
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    st.info("MediaPipe loaded")
    
    # YOLO model
    st.info("Loading YOLO model (may take 30-60 seconds)...")
    yolo = YOLO("yolov8n.pt")
    st.success("All models loaded!")
    
    return mp_hands, yolo

mp_hands, yolo_model = load_models()

# Sidebar controls
st.sidebar.title("Controls")
run_demo = st.sidebar.checkbox("Start Demo", value=True)

# Main camera frame
frame_placeholder = st.empty()

# Camera setup
try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("Camera not found! Please check your camera connection.")
        st.stop()
    
    frame_placeholder.info("Starting camera... (wait 2-3 seconds)")
    
    frame_count = 0
    last_results = None  # ← Cache YOLO results
    
    while run_demo:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot read from camera!")
            break
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pointing_obj = None
            
            # 1. HAND DETECTION (every frame)
            hands = mp_hands.process(frame_rgb)
            
            if hands.multi_hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hands.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
                
                # Index finger TIP (landmark 8)
                tip = hands.multi_hand_landmarks[0].landmark[8]
                tip_x, tip_y = int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])
                cv2.circle(frame, (tip_x, tip_y), 12, (0, 255, 0), -1)
                
                # Wrist (landmark 0)
                wrist = hands.multi_hand_landmarks[0].landmark[0]
                wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
                
                # POINTING RAY (Blue line from wrist to tip)
                cv2.line(frame, (wrist_x, wrist_y), (tip_x, tip_y), (255, 0, 0), 4)
                
                # Extend ray (Red line)
                dx = tip_x - wrist_x
                dy = tip_y - wrist_y
                ray_end_x = tip_x + int(dx * 2.5)
                ray_end_y = tip_y + int(dy * 2.5)
                cv2.line(frame, (tip_x, tip_y), (ray_end_x, ray_end_y), (0, 0, 255), 3)
                
                # 2. YOLO OBJECT DETECTION (every 5 frames = ~6fps)
                if frame_count % 5 == 0:
                    results = yolo_model(frame_rgb, verbose=False)
                    last_results = results
                
                # Draw ALL detected objects
                if last_results:
                    for r in last_results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                conf = box.conf[0].cpu().numpy()
                                cls_id = int(box.cls[0])
                                
                                if conf > 0.5:
                                    obj_name = r.names[cls_id]
                                    obj_center_x = (x1 + x2) // 2
                                    obj_center_y = (y1 + y2) // 2
                                    
                                    # Distance from ray to object (increased tolerance for 3D→2D perspective)
                                    dist_x = abs(ray_end_x - obj_center_x)
                                    dist_y = abs(ray_end_y - obj_center_y)
                                    is_pointing = (dist_x < 200 and dist_y < 200)
                                    
                                    # Draw box ONLY if pointing
                                    if is_pointing:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
                                        text = f"{obj_name} {conf:.1f}" + "✓"
                                        cv2.putText(frame, text, 
                                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   0.6, (255, 0, 255), 2)
                                        pointing_obj = obj_name
            
            # DISPLAY camera only
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            frame_count += 1
            time.sleep(0.03)  # 30 FPS
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            break
    
    cap.release()
    
except Exception as e:
    st.error(f"Setup error: {str(e)}")
