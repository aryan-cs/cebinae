import cv2
import time
import psutil
from collections import deque
import os
import mediapipe as mp
from datetime import datetime

'''
    Keymap
    - 'q': Quit the application
    - 'f': Flip the camera feed horizontally
    - 'd': Toggle diagnostics display
    - 'space': Capture n photos of detected faces
'''

MIRROR_FRAME = True
MIRROR_CAPTURE = True
SHOW_DIAGNOSTICS = True
display_diagnostic = {
    "FPS": True,
    "CPU": True
}

CAPTURES_DIR = "captures"
CAPTURE_PADDING = 0.5
NUM_CAPTURES = 5
CAPTURE_INTERVAL_SECONDS = 0.2

is_capturing = False
captures_left = 0
last_capture_time = 0

palette = { # Reversed for OpenCV
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'RED': (100, 10, 255),
    'GREEN': (10, 255, 137),
    'BLUE': (255, 210, 10),
    'YELLOW': (10, 202, 255),
}

if not os.path.exists(CAPTURES_DIR):
    os.makedirs(CAPTURES_DIR)

mp_face_mesh = mp.solutions.face_mesh
face_mesh_instance = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=5, 
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(color=palette['WHITE'], thickness=1)

video_capture = cv2.VideoCapture(0)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_dir = os.path.join(CAPTURES_DIR, timestamp)
os.makedirs(session_dir)
face_capture_count = 0

# print(f"Capture session started. Saving to: {session_dir}")
# print("Press 'q' to quit, 'f' to flip, 'm' for mesh/rect, 'd' for diagnostics, SPACE to capture.")

fps_buffer = deque(maxlen=60)

last_update_time = 0
update_interval = 1 # second(s)
cached_diagnostics = []

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame.")
        break

    clean_frame = frame.copy()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        MIRROR_FRAME = not MIRROR_FRAME
    elif key == ord('d'):
        SHOW_DIAGNOSTICS = not SHOW_DIAGNOSTICS
    elif key == 32:
        if not is_capturing:
            is_capturing = True
            captures_left = NUM_CAPTURES
            last_capture_time = 0
            print(f"Starting capture of {NUM_CAPTURES} photos...")

    if is_capturing:
        current_time = time.time()
        if captures_left > 0 and (current_time - last_capture_time >= CAPTURE_INTERVAL_SECONDS):
            clean_frame_for_capture = clean_frame.copy()
            
            results_capture = face_mesh_instance.process(cv2.cvtColor(clean_frame_for_capture, cv2.COLOR_BGR2RGB))
            if MIRROR_CAPTURE:
                clean_frame_for_capture = cv2.flip(clean_frame_for_capture, 1)
            if results_capture.multi_face_landmarks:
                for face_landmarks in results_capture.multi_face_landmarks:
                    face_capture_count += 1
                    
                    h, w, _ = clean_frame_for_capture.shape
                    x_coords = [landmark.x for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y for landmark in face_landmarks.landmark]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    box_x, box_y = int(x_min * w), int(y_min * h)
                    box_w, box_h = int((x_max - x_min) * w), int((y_max - y_min) * h)

                    pad_w = int(box_w * CAPTURE_PADDING)
                    pad_h = int(box_h * CAPTURE_PADDING)
                    
                    y1 = max(0, box_y - pad_h)
                    y2 = min(h, box_y + box_h + pad_h)
                    x1 = max(0, box_x - pad_w)
                    x2 = min(w, box_x + box_w + pad_w)

                    face_img = clean_frame_for_capture[y1:y2, x1:x2]
                    filename = os.path.join(session_dir, f"face_{face_capture_count}.png")
                    cv2.imwrite(filename, face_img)
                    print(f"Saved photo to {filename} ({NUM_CAPTURES - captures_left + 1}/{NUM_CAPTURES})")
            else:
                print(f"Capture {NUM_CAPTURES - captures_left + 1}/{NUM_CAPTURES}: No face detected.")

            captures_left -= 1
            last_capture_time = current_time

            if captures_left == 0:
                is_capturing = False
                print("Capture sequence finished.")

    if MIRROR_FRAME:
        frame = cv2.flip(frame, 1)

    results = face_mesh_instance.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mesh_drawing_spec
            )

    height, width, _ = frame.shape

    if SHOW_DIAGNOSTICS:
        current_time = time.time()
        fps_buffer.append(current_time)

        if current_time - last_update_time >= update_interval:
            last_update_time = current_time
            diagnostics_to_show = []

            if display_diagnostic["FPS"]:
                fps = 0
                if len(fps_buffer) > 1:
                    time_diff = fps_buffer[-1] - fps_buffer[0]
                    if time_diff > 0:
                        fps = (len(fps_buffer) - 1) / time_diff
                diagnostics_to_show.append(f"FPS: {fps:.0f}")

            if display_diagnostic["CPU"]:
                cpu_usage = psutil.cpu_percent()
                diagnostics_to_show.append(f"CPU: {cpu_usage:.0f}%")
            
            cached_diagnostics = diagnostics_to_show

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        line_type = 2
        line_height = 25

        for i, text in enumerate(cached_diagnostics):
            y = height - 15 - (i * line_height)
            cv2.putText(frame, text, (10, y), font, font_scale, palette['WHITE'], line_type)

    cv2.imshow('Cebinae', frame)

video_capture.release()
cv2.destroyAllWindows()
# print("Camera feed stopped.")
