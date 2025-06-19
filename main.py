import cv2
import time
import psutil
from collections import deque
import os
import mediapipe as mp

'''
Keymap
- 'q': Quit the application
- 'f': Flip the camera feed horizontally
- 'm': Toggle between face mesh and rectangle detection
- 'd': Toggle diagnostics display
'''

mirror_frame = True
master_diagnostics_enabled = True
show_diagnostics = {
    "FPS": True,
    "CPU": True
}
palette = { # Reversed for OpenCV
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'RED': (100, 10, 255),
    'GREEN': (10, 255, 137),
    'BLUE': (255, 210, 10),
    'YELLOW': (10, 202, 255),
}

face_mesh_enabled = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh_instance = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=5, 
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(color=palette['WHITE'], thickness=1)

face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

video_capture = cv2.VideoCapture(0)

# print("Showing camera feed. Press 'q' to quit, 'f' to flip, 'm' for mesh/rect, 'd' for diagnostics.")

fps_buffer = deque(maxlen=60)

last_update_time = 0
update_interval = 1 # second(s)
cached_diagnostics = []

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame.")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        mirror_frame = not mirror_frame
    elif key == ord('m'):
        face_mesh_enabled = not face_mesh_enabled
    elif key == ord('d'):
        master_diagnostics_enabled = not master_diagnostics_enabled

    if mirror_frame:
        frame = cv2.flip(frame, 1)

    if face_mesh_enabled:
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
                # mp_drawing.draw_landmarks(
                #     image=frame,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mesh_drawing_spec
                # )
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), palette['WHITE'], 2)

    height, width, _ = frame.shape

    if master_diagnostics_enabled:
        current_time = time.time()
        fps_buffer.append(current_time)

        if current_time - last_update_time >= update_interval:
            last_update_time = current_time
            diagnostics_to_show = []

            if show_diagnostics["FPS"]:
                fps = 0
                if len(fps_buffer) > 1:
                    time_diff = fps_buffer[-1] - fps_buffer[0]
                    if time_diff > 0:
                        fps = (len(fps_buffer) - 1) / time_diff
                diagnostics_to_show.append(f"FPS: {fps:.0f}")

            if show_diagnostics["CPU"]:
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
