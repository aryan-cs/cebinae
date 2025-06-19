import cv2
import time
import psutil
from collections import deque

mirror_frame = True
show_diagnostics = {
    "FPS": True,
    "CPU": True
}

video_capture = cv2.VideoCapture(0)

print("Showing camera feed. Press 'q' to quit or 'f' to flip camera.")

fps_buffer = deque(maxlen=60)

last_update_time = 0
update_interval = 1 # second(s)
cached_diagnostics = []

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        mirror_frame = not mirror_frame

    if mirror_frame:
        frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

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
    font_color = (255, 255, 255)
    line_type = 2
    line_height = 25

    for i, text in enumerate(cached_diagnostics):
        y = height - 15 - (i * line_height)
        cv2.putText(frame, text, (10, y), font, font_scale, font_color, line_type)

    cv2.imshow('Cebinae', frame)

video_capture.release()
cv2.destroyAllWindows()
print("Camera feed stopped.")
