import cv2
import numpy as np

# Input video files
video_paths = [
    "C:/Users/MYSEL/Desktop/Fwalk/front_fwalk.mp4",
    "C:/Users/MYSEL/Desktop/Fwalk/left_fwalk.mp4",
    "C:/Users/MYSEL/Desktop/Fwalk/right_fwalk.mp4"
]

# Open first video to get properties
cap = cv2.VideoCapture(video_paths[0])
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()

# Define output video properties
output_path = "C:/Users/MYSEL/Desktop/Fwalk/fwalk.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process each video one after another
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)

    # Verify FPS for consistency
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if video_fps == 0:
        video_fps = fps  # Set a default FPS if OpenCV fails to detect

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video, move to the next one

        # Resize frame to match dimensions (if needed)
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Write frame to output video
        out.write(frame)

    cap.release()

# Ensure all frames are properly flushed
out.release()

print(f"✅ Video successfully saved as {output_path}")

