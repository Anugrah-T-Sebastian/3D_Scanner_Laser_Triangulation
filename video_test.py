import cv2
import os

def extract_frames(video_name, extension = ".avi"):
    # Open the video file
    cap = cv2.VideoCapture('./Video_sample/Old_Samples/' + video_name + extension)
    # cap = cv2.VideoCapture('./reference.avi')

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frames = []
    frame_count = 0

    output_folder = f'./video_frames/{video_name}/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame, cv2.COLOR)
        # frame = cv2.cvtColor

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frames.append(frame)
        frame_count += 1
    print(f"Extracted {frame_count} frames from the video.")
    return frames

# extract_frames('reference_03','.mp4')
# extract_frames('grey_cube_30mm')