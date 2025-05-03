import os
import cv2

'''
The logic of this class is to extract a certain proportion of frames from the video and cache them as images.
'''

class FrameExtractor:
    def __init__(self, frame_interval=10):
        self.frame_interval = frame_interval

    def extract(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return

        os.makedirs(output_dir, exist_ok=True)

        frame_count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.frame_interval == 0:
                filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f"Frame saved: {saved_count}")
