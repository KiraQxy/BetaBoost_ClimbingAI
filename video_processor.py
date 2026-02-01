"""Video processing module - extracts pose/skeleton data from climbing videos."""

import tempfile
import cv2
import mediapipe as mp
import streamlit as st

from models import FrameData


class VideoProcessor:
    """Processes video files and extracts pose landmarks using MediaPipe."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )

    def process_video(self, video_file, sample_rate=3):
        """Process video and extract skeleton data."""
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_data = []
        frame_idx = 0
        progress_bar = st.progress(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            progress_bar.progress(min(1.0, frame_idx / total_frames))

            if frame_idx % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    keypoints = self._extract_keypoints(
                        results.pose_landmarks, frame_width, frame_height
                    )
                    frames_data.append(FrameData(
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        image=frame_rgb.copy(),
                        landmarks=results.pose_landmarks,
                        keypoints=keypoints
                    ))

            frame_idx += 1

        cap.release()
        progress_bar.empty()

        return frames_data

    def _extract_keypoints(self, landmarks, frame_width, frame_height):
        """Extract keypoints dictionary from MediaPipe results."""
        keypoints = {}
        for idx, landmark in enumerate(landmarks.landmark):
            keypoint = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
                "x_px": int(landmark.x * frame_width),
                "y_px": int(landmark.y * frame_height),
                "name": self._get_keypoint_name(idx)
            }
            keypoints[idx] = keypoint

        return keypoints

    def _get_keypoint_name(self, idx):
        """Get the name of a MediaPipe keypoint."""
        keypoint_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]

        if 0 <= idx < len(keypoint_names):
            return keypoint_names[idx]
        return f"unknown_{idx}"
