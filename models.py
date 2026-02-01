"""Data models for BetaBoost climbing analysis."""


class FrameData:
    """Container for processed frame data from video."""

    def __init__(self, frame_idx, timestamp, image, landmarks, keypoints):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.image = image
        self.landmarks = landmarks
        self.keypoints = keypoints
