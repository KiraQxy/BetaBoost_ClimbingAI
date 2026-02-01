"""Utility functions for video processing, visualization, and mapping."""

import os
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import streamlit as st


def create_skeleton_video(frames_data, output_path="skeleton_video.mp4"):
    """Create a video with skeleton overlay from frame data."""
    if not frames_data:
        return None

    first_frame = frames_data[0].image
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = 30
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

    for frame_data in frames_data:
        frame_with_skeleton = visualize_frame(frame_data)
        frame_bgr = cv2.cvtColor(frame_with_skeleton.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return output_path


def get_video_html(video_path):
    """Generate HTML for video player with custom controls and autoplay."""
    if not os.path.exists(video_path):
        return f"<p>Error: Video file not found at {video_path}</p>"

    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    video_base64 = base64.b64encode(video_bytes).decode()

    html = f"""
    <video id="climbing-video" width="100%" height="auto" controls autoplay loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <script>
        document.getElementById('climbing-video').style.maxHeight = '65vh';
    </script>
    """
    return html


def visualize_frame(frame_data, show_skeleton=True, show_com=True, trail_frames=10):
    """Create a visualization of a frame with skeleton and center of mass with trail effect."""
    image = frame_data.image.copy()
    h, w, _ = image.shape

    if show_skeleton and frame_data.landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        connection_style = mp_drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=2,
            circle_radius=1
        )
        left_point_style = mp_drawing.DrawingSpec(
            color=(255, 100, 100),
            thickness=2,
            circle_radius=4
        )
        right_point_style = mp_drawing.DrawingSpec(
            color=(100, 255, 100),
            thickness=2,
            circle_radius=4
        )

        landmark_style = {}
        for idx in range(33):
            if idx in [11, 13, 15, 23, 25, 27] or idx in [1, 3, 7]:
                landmark_style[idx] = left_point_style
            elif idx in [12, 14, 16, 24, 26, 28] or idx in [2, 4, 8]:
                landmark_style[idx] = right_point_style
            else:
                landmark_style[idx] = connection_style

        mp_drawing.draw_landmarks(
            image,
            frame_data.landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

    if show_com and hasattr(frame_data, 'frame_idx'):
        keypoints = frame_data.keypoints
        valid_points = [kp for kp in keypoints.values() if kp["visibility"] > 0.5]

        if valid_points:
            com_x = sum(kp["x"] for kp in valid_points) / len(valid_points)
            com_y = sum(kp["y"] for kp in valid_points) / len(valid_points)
            current_com = (int(com_x * w), int(com_y * h))

            if 'com_trail' not in st.session_state:
                st.session_state.com_trail = []
            st.session_state.com_trail.append(current_com)
            if len(st.session_state.com_trail) > trail_frames:
                st.session_state.com_trail.pop(0)

            for i, pos in enumerate(st.session_state.com_trail[:-1]):
                alpha = 0.2 + (i / len(st.session_state.com_trail)) * 0.6
                cv2.circle(image, pos, 3, (0, 0, 255, int(alpha * 255)), -1)
            cv2.circle(image, current_com, 6, (0, 0, 255), -1)

    return image


def generate_com_trajectory_plot(frames_data):
    """Generate center of mass trajectory plot."""
    if not frames_data:
        return None

    com_positions = []
    timestamps = []

    for frame in frames_data:
        keypoints = frame.keypoints
        valid_points = [kp for kp in keypoints.values() if kp["visibility"] > 0.5]
        if valid_points:
            com_x = sum(kp["x"] for kp in valid_points) / len(valid_points)
            com_y = sum(kp["y"] for kp in valid_points) / len(valid_points)
            com_positions.append((com_x, com_y))
            timestamps.append(frame.timestamp)

    if not com_positions:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    x_coords = [p[0] for p in com_positions]

    if timestamps:
        time_points = np.array(timestamps)
        time_normalized = (
            (time_points - min(time_points)) / (max(time_points) - min(time_points))
            if max(time_points) > min(time_points) else time_points
        )
    else:
        time_normalized = np.linspace(0, 1, len(x_coords))

    ax.plot(time_normalized, x_coords, 'g-', linewidth=3)
    ax.scatter(time_normalized[0], x_coords[0], color='blue', s=120, label='Start')
    ax.scatter(time_normalized[-1], x_coords[-1], color='red', s=120, label='End')

    quartiles = [int(len(time_normalized) * q) for q in [0.25, 0.5, 0.75]]
    for i, q in enumerate(quartiles):
        if q < len(time_normalized):
            ax.scatter(time_normalized[q], x_coords[q], color='purple', s=80,
                      label=f"{(i+1)*25}%" if i == 0 else None)

    ax.set_xlabel('Climb Progress', fontsize=12)
    ax.set_ylabel('Horizontal Position', fontsize=12)
    ax.set_title('Horizontal Center of Mass Trajectory', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    fig.tight_layout()

    return fig


def map_route_type(ui_route_type):
    """Map UI route type to API route type."""
    mapping = {
        "Vertical": "vertical",
        "Slab": "slab",
        "Overhang": "overhang",
        "Roof": "Roof"
    }
    return mapping.get(ui_route_type, "vertical")


def map_climber_level(ui_climber_level):
    """Map UI climber level to API climber level."""
    mapping = {
        "Beginner": "beginner",
        "Intermediate": "intermediate",
        "Advanced": "advanced"
    }
    return mapping.get(ui_climber_level, "intermediate")
