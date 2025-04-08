import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import tempfile
import os
from PIL import Image
from io import BytesIO
import anthropic
from typing import List, Dict, Any
import time
from pyngrok import ngrok

# 1. Define Data Class
class FrameData:
    def __init__(self, frame_idx, timestamp, image, landmarks, keypoints):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.image = image
        self.landmarks = landmarks
        self.keypoints = keypoints

# 2. Video Processing Module
class VideoProcessor:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )

    def process_video(self, video_file, sample_rate=3):
        """Process video and extract skeleton data"""
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        # Open the video
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Store processed frames
        frames_data = []

        # Process frames
        frame_idx = 0

        # Create a progress bar
        progress_bar = st.progress(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Update progress
            progress_bar.progress(min(1.0, frame_idx / total_frames))

            # Process every sample_rate frames
            if frame_idx % sample_rate == 0:
                # Convert to RGB for MediaPipe processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Set to non-writable for performance boost
                frame_rgb.flags.writeable = False
                results = self.pose.process(frame_rgb)

                # If pose landmarks are detected
                if results.pose_landmarks:
                    # Extract keypoints
                    keypoints = self._extract_keypoints(results.pose_landmarks, frame_width, frame_height)

                    # Store frame data
                    frames_data.append(FrameData(
                        frame_idx=frame_idx,
                        timestamp=frame_idx/fps,
                        image=frame_rgb.copy(),
                        landmarks=results.pose_landmarks,
                        keypoints=keypoints
                    ))

            frame_idx += 1

        cap.release()
        progress_bar.empty()

        return frames_data

    def _extract_keypoints(self, landmarks, frame_width, frame_height):
        """Extract keypoints dictionary from MediaPipe results"""
        keypoints = {}
        for idx, landmark in enumerate(landmarks.landmark):
            # Get keypoint coordinates (relative and pixel values)
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
        """Get the name of a MediaPipe keypoint"""
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

# 3. Feature Extraction Module
class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, frames_data):
        """Extract features from frame data"""
        # Check if there are enough frames
        if len(frames_data) < 5:
            st.error("Not enough frames to extract features. Need at least 5 frames.")
            return None

        # From all frames, select 5 frames randomly but ensuring good distribution
        if len(frames_data) >= 20:
            # If many frames, randomly select 5 frames
            selected_frames = sorted(random.sample(frames_data, 5),
                                     key=lambda x: x.frame_idx)
        else:
            # If fewer frames, ensure selection includes beginning, middle, and end
            segments = np.array_split(range(len(frames_data)), 5)
            selected_indices = [random.choice(segment.tolist()) for segment in segments]
            selected_frames = [frames_data[i] for i in sorted(selected_indices)]

        # Extract features for each frame
        frame_features = []
        for i, frame in enumerate(selected_frames):
            # Extract keypoints dictionary
            keypoints_dict = frame.keypoints

            # Calculate body features
            features = self._calculate_body_features(keypoints_dict)
            features['frame_index'] = frame.frame_idx
            features['frame_relative_position'] = frame.frame_idx / max(f.frame_idx for f in frames_data)

            frame_features.append(features)

        # Combine all frame features
        combined_features = {}
        for i, frame_feat in enumerate(frame_features):
            for feat_name, feat_value in frame_feat.items():
                combined_features[f"frame{i+1}_{feat_name}"] = feat_value

        # Add inter-frame features
        self._add_inter_frame_features(combined_features, frame_features)

        # Add dynamic features (as done in feature_extraction_xgboost.py)
        self._add_dynamic_features(combined_features)

        return combined_features

    def _calculate_body_features(self, keypoints_dict):
        """Calculate body features including joint angles, posture, balance, etc."""
        features = {}

        # Convert keypoints to the format needed for calculations
        body_points = {}
        for joint_idx, joint_data in keypoints_dict.items():
            if joint_data["visibility"] > 0.5:  # Only use points with good visibility
                body_points[int(joint_idx)] = {
                    'x': joint_data['x'],
                    'y': joint_data['y'],
                    'z': joint_data['z']
                }

        # Calculate joint angles
        # Elbow angles (shoulder-elbow-wrist)
        if all(k in body_points for k in [11, 13, 15]):  # Left shoulder, elbow, wrist
            features['left_elbow_angle'] = self._calculate_angle(
                body_points[11], body_points[13], body_points[15]
            )

        if all(k in body_points for k in [12, 14, 16]):  # Right shoulder, elbow, wrist
            features['right_elbow_angle'] = self._calculate_angle(
                body_points[12], body_points[14], body_points[16]
            )

        # Shoulder angles (elbow-shoulder-hip)
        if all(k in body_points for k in [13, 11, 23]):  # Left elbow, shoulder, hip
            features['left_shoulder_angle'] = self._calculate_angle(
                body_points[13], body_points[11], body_points[23]
            )

        if all(k in body_points for k in [14, 12, 24]):  # Right elbow, shoulder, hip
            features['right_shoulder_angle'] = self._calculate_angle(
                body_points[14], body_points[12], body_points[24]
            )

        # Hip angles (shoulder-hip-knee)
        if all(k in body_points for k in [11, 23, 25]):  # Left shoulder, hip, knee
            features['left_hip_angle'] = self._calculate_angle(
                body_points[11], body_points[23], body_points[25]
            )

        if all(k in body_points for k in [12, 24, 26]):  # Right shoulder, hip, knee
            features['right_hip_angle'] = self._calculate_angle(
                body_points[12], body_points[24], body_points[26]
            )

        # Knee angles (hip-knee-ankle)
        if all(k in body_points for k in [23, 25, 27]):  # Left hip, knee, ankle
            features['left_knee_angle'] = self._calculate_angle(
                body_points[23], body_points[25], body_points[27]
            )

        if all(k in body_points for k in [24, 26, 28]):  # Right hip, knee, ankle
            features['right_knee_angle'] = self._calculate_angle(
                body_points[24], body_points[26], body_points[28]
            )

        # Calculate trunk posture
        if all(k in body_points for k in [11, 12, 23, 24]):  # Shoulders and hips
            # Calculate trunk center points
            trunk_top = {
                'x': (body_points[11]['x'] + body_points[12]['x']) / 2,
                'y': (body_points[11]['y'] + body_points[12]['y']) / 2,
                'z': (body_points[11]['z'] + body_points[12]['z']) / 2
            }

            trunk_bottom = {
                'x': (body_points[23]['x'] + body_points[24]['x']) / 2,
                'y': (body_points[23]['y'] + body_points[24]['y']) / 2,
                'z': (body_points[23]['z'] + body_points[24]['z']) / 2
            }

            # Vertical direction vector (for angle calculation)
            vertical = {'x': 0, 'y': 1, 'z': 0}

            # Calculate trunk tilt angle (with respect to vertical)
            features['trunk_tilt_angle'] = self._calculate_angle(vertical, trunk_bottom, trunk_top)

            # Calculate trunk length
            trunk_length = math.sqrt(
                (trunk_top['x'] - trunk_bottom['x'])**2 +
                (trunk_top['y'] - trunk_bottom['y'])**2 +
                (trunk_top['z'] - trunk_bottom['z'])**2
            )
            features['trunk_length'] = trunk_length

            # Calculate hip rotation (angle between shoulder line and hip line)
            shoulder_vector = {
                'x': body_points[12]['x'] - body_points[11]['x'],
                'y': 0,  # Ignore y direction
                'z': body_points[12]['z'] - body_points[11]['z']
            }

            hip_vector = {
                'x': body_points[24]['x'] - body_points[23]['x'],
                'y': 0,  # Ignore y direction
                'z': body_points[24]['z'] - body_points[23]['z']
            }

            # Create virtual points for angle calculation
            origin = {'x': 0, 'y': 0, 'z': 0}
            shoulder_point = {'x': shoulder_vector['x'], 'y': 0, 'z': shoulder_vector['z']}
            hip_point = {'x': hip_vector['x'], 'y': 0, 'z': hip_vector['z']}

            features['hip_rotation_angle'] = self._calculate_angle(shoulder_point, origin, hip_point)

        # Calculate limb extensions
        # Arm extensions (shoulder to wrist distance)
        if all(k in body_points for k in [11, 15]):  # Left shoulder, wrist
            left_arm_extension = math.sqrt(
                (body_points[15]['x'] - body_points[11]['x'])**2 +
                (body_points[15]['y'] - body_points[11]['y'])**2 +
                (body_points[15]['z'] - body_points[11]['z'])**2
            )
            features['left_arm_extension'] = left_arm_extension

        if all(k in body_points for k in [12, 16]):  # Right shoulder, wrist
            right_arm_extension = math.sqrt(
                (body_points[16]['x'] - body_points[12]['x'])**2 +
                (body_points[16]['y'] - body_points[12]['y'])**2 +
                (body_points[16]['z'] - body_points[12]['z'])**2
            )
            features['right_arm_extension'] = right_arm_extension

        # Leg extensions (hip to ankle distance)
        if all(k in body_points for k in [23, 27]):  # Left hip, ankle
            left_leg_extension = math.sqrt(
                (body_points[27]['x'] - body_points[23]['x'])**2 +
                (body_points[27]['y'] - body_points[23]['y'])**2 +
                (body_points[27]['z'] - body_points[23]['z'])**2
            )
            features['left_leg_extension'] = left_leg_extension

        if all(k in body_points for k in [24, 28]):  # Right hip, ankle
            right_leg_extension = math.sqrt(
                (body_points[28]['x'] - body_points[24]['x'])**2 +
                (body_points[28]['y'] - body_points[24]['y'])**2 +
                (body_points[28]['z'] - body_points[24]['z'])**2
            )
            features['right_leg_extension'] = right_leg_extension

        # Calculate center of mass
        valid_points = [p for _, p in body_points.items()]
        if valid_points:
            com_x = sum(p['x'] for p in valid_points) / len(valid_points)
            com_y = sum(p['y'] for p in valid_points) / len(valid_points)
            com_z = sum(p['z'] for p in valid_points) / len(valid_points)

            features['center_of_mass_x'] = com_x
            features['center_of_mass_y'] = com_y
            features['center_of_mass_z'] = com_z

        # Calculate lateral balance (left-right balance)
        if all(k in body_points for k in [27, 28]) and 'center_of_mass_x' in features:  # Ankles
            support_center_x = (body_points[27]['x'] + body_points[28]['x']) / 2
            features['lateral_balance'] = features['center_of_mass_x'] - support_center_x

        # Calculate body stability (standard deviation of joint angles)
        angles = [v for k, v in features.items() if k.endswith('_angle') and v is not None]
        if angles:
            features['body_stability'] = np.std(angles)

        return features

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (p2 is the vertex)"""
        if not all([p1, p2, p3]):
            return None

        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return None

        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = min(1.0, max(-1.0, cos_angle))  # Ensure within [-1, 1] range

        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def _add_inter_frame_features(self, combined_features, frame_features):
        """Add inter-frame features like joint angle changes and COM movement"""
        # Calculate joint angle changes between frames
        for joint in ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle',
                     'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle',
                     'left_hip_angle', 'right_hip_angle']:
            for i in range(4):
                prev_feat = frame_features[i].get(joint)
                next_feat = frame_features[i+1].get(joint)

                if prev_feat is not None and next_feat is not None:
                    combined_features[f"change_{i+1}_to_{i+2}_{joint}"] = next_feat - prev_feat

        # Calculate center of mass movement
        for coord in ['x', 'y', 'z']:
            com_positions = []
            for i in range(5):
                key = f"center_of_mass_{coord}"
                if key in frame_features[i]:
                    com_positions.append(frame_features[i][key])

            if len(com_positions) == 5:
                # Total movement distance
                total_distance = sum(abs(com_positions[i] - com_positions[i-1]) for i in range(1, 5))
                combined_features[f"total_com_{coord}_movement"] = total_distance

                # Start-to-end distance
                direct_distance = abs(com_positions[4] - com_positions[0])
                combined_features[f"direct_com_{coord}_movement"] = direct_distance

                # Path efficiency (ratio of direct distance to total distance)
                if total_distance > 0:
                    combined_features[f"com_{coord}_efficiency"] = direct_distance / total_distance

        # Calculate trunk angle variation
        trunk_angles = []
        for i in range(5):
            if 'trunk_tilt_angle' in frame_features[i]:
                trunk_angles.append(frame_features[i]['trunk_tilt_angle'])

        if len(trunk_angles) == 5:
            combined_features["trunk_angle_variation"] = np.std(trunk_angles)

        # Calculate lateral balance variation
        lateral_balances = []
        for i in range(5):
            if 'lateral_balance' in frame_features[i]:
                lateral_balances.append(frame_features[i]['lateral_balance'])

        if len(lateral_balances) == 5:
            combined_features["lateral_balance_variation"] = np.std(lateral_balances)

    def _add_dynamic_features(self, combined_features):
        """Add advanced dynamic features similar to those in feature_extraction_xgboost.py"""
        # 1. Calculate distance-aware change rates for important features
        frame_features = ['hip_rotation_angle', 'trunk_tilt_angle', 'body_stability',
                         'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z',
                         'lateral_balance', 'trunk_length']

        for feature in frame_features:
            for i in range(1, 5):
                col1 = f'frame{i}_{feature}'
                col2 = f'frame{i+1}_{feature}'
                idx1 = f'frame{i}_frame_index'
                idx2 = f'frame{i+1}_frame_index'

                if all(col in combined_features for col in [col1, col2, idx1, idx2]):
                    # Calculate change amount
                    combined_features[f'change_{i}_to_{i+1}_{feature}'] = combined_features[col2] - combined_features[col1]

                    # Calculate frame-distance-aware change rate
                    frame_distance = combined_features[idx2] - combined_features[idx1]
                    if frame_distance > 0:
                        combined_features[f'rate_{i}_to_{i+1}_{feature}'] = (
                            (combined_features[col2] - combined_features[col1]) / frame_distance
                        )

        # 2. Calculate distance-aware change rates for limb features
        limb_features = ['left_arm_extension', 'right_arm_extension',
                        'left_leg_extension', 'right_leg_extension',
                        'left_elbow_angle', 'right_elbow_angle',
                        'left_shoulder_angle', 'right_shoulder_angle',
                        'left_knee_angle', 'right_knee_angle',
                        'left_hip_angle', 'right_hip_angle']

        for feature in limb_features:
            for i in range(1, 5):
                col1 = f'frame{i}_{feature}'
                col2 = f'frame{i+1}_{feature}'
                idx1 = f'frame{i}_frame_index'
                idx2 = f'frame{i+1}_frame_index'

                if all(col in combined_features for col in [col1, col2, idx1, idx2]):
                    # Calculate change amount
                    combined_features[f'change_{i}_to_{i+1}_{feature}'] = combined_features[col2] - combined_features[col1]

                    # Calculate frame-distance-aware change rate
                    frame_distance = combined_features[idx2] - combined_features[idx1]
                    if frame_distance > 0:
                        combined_features[f'rate_{i}_to_{i+1}_{feature}'] = (
                            (combined_features[col2] - combined_features[col1]) / frame_distance
                        )

        # 3. Calculate frame position weighted features
        important_features = ['hip_rotation_angle', 'trunk_tilt_angle', 'body_stability',
                             'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z',
                             'lateral_balance']

        for feature in important_features:
            weighted_values = 0
            weights = 0

            for i in range(1, 6):
                col = f'frame{i}_{feature}'
                pos_col = f'frame{i}_frame_relative_position'

                if col in combined_features and pos_col in combined_features:
                    # Use frame relative position as weight
                    weighted_values += combined_features[col] * combined_features[pos_col]
                    weights += combined_features[pos_col]

            # Calculate weighted average
            if weights > 0:
                combined_features[f'{feature}_weighted_avg'] = weighted_values / weights

        # 4. Calculate sequence coverage
        if 'frame1_frame_relative_position' in combined_features and 'frame5_frame_relative_position' in combined_features:
            combined_features['sequence_coverage'] = (
                combined_features['frame5_frame_relative_position'] -
                combined_features['frame1_frame_relative_position']
            )

        # 5. Create video phase indicators
        for i in range(1, 6):
            pos_col = f'frame{i}_frame_relative_position'

            if pos_col in combined_features:
                # Early/mid/late phase indicators
                combined_features[f'frame{i}_early_phase'] = 1 if combined_features[pos_col] <= 0.33 else 0
                combined_features[f'frame{i}_mid_phase'] = 1 if (combined_features[pos_col] > 0.33 and
                                                                combined_features[pos_col] <= 0.67) else 0
                combined_features[f'frame{i}_late_phase'] = 1 if combined_features[pos_col] > 0.67 else 0

        # 6. Calculate trajectory features
        for axis in ['x', 'y', 'z']:
            # Collect positions and frame indices
            positions = []
            indices = []

            for i in range(1, 6):
                pos_col = f'frame{i}_center_of_mass_{axis}'
                idx_col = f'frame{i}_frame_index'

                if pos_col in combined_features and idx_col in combined_features:
                    positions.append(combined_features[pos_col])
                    indices.append(combined_features[idx_col])

            # Calculate trajectory features if enough points
            if len(positions) >= 3 and len(indices) >= 3:
                try:
                    # Calculate trajectory slope using actual frame indices
                    slope, intercept = np.polyfit(indices, positions, 1)
                    combined_features[f'com_{axis}_trajectory_slope'] = slope

                    # Calculate nonlinearity (deviation from linear prediction)
                    linear_pred = slope * np.array(indices) + intercept
                    nonlinearity = np.mean((np.array(positions) - linear_pred)**2)
                    combined_features[f'com_{axis}_nonlinearity'] = nonlinearity
                except:
                    pass

        # 7. Calculate start-to-end change features
        key_features = ['hip_rotation_angle', 'trunk_tilt_angle', 'body_stability',
                       'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z']

        for feature in key_features:
            first_col = f'frame1_{feature}'
            last_col = f'frame5_{feature}'
            first_idx = 'frame1_frame_index'
            last_idx = 'frame5_frame_index'

            if all(col in combined_features for col in [first_col, last_col, first_idx, last_idx]):
                # Calculate total change
                combined_features[f'total_change_{feature}'] = combined_features[last_col] - combined_features[first_col]

                # Calculate time-aware change rate
                idx_diff = combined_features[last_idx] - combined_features[first_idx]
                if idx_diff > 0:
                    combined_features[f'total_rate_{feature}'] = (
                        (combined_features[last_col] - combined_features[first_col]) / idx_diff
                    )

        # 8. Calculate frame interval statistics
        frame_gaps = []

        for i in range(1, 5):
            idx1 = f'frame{i}_frame_index'
            idx2 = f'frame{i+1}_frame_index'

            if idx1 in combined_features and idx2 in combined_features:
                gap = combined_features[idx2] - combined_features[idx1]
                frame_gaps.append(gap)

        if frame_gaps:
            # Calculate average frame gap and standard deviation
            combined_features['avg_frame_gap'] = np.mean(frame_gaps)
            combined_features['std_frame_gap'] = np.std(frame_gaps)

            # Calculate coefficient of variation for frame gaps
            if combined_features['avg_frame_gap'] > 0:
                combined_features['cv_frame_gap'] = combined_features['std_frame_gap'] / combined_features['avg_frame_gap']


# 4. ClimbingRuleSystem
class ClimbingRuleSystem:
    def __init__(self):
        # Initialize various rules
        self.general_rules = self.initialize_general_rules()
        self.body_position_rules = self.initialize_body_position_rules()
        self.foot_placement_rules = self.initialize_foot_placement_rules()
        self.weight_distribution_rules = self.initialize_weight_distribution_rules()
        self.grip_technique_rules = self.initialize_grip_technique_rules()
        self.balance_issue_rules = self.initialize_balance_issue_rules()
        self.arm_extension_rules = self.initialize_arm_extension_rules()
        self.insufficient_core_rules = self.initialize_insufficient_core_rules()

        # Rule category mapping
        self.rule_categories = {
            "general": self.general_rules,
            "body_position": self.body_position_rules,
            "foot_placement": self.foot_placement_rules,
            "weight_distribution": self.weight_distribution_rules,
            "grip_technique": self.grip_technique_rules,
            "balance_issue": self.balance_issue_rules,
            "arm_extension": self.arm_extension_rules,
            "insufficient_core": self.insufficient_core_rules
        }

    def initialize_general_rules(self):
        """Initialize general rules (based on calculated data)"""
        return {
            "frame5_left_elbow_angle": {
                "threshold": [104.82, 183.59],
                "importance": 0.75,
                "explanation": "Left elbow angle affects arm control and overall stability",
                "suggestion": "Maintain appropriate left elbow angle, typically around 145 degrees",
                "optimal_value": 145
            },
            "direct_com_y_movement": {
                "threshold": [-0.023, 0.193],
                "importance": 0.85,
                "explanation": "Vertical center of mass direct movement reflects vertical control ability",
                "suggestion": "Control vertical ascent amplitude, optimal around 0.085",
                "optimal_value": 0.085
            },
            "total_com_x_movement": {
                "threshold": [-0.002, 0.352],
                "importance": 0.85,
                "explanation": "Total horizontal center of mass movement reflects lateral control ability",
                "suggestion": "Control horizontal movement range, optimal around 0.175",
                "optimal_value": 0.175
            },
            "com_y_efficiency": {
                "threshold": [0.386, 1.112],
                "importance": 0.85,
                "explanation": "Vertical center of mass movement efficiency reflects overall technical level",
                "suggestion": "Improve vertical movement efficiency, target around 0.75",
                "optimal_value": 0.75
            },
            "frame5_center_of_mass_x": {
                "threshold": [0.314, 0.743],
                "importance": 0.75,
                "explanation": "Horizontal center of mass position at end of movement reflects endpoint control ability",
                "suggestion": "Maintain center of mass at horizontal center at end of movement, around 0.53",
                "optimal_value": 0.53
            }
        }

    def initialize_body_position_rules(self):
        """Initialize body position rules (based on calculated data)"""
        return {
            "total_com_x_movement": {
                "threshold": [-0.002, 0.352],
                "importance": 0.87,
                "explanation": "Insufficient total horizontal center of mass movement indicates body position control issues",
                "suggestion": "Increase horizontal movement control, optimal around 0.175",
                "optimal_value": 0.175
            },
            "direct_com_x_movement": {
                "threshold": [-0.043, 0.304],
                "importance": 0.84,
                "explanation": "Insufficient direct horizontal center of mass movement indicates poor body position control",
                "suggestion": "Improve horizontal movement precision, optimal around 0.13",
                "optimal_value": 0.13
            },
            "trunk_angle_variation": {
                "threshold": [-0.80, 25.35],
                "importance": 0.81,
                "explanation": "Too small trunk angle variation indicates overly rigid body position",
                "suggestion": "Increase trunk angle dynamic variation, target around 12 degrees",
                "optimal_value": 12.0
            },
            "com_x_nonlinearity": {
                "threshold": [-0.001, 0.003],
                "importance": 0.77,
                "explanation": "Insufficient horizontal center of mass trajectory nonlinearity indicates lack of smoothness",
                "suggestion": "Increase movement smoothness and natural transition",
                "optimal_value": 0.001
            },
            "frame3_hip_rotation_angle": {
                "threshold": [-3.23, 29.63],
                "importance": 0.77,
                "explanation": "Insufficient hip rotation angle in middle stage affects body position",
                "suggestion": "Increase hip rotation angle, target around 13 degrees",
                "optimal_value": 13.0
            }
        }

    def initialize_foot_placement_rules(self):
        """Initialize foot placement rules (based on calculated data)"""
        return {
            "com_x_efficiency": {
                "threshold": [0.234, 1.185],
                "importance": 0.90,
                "explanation": "Low horizontal center of mass movement efficiency indicates unstable foot support",
                "suggestion": "Improve horizontal movement efficiency, maintain around 0.71",
                "optimal_value": 0.71
            },
            "frame4_center_of_mass_z": {
                "threshold": [-0.195, 0.206],
                "importance": 0.88,
                "explanation": "Fourth stage depth direction center of mass position affects foot control",
                "suggestion": "Maintain appropriate body distance from rock wall, close to 0",
                "optimal_value": 0.0
            }
        }

    def initialize_weight_distribution_rules(self):
        """Initialize weight distribution rules (based on calculated data)"""
        return {
            "total_com_z_movement": {
                "threshold": [-0.012, 0.591],
                "importance": 0.82,
                "explanation": "Total depth direction center of mass movement reflects weight distribution control",
                "suggestion": "Control body distance variation from rock wall, optimal around 0.29",
                "optimal_value": 0.29
            },
            "com_y_efficiency": {
                "threshold": [0.386, 1.112],
                "importance": 0.77,
                "explanation": "Vertical center of mass movement efficiency reflects weight distribution",
                "suggestion": "Improve vertical movement efficiency, target around 0.75",
                "optimal_value": 0.75
            },
            "frame2_trunk_length": {
                "threshold": [0.086, 0.358],
                "importance": 0.76,
                "explanation": "Second stage trunk length affects weight distribution",
                "suggestion": "Maintain appropriate trunk extension, around 0.22",
                "optimal_value": 0.22
            },
            "frame3_trunk_length": {
                "threshold": [0.077, 0.363],
                "importance": 0.76,
                "explanation": "Third stage trunk length affects weight distribution",
                "suggestion": "Maintain suitable trunk extension, around 0.22",
                "optimal_value": 0.22
            },
            "direct_com_y_movement": {
                "threshold": [-0.023, 0.193],
                "importance": 0.75,
                "explanation": "Direct vertical center of mass movement reflects weight ascent control",
                "suggestion": "Control vertical ascent amplitude, optimal around 0.085",
                "optimal_value": 0.085
            },
            "frame1_center_of_mass_z": {
                "threshold": [-0.181, 0.197],
                "importance": 0.93,
                "explanation": "Initial depth direction center of mass position affects foot placement stability",
                "suggestion": "Adjust initial distance from the rock wall, avoid being too far or too close",
                "optimal_value": 0.0
            },
            "frame2_center_of_mass_z": {
                "threshold": [-0.170, 0.219],
                "importance": 0.92,
                "explanation": "Second stage depth direction center of mass position affects foot support",
                "suggestion": "Maintain appropriate body distance from rock wall, around 0.02",
                "optimal_value": 0.02
            },
        }

    def initialize_grip_technique_rules(self):
        """Initialize grip technique rules (based on calculated data)"""
        return {
            "frame1_trunk_length": {
                "threshold": [0.084, 0.376],
                "importance": 0.85,
                "explanation": "Initial trunk length is directly related to grip posture",
                "suggestion": "Maintain appropriate trunk extension, around 0.23",
                "optimal_value": 0.23
            },
            "frame2_trunk_length": {
                "threshold": [0.086, 0.358],
                "importance": 0.84,
                "explanation": "Second stage trunk length affects grip stability",
                "suggestion": "Maintain suitable trunk extension, around 0.22",
                "optimal_value": 0.22
            },
            "frame3_trunk_length": {
                "threshold": [0.077, 0.363],
                "importance": 0.85,
                "explanation": "Third stage trunk length affects grip force application",
                "suggestion": "Maintain stable trunk extension, around 0.22",
                "optimal_value": 0.22
            },
            "frame1_right_leg_extension": {
                "threshold": [0.074, 0.512],
                "importance": 0.88,
                "explanation": "Initial right leg extension affects body stability during gripping",
                "suggestion": "Maintain appropriate right leg extension, around 0.29",
                "optimal_value": 0.29
            },
            "frame5_left_arm_extension": {
                "threshold": [0.084, 0.382],
                "importance": 0.88,
                "explanation": "Final left arm extension affects stability after grip completion",
                "suggestion": "Maintain appropriate left arm extension, around 0.23",
                "optimal_value": 0.23
            }
        }

    def initialize_balance_issue_rules(self):
        """Initialize balance issue rules (based on calculated data)"""
        return {
            "total_rate_center_of_mass_x": {
                "threshold": [-0.0025, 0.0038],
                "importance": 1.00,
                "explanation": "Abnormal horizontal center of mass change rate indicates balance issues",
                "suggestion": "Maintain stable horizontal center of mass change, close to 0",
                "optimal_value": 0.0
            },
            "com_x_trajectory_slope": {
                "threshold": [-0.0026, 0.0040],
                "importance": 1.00,
                "explanation": "Abnormal horizontal center of mass trajectory slope indicates instability",
                "suggestion": "Maintain stable horizontal center of mass trajectory, slope close to 0",
                "optimal_value": 0.0
            },
            "rate_4_to_5_center_of_mass_x": {
                "threshold": [-0.0040, 0.0070],
                "importance": 1.00,
                "explanation": "Abnormal horizontal center of mass change rate in final stage indicates imbalance",
                "suggestion": "Maintain horizontal stability in final stage, change rate close to 0.0015",
                "optimal_value": 0.0015
            },
            "frame5_frame_index": {
                "threshold": [37.91, 236.28],
                "importance": 0.99,
                "explanation": "Abnormal final frame index may indicate balance issues causing premature termination",
                "suggestion": "Complete entire movement sequence, do not interrupt due to balance problems",
                "optimal_value": 137.0
            },
            "rate_3_to_4_center_of_mass_x": {
                "threshold": [-0.0043, 0.0063],
                "importance": 0.99,
                "explanation": "Abnormal horizontal center of mass change rate between third and fourth stages indicates instability",
                "suggestion": "Maintain horizontal stability in mid-to-late stages, change rate close to 0.001",
                "optimal_value": 0.001
            }
        }

    def initialize_arm_extension_rules(self):
        """Initialize arm extension rules (based on calculated data)"""
        return {
            "frame1_center_of_mass_y": {
                "threshold": [0.394, 0.690],
                "importance": 0.96,
                "explanation": "Initial vertical center of mass position is related to arm extension posture",
                "suggestion": "Maintain appropriate initial vertical center of mass position, around 0.54",
                "optimal_value": 0.54
            },
            "hip_rotation_angle_weighted_avg": {
                "threshold": [0.074, 23.244],
                "importance": 0.95,
                "explanation": "Too small weighted average of hip rotation angle affects arm extension efficiency",
                "suggestion": "Increase hip rotation angle, average target around 11.66 degrees",
                "optimal_value": 11.66
            },
            "frame2_center_of_mass_y": {
                "threshold": [0.372, 0.680],
                "importance": 0.95,
                "explanation": "Second stage vertical center of mass position affects arm extension posture",
                "suggestion": "Maintain appropriate vertical center of mass position, around 0.53",
                "optimal_value": 0.53
            },
            "frame4_hip_rotation_angle": {
                "threshold": [-7.44, 31.46],
                "importance": 0.93,
                "explanation": "Too small hip rotation angle in fourth stage affects arm extension",
                "suggestion": "Increase hip rotation angle, target around 12 degrees",
                "optimal_value": 12.0
            },
            "frame3_center_of_mass_y": {
                "threshold": [0.353, 0.653],
                "importance": 0.92,
                "explanation": "Third stage vertical center of mass position affects arm extension efficiency",
                "suggestion": "Maintain appropriate vertical center of mass position, around 0.50",
                "optimal_value": 0.50
            },
            "frame4_right_arm_extension": {
                "threshold": [0.085, 0.123],
                "importance": 0.89,
                "explanation": "Right arm extension is related to foot placement balance",
                "suggestion": "Adjust right arm extension, maintain around 0.23 to maintain balance",
                "optimal_value": 0.23
            },
        }

    def initialize_insufficient_core_rules(self):
        """Initialize insufficient core strength rules (based on calculated data)"""
        return {
            "frame5_left_shoulder_angle": {
                "threshold": [44.91, 145.71],
                "importance": 0.98,
                "explanation": "Abnormal final left shoulder angle indicates insufficient core control",
                "suggestion": "Strengthen core control, maintain left shoulder angle around 95 degrees",
                "optimal_value": 95.0
            },
            "direct_com_y_movement": {
                "threshold": [-0.023, 0.193],
                "importance": 0.97,
                "explanation": "Too small direct vertical center of mass movement indicates insufficient core strength",
                "suggestion": "Enhance core strength, increase vertical movement to around 0.085",
                "optimal_value": 0.085
            },
            "center_of_mass_z_weighted_avg": {
                "threshold": [-0.154, 0.175],
                "importance": 0.96,
                "explanation": "Weighted average depth direction center of mass position too low indicates insufficient core support",
                "suggestion": "Strengthen core support, maintain appropriate body distance from rock wall",
                "optimal_value": 0.01
            },
            "total_com_y_movement": {
                "threshold": [-0.005, 0.221],
                "importance": 0.95,
                "explanation": "Too small total vertical center of mass movement indicates insufficient core strength",
                "suggestion": "Enhance core strength, increase total vertical movement to around 0.11",
                "optimal_value": 0.11
            },
            "frame5_center_of_mass_z": {
                "threshold": [-0.184, 0.191],
                "importance": 0.95,
                "explanation": "Final depth direction center of mass position too low indicates insufficient core support",
                "suggestion": "Strengthen core support at end of movement, maintain appropriate body distance from rock wall",
                "optimal_value": 0.0
            }
        }

class ClimbingRuleSystem:
    def __init__(self):
        # Initialize various rules
        self.general_rules = self.initialize_general_rules()
        self.body_position_rules = self.initialize_body_position_rules()
        self.foot_placement_rules = self.initialize_foot_placement_rules()
        self.weight_distribution_rules = self.initialize_weight_distribution_rules()
        self.grip_technique_rules = self.initialize_grip_technique_rules()
        self.balance_issue_rules = self.initialize_balance_issue_rules()
        self.arm_extension_rules = self.initialize_arm_extension_rules()
        self.insufficient_core_rules = self.initialize_insufficient_core_rules()

        # Rule category mapping
        self.rule_categories = {
            "general": self.general_rules,
            "body_position": self.body_position_rules,
            "foot_placement": self.foot_placement_rules,
            "weight_distribution": self.weight_distribution_rules,
            "grip_technique": self.grip_technique_rules,
            "balance_issue": self.balance_issue_rules,
            "arm_extension": self.arm_extension_rules,
            "insufficient_core": self.insufficient_core_rules
        }

    def initialize_general_rules(self):
        """Initialize general rules (overall movement quality)"""
        return {
            "com_y_efficiency": {
                "threshold": [0.386, 1.112],
                "importance": 0.75,
                "explanation": "Vertical movement efficiency reflects overall climbing economy",
                "suggestion": "Focus on smooth, direct vertical movements with minimal wasted motion",
                "optimal_value": 0.75
            },
            "com_x_efficiency": {
                "threshold": [0.234, 1.185],
                "importance": 0.70,
                "explanation": "Horizontal movement efficiency reflects route-reading and planning skills",
                "suggestion": "Plan your sequence to minimize unnecessary lateral movement",
                "optimal_value": 0.71
            },
            "total_com_y_movement": {
                "threshold": [-0.005, 0.221],
                "importance": 0.72,
                "explanation": "Total vertical movement reflects climbing progress and efficiency",
                "suggestion": "Maintain steady upward progress with controlled movements",
                "optimal_value": 0.11
            },
            "sequence_coverage": {
                "threshold": [0.4, 0.95],
                "importance": 0.70,
                "explanation": "Sequence coverage indicates how much of the climb was completed",
                "suggestion": "Complete the full climbing sequence with confidence",
                "optimal_value": 0.75
            },
            "body_stability": {
                "threshold": [0.5, 25.0],
                "importance": 0.73,
                "explanation": "Body stability indicates overall control during the climb",
                "suggestion": "Maintain controlled, deliberate movements throughout the climb",
                "optimal_value": 12.0
            }
        }

    def initialize_body_position_rules(self):
        """Initialize body position rules (focused on posture and positioning)"""
        return {
            "trunk_tilt_angle": {
                "threshold": [10.0, 45.0],
                "importance": 0.85,
                "explanation": "Trunk tilt angle affects body position relative to the wall",
                "suggestion": "Maintain an appropriate trunk angle to stay close to the wall without sacrificing mobility",
                "optimal_value": 25.0
            },
            "frame3_hip_rotation_angle": {
                "threshold": [-3.23, 29.63],
                "importance": 0.77,
                "explanation": "Hip rotation during mid-climb affects body positioning and reach",
                "suggestion": "Use appropriate hip rotation to maximize reach and stability",
                "optimal_value": 13.0
            },
            "hip_rotation_angle_weighted_avg": {
                "threshold": [0.074, 23.244],
                "importance": 0.80,
                "explanation": "Average hip rotation throughout the climb affects overall body positioning",
                "suggestion": "Engage in active hip positioning to optimize body position relative to holds",
                "optimal_value": 11.66
            },
            "trunk_angle_variation": {
                "threshold": [3.0, 25.35],
                "importance": 0.81,
                "explanation": "Trunk angle variation indicates adaptability of body position",
                "suggestion": "Allow your trunk angle to adapt to different moves rather than remaining rigid",
                "optimal_value": 12.0
            },
            "frame3_center_of_mass_y": {
                "threshold": [0.353, 0.653],
                "importance": 0.75,
                "explanation": "Mid-climb vertical center of mass position affects overall body configuration",
                "suggestion": "Position your center of mass at an appropriate height for maximum control",
                "optimal_value": 0.50
            }
        }

    def initialize_foot_placement_rules(self):
        """Initialize foot placement rules (focused on foot position and technique)"""
        return {
            "left_leg_extension": {
                "threshold": [0.15, 0.45],
                "importance": 0.85,
                "explanation": "Left leg extension affects foot placement stability and reach",
                "suggestion": "Adjust leg extension based on available footholds and required reach",
                "optimal_value": 0.30
            },
            "right_leg_extension": {
                "threshold": [0.15, 0.45],
                "importance": 0.85,
                "explanation": "Right leg extension affects foot placement stability and reach",
                "suggestion": "Adjust leg extension based on available footholds and required reach",
                "optimal_value": 0.30
            },
            "left_knee_angle": {
                "threshold": [80.0, 170.0],
                "importance": 0.80,
                "explanation": "Left knee angle affects foot pressure and positioning",
                "suggestion": "Adjust knee angle to optimize pressure and direction of force on footholds",
                "optimal_value": 120.0
            },
            "right_knee_angle": {
                "threshold": [80.0, 170.0],
                "importance": 0.80,
                "explanation": "Right knee angle affects foot pressure and positioning",
                "suggestion": "Adjust knee angle to optimize pressure and direction of force on footholds",
                "optimal_value": 120.0
            },
            "change_1_to_2_left_knee_angle": {
                "threshold": [-30.0, 30.0],
                "importance": 0.75,
                "explanation": "Change in left knee angle indicates foot adjustment technique",
                "suggestion": "Make deliberate foot adjustments rather than constant micro-adjustments",
                "optimal_value": 0.0
            }
        }

    def initialize_weight_distribution_rules(self):
        """Initialize weight distribution rules (balance between limbs)"""
        return {
            "lateral_balance": {
                "threshold": [-0.15, 0.15],
                "importance": 0.85,
                "explanation": "Lateral balance indicates weight distribution between left and right",
                "suggestion": "Center your weight appropriately to avoid overloading one side",
                "optimal_value": 0.0
            },
            "lateral_balance_variation": {
                "threshold": [0.01, 0.08],
                "importance": 0.85,
                "explanation": "Variation in lateral balance reflects weight shift control",
                "suggestion": "Make controlled weight shifts rather than erratic movements",
                "optimal_value": 0.04
            },
            "frame3_center_of_mass_x": {
                "threshold": [0.35, 0.65],
                "importance": 0.82,
                "explanation": "Mid-climb horizontal center of mass position affects weight distribution",
                "suggestion": "Position your center of mass horizontally to optimize weight distribution",
                "optimal_value": 0.50
            },
            "direct_com_y_movement": {
                "threshold": [-0.023, 0.193],
                "importance": 0.88,
                "explanation": "Direct vertical movement indicates weight transfer efficiency",
                "suggestion": "Transfer weight vertically with control for efficient upward progress",
                "optimal_value": 0.085
            },
            "total_com_z_movement": {
                "threshold": [0.01, 0.491],
                "importance": 0.84,
                "explanation": "Total depth movement indicates control of body distance from wall",
                "suggestion": "Maintain appropriate distance from the wall throughout the climb",
                "optimal_value": 0.27
            }
        }

    def initialize_grip_technique_rules(self):
        """Initialize grip technique rules (hand positions and engagement)"""
        return {
            "frame1_left_arm_extension": {
                "threshold": [0.084, 0.382],
                "importance": 0.85,
                "explanation": "Initial left arm extension affects grip setup and positioning",
                "suggestion": "Start with appropriate arm extension to optimize grip strength and position",
                "optimal_value": 0.23
            },
            "frame1_right_arm_extension": {
                "threshold": [0.084, 0.382],
                "importance": 0.85,
                "explanation": "Initial right arm extension affects grip setup and positioning",
                "suggestion": "Start with appropriate arm extension to optimize grip strength and position",
                "optimal_value": 0.23
            },
            "left_elbow_angle": {
                "threshold": [90.0, 170.0],
                "importance": 0.80,
                "explanation": "Left elbow angle affects grip force application and control",
                "suggestion": "Adjust elbow angle based on hold type and required grip strength",
                "optimal_value": 140.0
            },
            "right_elbow_angle": {
                "threshold": [90.0, 170.0],
                "importance": 0.80,
                "explanation": "Right elbow angle affects grip force application and control",
                "suggestion": "Adjust elbow angle based on hold type and required grip strength",
                "optimal_value": 140.0
            },
            "frame5_left_elbow_angle": {
                "threshold": [104.82, 183.59],
                "importance": 0.77,
                "explanation": "Final left elbow angle reflects grip control at the end of a move",
                "suggestion": "Complete moves with controlled elbow position for optimal grip security",
                "optimal_value": 145.0
            }
        }

    def initialize_balance_issue_rules(self):
        """Initialize balance issue rules (stability and control)"""
        return {
            "total_rate_center_of_mass_x": {
                "threshold": [-0.0025, 0.0038],
                "importance": 0.88,
                "explanation": "Rate of horizontal center of mass change indicates balance control",
                "suggestion": "Make controlled horizontal adjustments to maintain balance",
                "optimal_value": 0.0007
            },
            "com_x_trajectory_slope": {
                "threshold": [-0.0026, 0.0040],
                "importance": 0.85,
                "explanation": "Horizontal trajectory slope indicates consistent balance control",
                "suggestion": "Maintain a steady horizontal position during vertical progress",
                "optimal_value": 0.0007
            },
            "com_x_nonlinearity": {
                "threshold": [-0.001, 0.003],
                "importance": 0.77,
                "explanation": "Horizontal nonlinearity indicates balance adjustments",
                "suggestion": "Make smooth balance adjustments rather than jerky corrections",
                "optimal_value": 0.001
            },
            "rate_4_to_5_center_of_mass_x": {
                "threshold": [-0.0040, 0.0070],
                "importance": 0.82,
                "explanation": "Late-stage horizontal adjustment rate indicates final balance control",
                "suggestion": "Maintain control of horizontal position in the final phase of moves",
                "optimal_value": 0.0015
            },
            "rate_3_to_4_center_of_mass_x": {
                "threshold": [-0.0043, 0.0063],
                "importance": 0.80,
                "explanation": "Mid-stage horizontal adjustment rate indicates transitional balance",
                "suggestion": "Control horizontal position during the middle of climbing movements",
                "optimal_value": 0.001
            }
        }

    def initialize_arm_extension_rules(self):
        """Initialize arm extension rules (arm positioning and efficiency)"""
        return {
            "frame4_right_arm_extension": {
                "threshold": [0.10, 0.35],
                "importance": 0.85,
                "explanation": "Late-stage right arm extension affects movement efficiency",
                "suggestion": "Use appropriate arm extension during movement execution phase",
                "optimal_value": 0.23
            },
            "frame5_left_arm_extension": {
                "threshold": [0.10, 0.35],
                "importance": 0.83,
                "explanation": "Final left arm extension affects position after completing a move",
                "suggestion": "Complete movements with optimal arm extension for the next move",
                "optimal_value": 0.23
            },
            "frame1_center_of_mass_y": {
                "threshold": [0.394, 0.690],
                "importance": 0.78,
                "explanation": "Initial vertical position affects starting arm extension",
                "suggestion": "Start with an appropriate vertical position for optimal arm extension",
                "optimal_value": 0.54
            },
            "left_arm_extension": {
                "threshold": [0.10, 0.40],
                "importance": 0.85,
                "explanation": "Left arm extension affects energy efficiency and reach",
                "suggestion": "Use straight arms when static to conserve energy",
                "optimal_value": 0.25
            },
            "right_arm_extension": {
                "threshold": [0.10, 0.40],
                "importance": 0.85,
                "explanation": "Right arm extension affects energy efficiency and reach",
                "suggestion": "Use straight arms when static to conserve energy",
                "optimal_value": 0.25
            }
        }

    def initialize_insufficient_core_rules(self):
        """Initialize insufficient core strength rules (core engagement and stability)"""
        return {
            "frame5_left_shoulder_angle": {
                "threshold": [44.91, 145.71],
                "importance": 0.85,
                "explanation": "Final shoulder angle indicates core control at move completion",
                "suggestion": "Engage core to maintain proper shoulder position throughout movements",
                "optimal_value": 95.0
            },
            "frame2_trunk_length": {
                "threshold": [0.15, 0.358],
                "importance": 0.82,
                "explanation": "Early trunk extension indicates core engagement level",
                "suggestion": "Maintain appropriate trunk extension through core engagement",
                "optimal_value": 0.22
            },
            "frame3_trunk_length": {
                "threshold": [0.15, 0.363],
                "importance": 0.82,
                "explanation": "Mid-climb trunk extension indicates sustained core engagement",
                "suggestion": "Maintain consistent core engagement throughout the climb",
                "optimal_value": 0.22
            },
            "center_of_mass_z_weighted_avg": {
                "threshold": [-0.154, 0.175],
                "importance": 0.80,
                "explanation": "Average distance from wall indicates core support strength",
                "suggestion": "Use core strength to maintain optimal distance from the wall",
                "optimal_value": 0.01
            },
            "frame5_center_of_mass_z": {
                "threshold": [-0.184, 0.191],
                "importance": 0.80,
                "explanation": "Final distance from wall reflects core control at move completion",
                "suggestion": "Complete moves with proper core engagement to stay close to the wall",
                "optimal_value": 0.0
            }
        }

    def calculate_score(self, violations):
        """Calculate overall score based on rule violations

        Args:
            violations: List of violated rules

        Returns:
            Score from 0-100, 100 means no issues
        """
        if not violations:
            return 100.0

        total_deduction = 0.0
        total_weight = 0.0

        for violation in violations:
            importance = violation["importance"]
            deviation = violation["relative_deviation"]

            # Use penalty multiplier of 8 as requested
            deduction = importance * deviation * 8
            total_deduction += deduction
            total_weight += importance

        # Normalize total deduction
        if total_weight > 0:
            normalized_deduction = total_deduction / total_weight * 10
        else:
            normalized_deduction = total_deduction

        # Use base score of 50 as requested
        base_score = 50.0
        
        # Calculate final score without bonus
        final_score = max(base_score, 100.0 - normalized_deduction)
        
        # Ensure score doesn't exceed 100
        final_score = min(100.0, final_score)

        return round(final_score, 1)

    def check_rules(self, features, category=None):
        """Check if features violate rules

        Args:
            features: Extracted feature dictionary
            category: Rule category to check, if None checks all rules

        Returns:
            List of violated rules
        """
        violations = []

        # Determine rule categories to check
        if category and category in self.rule_categories:
            categories_to_check = {category: self.rule_categories[category]}
        else:
            categories_to_check = self.rule_categories

        # Check various rule categories
        for category_name, rules in categories_to_check.items():
            for feature_name, rule in rules.items():
                if feature_name in features:
                    feature_value = features[feature_name]
                    threshold = rule["threshold"]
                    optimal_value = rule.get("optimal_value", None)

                    # Method 1: Check if within valid range
                    in_valid_range = True
                    if isinstance(threshold, list) and len(threshold) == 2:
                        if feature_value < threshold[0] or feature_value > threshold[1]:
                            in_valid_range = False

                    # Method 2: Check deviation from optimal value (less sensitive)
                    significant_deviation = False
                    deviation_ratio = 0.0
                    deviation_direction = None

                    if optimal_value is not None:
                        # Calculate relative deviation
                        deviation = abs(feature_value - optimal_value)

                        if isinstance(threshold, list) and len(threshold) == 2:
                            threshold_range = threshold[1] - threshold[0]
                            if threshold_range > 0:
                                deviation_ratio = deviation / threshold_range

                            # Increase sensitivity threshold from 0.3 to 0.4 to reduce false positives
                            if deviation_ratio > 0.3:
                                significant_deviation = True

                                # Record deviation direction for error type determination
                                if feature_value < optimal_value:
                                    deviation_direction = "below optimal value"
                                else:
                                    deviation_direction = "above optimal value"

                    # Combined judgment: outside range or significantly deviating from optimal value
                    violated = not in_valid_range or significant_deviation

                    # Calculate relative deviation for sorting
                    if not in_valid_range:
                        # For cases outside the range
                        if isinstance(threshold, list) and len(threshold) == 2:
                            if feature_value < threshold[0]:
                                deviation = threshold[0] - feature_value
                            else:
                                deviation = feature_value - threshold[1]

                            threshold_range = threshold[1] - threshold[0]
                            if threshold_range > 0:
                                relative_deviation = deviation / threshold_range
                            else:
                                relative_deviation = deviation
                    else:
                        # For cases within range but deviating from optimal value
                        relative_deviation = deviation_ratio

                    if violated:
                        violations.append({
                            "category": category_name,
                            "feature": feature_name,
                            "value": feature_value,
                            "threshold": threshold,
                            "optimal_value": optimal_value,
                            "importance": rule["importance"],
                            "explanation": rule["explanation"],
                            "suggestion": rule["suggestion"],
                            "relative_deviation": min(1.0, relative_deviation),  # Limit max deviation to 1.0
                            "deviation_direction": deviation_direction
                        })

        # Sort by importance * deviation degree
        violations.sort(key=lambda x: x["importance"] * x["relative_deviation"], reverse=True)

        return violations

    def predict_error_type(self, features):
        """Predict main error type

        Args:
            features: Extracted feature dictionary

        Returns:
            Predicted error type and its probability
        """
        # Check rule violations for each category
        category_violations = {}
        category_scores = {}

        # Check only main error types, exclude general
        error_categories = [cat for cat in self.rule_categories.keys() if cat != "general"]

        for category in error_categories:
            violations = self.check_rules(features, category)
            category_violations[category] = violations

            # Calculate score for this category
            if violations:
                category_scores[category] = self.calculate_score(violations)
            else:
                category_scores[category] = 100.0

        # Find category with lowest score as prediction result
        if category_scores:
            predicted_type = min(category_scores.items(), key=lambda x: x[1])[0]

            # Calculate probabilities for each category (based on inverse of scores)
            # More balanced approach to prevent one category from dominating
            total_inverse_score = sum(max(0, 100 - score) for score in category_scores.values())

            error_probabilities = {}
            if total_inverse_score > 0:
                for category, score in category_scores.items():
                    inverse_score = max(0, 100 - score)
                    # Apply mild softening to prevent extreme probabilities
                    adjusted_inverse = inverse_score ** 0.9
                    error_probabilities[category] = adjusted_inverse / total_inverse_score
            else:
                # If all categories have 100 points, distribute probability evenly
                for category in category_scores:
                    error_probabilities[category] = 1.0 / len(category_scores)

            return {
                "predicted_type": predicted_type,
                "probabilities": error_probabilities,
                "violations": category_violations[predicted_type]
            }
        else:
            return {
                "predicted_type": None,
                "probabilities": {},
                "violations": []
            }

# Create rule system
rule_system = ClimbingRuleSystem()


# 5. ClimbingKnowledgeBase
# A structured knowledge base built from expert comments and professional knowledge
class ClimbingKnowledgeBase:
    def __init__(self):
        # Initialize knowledge base
        self.knowledge = {
            "weight_distribution": self.initialize_weight_distribution_knowledge(),
            "body_position": self.initialize_body_position_knowledge(),
            "foot_placement": self.initialize_foot_placement_knowledge(),
            "grip_technique": self.initialize_grip_technique_knowledge(),
            "balance_issue": self.initialize_balance_issue_knowledge(),
            "arm_extension": self.initialize_arm_extension_knowledge(),
            "insufficient_core": self.initialize_insufficient_core_knowledge(),
            "general": self.initialize_general_knowledge()
        }

    def initialize_weight_distribution_knowledge(self):
        """Initialize weight distribution related knowledge"""
        return {
            "problems": [
                "Excessive forward lean causing arms to bear too much pressure",
                "Excessive lateral center of mass movement causing instability",
                "Vertical center of mass fluctuations causing jerky movements",
                "Center of mass too high causing instability",
                "Uneven weight distribution causing one side to bear excessive load"
            ],
            "causes": [
                "Insufficient core muscle control",
                "Incorrect center of gravity perception",
                "Over-reliance on upper body strength rather than legs",
                "Improper distance from the wall",
                "Insufficient or excessive hip rotation"
            ],
            "suggestions": [
                "Keep your center of mass directly above your support points",
                "Plan your next center of mass position before moving",
                "Use hip rotation to assist with center of mass movement",
                "Maintain Z-axis movement around 0.29 for optimal stability",
                "Aim for a vertical movement efficiency (com_y_efficiency) of around 0.75"
            ],
            "route_specific": {
                "slab": {
                    "explanation": "On slab routes, maintain your center of mass close to the wall with a slightly higher position",
                    "key_focus": "Foot friction and fore-aft center of mass position",
                    "suggestions": [
                        "Keep your center of mass directly above your feet",
                        "Use micro-adjustments to prevent slipping"
                    ]
                },
                "vertical": {
                    "explanation": "On vertical walls, center of mass should be directly above support points",
                    "key_focus": "Core stability and center of mass height",
                    "suggestions": [
                        "Maintain moderate center of mass height",
                        "Use hip micro-adjustments to control center of mass position"
                    ]
                },
                "overhang": {
                    "explanation": "On overhang routes, keep center of mass close to the wall to avoid swinging",
                    "key_focus": "Core tension and arm extension",
                    "suggestions": [
                        "Maintain core tension to prevent body moving away from the wall",
                        "Use foot support to reduce arm load"
                    ]
                }
            },
            "training_exercises": [
                {
                    "name": "Single Leg Balance Training",
                    "description": "Practice standing on one leg on the ground or low climbing wall",
                    "benefit": "Improves control of weight distribution on single sides"
                },
                {
                    "name": "Hanging Core Exercises",
                    "description": "Hang from climbing holds while performing leg raises",
                    "benefit": "Enhances core strength for better center of mass control"
                },
                {
                    "name": "Blind Climbing Practice",
                    "description": "Complete easy routes with eyes closed, focusing on feeling weight distribution",
                    "benefit": "Improves proprioception and weight distribution awareness"
                }
            ]
        }

    def initialize_body_position_knowledge(self):
        """Initialize body position related knowledge"""
        return {
            "problems": [
                "Excessive body twisting causing reduced stability",
                "Inappropriate trunk tilt angle causing center of mass displacement",
                "Uncoordinated limb extension resulting in awkward positioning",
                "Inefficient joint angles reducing force application",
                "Improper distance from the wall"
            ],
            "causes": [
                "Insufficient body position awareness",
                "Limited joint flexibility",
                "Inadequate core strength to maintain ideal posture",
                "Incorrect movement habits",
                "Poor route reading leading to inadequate preparation"
            ],
            "suggestions": [
                "Maintain your body facing the wall, avoid excessive twisting",
                "Adjust your trunk tilt angle, typically maintaining a slight forward lean",
                "Coordinate limb movements, with upper body actions supporting lower body positioning",
                "Aim for a horizontal center of mass movement (total_com_x_movement) of around 0.175",
                "Maintain trunk angle variation around 12 degrees for optimal body positioning"
            ],
            "route_specific": {
                "slab": {
                    "explanation": "On slab routes, maintain a more upright posture with center of mass above feet",
                    "key_focus": "Center of mass position and trunk angle",
                    "suggestions": [
                        "Maintain higher center of mass and more upright trunk",
                        "Keep arms relaxed and extended, avoid over-gripping"
                    ]
                },
                "vertical": {
                    "explanation": "On vertical walls, maintain balanced posture with coordinated hand and foot movements",
                    "key_focus": "Trunk stability and limb coordination",
                    "suggestions": [
                        "Maintain stable trunk position",
                        "Keep arms moderately extended, neither fully bent nor completely straight"
                    ]
                },
                "overhang": {
                    "explanation": "On overhang routes, engage trunk more actively, maintaining core tension",
                    "key_focus": "Trunk-to-wall angle and core control",
                    "suggestions": [
                        "Keep trunk parallel to the wall, prevent hips from sagging",
                        "Actively use leg pushing to reduce arm load"
                    ]
                }
            },
            "training_exercises": [
                {
                    "name": "Yoga Pose Practice",
                    "description": "Practice basic yoga poses to improve body control and posture awareness",
                    "benefit": "Enhances body position awareness and control"
                },
                {
                    "name": "Posture Mirroring Training",
                    "description": "Observe and mirror professional climbers' postures, using a mirror for feedback",
                    "benefit": "Develops correct posture memory and movement patterns"
                },
                {
                    "name": "Static Position Holds",
                    "description": "Hold challenging positions on the climbing wall for 10-30 seconds",
                    "benefit": "Increases muscular endurance for maintaining correct positions"
                }
            ]
        }

    def initialize_foot_placement_knowledge(self):
        """Initialize foot placement related knowledge"""
        return {
            "problems": [
                "Inaccurate foot positioning leading to unstable support",
                "Hasty foot placement resulting in insufficient friction",
                "Improper foot angle reducing contact area",
                "Noisy foot placement indicating lack of precision control",
                "Uncoordinated hand and foot movements"
            ],
            "causes": [
                "Insufficient visual attention to foot positioning",
                "Inadequate leg flexibility or strength",
                "Over-reliance on arms in climbing technique",
                "Lack of foot precision training",
                "Inappropriate climbing shoe selection or fit"
            ],
            "suggestions": [
                "Visually confirm target points before placing feet",
                "Place feet gently and precisely, not hastily",
                "Adjust foot angle based on hold shape to maximize contact area",
                "Practice silent footwork to improve foot control precision",
                "Coordinate hand and foot movements, stabilize feet before moving hands"
            ],
            "route_specific": {
                "slab": {
                    "explanation": "On slab routes, foot friction is crucial, maximize contact area",
                    "key_focus": "Foot pressure and angle",
                    "suggestions": [
                        "Use inside edge of foot to maximize friction",
                        "Apply downward pressure, avoid lateral forces that cause slipping"
                    ]
                },
                "vertical": {
                    "explanation": "On vertical walls, precise foot placement is essential, combining toe and inside edge",
                    "key_focus": "Foot position precision and pressure direction",
                    "suggestions": [
                        "Combine use of toe and inside edge of foot",
                        "Apply pressure downward with slight inward direction toward the wall"
                    ]
                },
                "overhang": {
                    "explanation": "On overhang routes, active toe hooking and heel hooking is essential",
                    "key_focus": "Heel hooks and toe precision",
                    "suggestions": [
                        "Actively engage heel hooks on suitable holds",
                        "Point toes toward the wall to help reduce arm load"
                    ]
                }
            },
            "training_exercises": [
                {
                    "name": "Precision Footwork Drills",
                    "description": "Practice quickly and accurately placing feet on specific points on ground or low walls",
                    "benefit": "Improves foot placement precision and speed"
                },
                {
                    "name": "Silent Climbing",
                    "description": "Complete routes with completely silent footwork, focusing on placement",
                    "benefit": "Enhances foot control precision and body balance"
                },
                {
                    "name": "One-Foot Support Training",
                    "description": "Practice hand movements and body adjustments while supported on a single foot",
                    "benefit": "Improves single-foot support capacity and balance"
                }
            ]
        }

    def initialize_grip_technique_knowledge(self):
        """Initialize grip technique related knowledge"""
        return {
            "problems": [
                "Overgripping causing premature forearm fatigue",
                "Incorrect hand position on holds reducing effectiveness",
                "Improper body positioning making holds feel worse than they are",
                "Insufficient use of open hand techniques",
                "Inappropriate trunk length adjustment when gripping"
            ],
            "causes": [
                "Anxiety or fear leading to excessive gripping force",
                "Lack of experience with various hold types",
                "Poor understanding of body position's impact on grip effectiveness",
                "Limited grip strength or endurance",
                "Incorrect perception of necessary grip force"
            ],
            "suggestions": [
                "Use minimum necessary force when gripping holds",
                "Position body to optimize the angle of pull on holds",
                "Maintain appropriate trunk length (around 0.22-0.23) for optimal grip leverage",
                "Practice various grip types (crimp, open hand, pinch) appropriate to hold types",
                "Use proper arm extension (around 0.29 for right leg, 0.23 for left arm)"
            ],
            "route_specific": {
                "slab": {
                    "explanation": "On slab routes, focus on balance rather than grip strength, using open hand grips",
                    "key_focus": "Minimal gripping force and body balance",
                    "suggestions": [
                        "Use open hand grips whenever possible",
                        "Focus more on foot placement than hand strength"
                    ]
                },
                "vertical": {
                    "explanation": "On vertical walls, combine various grip techniques based on hold types",
                    "key_focus": "Grip matching to hold type",
                    "suggestions": [
                        "Match grip technique to hold type (crimps, slopers, pinches)",
                        "Maintain proper trunk position to optimize grip angle"
                    ]
                },
                "overhang": {
                    "explanation": "On overhang routes, focus on engagement and body tension to reduce grip strain",
                    "key_focus": "Body tension and toe pressure",
                    "suggestions": [
                        "Engage core to reduce load on fingers",
                        "Use toe pressure against wall to reduce grip force needed"
                    ]
                }
            },
            "training_exercises": [
                {
                    "name": "Grip Type Progression",
                    "description": "Practice progressively challenging versions of open hand, half crimp, and full crimp on hangboard",
                    "benefit": "Develops grip strength across all grip types"
                },
                {
                    "name": "Minimum Force Climbing",
                    "description": "Complete easy routes using the minimum possible gripping force",
                    "benefit": "Improves grip efficiency and reduces overgripping"
                },
                {
                    "name": "Grip Position Drills",
                    "description": "Practice finding optimal hand positions on various hold types",
                    "benefit": "Enhances hand placement precision and effectiveness"
                }
            ]
        }

    def initialize_balance_issue_knowledge(self):
        """Initialize balance issue related knowledge"""
        return {
            "problems": [
                "Lateral center of mass movement causing instability",
                "Unsteady center of mass trajectory with sudden shifts",
                "Uncontrolled weight shifts between movements",
                "Inability to maintain stable positions during reaches",
                "Poor dynamic balance during movement transitions"
            ],
            "causes": [
                "Underdeveloped proprioception",
                "Insufficient core strength and control",
                "Poor weight distribution awareness",
                "Rushed movements without establishing balance",
                "Inadequate base of support positioning"
            ],
            "suggestions": [
                "Maintain horizontal center of mass change rate near zero",
                "Keep your center of mass trajectory smooth with minimal slope (around 0.0007)",
                "Control late-stage horizontal adjustments, maintaining a rate near 0.0015",
                "Complete full movement sequences without interruptions due to balance issues",
                "Keep mid-stage horizontal adjustments stable with a rate near 0.001"
            ],
            "route_specific": {
                "slab": {
                    "explanation": "On slab routes, balance is paramount, focus on subtle weight shifts and high foot friction",
                    "key_focus": "Micro-adjustments and center of gravity control",
                    "suggestions": [
                        "Make very small, controlled weight shifts",
                        "Keep weight centered over the highest friction parts of your shoes"
                    ]
                },
                "vertical": {
                    "explanation": "On vertical walls, maintain three-point stability before moving limbs",
                    "key_focus": "Three-point contact and controlled movement",
                    "suggestions": [
                        "Establish solid three-point contact before moving",
                        "Keep center of mass directly over your base of support"
                    ]
                },
                "overhang": {
                    "explanation": "On overhang routes, dynamic balance requires core tension and momentum control",
                    "key_focus": "Core tension and momentum management",
                    "suggestions": [
                        "Maintain constant core engagement to control swinging",
                        "Use controlled momentum rather than fighting against it"
                    ]
                }
            },
            "training_exercises": [
                {
                    "name": "Slackline Training",
                    "description": "Practice balancing on a slackline to improve proprioception",
                    "benefit": "Enhances dynamic balance and body awareness"
                },
                {
                    "name": "One-leg Balance Drills",
                    "description": "Balance on one leg while performing upper body movements",
                    "benefit": "Improves static balance and stability during reaching movements"
                },
                {
                    "name": "Hover Drills",
                    "description": "Practice removing hands or feet briefly from holds while maintaining position",
                    "benefit": "Develops core control and balance during movement transitions"
                }
            ]
        }

    def initialize_arm_extension_knowledge(self):
        """Initialize arm extension related knowledge"""
        return {
            "problems": [
                "Insufficient arm extension causing unnecessary muscular strain",
                "Overextended arms reducing force application potential",
                "Poor coordination between arm extension and body positioning",
                "Inefficient arm positions relative to center of mass",
                "Inappropriate timing of arm extension during movements"
            ],
            "causes": [
                "Habit of climbing with bent arms",
                "Insufficient understanding of optimal arm mechanics",
                "Limited shoulder mobility or strength",
                "Inadequate hip rotation reducing effective reach",
                "Improper center of mass positioning relative to holds"
            ],
            "suggestions": [
                "Maintain appropriate vertical center of mass position (around 0.54 initially)",
                "Incorporate adequate hip rotation (average of 11.66 degrees) to optimize arm positioning",
                "Adjust vertical center of mass throughout movement sequence (0.53 in second stage, 0.50 in third stage)",
                "Increase hip rotation in later stages (aim for around 12 degrees in fourth stage)",
                "Focus on keeping arms straight when static and bend only when necessary for movement"
            ],
            "route_specific": {
                "slab": {
                    "explanation": "On slab routes, extended arms help maintain balance and reduce unnecessary force",
                    "key_focus": "Arm extension and balance",
                    "suggestions": [
                        "Keep arms fully extended to maintain balance",
                        "Focus on pushing with extended arms rather than pulling"
                    ]
                },
                "vertical": {
                    "explanation": "On vertical walls, alternate between straight arms for rest and bent arms for movement",
                    "key_focus": "Strategic arm bending and straightening",
                    "suggestions": [
                        "Straighten arms when static to conserve energy",
                        "Bend arms only during the actual movement phase"
                    ]
                },
                "overhang": {
                    "explanation": "On overhang routes, careful arm extension management is crucial to prevent barn-dooring",
                    "key_focus": "Body tension with arm extension",
                    "suggestions": [
                        "Combine straight arms with active core engagement",
                        "Use opposing tension between extended arms to maintain stability"
                    ]
                }
            },
            "training_exercises": [
                {
                    "name": "Straight Arm Lockoffs",
                    "description": "Practice maintaining position with straight arms in various body positions",
                    "benefit": "Builds strength and comfort in extended arm positions"
                },
                {
                    "name": "Extension Awareness Drills",
                    "description": "Climb easy routes focusing exclusively on maximizing arm extension",
                    "benefit": "Develops habits of proper arm extension"
                },
                {
                    "name": "Shoulder Mobility Exercises",
                    "description": "Perform targeted mobility exercises for shoulder joints",
                    "benefit": "Increases range of motion needed for optimal arm extension"
                }
            ]
        }

    def initialize_insufficient_core_knowledge(self):
        """Initialize insufficient core strength related knowledge"""
        return {
            "problems": [
                "Inability to maintain body tension, especially on overhangs",
                "Sagging hips when reaching for holds",
                "Difficulty maintaining proper body position during dynamic movements",
                "Feet cutting loose unintentionally during movement",
                "Inefficient transfer of force between upper and lower body"
            ],
            "causes": [
                "Underdeveloped core musculature",
                "Lack of core engagement awareness during climbing",
                "Insufficient core endurance for sustained climbing",
                "Poor understanding of how to activate core during specific moves",
                "Overreliance on arm strength compensating for weak core"
            ],
            "suggestions": [
                "Maintain appropriate shoulder angles (around 95 degrees for left shoulder)",
                "Increase vertical center of mass direct movement to around 0.085",
                "Keep center of mass at an appropriate distance from the wall",
                "Work on improving total vertical movement to around 0.11",
                "Maintain appropriate final center of mass position relative to the wall"
            ],
            "route_specific": {
                "slab": {
                    "explanation": "On slab routes, subtle core engagement helps maintain balance and precision",
                    "key_focus": "Fine core control for balance",
                    "suggestions": [
                        "Maintain constant but gentle core engagement",
                        "Focus on rotational stability during high steps"
                    ]
                },
                "vertical": {
                    "explanation": "On vertical walls, core engagement facilitates efficient movement and stability",
                    "key_focus": "Core stabilization during movement",
                    "suggestions": [
                        "Engage core before initiating movement",
                        "Maintain stable midsection during reaches"
                    ]
                },
                "overhang": {
                    "explanation": "On overhang routes, maximum core tension is critical to prevent feet cutting loose",
                    "key_focus": "High tension core activation",
                    "suggestions": [
                        "Maintain constant high tension through core and posterior chain",
                        "Engage lower abs to keep feet on during reaches"
                    ]
                }
            },
            "training_exercises": [
                {
                    "name": "Front Lever Progressions",
                    "description": "Practice progressive front lever variations to build climbing-specific core strength",
                    "benefit": "Develops anterior core strength critical for overhanging climbing"
                },
                {
                    "name": "Tension Board Training",
                    "description": "Practice on a steep tension board focusing on maintaining body tension",
                    "benefit": "Builds climbing-specific core strength and tension awareness"
                },
                {
                    "name": "Toes-to-Bar Exercises",
                    "description": "Perform hanging leg raises focusing on controlled movement",
                    "benefit": "Strengthens lower core needed for foot retention on steep terrain"
                }
            ]
        }

    def initialize_general_knowledge(self):
        """Initialize general climbing knowledge"""
        return {
            "technique_principles": [
                "Maintain three-point support principle, keeping three limbs stable when moving one",
                "Straight arm climbing principle, use extended arms whenever possible to reduce muscular fatigue",
                "Feet-first principle, establish stable foot positions before moving hands",
                "Route reading principle, thoroughly observe the route and plan movement sequences before climbing",
                "Breathing control principle, maintain steady breathing and avoid unconscious breath-holding"
            ],
            "common_mistakes": [
                "Over-reliance on upper body strength while neglecting leg drive",
                "Keeping body too far from the wall causing center of mass displacement",
                "Climbing too quickly reducing precision and control",
                "Excessive arm bending leading to rapid fatigue",
                "Insufficient route reading leading to poor tactical choices"
            ],
            "progression_tips": [
                "Progressively increase difficulty, avoid jumping to routes that are too challenging",
                "Focus on technique improvement rather than pure strength gains",
                "Seek feedback from experienced climbers",
                "Record videos of your climbing for analysis",
                "Regularly return to easier routes to refine basic techniques"
            ],
            "injury_prevention": [
                "Warm up thoroughly, especially focusing on shoulders, fingers and forearms",
                "Avoid overtraining, schedule adequate recovery time",
                "Gradually increase training volume and intensity",
                "Learn proper falling techniques to reduce injury risk",
                "Regularly perform antagonist training for joint stability"
            ]
        }

    def get_knowledge(self, error_type, aspect=None):
        """Get knowledge for a specific error type

        Args:
            error_type: The error type
            aspect: Specific aspect of knowledge like "problems", "causes", etc.

        Returns:
            Requested knowledge content
        """
        if error_type not in self.knowledge:
            # If specific error type not found, return general knowledge
            error_type = "general"

        if aspect:
            return self.knowledge[error_type].get(aspect, {})
        else:
            return self.knowledge[error_type]

    def get_route_specific_knowledge(self, error_type, route_type):
        """Get knowledge specific to error type and route type

        Args:
            error_type: The error type
            route_type: Route type like "slab", "vertical", "overhang"

        Returns:
            Route-specific knowledge
        """
        if error_type not in self.knowledge:
            return {}

        route_specific = self.knowledge[error_type].get("route_specific", {})
        return route_specific.get(route_type, {})

    def get_training_exercises(self, error_type):
        """Get training suggestions for a specific error type

        Args:
            error_type: The error type

        Returns:
            List of training suggestions
        """
        if error_type not in self.knowledge:
            return []

        return self.knowledge[error_type].get("training_exercises", [])

    def search_knowledge(self, query):
        """Search knowledge base

        Args:
            query: Search keywords

        Returns:
            Relevant knowledge entries
        """
        results = []
        query = query.lower()

        # Search across all knowledge categories
        for category, content in self.knowledge.items():
            # Search problem descriptions
            for problem in content.get("problems", []):
                if query in problem.lower():
                    results.append({
                        "category": category,
                        "type": "problem",
                        "content": problem
                    })

            # Search causes
            for cause in content.get("causes", []):
                if query in cause.lower():
                    results.append({
                        "category": category,
                        "type": "cause",
                        "content": cause
                    })

            # Search suggestions
            for suggestion in content.get("suggestions", []):
                if query in suggestion.lower():
                    results.append({
                        "category": category,
                        "type": "suggestion",
                        "content": suggestion
                    })

            # Search route-specific knowledge
            for route_type, route_info in content.get("route_specific", {}).items():
                if query in route_type.lower() or query in route_info.get("explanation", "").lower():
                    results.append({
                        "category": category,
                        "type": "route_specific",
                        "route_type": route_type,
                        "content": route_info.get("explanation", "")
                    })

            # Search training suggestions
            for exercise in content.get("training_exercises", []):
                if query in exercise.get("name", "").lower() or query in exercise.get("description", "").lower():
                    results.append({
                        "category": category,
                        "type": "training_exercise",
                        "content": exercise.get("name", ""),
                        "description": exercise.get("description", "")
                    })

        return results
# Create knowledge base
knowledge_base = ClimbingKnowledgeBase()


# 6. AnalysisEngine
class AnalysisEngine:
    def __init__(self, rule_system):
        self.rule_system = rule_system

    def analyze(self, features, route_info):
        """Analyze feature data using the rule system"""
        # Check rules violations
        violations = self.rule_system.check_rules(features)

        # Calculate overall score
        score = self.rule_system.calculate_score(violations)

        # Predict error type
        error_prediction = self.rule_system.predict_error_type(features)

        # Return analysis results
        return {
            "violations": violations,
            "score": score,
            "error_prediction": error_prediction,
            "features": features,
            "route_info": route_info
        }


# 7. FeedbackGenerator
class FeedbackGenerator:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

        # Initialize Claude client (if API key is available)
        self.claude_client = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate_feedback(self, analysis_results, route_info):
        """Generate complete layered feedback"""
        violations = analysis_results["violations"]
        score = analysis_results["score"]
        error_prediction = analysis_results["error_prediction"]

        # Prepare feedback structure
        feedback = {
            "summary": self._generate_summary(score, error_prediction, violations),
            "technical_assessment": self._generate_technical_assessment(violations, score),
            "error_analysis": self._generate_error_analysis(error_prediction, route_info),
            "improvement_suggestions": self._generate_improvement_suggestions(error_prediction, violations, route_info),
            "training_recommendations": self._generate_training_recommendations(error_prediction, route_info)
        }

        # Use Claude to enhance feedback (if available)
        if self.claude_client:
            claude_feedback = self._generate_claude_feedback(feedback, route_info)
            feedback["claude_enhanced"] = claude_feedback

        return feedback

    def _generate_summary(self, score, error_prediction, violations):
        """Generate overall assessment summary"""
        # Determine performance level based on score
        if score >= 90:
            performance_level = "Excellent"
        elif score >= 80:
            performance_level = "Good"
        elif score >= 70:
            performance_level = "Satisfactory"
        elif score >= 60:
            performance_level = "Needs Improvement"
        else:
            performance_level = "Needs Significant Improvement"

        # Determine main issue
        main_issue = "No significant issues detected"
        if violations and error_prediction["predicted_type"]:
            error_type = error_prediction["predicted_type"]
            error_mapping = {
                "weight_distribution": "Weight Distribution Control",
                "body_position": "Body Position Control",
                "foot_placement": "Foot Placement Precision",
                "grip_technique": "Grip Technique",
                "balance_issue": "Balance Control",
                "arm_extension": "Arm Extension Efficiency",
                "insufficient_core": "Core Strength and Engagement"
            }
            main_issue = error_mapping.get(error_type, error_type.replace("_", " ").title())

        return {
            "score": score,
            "performance_level": performance_level,
            "main_issue": main_issue,
            "violation_count": len(violations)
        }

    def _generate_technical_assessment(self, violations, score):
        """Generate detailed technical assessment"""
        # Group violations by category
        categorized_violations = {}
        for violation in violations:
            category = violation["category"]
            if category not in categorized_violations:
                categorized_violations[category] = []
            categorized_violations[category].append(violation)

        # Generate assessments for each category
        assessments = []
        for category, category_violations in categorized_violations.items():
            # Calculate average deviation for this category
            avg_deviation = sum(v["relative_deviation"] for v in category_violations) / len(category_violations)

            # Determine assessment level
            if avg_deviation < 0.2:
                level = "Minor Issue"
            elif avg_deviation < 0.5:
                level = "Moderate Issue"
            else:
                level = "Significant Issue"

            # Add assessment
            assessments.append({
                "category": category,
                "level": level,
                "violation_count": len(category_violations),
                "details": [self._format_violation(v) for v in category_violations[:3]]  # Only include top 3 most severe violations
            })

        # Sort assessments, most severe first
        assessments.sort(key=lambda x: x["violation_count"], reverse=True)

        # Add overall assessment
        if score >= 90:
            overall = "Technical movement is very fluid with almost no areas for improvement."
        elif score >= 80:
            overall = "Technical movement is good with only minor details that can be improved."
        elif score >= 70:
            overall = "Technical movement is adequate with several areas that need attention."
        elif score >= 60:
            overall = "Technical movement has notable issues that require targeted improvement."
        else:
            overall = "Technical movement has significant issues requiring fundamental technique practice."

        return {
            "overall": overall,
            "detailed_assessments": assessments
        }

    def _format_violation(self, violation):
        """Format violation details"""
        feature = violation["feature"]
        value = violation["value"]
        threshold = violation["threshold"]
        explanation = violation["explanation"]

        # Determine violation direction
        direction = ""
        if isinstance(threshold, list) and len(threshold) == 2:
            if value < threshold[0]:
                direction = "too low"
            elif value > threshold[1]:
                direction = "too high"

        return {
            "feature": feature,
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "explanation": explanation
        }

    def _generate_error_analysis(self, error_prediction, route_info):
        """Generate error analysis"""
        if not error_prediction["predicted_type"]:
            return {
                "error_type": None,
                "probability": 0,
                "explanation": "No significant technical issues detected."
            }

        # Get predicted error type
        error_type = error_prediction["predicted_type"]
        probability = error_prediction["probabilities"][error_type]

        # Get error explanation from knowledge base
        knowledge = self.knowledge_base.get_knowledge(error_type)
        problems = knowledge.get("problems", [])
        causes = knowledge.get("causes", [])

        # Get route-specific knowledge
        route_type = route_info.get("route_type", "vertical")
        route_specific = self.knowledge_base.get_route_specific_knowledge(error_type, route_type)

        # Build explanation
        if problems and causes:
            explanation = f"{problems[0]}. This is typically caused by {causes[0].lower()}."
            if len(problems) > 1:
                explanation += f" Additionally, {problems[1].lower()}."
        else:
            explanation = "Could not find detailed explanation."

        # Add route-specific explanation
        if route_specific and "explanation" in route_specific:
            explanation += f" On {route_type} routes, {route_specific['explanation'].lower()}."

        return {
            "error_type": error_type,
            "probability": probability,
            "explanation": explanation,
            "common_problems": problems[:3],
            "common_causes": causes[:3]
        }

    def _generate_improvement_suggestions(self, error_prediction, violations, route_info):
        """Generate improvement suggestions"""
        if not error_prediction["predicted_type"] and not violations:
            return {
                "general_suggestions": ["Continue maintaining good technical movement, and consider trying more challenging routes."],
                "specific_suggestions": []
            }

        # Get error type
        error_type = error_prediction.get("predicted_type")

        # Get suggestions from knowledge base
        general_suggestions = []
        if error_type:
            knowledge = self.knowledge_base.get_knowledge(error_type)
            suggestions = knowledge.get("suggestions", [])
            general_suggestions = suggestions[:3]

        # Get route-specific suggestions
        route_type = route_info.get("route_type", "vertical")
        route_specific = {}
        if error_type:
            route_specific = self.knowledge_base.get_route_specific_knowledge(error_type, route_type)

        if route_specific and "suggestions" in route_specific:
            for suggestion in route_specific["suggestions"]:
                if suggestion not in general_suggestions:
                    general_suggestions.append(suggestion)

        # Get specific suggestions from violations
        specific_suggestions = []
        for violation in violations[:5]:  # Only use top 5 most severe violations
            suggestion = violation["suggestion"]
            if suggestion not in specific_suggestions:
                specific_suggestions.append({
                    "feature": violation["feature"],
                    "suggestion": suggestion
                })

        return {
            "general_suggestions": general_suggestions,
            "specific_suggestions": specific_suggestions
        }

    def _generate_training_recommendations(self, error_prediction, route_info):
        """Generate training recommendations"""
        recommendations = []

        # Get error type specific training recommendations
        if error_prediction.get("predicted_type"):
            error_type = error_prediction["predicted_type"]
            training_exercises = self.knowledge_base.get_training_exercises(error_type)

            for exercise in training_exercises[:2]:  # Choose top two training recommendations
                recommendations.append({
                    "name": exercise["name"],
                    "description": exercise["description"],
                    "benefit": exercise["benefit"]
                })

        # If not enough specific recommendations, add general training recommendations
        if len(recommendations) < 2:
            general_knowledge = self.knowledge_base.get_knowledge("general")
            principles = general_knowledge.get("technique_principles", [])

            if principles:
                recommendations.append({
                    "name": "Basic Technique Principle Practice",
                    "description": f"Focus on practicing this principle: {principles[0]}",
                    "benefit": "Improves overall technical foundation"
                })

        # Add progression recommendation based on difficulty
        difficulty = route_info.get("route_difficulty", "V3")
        climber_level = route_info.get("climber_level", "intermediate")

        if difficulty >= "V4" and climber_level == "intermediate":
            recommendations.append({
                "name": "Return to Fundamentals",
                "description": "Spend time on lower difficulty routes (V2-V3) perfecting technical basics",
                "benefit": "Solidifies technical foundation before progressing to higher difficulties"
            })

        return recommendations

    def _generate_claude_feedback(self, feedback, route_info):
        """Use Claude API to generate enhanced feedback"""
        try:
            # Prepare prompt to send to Claude
            prompt = self._prepare_claude_prompt(feedback, route_info)

            # Call Claude API
            response = self.claude_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                temperature=0.3,
                system="You are an expert climbing coach skilled at observing technical details in climbers and providing personalized professional feedback. Your advice should be specific, practical, and encouraging.",
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract content
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, list):
                    extracted_text = ""
                    for block in content:
                        if isinstance(block, dict) and 'text' in block:
                            extracted_text += block['text']
                        elif hasattr(block, 'text'):
                            extracted_text += block.text
                    return extracted_text
                elif isinstance(content, str):
                    return content
                else:
                    return str(content)

            return str(response.completion)

        except Exception as e:
            print(f"Claude API error: {e}")
            return None

    def _prepare_claude_prompt(self, feedback, route_info):
        """Prepare prompt to send to Claude API"""
        # Extract key information
        summary = feedback["summary"]
        error_analysis = feedback["error_analysis"]
        violations = feedback.get("technical_assessment", {}).get("detailed_assessments", [])
        suggestions = feedback["improvement_suggestions"]

        # Get the top 2 most important violations by category
        top_violations = []
        if violations:
            # Sort by violation count (higher is more severe)
            sorted_violations = sorted(violations, key=lambda x: x["violation_count"], reverse=True)
            top_violations = sorted_violations[:2]  # Get only top 2 categories

        # Format the top violations with specific values
        violations_text = ""
        for category in top_violations:
            violations_text += f"- {category['category'].replace('_', ' ').title()}: {category['level']} ({category['violation_count']} issues)\n"
            for detail in category.get("details", [])[:2]:
                # Include specific values and thresholds
                value = detail.get("value", "N/A")
                threshold = detail.get("threshold", "N/A")
                direction = detail.get("direction", "")

                violations_text += f"  * {detail['explanation']}\n"
                violations_text += f"    Current value: {value} ({direction}), optimal range: {threshold}\n"

        # Extract specific suggestions related to the top violations
        specific_suggestions = []
        for violation_category in top_violations:
            category_name = violation_category["category"]
            # Find suggestions related to this category
            for suggestion in suggestions.get("specific_suggestions", []):
                if suggestion["feature"] in [detail["feature"] for detail in violation_category.get("details", [])]:
                    specific_suggestions.append({
                        "category": category_name,
                        "suggestion": suggestion["suggestion"],
                        "feature": suggestion["feature"]
                    })

        # Format specific suggestions
        suggestions_text = ""
        for suggestion in specific_suggestions[:4]:  # Limit to top 4 most relevant
            suggestions_text += f"- {suggestion['suggestion']} (for {suggestion['feature']})\n"

        # Build prompt
        prompt = f"""
Based on video analysis of a climber on a {route_info.get('route_type', 'vertical')} route, I've detected the following technical issues:

Overall technical score: {summary['score']}/100 ({summary['performance_level']})
Main issue: {summary['main_issue']}

Top technical issues with specific measurements:
{violations_text}

Specific technical suggestions:
{suggestions_text}

Climber information:
- Experience level: {route_info.get('climber_level', 'intermediate')}
- Route difficulty: {route_info.get('route_difficulty', 'V3')}
- Route type: {route_info.get('route_type', 'vertical')}

{"Problem analysis: " + error_analysis.get("explanation", "") if error_analysis.get("error_type") else "No significant issues detected."}

Please provide a CONCISE climbing coach feedback focusing ONLY on the top 2 technical issues identified. Your response must:
1. Be brief and to the point (maximum 250 words total)
2. Focus ONLY on the 2 most important issues detected
3. Include the specific numerical values measured (like "{summary['score']}/100" or actual angles/measurements)
4. Provide only the most relevant, actionable advice for these 2 specific issues
5. Use direct, coach-like language as if talking to the climber during a session
6. Skip general training plans - focus only on immediate technique corrections

Structure your response in this format:
- Brief summary (2-3 sentences max)
- First issue with specific measured values
- Second issue with specific measured values
- 3-4 specific, actionable corrections

Do not go beyond this structure or add additional sections.
"""
        return prompt


# All function definitions
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import tempfile
import os
import time
from PIL import Image
from io import BytesIO
import anthropic
from typing import List, Dict, Any
import base64
from pyngrok import ngrok

# Reuse your existing classes: FrameData, VideoProcessor, FeatureExtractor,
# ClimbingRuleSystem, ClimbingKnowledgeBase, AnalysisEngine, FeedbackGenerator
# (These are already defined in your app.py)

def render_ui():
    """Main UI rendering function"""
    st.set_page_config(layout="wide", page_title="BetaBoost Climbing AI", page_icon="🧗‍♀️")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stVideo {
            max-height: 70vh !important;
        }
        .stVideo > video {
            max-height: 70vh !important;
        }
        .section-title {
            font-size: 0.95rem !important;
            font-weight: 600;
            color: #1f2937;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1rem;
        }
        .upload-section {
            display: flex;
            align-items: flex-end;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .upload-section > div {
            display: flex;
            flex-direction: column;
        }
        div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
            align-items: flex-end;
        }
        .stButton {
            margin-top: 0 !important;
        }
        .stSelectbox {
            margin-bottom: 0 !important;
        }
        /* Full height styling for file uploader */
        .stFileUploader > div:first-child {
            width: 100%;
            height: 100%;
            min-height: 60px;
        }
        /* Make dropdowns same height */
        .stSelectbox > div:first-child {
            min-height: 60px;
        }
        /* File uploader has fixed height */
        .stFileUploader {
            min-height: 60px;
        }
        /* Container for file info */
        .file-info-container {
            min-height: 55px;
            padding: 10px;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            margin-top: 5px;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
        }
        /* Empty file info container */
        .empty-container {
            min-height: 55px;
            margin-top: 5px;
            margin-bottom: 5px;
            visibility: hidden;
        }
        .score-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            height: 100%;
            min-height: 120px;
        }
        .score-value {
            font-size: 2.5rem;
            font-weight: 700;
        }
        .score-good {
            color: #10b981;
        }
        .score-medium {
            color: #f59e0b;
        }
        .score-poor {
            color: #ef4444;
        }
        .main-issue-card {
            background-color: #fee2e2;
            border-radius: 0.375rem;
            padding: 1rem;
            border: 1px solid #fecaca;
            min-height: 120px;
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        .main-issue-card::before {
            content: "🪨";
            position: absolute;
            font-size: 80px;
            right: -15px;
            bottom: -15px;
            opacity: 0.08;
            z-index: 0;
            pointer-events: none;
        }
        </style>
    """, unsafe_allow_html=True)

    # Custom title with larger BetaBoost text
    st.markdown("""
        <h1>
            <span style="font-size: 3rem; font-weight: 700; background-image: linear-gradient(to right, #3b82f6, #8b5cf6, #f97316); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                BetaBoost
            </span>
            <span style="margin-left: 0.5rem; font-size: 0.875rem; font-weight: 500; background-color: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 0.375rem; padding: 0.125rem 0.5rem;">
                Climbing AI
            </span>
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("<p style='margin-top: -0.5rem; color: #6b7280;'>Analyze your climbing technique with AI-powered feedback</p>", unsafe_allow_html=True)

    # Create two rows of grid layout
    # Row 1: Upload area, Route Type, Climber Level
    row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 1])
    
    # Row 2: File info, Difficulty, Start Analysis button
    row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])
    
    # Row 1 - Column 1: File upload area
    with row1_col1:
        st.markdown("##### Upload climbing video")
        uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"])

    # Row 1 - Column 2: Route Type dropdown
    with row1_col2:
        st.markdown("##### Route Type")
        route_type = st.selectbox("", ["Vertical", "Slab", "Overhang", "Roof"], label_visibility="collapsed")

    # Row 1 - Column 3: Climber Level dropdown
    with row1_col3:
        st.markdown("##### Climber Level")
        climber_level = st.selectbox("", ["Beginner", "Intermediate", "Advanced"], label_visibility="collapsed")

    # Row 2 - Column 1: File info display
    with row2_col1:
        # Display file info if uploaded, otherwise show empty space
        if uploaded_file:
            st.markdown(f"""
            <div class="file-info-container">
                <span style="margin-right: 10px;"><i class="fas fa-file-video"></i></span>
                <span style="font-weight: 500;">{uploaded_file.name}</span>
                <span style="margin-left: 10px; color: #6b7280; font-size: 0.875rem;">
                    {round(uploaded_file.size / (1024 * 1024), 2)} MB
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Empty container to maintain layout
            st.markdown("""<div class="empty-container"></div>""", unsafe_allow_html=True)

    # Row 2 - Column 2: Difficulty dropdown
    with row2_col2:
        st.markdown("##### Difficulty")
        route_difficulty = st.selectbox("", ["V1", "V2", "V3", "V4", "V5", "V6"], label_visibility="collapsed")

    # Row 2 - Column 3: Analysis button (always visible)
    with row2_col3:
        st.markdown("##### &nbsp;")  # Empty header for alignment
        analyze_button = st.button("Start Analysis", use_container_width=True)
        if analyze_button and uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.route_type = route_type
            st.session_state.route_difficulty = route_difficulty
            st.session_state.climber_level = climber_level
            st.session_state.should_analyze = True

    # Horizontal divider
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

    # Main interface divided into left and right sections
    left_col, right_col = st.columns([1.5, 2])

    # Left side - Video and trajectory plot
    with left_col:
        # Video playback area
        st.markdown("<div class='section-title'>Video Analysis</div>", unsafe_allow_html=True)
        video_container = st.container()
        with video_container:
            video_placeholder = st.empty()
            controls_placeholder = st.empty()  # For video controls
        
        # Center of mass trajectory - Below video
        st.markdown("<div class='section-title'>Horizontal Center of Mass Trajectory</div>", unsafe_allow_html=True)
        gravity_center_plot = st.empty()

    # Right side - Analysis results
    with right_col:
        # AI analysis area
        st.markdown("<div class='section-title'>AI Technical Analysis</div>", unsafe_allow_html=True)
        
        # Create 2-column layout for score and main issue
        score_col, issue_col = st.columns([1, 2])
        
        with score_col:
            # Use custom HTML for color-coded score
            score_placeholder = st.empty()
        
        with issue_col:
            main_issue = st.empty()
        
        # Key findings
        findings_container = st.container()
        with findings_container:
            findings_header = st.markdown("<div class='section-title'>Key Findings</div>", unsafe_allow_html=True)
            findings_list = st.empty()
        
        # Detailed feedback
        detail_expander = st.expander("Detailed Technical Feedback", expanded=True)
        with detail_expander:
            detailed_feedback = st.empty()
        
        # Training recommendations
        training_expander = st.expander("Training Recommendations", expanded=False)
        with training_expander:
            training_recs = st.empty()

    return {
        'video_placeholder': video_placeholder,
        'controls_placeholder': controls_placeholder,
        'score_placeholder': score_placeholder,
        'main_issue': main_issue,
        'findings_list': findings_list,
        'detailed_feedback': detailed_feedback,
        'training_recs': training_recs,
        'gravity_center_plot': gravity_center_plot
    }

    # Custom title with larger BetaBoost text
    st.markdown("""
        <h1>
            <span style="font-size: 3rem; font-weight: 700; background-image: linear-gradient(to right, #3b82f6, #8b5cf6, #f97316); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                BetaBoost
            </span>
            <span style="margin-left: 0.5rem; font-size: 0.875rem; font-weight: 500; background-color: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 0.375rem; padding: 0.125rem 0.5rem;">
                Climbing AI
            </span>
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("<p style='margin-top: -0.5rem; color: #6b7280;'>Analyze your climbing technique with AI-powered feedback</p>", unsafe_allow_html=True)



def create_skeleton_video(frames_data, output_path="skeleton_video.mp4"):
    """Create a video with skeleton overlay from frame data"""
    if not frames_data:
        return None
    
    # Get dimensions from the first frame
    first_frame = frames_data[0].image
    h, w, _ = first_frame.shape
    
    # Use a widely supported codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # More compatible codec
    fps = 30  # You can adjust this based on original video
    
    # Ensure color conversion and video writer initialization 
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)
    
    # Process each frame
    for frame_data in frames_data:
        # Visualize frame with skeleton
        frame_with_skeleton = visualize_frame(frame_data)
        
        # Convert RGB to BGR (OpenCV uses BGR) and ensure uint8
        frame_bgr = cv2.cvtColor(frame_with_skeleton.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Write to video file
        out.write(frame_bgr)
    
    # Release video writer
    out.release()
    
    return output_path



def get_video_html(video_path):
    """Generate HTML for video player with custom controls and autoplay"""
    if not os.path.exists(video_path):
        return f"<p>Error: Video file not found at {video_path}</p>"

    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    video_base64 = base64.b64encode(video_bytes).decode()

    # Custom HTML for video with autoplay and loop
    html = f"""
    <video id="climbing-video" width="100%" height="auto" controls autoplay loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <script>
        // Auto-resize video to fit container while maintaining aspect ratio
        document.getElementById('climbing-video').style.maxHeight = '65vh';
    </script>
    """
    return html

def visualize_frame(frame_data, show_skeleton=True, show_com=True, trail_frames=10):
    """Create a visualization of a frame with skeleton and center of mass with trail effect"""
    # Get the image
    image = frame_data.image.copy()
    h, w, _ = image.shape

    # Draw skeleton
    if show_skeleton and frame_data.landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # Define connection line style (use white for all)
        connection_style = mp_drawing.DrawingSpec(
            color=(255, 255, 255),  # White
            thickness=2,
            circle_radius=1  # Smaller endpoints for connections
        )
        
        # Define left/right joint point styles
        left_point_style = mp_drawing.DrawingSpec(
            color=(255, 100, 100),  # Blue color (BGR)
            thickness=2,
            circle_radius=4  # Slightly larger joint points
        )
        right_point_style = mp_drawing.DrawingSpec(
            color=(100, 255, 100),  # Green color (BGR)
            thickness=2,
            circle_radius=4  # Slightly larger joint points
        )

        # Create joint point style mapping
        landmark_style = {}
        for idx in range(33):  # MediaPipe Pose has 33 joint points
            # Left side joints (including left torso points)
            if idx in [11, 13, 15, 23, 25, 27] or idx in [1, 3, 7]:  # Left limbs + left torso
                landmark_style[idx] = left_point_style
            # Right side joints (including right torso points)
            elif idx in [12, 14, 16, 24, 26, 28] or idx in [2, 4, 8]:  # Right limbs + right torso
                landmark_style[idx] = right_point_style
            # Center line joints (nose, mouth, neck center, etc.) remain white
            else:
                landmark_style[idx] = connection_style

        # Draw skeleton
        mp_drawing.draw_landmarks(
            image,
            frame_data.landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

    # Draw center of mass with trail effect
    if show_com and hasattr(frame_data, 'frame_idx'):
        # Calculate current frame's COM
        keypoints = frame_data.keypoints
        valid_points = [kp for kp in keypoints.values() if kp["visibility"] > 0.5]

        if valid_points:
            com_x = sum(kp["x"] for kp in valid_points) / len(valid_points)
            com_y = sum(kp["y"] for kp in valid_points) / len(valid_points)
            
            # Convert normalized coordinates to pixel coordinates
            current_com = (int(com_x * w), int(com_y * h))
            
            # Store COM position in session state if not exists
            if 'com_trail' not in st.session_state:
                st.session_state.com_trail = []
            
            # Add current COM to trail
            st.session_state.com_trail.append(current_com)
            
            # Keep only recent positions for trail effect
            if len(st.session_state.com_trail) > trail_frames:
                st.session_state.com_trail.pop(0)
            
            # Draw trail with fading effect
            for i, pos in enumerate(st.session_state.com_trail[:-1]):
                # Calculate alpha for fading effect (0.2 to 0.8)
                alpha = 0.2 + (i / len(st.session_state.com_trail)) * 0.6
                # Use red color (BGR format)
                cv2.circle(image, pos, 3, (0, 0, 255, int(alpha * 255)), -1)
            
            # Draw current COM position (larger and more visible)
            cv2.circle(image, current_com, 6, (0, 0, 255), -1)  # Red current point, larger

    return image

def generate_com_trajectory_plot(frames_data):
    """Generate center of mass trajectory plot"""
    if not frames_data:
        return None

    # Extract COM positions from all frames
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

    # Create the plot with a larger figure size
    fig, ax = plt.subplots(figsize=(10, 4))

    # Extract x-coordinates for horizontal COM motion
    x_coords = [p[0] for p in com_positions]

    # Create a time axis
    if timestamps:
        time_points = np.array(timestamps)
        time_normalized = (time_points - min(time_points)) / (max(time_points) - min(time_points)) \
                         if max(time_points) > min(time_points) else time_points
    else:
        time_normalized = np.linspace(0, 1, len(x_coords))

    # Plot horizontal COM trajectory with improved styling
    ax.plot(time_normalized, x_coords, 'g-', linewidth=3)

    # Add markers for start and end
    ax.scatter(time_normalized[0], x_coords[0], color='blue', s=120, label='Start')
    ax.scatter(time_normalized[-1], x_coords[-1], color='red', s=120, label='End')

    # Add markers for each 25% of the climb
    quartiles = [int(len(time_normalized) * q) for q in [0.25, 0.5, 0.75]]
    for i, q in enumerate(quartiles):
        if q < len(time_normalized):
            ax.scatter(time_normalized[q], x_coords[q], color='purple', s=80,
                      label=f"{(i+1)*25}%" if i == 0 else None)

    # Set labels and title
    ax.set_xlabel('Climb Progress', fontsize=12)
    ax.set_ylabel('Horizontal Position', fontsize=12)
    ax.set_title('Horizontal Center of Mass Trajectory', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add a horizontal reference line at the middle
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Make sure the figure has tight layout
    fig.tight_layout()

    return fig

def update_ui_with_results(ui_elements, frames_data, features, analysis_results, feedback):
    # Create skeleton video
    with st.spinner("Creating video with skeleton overlay..."):
        video_path = create_skeleton_video(frames_data)

    # Display video with skeleton
    if video_path:
        try:
            # Prefer using Streamlit's video method
            ui_elements['video_placeholder'].video(video_path)
        except Exception as e:
            # If failed, try custom HTML
            video_html = get_video_html(video_path)
            ui_elements['video_placeholder'].markdown(video_html, unsafe_allow_html=True)

    # Update technical score and main issue
    summary = feedback["summary"]
    score = summary['score']
    
    # Determine color class based on score
    score_class = "score-poor"
    if score >= 80:
        score_class = "score-good"
    elif score >= 60:
        score_class = "score-medium"
    
    # Use custom HTML to display score
    score_html = f"""
    <div class="score-card">
        <div class="score-value {score_class}">{score}</div>
        <div style="font-size: 0.875rem; color: #6b7280;">/100</div>
    </div>
    """
    ui_elements['score_placeholder'].markdown(score_html, unsafe_allow_html=True)
    
    # Display main issue - with rock emoji in background
    ui_elements['main_issue'].markdown(f"""
    <div class="main-issue-card">
        <div style="display: flex; align-items: flex-start; position: relative; z-index: 1;">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="#f87171" style="width: 1.25rem; height: 1.25rem; margin-right: 0.5rem; flex-shrink: 0;">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
            <div>
                <div style="font-weight: 600; color: #991b1b;">Main Issue:</div>
                <div style="color: #b91c1c;">{summary['main_issue']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Update findings
    findings_md = ""
    if feedback.get("error_analysis") and feedback["error_analysis"].get("common_problems"):
        problems = feedback["error_analysis"]["common_problems"]
        for problem in problems:
            findings_md += f"<li style='display: flex; align-items: flex-start; margin-bottom: 0.75rem;'><span style='color: #f97316; margin-right: 0.5rem;'>•</span> <span>{problem}</span></li>"
    else:
        findings_md = "<li style='display: flex; align-items: flex-start;'><span style='color: #f97316; margin-right: 0.5rem;'>•</span> <span>No significant issues detected</span></li>"

    ui_elements['findings_list'].markdown(f"<ul style='list-style-type: none; padding-left: 0;'>{findings_md}</ul>", unsafe_allow_html=True)

    # Update detailed feedback
    detailed_md = ""
    if feedback.get("claude_enhanced"):
        detailed_md = feedback["claude_enhanced"]
    elif feedback.get("error_analysis") and feedback["error_analysis"].get("explanation"):
        detailed_md = feedback["error_analysis"]["explanation"]

        # Add improvement suggestions
        detailed_md += "\n\n**Improvement Suggestions:**\n"
        suggestions = feedback.get("improvement_suggestions", {})
        for suggestion in suggestions.get("general_suggestions", []):
            detailed_md += f"- {suggestion}\n"

    ui_elements['detailed_feedback'].markdown(detailed_md)

    # Update training recommendations
    training_md = ""
    for rec in feedback.get("training_recommendations", []):
        training_md += f"""
        <div style="margin-bottom: 1rem; padding: 0.75rem; background-color: #f3f4f6; border-radius: 0.375rem;">
            <p style="font-weight: 600; color: #1f2937; margin-bottom: 0.5rem;">{rec['name']}</p>
            <p style="color: #4b5563; margin-bottom: 0.5rem;">{rec['description']}</p>
            <p style="font-style: italic; color: #6b7280; font-size: 0.875rem;">Benefit: {rec['benefit']}</p>
        </div>
        """

    ui_elements['training_recs'].markdown(training_md, unsafe_allow_html=True)

    # Generate and display center of mass trajectory
    com_trajectory_plot = generate_com_trajectory_plot(frames_data)
    if com_trajectory_plot:
        ui_elements['gravity_center_plot'].pyplot(com_trajectory_plot)

def map_route_type(ui_route_type):
    """Map UI route type to API route type"""
    mapping = {
        "Vertical": "vertical",
        "Slab": "slab",
        "Overhang": "overhang",
        "Roof": "Roof"
    }
    return mapping.get(ui_route_type, "vertical")

def map_climber_level(ui_climber_level):
    """Map UI climber level to API climber level"""
    mapping = {
        "Beginner": "beginner",
        "Intermediate": "intermediate",
        "Advanced": "advanced"
    }
    return mapping.get(ui_climber_level, "intermediate")

# Main application
def main():
    # Initialize session state variables if they don't exist
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'should_analyze' not in st.session_state:
        st.session_state.should_analyze = False
    if 'frames_data' not in st.session_state:
        st.session_state.frames_data = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None

    # Render the UI
    ui_elements = render_ui()

    # Process video if analysis is requested
    if st.session_state.should_analyze and not st.session_state.processed:
        with st.spinner("Processing video..."):
            # 1. Initialize components
            video_processor = VideoProcessor()
            feature_extractor = FeatureExtractor()
            rule_system = ClimbingRuleSystem()
            knowledge_base = ClimbingKnowledgeBase()
            analysis_engine = AnalysisEngine(rule_system)
            feedback_generator = FeedbackGenerator(knowledge_base)

            # 2. Process video
            frames_data = video_processor.process_video(st.session_state.uploaded_file)
            st.session_state.frames_data = frames_data

            # 3. Extract features
            if frames_data:
                features = feature_extractor.extract_features(frames_data)
                st.session_state.features = features

                # 4. Prepare route info
                route_info = {
                    "route_type": map_route_type(st.session_state.route_type),
                    "route_difficulty": st.session_state.route_difficulty,
                    "climber_level": map_climber_level(st.session_state.climber_level)
                }

                # 5. Analyze features
                analysis_results = analysis_engine.analyze(features, route_info)
                st.session_state.analysis_results = analysis_results

                # 6. Generate feedback
                feedback = feedback_generator.generate_feedback(analysis_results, route_info)
                st.session_state.feedback = feedback

                # Mark as processed
                st.session_state.processed = True
                st.session_state.should_analyze = False

    # Update UI with results if processing is complete
    if st.session_state.processed:
        update_ui_with_results(
            ui_elements,
            st.session_state.frames_data,
            st.session_state.features,
            st.session_state.analysis_results,
            st.session_state.feedback
        )

# Application entry point
if __name__ == "__main__":
    # Set API key if available
    if os.environ.get("ANTHROPIC_API_KEY"):
        pass  # Use the environment variable
    else:
        # You can set a default API key for testing, but this should be removed in production
        os.environ["ANTHROPIC_API_KEY"] = "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"

    # Run the main application
    main()

    # Create ngrok tunnel if running in Colab
    # public_url = ngrok.connect(8501)
    # st.write(f"Public URL: {public_url}")
