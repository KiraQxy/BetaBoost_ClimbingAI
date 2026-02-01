"""Feature extraction module - computes body/pose features from frame data."""

import math
import random
import numpy as np
import streamlit as st


class FeatureExtractor:
    """Extracts climbing-relevant features from processed frame data."""

    def __init__(self):
        pass

    def extract_features(self, frames_data):
        """Extract features from frame data."""
        if len(frames_data) < 5:
            st.error("Not enough frames to extract features. Need at least 5 frames.")
            return None

        if len(frames_data) >= 20:
            selected_frames = sorted(random.sample(frames_data, 5), key=lambda x: x.frame_idx)
        else:
            segments = np.array_split(range(len(frames_data)), 5)
            selected_indices = [random.choice(segment.tolist()) for segment in segments]
            selected_frames = [frames_data[i] for i in sorted(selected_indices)]

        frame_features = []
        for i, frame in enumerate(selected_frames):
            features = self._calculate_body_features(frame.keypoints)
            features['frame_index'] = frame.frame_idx
            features['frame_relative_position'] = frame.frame_idx / max(f.frame_idx for f in frames_data)
            frame_features.append(features)

        combined_features = {}
        for i, frame_feat in enumerate(frame_features):
            for feat_name, feat_value in frame_feat.items():
                combined_features[f"frame{i+1}_{feat_name}"] = feat_value

        self._add_inter_frame_features(combined_features, frame_features)
        self._add_dynamic_features(combined_features)

        return combined_features

    def _calculate_body_features(self, keypoints_dict):
        """Calculate body features including joint angles, posture, balance, etc."""
        features = {}
        body_points = {}

        for joint_idx, joint_data in keypoints_dict.items():
            if joint_data["visibility"] > 0.5:
                body_points[int(joint_idx)] = {
                    'x': joint_data['x'],
                    'y': joint_data['y'],
                    'z': joint_data['z']
                }

        # Elbow angles
        if all(k in body_points for k in [11, 13, 15]):
            features['left_elbow_angle'] = self._calculate_angle(
                body_points[11], body_points[13], body_points[15]
            )
        if all(k in body_points for k in [12, 14, 16]):
            features['right_elbow_angle'] = self._calculate_angle(
                body_points[12], body_points[14], body_points[16]
            )

        # Shoulder angles
        if all(k in body_points for k in [13, 11, 23]):
            features['left_shoulder_angle'] = self._calculate_angle(
                body_points[13], body_points[11], body_points[23]
            )
        if all(k in body_points for k in [14, 12, 24]):
            features['right_shoulder_angle'] = self._calculate_angle(
                body_points[14], body_points[12], body_points[24]
            )

        # Hip angles
        if all(k in body_points for k in [11, 23, 25]):
            features['left_hip_angle'] = self._calculate_angle(
                body_points[11], body_points[23], body_points[25]
            )
        if all(k in body_points for k in [12, 24, 26]):
            features['right_hip_angle'] = self._calculate_angle(
                body_points[12], body_points[24], body_points[26]
            )

        # Knee angles
        if all(k in body_points for k in [23, 25, 27]):
            features['left_knee_angle'] = self._calculate_angle(
                body_points[23], body_points[25], body_points[27]
            )
        if all(k in body_points for k in [24, 26, 28]):
            features['right_knee_angle'] = self._calculate_angle(
                body_points[24], body_points[26], body_points[28]
            )

        # Trunk posture
        if all(k in body_points for k in [11, 12, 23, 24]):
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
            vertical = {'x': 0, 'y': 1, 'z': 0}
            features['trunk_tilt_angle'] = self._calculate_angle(vertical, trunk_bottom, trunk_top)
            features['trunk_length'] = math.sqrt(
                (trunk_top['x'] - trunk_bottom['x'])**2 +
                (trunk_top['y'] - trunk_bottom['y'])**2 +
                (trunk_top['z'] - trunk_bottom['z'])**2
            )

            shoulder_vector = {
                'x': body_points[12]['x'] - body_points[11]['x'],
                'y': 0,
                'z': body_points[12]['z'] - body_points[11]['z']
            }
            hip_vector = {
                'x': body_points[24]['x'] - body_points[23]['x'],
                'y': 0,
                'z': body_points[24]['z'] - body_points[23]['z']
            }
            origin = {'x': 0, 'y': 0, 'z': 0}
            shoulder_point = {'x': shoulder_vector['x'], 'y': 0, 'z': shoulder_vector['z']}
            hip_point = {'x': hip_vector['x'], 'y': 0, 'z': hip_vector['z']}
            features['hip_rotation_angle'] = self._calculate_angle(shoulder_point, origin, hip_point)

        # Limb extensions
        if all(k in body_points for k in [11, 15]):
            features['left_arm_extension'] = math.sqrt(
                (body_points[15]['x'] - body_points[11]['x'])**2 +
                (body_points[15]['y'] - body_points[11]['y'])**2 +
                (body_points[15]['z'] - body_points[11]['z'])**2
            )
        if all(k in body_points for k in [12, 16]):
            features['right_arm_extension'] = math.sqrt(
                (body_points[16]['x'] - body_points[12]['x'])**2 +
                (body_points[16]['y'] - body_points[12]['y'])**2 +
                (body_points[16]['z'] - body_points[12]['z'])**2
            )
        if all(k in body_points for k in [23, 27]):
            features['left_leg_extension'] = math.sqrt(
                (body_points[27]['x'] - body_points[23]['x'])**2 +
                (body_points[27]['y'] - body_points[23]['y'])**2 +
                (body_points[27]['z'] - body_points[23]['z'])**2
            )
        if all(k in body_points for k in [24, 28]):
            features['right_leg_extension'] = math.sqrt(
                (body_points[28]['x'] - body_points[24]['x'])**2 +
                (body_points[28]['y'] - body_points[24]['y'])**2 +
                (body_points[28]['z'] - body_points[24]['z'])**2
            )

        # Center of mass
        valid_points = [p for _, p in body_points.items()]
        if valid_points:
            features['center_of_mass_x'] = sum(p['x'] for p in valid_points) / len(valid_points)
            features['center_of_mass_y'] = sum(p['y'] for p in valid_points) / len(valid_points)
            features['center_of_mass_z'] = sum(p['z'] for p in valid_points) / len(valid_points)

        # Lateral balance
        if all(k in body_points for k in [27, 28]) and 'center_of_mass_x' in features:
            support_center_x = (body_points[27]['x'] + body_points[28]['x']) / 2
            features['lateral_balance'] = features['center_of_mass_x'] - support_center_x

        # Body stability
        angles = [v for k, v in features.items() if k.endswith('_angle') and v is not None]
        if angles:
            features['body_stability'] = np.std(angles)

        return features

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (p2 is the vertex)."""
        if not all([p1, p2, p3]):
            return None

        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return None

        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = min(1.0, max(-1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    def _add_inter_frame_features(self, combined_features, frame_features):
        """Add inter-frame features like joint angle changes and COM movement."""
        for joint in ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle',
                     'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle',
                     'left_hip_angle', 'right_hip_angle']:
            for i in range(4):
                prev_feat = frame_features[i].get(joint)
                next_feat = frame_features[i+1].get(joint)
                if prev_feat is not None and next_feat is not None:
                    combined_features[f"change_{i+1}_to_{i+2}_{joint}"] = next_feat - prev_feat

        for coord in ['x', 'y', 'z']:
            com_positions = []
            for i in range(5):
                key = f"center_of_mass_{coord}"
                if key in frame_features[i]:
                    com_positions.append(frame_features[i][key])

            if len(com_positions) == 5:
                total_distance = sum(abs(com_positions[i] - com_positions[i-1]) for i in range(1, 5))
                combined_features[f"total_com_{coord}_movement"] = total_distance
                direct_distance = abs(com_positions[4] - com_positions[0])
                combined_features[f"direct_com_{coord}_movement"] = direct_distance
                if total_distance > 0:
                    combined_features[f"com_{coord}_efficiency"] = direct_distance / total_distance

        trunk_angles = [frame_features[i]['trunk_tilt_angle'] for i in range(5)
                        if 'trunk_tilt_angle' in frame_features[i]]
        if len(trunk_angles) == 5:
            combined_features["trunk_angle_variation"] = np.std(trunk_angles)

        lateral_balances = [frame_features[i]['lateral_balance'] for i in range(5)
                           if 'lateral_balance' in frame_features[i]]
        if len(lateral_balances) == 5:
            combined_features["lateral_balance_variation"] = np.std(lateral_balances)

    def _add_dynamic_features(self, combined_features):
        """Add advanced dynamic features."""
        frame_features = ['hip_rotation_angle', 'trunk_tilt_angle', 'body_stability',
                         'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z',
                         'lateral_balance', 'trunk_length']

        for feature in frame_features:
            for i in range(1, 5):
                col1, col2 = f'frame{i}_{feature}', f'frame{i+1}_{feature}'
                idx1, idx2 = f'frame{i}_frame_index', f'frame{i+1}_frame_index'
                if all(col in combined_features for col in [col1, col2, idx1, idx2]):
                    combined_features[f'change_{i}_to_{i+1}_{feature}'] = (
                        combined_features[col2] - combined_features[col1]
                    )
                    frame_distance = combined_features[idx2] - combined_features[idx1]
                    if frame_distance > 0:
                        combined_features[f'rate_{i}_to_{i+1}_{feature}'] = (
                            (combined_features[col2] - combined_features[col1]) / frame_distance
                        )

        limb_features = ['left_arm_extension', 'right_arm_extension',
                         'left_leg_extension', 'right_leg_extension',
                         'left_elbow_angle', 'right_elbow_angle',
                         'left_shoulder_angle', 'right_shoulder_angle',
                         'left_knee_angle', 'right_knee_angle',
                         'left_hip_angle', 'right_hip_angle']

        for feature in limb_features:
            for i in range(1, 5):
                col1, col2 = f'frame{i}_{feature}', f'frame{i+1}_{feature}'
                idx1, idx2 = f'frame{i}_frame_index', f'frame{i+1}_frame_index'
                if all(col in combined_features for col in [col1, col2, idx1, idx2]):
                    combined_features[f'change_{i}_to_{i+1}_{feature}'] = (
                        combined_features[col2] - combined_features[col1]
                    )
                    frame_distance = combined_features[idx2] - combined_features[idx1]
                    if frame_distance > 0:
                        combined_features[f'rate_{i}_to_{i+1}_{feature}'] = (
                            (combined_features[col2] - combined_features[col1]) / frame_distance
                        )

        important_features = ['hip_rotation_angle', 'trunk_tilt_angle', 'body_stability',
                             'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z',
                             'lateral_balance']

        for feature in important_features:
            weighted_values = weights = 0
            for i in range(1, 6):
                col, pos_col = f'frame{i}_{feature}', f'frame{i}_frame_relative_position'
                if col in combined_features and pos_col in combined_features:
                    weighted_values += combined_features[col] * combined_features[pos_col]
                    weights += combined_features[pos_col]
            if weights > 0:
                combined_features[f'{feature}_weighted_avg'] = weighted_values / weights

        if 'frame1_frame_relative_position' in combined_features and 'frame5_frame_relative_position' in combined_features:
            combined_features['sequence_coverage'] = (
                combined_features['frame5_frame_relative_position'] -
                combined_features['frame1_frame_relative_position']
            )

        for i in range(1, 6):
            pos_col = f'frame{i}_frame_relative_position'
            if pos_col in combined_features:
                pos = combined_features[pos_col]
                combined_features[f'frame{i}_early_phase'] = 1 if pos <= 0.33 else 0
                combined_features[f'frame{i}_mid_phase'] = 1 if 0.33 < pos <= 0.67 else 0
                combined_features[f'frame{i}_late_phase'] = 1 if pos > 0.67 else 0

        for axis in ['x', 'y', 'z']:
            positions, indices = [], []
            for i in range(1, 6):
                pos_col, idx_col = f'frame{i}_center_of_mass_{axis}', f'frame{i}_frame_index'
                if pos_col in combined_features and idx_col in combined_features:
                    positions.append(combined_features[pos_col])
                    indices.append(combined_features[idx_col])
            if len(positions) >= 3 and len(indices) >= 3:
                try:
                    slope, intercept = np.polyfit(indices, positions, 1)
                    combined_features[f'com_{axis}_trajectory_slope'] = slope
                    linear_pred = slope * np.array(indices) + intercept
                    combined_features[f'com_{axis}_nonlinearity'] = np.mean((np.array(positions) - linear_pred)**2)
                except Exception:
                    pass

        key_features = ['hip_rotation_angle', 'trunk_tilt_angle', 'body_stability',
                        'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z']
        for feature in key_features:
            first_col, last_col = f'frame1_{feature}', f'frame5_{feature}'
            first_idx, last_idx = 'frame1_frame_index', 'frame5_frame_index'
            if all(col in combined_features for col in [first_col, last_col, first_idx, last_idx]):
                combined_features[f'total_change_{feature}'] = (
                    combined_features[last_col] - combined_features[first_col]
                )
                idx_diff = combined_features[last_idx] - combined_features[first_idx]
                if idx_diff > 0:
                    combined_features[f'total_rate_{feature}'] = (
                        (combined_features[last_col] - combined_features[first_col]) / idx_diff
                    )

        frame_gaps = []
        for i in range(1, 5):
            idx1, idx2 = f'frame{i}_frame_index', f'frame{i+1}_frame_index'
            if idx1 in combined_features and idx2 in combined_features:
                frame_gaps.append(combined_features[idx2] - combined_features[idx1])
        if frame_gaps:
            combined_features['avg_frame_gap'] = np.mean(frame_gaps)
            combined_features['std_frame_gap'] = np.std(frame_gaps)
            if combined_features['avg_frame_gap'] > 0:
                combined_features['cv_frame_gap'] = (
                    combined_features['std_frame_gap'] / combined_features['avg_frame_gap']
                )
