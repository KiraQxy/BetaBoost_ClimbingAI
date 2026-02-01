"""Climbing rule system - evaluates features against climbing technique rules."""


class ClimbingRuleSystem:
    """Evaluates extracted features against climbing technique rules."""

    def __init__(self):
        self.general_rules = self.initialize_general_rules()
        self.body_position_rules = self.initialize_body_position_rules()
        self.foot_placement_rules = self.initialize_foot_placement_rules()
        self.weight_distribution_rules = self.initialize_weight_distribution_rules()
        self.grip_technique_rules = self.initialize_grip_technique_rules()
        self.balance_issue_rules = self.initialize_balance_issue_rules()
        self.arm_extension_rules = self.initialize_arm_extension_rules()
        self.insufficient_core_rules = self.initialize_insufficient_core_rules()

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
        """Initialize general rules (overall movement quality)."""
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
        """Initialize body position rules (focused on posture and positioning)."""
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
        """Initialize foot placement rules (focused on foot position and technique)."""
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
        """Initialize weight distribution rules (balance between limbs)."""
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
        """Initialize grip technique rules (hand positions and engagement)."""
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
        """Initialize balance issue rules (stability and control)."""
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
        """Initialize arm extension rules (arm positioning and efficiency)."""
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
        """Initialize insufficient core strength rules (core engagement and stability)."""
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
        """Calculate overall score based on rule violations."""
        if not violations:
            return 100.0

        total_deduction = 0.0
        total_weight = 0.0

        for violation in violations:
            importance = violation["importance"]
            deviation = violation["relative_deviation"]
            deduction = importance * deviation * 8
            total_deduction += deduction
            total_weight += importance

        if total_weight > 0:
            normalized_deduction = total_deduction / total_weight * 10
        else:
            normalized_deduction = total_deduction

        base_score = 50.0
        final_score = max(base_score, 100.0 - normalized_deduction)
        final_score = min(100.0, final_score)

        return round(final_score, 1)

    def check_rules(self, features, category=None):
        """Check if features violate rules."""
        violations = []

        if category and category in self.rule_categories:
            categories_to_check = {category: self.rule_categories[category]}
        else:
            categories_to_check = self.rule_categories

        for category_name, rules in categories_to_check.items():
            for feature_name, rule in rules.items():
                if feature_name in features:
                    feature_value = features[feature_name]
                    threshold = rule["threshold"]
                    optimal_value = rule.get("optimal_value", None)

                    in_valid_range = True
                    if isinstance(threshold, list) and len(threshold) == 2:
                        if feature_value < threshold[0] or feature_value > threshold[1]:
                            in_valid_range = False

                    significant_deviation = False
                    deviation_ratio = 0.0
                    deviation_direction = None

                    if optimal_value is not None:
                        deviation = abs(feature_value - optimal_value)
                        if isinstance(threshold, list) and len(threshold) == 2:
                            threshold_range = threshold[1] - threshold[0]
                            if threshold_range > 0:
                                deviation_ratio = deviation / threshold_range
                            if deviation_ratio > 0.3:
                                significant_deviation = True
                                deviation_direction = (
                                    "below optimal value" if feature_value < optimal_value
                                    else "above optimal value"
                                )

                    violated = not in_valid_range or significant_deviation

                    if not in_valid_range:
                        if isinstance(threshold, list) and len(threshold) == 2:
                            deviation = threshold[0] - feature_value if feature_value < threshold[0] else feature_value - threshold[1]
                            threshold_range = threshold[1] - threshold[0]
                            relative_deviation = deviation / threshold_range if threshold_range > 0 else deviation
                        else:
                            relative_deviation = 0
                    else:
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
                            "relative_deviation": min(1.0, relative_deviation),
                            "deviation_direction": deviation_direction
                        })

        violations.sort(key=lambda x: x["importance"] * x["relative_deviation"], reverse=True)
        return violations

    def predict_error_type(self, features):
        """Predict main error type from features."""
        category_violations = {}
        category_scores = {}
        error_categories = [cat for cat in self.rule_categories.keys() if cat != "general"]

        for category in error_categories:
            violations = self.check_rules(features, category)
            category_violations[category] = violations
            category_scores[category] = self.calculate_score(violations) if violations else 100.0

        if category_scores:
            predicted_type = min(category_scores.items(), key=lambda x: x[1])[0]
            total_inverse_score = sum(max(0, 100 - score) for score in category_scores.values())
            error_probabilities = {}

            if total_inverse_score > 0:
                for category, score in category_scores.items():
                    inverse_score = max(0, 100 - score)
                    adjusted_inverse = inverse_score ** 0.9
                    error_probabilities[category] = adjusted_inverse / total_inverse_score
            else:
                for category in category_scores:
                    error_probabilities[category] = 1.0 / len(category_scores)

            return {
                "predicted_type": predicted_type,
                "probabilities": error_probabilities,
                "violations": category_violations[predicted_type]
            }

        return {
            "predicted_type": None,
            "probabilities": {},
            "violations": []
        }
