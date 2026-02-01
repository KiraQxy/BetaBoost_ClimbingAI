"""Climbing knowledge base - expert knowledge for feedback generation."""

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


