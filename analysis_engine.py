"""Analysis engine - orchestrates rule checking and score calculation."""

from rule_system import ClimbingRuleSystem


class AnalysisEngine:
    """Analyzes features using the rule system and produces analysis results."""

    def __init__(self, rule_system: ClimbingRuleSystem):
        self.rule_system = rule_system

    def analyze(self, features, route_info):
        """Analyze feature data using the rule system."""
        violations = self.rule_system.check_rules(features)
        score = self.rule_system.calculate_score(violations)
        error_prediction = self.rule_system.predict_error_type(features)

        return {
            "violations": violations,
            "score": score,
            "error_prediction": error_prediction,
            "features": features,
            "route_info": route_info
        }
