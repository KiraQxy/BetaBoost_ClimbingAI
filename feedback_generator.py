"""Feedback generator - produces human-readable feedback from analysis results."""

import os
import anthropic

from knowledge_base import ClimbingKnowledgeBase


class FeedbackGenerator:
    """Generates layered feedback from analysis results using the knowledge base."""

    def __init__(self, knowledge_base: ClimbingKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.claude_client = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate_feedback(self, analysis_results, route_info):
        """Generate complete layered feedback."""
        violations = analysis_results["violations"]
        score = analysis_results["score"]
        error_prediction = analysis_results["error_prediction"]

        feedback = {
            "summary": self._generate_summary(score, error_prediction, violations),
            "technical_assessment": self._generate_technical_assessment(violations, score),
            "error_analysis": self._generate_error_analysis(error_prediction, route_info),
            "improvement_suggestions": self._generate_improvement_suggestions(
                error_prediction, violations, route_info
            ),
            "training_recommendations": self._generate_training_recommendations(
                error_prediction, route_info
            )
        }

        if self.claude_client:
            claude_feedback = self._generate_claude_feedback(feedback, route_info)
            feedback["claude_enhanced"] = claude_feedback

        return feedback

    def _generate_summary(self, score, error_prediction, violations):
        """Generate overall assessment summary."""
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
        """Generate detailed technical assessment."""
        categorized_violations = {}
        for violation in violations:
            category = violation["category"]
            if category not in categorized_violations:
                categorized_violations[category] = []
            categorized_violations[category].append(violation)

        assessments = []
        for category, category_violations in categorized_violations.items():
            avg_deviation = sum(v["relative_deviation"] for v in category_violations) / len(category_violations)
            if avg_deviation < 0.2:
                level = "Minor Issue"
            elif avg_deviation < 0.5:
                level = "Moderate Issue"
            else:
                level = "Significant Issue"

            assessments.append({
                "category": category,
                "level": level,
                "violation_count": len(category_violations),
                "details": [self._format_violation(v) for v in category_violations[:3]]
            })

        assessments.sort(key=lambda x: x["violation_count"], reverse=True)

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
        """Format violation details."""
        feature = violation["feature"]
        value = violation["value"]
        threshold = violation["threshold"]
        explanation = violation["explanation"]

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
        """Generate error analysis."""
        if not error_prediction["predicted_type"]:
            return {
                "error_type": None,
                "probability": 0,
                "explanation": "No significant technical issues detected."
            }

        error_type = error_prediction["predicted_type"]
        probability = error_prediction["probabilities"][error_type]
        knowledge = self.knowledge_base.get_knowledge(error_type)
        problems = knowledge.get("problems", [])
        causes = knowledge.get("causes", [])
        route_type = route_info.get("route_type", "vertical")
        route_specific = self.knowledge_base.get_route_specific_knowledge(error_type, route_type)

        if problems and causes:
            explanation = f"{problems[0]}. This is typically caused by {causes[0].lower()}."
            if len(problems) > 1:
                explanation += f" Additionally, {problems[1].lower()}."
        else:
            explanation = "Could not find detailed explanation."

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
        """Generate improvement suggestions."""
        if not error_prediction["predicted_type"] and not violations:
            return {
                "general_suggestions": ["Continue maintaining good technical movement, and consider trying more challenging routes."],
                "specific_suggestions": []
            }

        error_type = error_prediction.get("predicted_type")
        general_suggestions = []
        if error_type:
            knowledge = self.knowledge_base.get_knowledge(error_type)
            suggestions = knowledge.get("suggestions", [])
            general_suggestions = suggestions[:3]

        route_type = route_info.get("route_type", "vertical")
        route_specific = {}
        if error_type:
            route_specific = self.knowledge_base.get_route_specific_knowledge(error_type, route_type)

        if route_specific and "suggestions" in route_specific:
            for suggestion in route_specific["suggestions"]:
                if suggestion not in general_suggestions:
                    general_suggestions.append(suggestion)

        specific_suggestions = []
        for violation in violations[:5]:
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
        """Generate training recommendations."""
        recommendations = []

        if error_prediction.get("predicted_type"):
            error_type = error_prediction["predicted_type"]
            training_exercises = self.knowledge_base.get_training_exercises(error_type)
            for exercise in training_exercises[:2]:
                recommendations.append({
                    "name": exercise["name"],
                    "description": exercise["description"],
                    "benefit": exercise["benefit"]
                })

        if len(recommendations) < 2:
            general_knowledge = self.knowledge_base.get_knowledge("general")
            principles = general_knowledge.get("technique_principles", [])
            if principles:
                recommendations.append({
                    "name": "Basic Technique Principle Practice",
                    "description": f"Focus on practicing this principle: {principles[0]}",
                    "benefit": "Improves overall technical foundation"
                })

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
        """Use Claude API to generate enhanced feedback."""
        try:
            prompt = self._prepare_claude_prompt(feedback, route_info)
            response = self.claude_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                temperature=0.3,
                system="You are an expert climbing coach skilled at observing technical details in climbers and providing personalized professional feedback. Your advice should be specific, practical, and encouraging.",
                messages=[{"role": "user", "content": prompt}]
            )

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
        """Prepare prompt to send to Claude API."""
        summary = feedback["summary"]
        error_analysis = feedback["error_analysis"]
        violations = feedback.get("technical_assessment", {}).get("detailed_assessments", [])
        suggestions = feedback["improvement_suggestions"]

        top_violations = []
        if violations:
            sorted_violations = sorted(violations, key=lambda x: x["violation_count"], reverse=True)
            top_violations = sorted_violations[:2]

        violations_text = ""
        for category in top_violations:
            violations_text += f"- {category['category'].replace('_', ' ').title()}: {category['level']} ({category['violation_count']} issues)\n"
            for detail in category.get("details", [])[:2]:
                violations_text += f"  * {detail['explanation']}\n"
                violations_text += f"    Current value: {detail.get('value', 'N/A')} ({detail.get('direction', '')}), optimal range: {detail.get('threshold', 'N/A')}\n"

        specific_suggestions = []
        for violation_category in top_violations:
            category_name = violation_category["category"]
            for suggestion in suggestions.get("specific_suggestions", []):
                if suggestion["feature"] in [d["feature"] for d in violation_category.get("details", [])]:
                    specific_suggestions.append({
                        "category": category_name,
                        "suggestion": suggestion["suggestion"],
                        "feature": suggestion["feature"]
                    })

        suggestions_text = ""
        for suggestion in specific_suggestions[:4]:
            suggestions_text += f"- {suggestion['suggestion']} (for {suggestion['feature']})\n"

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
