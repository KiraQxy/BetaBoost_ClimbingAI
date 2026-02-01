"""BetaBoost - AI-powered climbing technique analysis app.

Main entry point. Run with: streamlit run BetaBoost_app.py
"""

import os

import streamlit as st

from video_processor import VideoProcessor
from feature_extractor import FeatureExtractor
from rule_system import ClimbingRuleSystem
from knowledge_base import ClimbingKnowledgeBase
from analysis_engine import AnalysisEngine
from feedback_generator import FeedbackGenerator
from ui import render_ui, update_ui_with_results
from utils import map_route_type, map_climber_level


def main():
    """Main application entry point."""
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

    ui_elements = render_ui()

    if st.session_state.should_analyze and not st.session_state.processed:
        with st.spinner("Processing video..."):
            video_processor = VideoProcessor()
            feature_extractor = FeatureExtractor()
            rule_system = ClimbingRuleSystem()
            knowledge_base = ClimbingKnowledgeBase()
            analysis_engine = AnalysisEngine(rule_system)
            feedback_generator = FeedbackGenerator(knowledge_base)

            frames_data = video_processor.process_video(st.session_state.uploaded_file)
            st.session_state.frames_data = frames_data

            if frames_data:
                features = feature_extractor.extract_features(frames_data)
                st.session_state.features = features

                route_info = {
                    "route_type": map_route_type(st.session_state.route_type),
                    "route_difficulty": st.session_state.route_difficulty,
                    "climber_level": map_climber_level(st.session_state.climber_level)
                }

                analysis_results = analysis_engine.analyze(features, route_info)
                st.session_state.analysis_results = analysis_results

                feedback = feedback_generator.generate_feedback(analysis_results, route_info)
                st.session_state.feedback = feedback

                st.session_state.processed = True
                st.session_state.should_analyze = False

    if st.session_state.processed:
        update_ui_with_results(
            ui_elements,
            st.session_state.frames_data,
            st.session_state.features,
            st.session_state.analysis_results,
            st.session_state.feedback
        )


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = (
            "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"
        )

    main()
