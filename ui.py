"""UI components - Streamlit layout and result display."""

import streamlit as st

from utils import (
    create_skeleton_video,
    get_video_html,
    generate_com_trajectory_plot,
)


def render_ui():
    """Main UI rendering function."""
    st.set_page_config(layout="wide", page_title="BetaBoost Climbing AI", page_icon="🧗‍♀️")

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
        .stFileUploader > div:first-child {
            width: 100%;
            height: 100%;
            min-height: 60px;
        }
        .stSelectbox > div:first-child {
            min-height: 60px;
        }
        .stFileUploader {
            min-height: 60px;
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

    row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 1])
    row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])

    with row1_col1:
        st.markdown("##### Upload climbing video")
        uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"], label_visibility="collapsed")

    with row1_col2:
        st.markdown("##### Route Type")
        route_type = st.selectbox("", ["Vertical", "Slab", "Overhang", "Roof"], label_visibility="collapsed")

    with row1_col3:
        st.markdown("##### Climber Level")
        climber_level = st.selectbox("", ["Beginner", "Intermediate", "Advanced"], label_visibility="collapsed")

    with row2_col1:
        pass  # File info shown by file_uploader

    with row2_col2:
        st.markdown("##### Difficulty")
        route_difficulty = st.selectbox("", ["V1", "V2", "V3", "V4", "V5", "V6"], label_visibility="collapsed")

    with row2_col3:
        st.markdown("##### &nbsp;")
        analyze_button = st.button("Start Analysis", use_container_width=True)
        if analyze_button and uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.route_type = route_type
            st.session_state.route_difficulty = route_difficulty
            st.session_state.climber_level = climber_level
            st.session_state.should_analyze = True

    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.5, 2])

    with left_col:
        st.markdown("<div class='section-title'>Video Analysis</div>", unsafe_allow_html=True)
        video_container = st.container()
        with video_container:
            video_placeholder = st.empty()
            controls_placeholder = st.empty()
        st.markdown("<div class='section-title'>Horizontal Center of Mass Trajectory</div>", unsafe_allow_html=True)
        gravity_center_plot = st.empty()

    with right_col:
        st.markdown("<div class='section-title'>AI Technical Analysis</div>", unsafe_allow_html=True)
        score_col, issue_col = st.columns([1, 2])
        with score_col:
            score_placeholder = st.empty()
        with issue_col:
            main_issue = st.empty()
        findings_container = st.container()
        with findings_container:
            st.markdown("<div class='section-title'>Key Findings</div>", unsafe_allow_html=True)
            findings_list = st.empty()
        detail_expander = st.expander("Detailed Technical Feedback", expanded=True)
        with detail_expander:
            detailed_feedback = st.empty()
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


def update_ui_with_results(ui_elements, frames_data, features, analysis_results, feedback):
    """Update UI placeholders with analysis results."""
    with st.spinner("Creating video with skeleton overlay..."):
        video_path = create_skeleton_video(frames_data)

    if video_path:
        try:
            ui_elements['video_placeholder'].video(video_path)
        except Exception:
            video_html = get_video_html(video_path)
            ui_elements['video_placeholder'].markdown(video_html, unsafe_allow_html=True)

    summary = feedback["summary"]
    score = summary['score']

    score_class = "score-poor"
    if score >= 80:
        score_class = "score-good"
    elif score >= 60:
        score_class = "score-medium"

    score_html = f"""
    <div class="score-card">
        <div class="score-value {score_class}">{score}</div>
        <div style="font-size: 0.875rem; color: #6b7280;">/100</div>
    </div>
    """
    ui_elements['score_placeholder'].markdown(score_html, unsafe_allow_html=True)

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

    findings_md = ""
    if feedback.get("error_analysis") and feedback["error_analysis"].get("common_problems"):
        problems = feedback["error_analysis"]["common_problems"]
        for problem in problems:
            findings_md += f"<li style='display: flex; align-items: flex-start; margin-bottom: 0.75rem;'><span style='color: #f97316; margin-right: 0.5rem;'>•</span> <span>{problem}</span></li>"
    else:
        findings_md = "<li style='display: flex; align-items: flex-start;'><span style='color: #f97316; margin-right: 0.5rem;'>•</span> <span>No significant issues detected</span></li>"

    ui_elements['findings_list'].markdown(f"<ul style='list-style-type: none; padding-left: 0;'>{findings_md}</ul>", unsafe_allow_html=True)

    detailed_md = ""
    if feedback.get("claude_enhanced"):
        detailed_md = feedback["claude_enhanced"]
    elif feedback.get("error_analysis") and feedback["error_analysis"].get("explanation"):
        detailed_md = feedback["error_analysis"]["explanation"]
        detailed_md += "\n\n**Improvement Suggestions:**\n"
        suggestions = feedback.get("improvement_suggestions", {})
        for suggestion in suggestions.get("general_suggestions", []):
            detailed_md += f"- {suggestion}\n"

    ui_elements['detailed_feedback'].markdown(detailed_md)

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

    com_trajectory_plot = generate_com_trajectory_plot(frames_data)
    if com_trajectory_plot:
        ui_elements['gravity_center_plot'].pyplot(com_trajectory_plot)
