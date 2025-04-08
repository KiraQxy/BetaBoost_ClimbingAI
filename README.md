# 🧗‍♀️ BetaBoost: AI-Powered Climbing Coach

**BetaBoost** is an AI climbing coach that analyzes user-submitted climbing videos to provide personalized technical feedback. Using pose estimation, dynamic biomechanical features, and an expert rule system, the tool delivers feedback at a fraction of professional coaching costs.

> 🧠 “Climbing harder, but not climbing better.” — BetaBoost bridges this gap with intelligent, visual feedback.

---

## 🔍 Key Features

- 🎯 **Pose Estimation** with MediaPipe for 33 body landmarks per frame
- 📊 **327 Biomechanical Features**: joint angles, balance, trunk rotation, center of mass, and more
- 🔁 **Dynamic Movement Analysis** using smart frame sampling
- 🧠 **Rule-Based System**: Categorizes errors in 7 technique areas
- 🤖 **Natural Language Feedback** via Claude API (e.g. "twist your hips more on vertical routes")
- 🖥️ **Streamlit Web App** for end-to-end interaction
- 💡 **Deployable + Scalable** for gym or personal use

---

## 📸 Project Interface

### Home UI with Personalized Feedback

![BetaBoost Screenshot](assets/beta_screenshot.png)

Users upload a video, select route type and difficulty, and receive:
- Visual analysis (skeletal overlay, trajectory)
- Text feedback on technique issues
- Suggested drills and corrections

---

## 🧰 Tech Stack

- **Language**: Python
- **Frontend**: Streamlit
- **CV**: MediaPipe
- **Modeling**: XGBoost
- **Logic System**: Rule engine + knowledge base
- **NLP**: Claude API
- **Visualization**: Matplotlib, OpenCV

---

## 🚀 Usage

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
