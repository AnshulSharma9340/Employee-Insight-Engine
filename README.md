<div align="center">

# 🧠 Employee Insight Engine
### _Intelligent Predictive Analytics for Workforce Decisions_

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A unified Streamlit dashboard that hosts **three machine-learning models** for HR analytics — all behind a single sidebar selector.

</div>

---

## 🌟 What it does

| Page | Model | What it predicts |
| --- | --- | --- |
| 🧠 **Full Predictive Analytics** | 4× RandomForest (3 classifiers + 1 regressor) | Attrition risk, performance score, satisfaction, rating (1-5) — all at once |
| 👨‍💼 **Attrition (Pipeline)** | Logistic Regression pipeline (IBM HR-Analytics dataset) | Probability of an employee leaving the company |
| 🏭 **Garment Productivity** | XGBoost RandomForest regressor | Predicted worker productivity (0-1) from factory-floor metrics |

---

## 📁 Repo layout

```
Employee-Insight-Engine/
├── streamlit_app.py                ← unified entry point (deploy this)
├── requirements.txt                ← pinned ML deps
├── runtime.txt                     ← Python 3.11
├── .streamlit/config.toml          ← theme
│
├── All Four model prediction/      ← page 1 artifacts
│   ├── app.py
│   ├── rf_attrition_model.joblib
│   ├── rf_performance_model.joblib
│   ├── rf_satisfaction_model.joblib
│   ├── rf_rating_model.joblib
│   ├── attrition_features.joblib
│   ├── performance_features.joblib
│   ├── satisfaction_features.joblib
│   ├── rating_features.joblib
│   ├── scaler.joblib
│   └── main.ipynb
│
├── Employee Attrition Prediction/  ← page 2 artifacts
│   ├── app.py
│   ├── hr_attrition_log_reg_pipeline.joblib
│   └── HR Analytics - Predict Employee Attrition.ipynb
│
└── Employee_Performance_Prediction/ ← page 3 artifacts
    ├── app.py
    ├── model.h5                    (actually a pickled XGBRFRegressor)
    └── Emp_Performance_Analysis.ipynb
```

---

## 🚀 Run locally

```bash
git clone https://github.com/AnshulSharma9340/Employee-Insight-Engine.git
cd Employee-Insight-Engine

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501. The sidebar lets you switch between the three pages.

---

## ☁️ Deploy on Streamlit Community Cloud

It's free and takes 5 clicks.

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **"Create app" → "Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `AnshulSharma9340/Employee-Insight-Engine`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **Deploy**.

Streamlit Cloud reads `requirements.txt`, installs the pinned dependencies, and serves the app at a `https://<your-slug>.streamlit.app` URL. First build takes about 2-3 minutes (scikit-learn + xgboost).

### ⚠️ Important: version pinning

The model artifacts were saved with **scikit-learn 1.5.1**. `requirements.txt` pins that exact version — if you bump sklearn later, the attrition pipeline (`hr_attrition_log_reg_pipeline.joblib`) will fail to unpickle because it references private classes (`_RemainderColsList`) that change between minor versions. If you want to upgrade, retrain the models from the notebooks and re-export the artifacts.

---

## 🧪 Retraining

Each model has its own Jupyter notebook in its subfolder:

- `All Four model prediction/main.ipynb`
- `Employee Attrition Prediction/HR Analytics - Predict Employee Attrition.ipynb`
- `Employee_Performance_Prediction/Emp_Performance_Analysis.ipynb`

Run those to regenerate the `.joblib` / `.h5` artifacts.

---

## 👨‍💻 Author

**Anshul Sharma** — B.Tech Data Science
[@AnshulSharma9340](https://github.com/AnshulSharma9340)

## 🪪 License

MIT — see [LICENSE](LICENSE).
