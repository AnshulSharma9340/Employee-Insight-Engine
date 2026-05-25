"""Employee Insight Engine — unified Streamlit entry point.

Hosts three ML demos behind a sidebar selector:
  1. All Four Predictions  (attrition, performance, satisfaction, rating)
  2. Attrition (Pipeline)   (Logistic-Regression pipeline trained on HR-Analytics dataset)
  3. Productivity           (Garment-worker productivity regression)

Each subpage loads its own model artifacts from the matching subfolder.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
DIR_ALL = BASE / "All Four model prediction"
DIR_ATTRITION = BASE / "Employee Attrition Prediction"
DIR_PERF = BASE / "Employee_Performance_Prediction"


st.set_page_config(
    page_title="Employee Insight Engine",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_all_four():
    return {
        "att": joblib.load(DIR_ALL / "rf_attrition_model.joblib"),
        "perf": joblib.load(DIR_ALL / "rf_performance_model.joblib"),
        "sat": joblib.load(DIR_ALL / "rf_satisfaction_model.joblib"),
        "rate": joblib.load(DIR_ALL / "rf_rating_model.joblib"),
        "scaler": joblib.load(DIR_ALL / "scaler.joblib"),
        "att_features": joblib.load(DIR_ALL / "attrition_features.joblib"),
        "perf_features": joblib.load(DIR_ALL / "performance_features.joblib"),
        "sat_features": joblib.load(DIR_ALL / "satisfaction_features.joblib"),
        "rate_features": joblib.load(DIR_ALL / "rating_features.joblib"),
    }


@st.cache_resource
def load_attrition_pipeline():
    return joblib.load(DIR_ATTRITION / "hr_attrition_log_reg_pipeline.joblib")


@st.cache_resource
def load_performance_model():
    with open(DIR_PERF / "model.h5", "rb") as f:
        return pickle.load(f)


SATISFACTION_MAP = defaultdict(
    lambda: 3,
    {
        "very low": 1, "Low": 2, "Medium": 3, "High": 4,
        "Poor": 1, "Average": 2, "Good": 3, "Excellent": 4,
    },
)
PERFORMANCE_MAP = {4: "Exceeds Expectations", 3: "Fully Meets", 2: "Needs Improvement", 1: "NA/Average", 0: "PIP"}
SATISFACTION_OUTPUT = {4: "High", 3: "Medium", 2: "Low", 1: "Very Low"}
ATTRITION_LABELS = {0: "Will Stay (Low Risk)", 1: "High Attrition Risk"}

TITLES = ["Production Technician I", "Area Sales Manager", "Production Manager", "Software Engineer", "Data Analyst"]
DEPARTMENTS = ["Production", "Sales", "Software Engineering", "Finance & Accounting", "IT"]
MARITAL = ["Married", "Single", "Widowed"]
GENDER = ["Male", "Female"]
SATISFACTION_LEVELS = ["High", "Medium", "Low", "very low"]
HYGIENE_LEVELS = ["Good", "Average", "Poor"]


def preprocess_all_four(input_data, feature_list, scaler):
    df = pd.DataFrame([input_data])
    df["Overall_Satisfaction_Encoded"] = df["Overall Satisfaction"].map(SATISFACTION_MAP)
    df["Hygiene_Encoded"] = df["Hygiene"].map(SATISFACTION_MAP)

    cat_cols = ["Title", "DepartmentType", "GenderCode", "MaritalDesc",
                "EmployeeType", "PayZone", "RaceDesc", "TerminationType"]
    df = df.drop(columns=["Overall Satisfaction", "Hygiene"], errors="ignore")
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    final = pd.DataFrame(0, index=[0], columns=feature_list)
    for col in encoded.columns:
        if col in final.columns:
            final[col] = encoded[col].values[0]

    final[["Tenure_Years", "Age"]] = scaler.transform(final[["Tenure_Years", "Age"]])
    return final


def page_all_four():
    st.title("👨‍💻 Full Predictive Analytics")
    st.caption("Predict attrition, performance, satisfaction, and rating in one shot.")

    models = load_all_four()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("👤 Demographics")
        age = st.slider("Age", 20, 65, 30)
        tenure = st.slider("Tenure (years)", 0.0, 30.0, 3.5)
        gender = st.selectbox("Gender", GENDER)
        marital = st.selectbox("Marital status", MARITAL)
    with c2:
        st.subheader("🏢 Job & environment")
        department = st.selectbox("Department", DEPARTMENTS)
        title = st.selectbox("Job title", TITLES)
        satisfaction = st.selectbox("Overall satisfaction survey", SATISFACTION_LEVELS)
        hygiene = st.selectbox("Hygiene / facilities survey", HYGIENE_LEVELS)

    if st.button("🔮 Predict all four outcomes", type="primary"):
        input_data = {
            "Age": age, "Tenure_Years": tenure, "GenderCode": gender,
            "MaritalDesc": marital, "DepartmentType": department, "Title": title,
            "Overall Satisfaction": satisfaction, "Hygiene": hygiene,
            "EmployeeType": "Full-Time", "PayZone": "Zone A",
            "RaceDesc": "White", "TerminationType": "Unk",
        }

        x_att = preprocess_all_four(input_data, models["att_features"], models["scaler"])
        x_perf = preprocess_all_four(input_data, models["perf_features"], models["scaler"])
        x_sat = preprocess_all_four(input_data, models["sat_features"], models["scaler"])
        x_rate = preprocess_all_four(input_data, models["rate_features"], models["scaler"])

        att_pred = models["att"].predict(x_att)[0]
        att_proba = models["att"].predict_proba(x_att)[0][1]
        perf_pred = models["perf"].predict(x_perf)[0]
        sat_pred = models["sat"].predict(x_sat)[0]
        rate_pred = models["rate"].predict(x_rate)[0]

        st.markdown("---")
        st.header("📊 Results")
        a, b, c, d = st.columns(4)
        a.metric("Attrition risk", ATTRITION_LABELS[att_pred], f"{att_proba * 100:.1f}%")
        b.metric("Performance score", PERFORMANCE_MAP.get(perf_pred, "?"), f"encoded={perf_pred}")
        c.metric("Satisfaction", SATISFACTION_OUTPUT.get(sat_pred, "?"), f"encoded={sat_pred}")
        d.metric("Rating (1-5)", f"{rate_pred:.2f}", f"rounded={round(rate_pred)}")


def get_income_band(monthly_income: float) -> str:
    if monthly_income <= 3279.88:
        return "Low"
    if monthly_income <= 7123.00:
        return "Medium"
    return "High"


def page_attrition():
    st.title("👨‍💼 Employee Attrition (Pipeline)")
    st.caption("Logistic-regression pipeline trained on the IBM HR-Analytics dataset.")

    pipeline = load_attrition_pipeline()

    with st.form("attrition_form"):
        st.subheader("Employee data")
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 18, 60, 35)
        daily_rate = c2.number_input("Daily rate", 102, 1499, 802)
        distance = c3.slider("Distance from home (mi)", 1, 29, 10)

        c4, c5, c6 = st.columns(3)
        education = c4.selectbox("Education level", (1, 2, 3, 4, 5), index=2)
        env_sat = c5.selectbox("Environment satisfaction", (1, 2, 3, 4), index=2)
        hourly_rate = c6.number_input("Hourly rate", 30, 100, 65)

        c7, c8, c9 = st.columns(3)
        job_inv = c7.selectbox("Job involvement", (1, 2, 3, 4), index=2)
        job_level = c8.selectbox("Job level", (1, 2, 3, 4, 5), index=0)
        job_sat = c9.selectbox("Job satisfaction", (1, 2, 3, 4), index=2)

        c10, c11, c12 = st.columns(3)
        monthly_income = c10.number_input("Monthly income (USD)", 1000, 20000, 5000, step=100)
        monthly_rate = c11.number_input("Monthly rate", 2094, 26999, 14261)
        n_companies = c12.slider("Num companies worked", 0, 9, 2)

        c13, c14, c15 = st.columns(3)
        pct_hike = c13.slider("Percent salary hike", 11, 25, 15)
        perf_rating = c14.selectbox("Performance rating", (3, 4), index=0)
        rel_sat = c15.selectbox("Relationship satisfaction", (1, 2, 3, 4), index=2)

        c16, c17, c18 = st.columns(3)
        stock = c16.slider("Stock option level", 0, 3, 0)
        total_years = c17.slider("Total working years", 0, 40, 8)
        trainings = c18.slider("Trainings last year", 0, 6, 3)

        c19, c20, c21 = st.columns(3)
        wlb = c19.selectbox("Work-life balance", (1, 2, 3, 4), index=2)
        years_co = c20.slider("Years at company", 0, 40, 5)
        years_role = c21.slider("Years in current role", 0, 18, 3)

        c22, c23 = st.columns(2)
        years_promo = c22.slider("Years since last promotion", 0, 15, 1)
        years_mgr = c23.slider("Years with current manager", 0, 17, 3)

        travel = st.selectbox("Business travel", ("Travel_Rarely", "Travel_Frequently", "Non-Travel"))
        dept = st.selectbox("Department", ("Research & Development", "Sales", "Human Resources"))
        ed_field = st.selectbox("Education field",
                                ("Life Sciences", "Medical", "Marketing",
                                 "Technical Degree", "Human Resources", "Other"))
        gender = st.selectbox("Gender", ("Male", "Female"))
        role = st.selectbox("Job role",
                            ("Sales Executive", "Research Scientist", "Laboratory Technician",
                             "Manufacturing Director", "Healthcare Representative", "Manager",
                             "Sales Representative", "Research Director", "Human Resources"),
                            index=1)
        marital = st.selectbox("Marital status", ("Married", "Single", "Divorced"))
        overtime = st.selectbox("Over time?", ("Yes", "No"))

        submitted = st.form_submit_button("Predict attrition", type="primary")

    if submitted:
        tenure = years_co / max(total_years, 1)
        promo_gap = years_promo / max(years_co, 1)
        income_band = get_income_band(monthly_income)
        input_data = {
            "Age": age, "BusinessTravel": travel, "DailyRate": daily_rate, "Department": dept,
            "DistanceFromHome": distance, "Education": education, "EducationField": ed_field,
            "EnvironmentSatisfaction": env_sat, "Gender": gender, "HourlyRate": hourly_rate,
            "JobInvolvement": job_inv, "JobLevel": job_level, "JobRole": role,
            "JobSatisfaction": job_sat, "MaritalStatus": marital, "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate, "NumCompaniesWorked": n_companies, "OverTime": overtime,
            "PercentSalaryHike": pct_hike, "PerformanceRating": perf_rating,
            "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock,
            "TotalWorkingYears": total_years, "TrainingTimesLastYear": trainings,
            "WorkLifeBalance": wlb, "YearsAtCompany": years_co,
            "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_mgr,
            "Tenure": tenure, "PromotionGap": promo_gap, "IncomeBand": income_band,
        }
        df = pd.DataFrame([input_data])
        proba = pipeline.predict_proba(df)[:, 1][0]
        pred = pipeline.predict(df)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"**HIGH** likelihood of attrition — probability {proba:.2%}")
        else:
            st.success(f"**LOW** likelihood of attrition — probability {proba:.2%}")
        st.metric("Raw probability", f"{proba:.4f}")


QUARTER_MAP = {"Q1 (Jan-Mar)": 1, "Q2 (Apr-Jun)": 2, "Q3 (Jul-Sep)": 3, "Q4 (Oct-Dec)": 4}
DEPT_MAP = {"Finishing": 1, "Sewing": 2}
DAY_MAP = {d: i + 1 for i, d in enumerate(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}
TEAM_MAP = {f"Team {i}": i for i in range(1, 13)}
MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}


def page_productivity():
    st.title("🏭 Garment Worker Productivity")
    st.caption("Regression model predicting productivity (0–1) from factory floor metrics.")

    model = load_performance_model()

    with st.form("productivity_form"):
        st.subheader("Time & location")
        c1, c2, c3 = st.columns(3)
        quarter = c1.selectbox("Quarter", list(QUARTER_MAP.keys()))
        month = c2.selectbox("Month", list(MONTH_MAP.keys()))
        day = c3.selectbox("Day of week", list(DAY_MAP.keys()))

        c1, c2, c3 = st.columns(3)
        dept = c1.selectbox("Department", list(DEPT_MAP.keys()))
        team = c2.selectbox("Team", list(TEAM_MAP.keys()))
        workers = c3.number_input("Number of workers", min_value=1.0, value=30.0, step=0.1)

        st.subheader("Performance & resources")
        c1, c2, c3 = st.columns(3)
        target = c1.number_input("Targeted productivity", 0.0, 1.0, 0.45, 0.01)
        smv = c2.number_input("Standard minute value (SMV)", 0.1, 100.0, 15.69, 0.01)
        overtime = c3.number_input("Over-time (minutes)", min_value=0, value=1800, step=60)

        c1, c2, _ = st.columns(3)
        incentive = c1.number_input("Incentive ($)", min_value=0, value=100)
        style_changes = c2.number_input("Style changes", min_value=0, value=0)

        st.subheader("Idle time")
        c1, c2 = st.columns(2)
        idle_time = c1.number_input("Idle time (minutes)", min_value=0.0, value=0.0)
        idle_men = c2.number_input("Idle men", min_value=0, value=0)

        submitted = st.form_submit_button("Predict productivity", type="primary")

    if submitted:
        features = np.array([[
            QUARTER_MAP[quarter], DEPT_MAP[dept], DAY_MAP[day], TEAM_MAP[team],
            float(target), float(smv), int(overtime), int(incentive),
            float(idle_time), int(idle_men), int(style_changes),
            float(workers), MONTH_MAP[month],
        ]])
        pred = float(model.predict(features)[0])

        st.markdown("---")
        if pred > 0.8:
            st.success(f"**High productivity**: predicted score {pred:.3f}")
        elif pred > 0.3:
            st.info(f"**Medium productivity**: predicted score {pred:.3f}")
        else:
            st.warning(f"**Average productivity**: predicted score {pred:.3f}")
        st.progress(min(max(pred, 0.0), 1.0))


PAGES = {
    "🧠 Full predictive analytics (all four)": page_all_four,
    "👨‍💼 Attrition (HR-Analytics pipeline)": page_attrition,
    "🏭 Garment worker productivity": page_productivity,
}

with st.sidebar:
    st.title("🧠 Employee Insight Engine")
    st.caption("Three ML demos — choose one to start.")
    choice = st.radio("Model", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.markdown(
        "**Author:** Anshul Sharma  \n"
        "**Repo:** [GitHub](https://github.com/AnshulSharma9340/Employee-Insight-Engine)"
    )

PAGES[choice]()
