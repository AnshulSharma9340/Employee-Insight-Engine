import streamlit as st
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict

# --- CRITICAL: set_page_config MUST be the first Streamlit command ---
st.set_page_config(page_title="Employee Predictive Analytics", layout="wide")
# ----------------------------------------------------------------------


# --- 1. Load All Models and Assets ---

@st.cache_resource
def load_models():
    """Loads all four models, scaler, and feature lists."""
    try:
        # Load Classification Models
        model_att = joblib.load('rf_attrition_model.joblib')
        model_perf = joblib.load('rf_performance_model.joblib')
        model_sat = joblib.load('rf_satisfaction_model.joblib')
        
        # Load Regression Model
        model_rate = joblib.load('rf_rating_model.joblib')
        
        # Load Scaler
        scaler = joblib.load('scaler.joblib')
        
        # Load Feature Names (8 files total)
        att_features = joblib.load('attrition_features.joblib')
        perf_features = joblib.load('performance_features.joblib')
        sat_features = joblib.load('satisfaction_features.joblib')
        rate_features = joblib.load('rating_features.joblib')

        return model_att, model_perf, model_sat, model_rate, scaler, att_features, perf_features, sat_features, rate_features
    except FileNotFoundError:
        st.error("Error loading models: Please run the training script (main.ipynb steps) to create all 8 .joblib files first.")
        st.stop()


rf_att, rf_perf, rf_sat, rf_rate, scaler, att_features, perf_features, sat_features, rate_features = load_models()

# --- 2. Static Mappings and UI Data ---

# Ordinal Encoding Mapping (Used during training)
SATISFACTION_MAP = defaultdict(lambda: 3, { 
    'very low': 1, 'Low': 2, 'Medium': 3, 'High': 4,
    'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4
})

# Output Mappings (to display text instead of just numbers)
PERFORMANCE_MAP = {4: "Exceeds Expectations", 3: "Fully Meets", 2: "Needs Improvement", 1: "NA/Average", 0: "PIP"}
SATISFACTION_OUTPUT = {4: "High", 3: "Medium", 2: "Low", 1: "Very Low"} # Match classification output

# Prediction Labels
ATTRITION_LABELS = {0: "Employee Will Stay (Low Risk)", 1: "High Attrition Risk (Action Needed)"}

# Inferred unique values for UI select boxes
TITLES = ['Production Technician I', 'Area Sales Manager', 'Production Manager', 'Software Engineer', 'Data Analyst']
DEPARTMENTS = ['Production', 'Sales', 'Software Engineering', 'Finance & Accounting', 'IT']
MARITAL = ['Married', 'Single', 'Widowed']
GENDER = ['Male', 'Female']
SATISFACTION_LEVELS = ['High', 'Medium', 'Low', 'very low']
HYGIENE_LEVELS = ['Good', 'Average', 'Poor']


# --- 3. Function: Preprocess User Input (The core of data transformation) ---

def preprocess_input(input_data, feature_list, scaler):
    """Prepares user input for prediction by encoding and scaling."""
    
    data_df = pd.DataFrame([input_data])
    
    # 1. Ordinal Features Encode
    data_df['Overall_Satisfaction_Encoded'] = data_df['Overall Satisfaction'].map(SATISFACTION_MAP)
    data_df['Hygiene_Encoded'] = data_df['Hygiene'].map(SATISFACTION_MAP)
    
    # 2. One-Hot Encoding preparation
    categorical_cols = ['Title', 'DepartmentType', 'GenderCode', 'MaritalDesc', 
                        'EmployeeType', 'PayZone', 'RaceDesc', 'TerminationType']
    
    data_df = data_df.drop(columns=['Overall Satisfaction', 'Hygiene'], errors='ignore')
    data_encoded = pd.get_dummies(data_df, columns=categorical_cols, drop_first=True)
    
    # 3. Align Columns (CRITICAL: Ensures input columns match training columns)
    final_features = pd.DataFrame(0, index=[0], columns=feature_list)
    
    for col in data_encoded.columns:
        if col in final_features.columns:
            final_features[col] = data_encoded[col].values[0]

    # 4. Scale Numerical Features
    final_features[['Tenure_Years', 'Age']] = scaler.transform(
        final_features[['Tenure_Years', 'Age']]
    )

    return final_features


# --- 4. Streamlit UI (User Interface) ---

st.title("👨‍💻 Employee Predictive Analytics Dashboard")
st.subheader("Predicting Attrition, Performance, Satisfaction, and Rating for HR Insights")

# Two-column layout for input
col1, col2 = st.columns(2)

with col1:
    st.header("👤 Employee Demographics")
    age = st.slider("Age", 20, 65, 30)
    tenure = st.slider("Tenure (Years at Company)", 0.0, 30.0, 3.5)
    gender = st.selectbox("Gender Code", GENDER)
    marital = st.selectbox("Marital Status", MARITAL)

with col2:
    st.header("🏢 Job & Environment")
    department = st.selectbox("Department Type", DEPARTMENTS)
    title = st.selectbox("Job Title", TITLES)
    satisfaction = st.selectbox("Overall Satisfaction Survey", SATISFACTION_LEVELS)
    hygiene = st.selectbox("Hygiene/Facilities Survey", HYGIENE_LEVELS)
    
# --- 5. Prediction Button and Results ---

if st.button("Predict All Four Employee Outcomes"):
    
    # 1. Prepare User Input Data
    input_data = {
        'Age': age, 'Tenure_Years': tenure, 'GenderCode': gender, 
        'MaritalDesc': marital, 'DepartmentType': department, 'Title': title,
        'Overall Satisfaction': satisfaction, 'Hygiene': hygiene,
        # Default OHE values (must match training data)
        'EmployeeType': 'Full-Time', 'PayZone': 'Zone A', 'RaceDesc': 'White', 
        'TerminationType': 'Unk' 
    }

    # 2. Run Predictions for all four models
    
    # Model 1: Attrition (Classification)
    X_att_pred = preprocess_input(input_data, att_features, scaler)
    att_pred = rf_att.predict(X_att_pred)[0]
    att_proba = rf_att.predict_proba(X_att_pred)[0][1] 
    
    # Model 4: Performance Score (Classification)
    X_perf_pred = preprocess_input(input_data, perf_features, scaler)
    perf_pred = rf_perf.predict(X_perf_pred)[0]
    
    # Model 3: Satisfaction (Classification)
    X_sat_pred = preprocess_input(input_data, sat_features, scaler)
    sat_pred = rf_sat.predict(X_sat_pred)[0]
    
    # Model 2: Rating (Regression - predicts a float)
    X_rate_pred = preprocess_input(input_data, rate_features, scaler)
    rate_pred_float = rf_rate.predict(X_rate_pred)[0]

    # 3. Display Results
    st.markdown("---")
    st.header("📊 Full Predictive Analytics Results")

    # Display all four results side-by-side
    col_att, col_perf, col_sat, col_rate = st.columns(4) 

    # RESULT 1: Attrition
    with col_att:
        st.subheader("1. Attrition Risk")
        st.metric(label="Risk Level", value=ATTRITION_LABELS[att_pred])
        st.markdown(f"**Probability:** `{att_proba*100:.1f}%`")

    # RESULT 2: Performance Score
    with col_perf:
        st.subheader("2. Performance Score")
        st.metric(label="Predicted Score", value=PERFORMANCE_MAP.get(perf_pred, "Unknown"))
        st.markdown(f"**Encoded Value:** `{perf_pred}`")

    # RESULT 3: Overall Satisfaction
    with col_sat:
        st.subheader("3. Satisfaction Level")
        st.metric(label="Predicted Level", value=SATISFACTION_OUTPUT.get(sat_pred, "Unknown"))
        st.markdown(f"**Encoded Value:** `{sat_pred}`")
    
    # RESULT 4: Employee Rating
    with col_rate:
        st.subheader("4. Current Rating")
        st.metric(label="Predicted Rating (1-5)", value=f"{rate_pred_float:.2f}")
        st.markdown(f"**Rounded Rating:** `{round(rate_pred_float)}`")