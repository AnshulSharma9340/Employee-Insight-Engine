import streamlit as st
import numpy as np
import pickle
import os

# --- CRITICAL: set_page_config MUST be the very first Streamlit command ---
st.set_page_config(page_title="Employee Performance Prediction", layout="centered")

# --- Configuration and Model Loading ---

# Use st.cache_resource to load the model only once across all user sessions
@st.cache_resource
def load_model():
    """
    Loads the pickled machine learning model from the model.h5 file.
    """
    model_path = 'model.h5'
    try:
        if not os.path.exists(model_path):
            # This st.error is fine now because st.set_page_config has already run
            st.error(f"Model file not found at: {model_path}. Please ensure 'model.h5' is in the same directory.")
            return None
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Placeholder Mappings for Categorical Data (Adjust as needed) ---
# Assuming these inputs were integer-encoded in the original model training
QUARTER_MAPPING = {
    "Quarter 1 (Jan-Mar)": 1, 
    "Quarter 2 (Apr-Jun)": 2, 
    "Quarter 3 (Jul-Sep)": 3, 
    "Quarter 4 (Oct-Dec)": 4
}
DEPARTMENT_MAPPING = {
    "Finishing": 1, 
    "Sewing": 2
}
DAY_MAPPING = {
    "Monday": 1, 
    "Tuesday": 2, 
    "Wednesday": 3, 
    "Thursday": 4, 
    "Friday": 5, 
    "Saturday": 6, 
    "Sunday": 7
}
TEAM_MAPPING = {f"Team {i}": i for i in range(1, 13)} # Teams 1 through 12
MONTH_MAPPING = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, 
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}


# --- Prediction Logic Function ---

def predict_performance(data):
    """
    Takes input features, makes a prediction for employee performance, and returns the result text.
    """
    try:
        # The model expects a 2D numpy array: [[feature1, feature2, ...]]
        prediction = model.predict(data)[0]
        
        # Apply the original logic for determining productivity text
        if prediction <= 0.3:
            text = f'Prediction: {prediction:.2f}. The employee is averagely productive.'
        elif 0.3 < prediction <= 0.8:
            text = f'Prediction: {prediction:.2f}. The employee is medium productive.'
        else:
            text = f'Prediction: {prediction:.2f}. The employee is highly productive.'
            
        return text, prediction
        
    except Exception as e:
        return f"An error occurred during prediction: {e}", None

# --- Streamlit UI ---

st.title("🏭 Employee Performance Prediction App")
st.markdown("Use the input form below to set the parameters for **performance** prediction.")

if model is None:
    st.stop() # Stop the app if model loading failed

# Create the input form
with st.form("prediction_form"):
    st.subheader("Time and Location Features")
    col1, col2, col3 = st.columns(3)
    
    # Row 1
    quarter_str = col1.selectbox("Quarter", options=list(QUARTER_MAPPING.keys()), index=0)
    month_str = col2.selectbox("Month", options=list(MONTH_MAPPING.keys()), index=0)
    day_str = col3.selectbox("Day of Week", options=list(DAY_MAPPING.keys()), index=0)
    
    # Row 2
    department_str = col1.selectbox("Department", options=list(DEPARTMENT_MAPPING.keys()), index=0)
    team_str = col2.selectbox("Team Number", options=list(TEAM_MAPPING.keys()), index=0)
    no_of_workers = col3.number_input("Number of Workers", min_value=1.0, value=30.0, step=0.1, format="%.1f")

    st.subheader("Performance & Resource Allocation")
    col4, col5, col6 = st.columns(3)
    
    # Row 3
    targeted_productivity = col4.number_input("Targeted Productivity", min_value=0.0, max_value=1.0, value=0.45, step=0.01, format="%.2f")
    smv = col5.number_input("Standard Minute Value (SMV)", min_value=0.1, value=15.69, step=0.01, format="%.2f")
    over_time = col6.number_input("Over Time (minutes)", min_value=0, value=1800, step=60)
    
    # Row 4
    incentive = col4.number_input("Incentive ($)", min_value=0, value=100, step=1)
    no_of_style_change = col5.number_input("No. of Style Changes", min_value=0, value=0, step=1)

    st.subheader("Idle Time Metrics")
    col7, col8 = st.columns(2)

    # Row 5
    idle_time = col7.number_input("Idle Time (minutes)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    idle_men = col8.number_input("Idle Men", min_value=0, value=0, step=1)
    
    # Submit button
    submitted = st.form_submit_button("Predict Performance")

if submitted:
    # 1. Map string inputs back to model's expected integers/floats
    # Use .get() for safe retrieval, falling back to 0 if not found (shouldn't happen with selectbox)
    quarter = QUARTER_MAPPING.get(quarter_str, 0)
    department = DEPARTMENT_MAPPING.get(department_str, 0)
    day = DAY_MAPPING.get(day_str, 0)
    team = TEAM_MAPPING.get(team_str, 0)
    month = MONTH_MAPPING.get(month_str, 0)

    # 2. Collect all features in the correct order and data type as the original Flask code:
    # [quarter, department, day, team, targeted_productivity, smv, over_time, incentive, idle_time, idle_men, no_of_style_change, no_of_workers, month]
    input_features = [
        quarter, 
        department, 
        day, 
        team, 
        float(targeted_productivity), 
        float(smv), 
        int(over_time), 
        int(incentive), 
        float(idle_time), 
        int(idle_men), 
        int(no_of_style_change), 
        float(no_of_workers), 
        month
    ]
    
    # Convert to NumPy array for prediction
    final_features = np.array([input_features])
    
    # 3. Get the prediction
    result_text, prediction_value = predict_performance(final_features)
    
    # 4. Display the result
    st.markdown("---")
    if prediction_value is not None:
        if prediction_value > 0.8:
            st.success(f"**High Productivity Status:** {result_text}")
        elif prediction_value > 0.3:
            st.info(f"**Medium Productivity Status:** {result_text}")
        else:
            st.warning(f"**Average Productivity Status:** {result_text}")
    else:
        st.error(result_text)

st.sidebar.markdown(f"**Model Status:** {'Ready' if model else 'Error'}")

st.markdown("---")
st.caption("This app is a Streamlit conversion of a Flask application for predicting employee performance based on various factory metrics.")
