import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Model Pipeline ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model pipeline using joblib."""
    try:
        # Ensure your model filename is correct
        pipeline = joblib.load('hr_attrition_log_reg_pipeline.joblib')
        return pipeline
    except FileNotFoundError:
        st.error("Error: Model file not found. "
                 "Please ensure 'hr_attrition_log_reg_pipeline.joblib' is in the same directory.")
        return None

model_pipeline = load_model()

# --- Feature Engineering Function (CRITICAL FIX) ---
# This function replicates the pd.qcut logic from the training phase.
# NOTE: The bin edges below are the EXACT quantiles calculated on the full dataset,
# which ensures the 'IncomeBand' feature is generated correctly for the pipeline.
def get_income_band(monthly_income):
    """Assigns the IncomeBand label based on the MonthlyIncome value."""
    # Monthly Income Quantiles (tertiles) from the original dataset:
    # 33.3% Quantile (approx): 3279.88
    # 66.6% Quantile (approx): 7123.00
    
    if monthly_income <= 3279.88:
        return 'Low'
    elif monthly_income <= 7123.00:
        return 'Medium'
    else:
        return 'High'

# --- 2. Streamlit App Layout ---

st.title("👨‍💼 Employee Attrition Prediction App")
st.markdown("Predict the likelihood of an employee leaving the company.")

if model_pipeline is not None:
    # --- Input Form ---
    with st.form(key='attrition_form'):
        st.header("Employee Data Input")
        st.markdown("---")

        # Define all user inputs
        col1, col2, col3 = st.columns(3)
        
        # Numeric Inputs
        age = col1.slider("Age", min_value=18, max_value=60, value=35)
        daily_rate = col2.number_input("Daily Rate", min_value=102, max_value=1499, value=802)
        distance_from_home = col3.slider("Distance From Home (miles)", min_value=1, max_value=29, value=10)

        col4, col5, col6 = st.columns(3)
        education = col4.selectbox("Education Level", (1, 2, 3, 4, 5), index=2, help="1: Below College, 5: Doctor")
        environment_satisfaction = col5.selectbox("Environment Satisfaction", (1, 2, 3, 4), index=2, help="1: Low, 4: High")
        hourly_rate = col6.number_input("Hourly Rate", min_value=30, max_value=100, value=65)

        col7, col8, col9 = st.columns(3)
        job_involvement = col7.selectbox("Job Involvement", (1, 2, 3, 4), index=2, help="1: Low, 4: High")
        job_level = col8.selectbox("Job Level", (1, 2, 3, 4, 5), index=0)
        job_satisfaction = col9.selectbox("Job Satisfaction", (1, 2, 3, 4), index=2, help="1: Low, 4: High")

        col10, col11, col12 = st.columns(3)
        monthly_income = col10.number_input("Monthly Income (USD)", min_value=1000, max_value=20000, value=5000, step=100)
        monthly_rate = col11.number_input("Monthly Rate", min_value=2094, max_value=26999, value=14261)
        num_companies_worked = col12.slider("Num Companies Worked", min_value=0, max_value=9, value=2)

        col13, col14, col15 = st.columns(3)
        percent_salary_hike = col13.slider("Percent Salary Hike", min_value=11, max_value=25, value=15)
        performance_rating = col14.selectbox("Performance Rating", (3, 4), index=0, help="3: Excellent, 4: Outstanding")
        relationship_satisfaction = col15.selectbox("Relationship Satisfaction", (1, 2, 3, 4), index=2, help="1: Low, 4: High")

        col16, col17, col18 = st.columns(3)
        stock_option_level = col16.slider("Stock Option Level", min_value=0, max_value=3, value=0)
        total_working_years = col17.slider("Total Working Years", min_value=0, max_value=40, value=8)
        training_times_last_year = col18.slider("Training Times Last Year", min_value=0, max_value=6, value=3)

        col19, col20, col21 = st.columns(3)
        work_life_balance = col19.selectbox("Work Life Balance", (1, 2, 3, 4), index=2, help="1: Bad, 4: Best")
        years_at_company = col20.slider("Years At Company", min_value=0, max_value=40, value=5)
        years_in_current_role = col21.slider("Years In Current Role", min_value=0, max_value=18, value=3)
        
        col22, col23 = st.columns(2)
        years_since_last_promotion = col22.slider("Years Since Last Promotion", min_value=0, max_value=15, value=1)
        years_with_curr_manager = col23.slider("Years With Current Manager", min_value=0, max_value=17, value=3)
        
        # Categorical Inputs
        business_travel = st.selectbox("Business Travel", ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'), index=0)
        department = st.selectbox("Department", ('Research & Development', 'Sales', 'Human Resources'), index=0)
        education_field = st.selectbox("Education Field", ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'), index=0)
        gender = st.selectbox("Gender", ('Male', 'Female'), index=0)
        job_role = st.selectbox("Job Role", ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'), index=1)
        marital_status = st.selectbox("Marital Status", ('Married', 'Single', 'Divorced'), index=0)
        over_time = st.selectbox("Works Over Time?", ("Yes", "No"))

        # Submit button
        submit_button = st.form_submit_button(label='Predict Attrition')

    # --- 3. Prediction Logic ---
    if submit_button:
        # --- Feature Engineering (CRITICAL: Must be done here) ---
        tenure = years_at_company / np.maximum(total_working_years, 1)
        promotion_gap = years_since_last_promotion / np.maximum(years_at_company, 1)
        income_band = get_income_band(monthly_income)
        
        # --- Final Dataframe Creation ---
        input_data = {
            'Age': age,
            'BusinessTravel': business_travel,
            'DailyRate': daily_rate,
            'Department': department,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'EducationField': education_field,
            'EnvironmentSatisfaction': environment_satisfaction,
            'Gender': gender,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies_worked,
            'OverTime': over_time,
            'PercentSalaryHike': percent_salary_hike,
            'PerformanceRating': performance_rating,
            'RelationshipSatisfaction': relationship_satisfaction,
            'StockOptionLevel': stock_option_level,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times_last_year,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager,
            'Tenure': tenure, # Engineered Feature
            'PromotionGap': promotion_gap, # Engineered Feature
            'IncomeBand': income_band # Engineered Feature (Categorical)
        }
        
        # Ensure the DataFrame is created with the correct list of values
        # and all columns are present.
        input_df = pd.DataFrame([input_data])
        
        # Predict probability
        prediction_proba = model_pipeline.predict_proba(input_df)[:, 1][0]
        
        # Get the prediction label
        prediction = model_pipeline.predict(input_df)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"The model predicts **HIGH** likelihood of Attrition (Employee may leave).")
            st.metric(label="Predicted Attrition Probability (Yes)", value=f"{prediction_proba:.2%}")
        else:
            st.success(f"The model predicts **LOW** likelihood of Attrition (Employee likely to stay).")
            st.metric(label="Predicted Attrition Probability (Yes)", value=f"{prediction_proba:.2%}")

        st.markdown(f"***Model Note:*** The raw prediction probability of the employee leaving is **{prediction_proba:.2f}**.")
        
# --- Instructions for the user ---
st.sidebar.title("How to Run")
st.sidebar.markdown(f"""
1.  **Save the Model:** Ensure you have saved your pipeline using `joblib.dump(log_reg_model, 'hr_attrition_log_reg_pipeline.joblib')`.
2.  **Create App File:** Save the corrected code above as `app.py` in the same directory as the `.joblib` file.
3.  **Run Streamlit:** Open your terminal in that directory and run:
    ```bash
    streamlit run app.py
    ```
""")