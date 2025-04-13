import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Load Model and Prepare Encoders ---
@st.cache_resource
def load_model_and_encoders():
    # Load trained model
    with open("my_random_forest_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load training dataset
    dataset = pd.read_csv("Salary Data.csv").dropna()

    # Encode Gender
    gender_encoder = LabelEncoder()
    dataset["Gender"] = gender_encoder.fit_transform(dataset["Gender"])

    # Encode Education
    edu_level_encoder = LabelEncoder()
    dataset["Education Level"] = edu_level_encoder.fit_transform(dataset["Education Level"])

    # One-hot encoding of Job Title
    dataset = pd.get_dummies(dataset, columns=["Job Title"], prefix="JobTitle", drop_first=False)
    feature_columns = dataset.drop(columns=["Salary"]).columns

    # Scale numerical features
    scaler = StandardScaler()
    dataset[["Age", "Years of Experience"]] = scaler.fit_transform(dataset[["Age", "Years of Experience"]])

    # Get salary mean and std for inverse transform
    salary_mean = dataset["Salary"].mean()
    salary_std = dataset["Salary"].std()

    return model, gender_encoder, edu_level_encoder, scaler, feature_columns, salary_mean, salary_std

model, gender_enc, edu_enc, scaler, feat_cols, sal_mean, sal_std = load_model_and_encoders()

# --- Streamlit UI ---
st.title("💼 Salary Prediction App")
st.markdown("Enter the employee details to predict the expected salary.")
# --------------------
# Dropdown Job Titles
job_titles = [
    'Account Manager', 'Accountant', 'Administrative Assistant', 'Business Analyst',
    'Business Development Manager', 'Business Intelligence Analyst', 'CEO',
    'Chief Data Officer', 'Chief Technology Officer', 'Content Marketing Manager',
    'Copywriter', 'Creative Director', 'Customer Service Manager', 'Customer Service Rep',
    'Customer Service Representative', 'Customer Success Manager', 'Customer Success Rep',
    'Data Analyst', 'Data Entry Clerk', 'Data Scientist', 'Digital Content Producer',
    'Digital Marketing Manager', 'Director', 'Director of Business Development',
    'Director of Engineering', 'Director of Finance', 'Director of HR',
    'Director of Human Capital', 'Director of Human Resources', 'Director of Marketing',
    'Director of Operations', 'Director of Product Management', 'Director of Sales',
    'Director of Sales and Marketing', 'Event Coordinator', 'Financial Advisor',
    'Financial Analyst', 'Financial Manager', 'Graphic Designer', 'HR Generalist',
    'HR Manager', 'Help Desk Analyst', 'Human Resources Director', 'IT Manager',
    'IT Support', 'IT Support Specialist', 'Junior Account Manager', 'Junior Accountant',
    'Junior Advertising Coordinator', 'Junior Business Analyst',
    'Junior Business Development Associate', 'Junior Business Operations Analyst',
    'Junior Copywriter', 'Junior Customer Support Specialist', 'Junior Data Analyst',
    'Junior Data Scientist', 'Junior Designer', 'Junior Developer',
    'Junior Financial Advisor', 'Junior Financial Analyst', 'Junior HR Coordinator',
    'Junior HR Generalist', 'Junior Marketing Analyst', 'Junior Marketing Coordinator',
    'Junior Marketing Manager', 'Junior Marketing Specialist', 'Junior Operations Analyst',
    'Junior Operations Coordinator', 'Junior Operations Manager', 'Junior Product Manager',
    'Junior Project Manager', 'Junior Recruiter', 'Junior Research Scientist',
    'Junior Sales Representative', 'Junior Social Media Manager',
    'Junior Social Media Specialist', 'Junior Software Developer',
    'Junior Software Engineer', 'Junior UX Designer', 'Junior Web Designer',
    'Junior Web Developer', 'Marketing Analyst', 'Marketing Coordinator',
    'Marketing Manager', 'Marketing Specialist', 'Network Engineer', 'Office Manager',
    'Operations Analyst', 'Operations Director', 'Operations Manager',
    'Principal Engineer', 'Principal Scientist', 'Product Designer', 'Product Manager',
    'Product Marketing Manager', 'Project Engineer', 'Project Manager',
    'Public Relations Manager', 'Recruiter', 'Research Director', 'Research Scientist',
    'Sales Associate', 'Sales Director', 'Sales Executive', 'Sales Manager',
    'Sales Operations Manager', 'Sales Representative', 'Senior Account Executive',
    'Senior Account Manager', 'Senior Accountant', 'Senior Business Analyst',
    'Senior Business Development Manager', 'Senior Consultant', 'Senior Data Analyst',
    'Senior Data Engineer', 'Senior Data Scientist', 'Senior Engineer',
    'Senior Financial Advisor', 'Senior Financial Analyst', 'Senior Financial Manager',
    'Senior Graphic Designer', 'Senior HR Generalist', 'Senior HR Manager',
    'Senior HR Specialist', 'Senior Human Resources Coordinator',
    'Senior Human Resources Manager', 'Senior Human Resources Specialist',
    'Senior IT Consultant', 'Senior IT Project Manager', 'Senior IT Support Specialist',
    'Senior Manager', 'Senior Marketing Analyst', 'Senior Marketing Coordinator',
    'Senior Marketing Director', 'Senior Marketing Manager', 'Senior Marketing Specialist',
    'Senior Operations Analyst', 'Senior Operations Coordinator', 'Senior Operations Manager',
    'Senior Product Designer', 'Senior Product Development Manager', 'Senior Product Manager',
    'Senior Product Marketing Manager', 'Senior Project Coordinator', 'Senior Project Manager',
    'Senior Quality Assurance Analyst', 'Senior Research Scientist', 'Senior Researcher',
    'Senior Sales Manager', 'Senior Sales Representative', 'Senior Scientist',
    'Senior Software Architect', 'Senior Software Developer', 'Senior Software Engineer',
    'Senior Training Specialist', 'Senior UX Designer', 'Social Media Manager',
    'Social Media Specialist', 'Software Developer', 'Software Engineer', 'Software Manager',
    'Software Project Manager', 'Strategy Consultant', 'Supply Chain Analyst',
    'Supply Chain Manager', 'Technical Recruiter', 'Technical Support Specialist',
    'Technical Writer', 'Training Specialist', 'UX Designer', 'UX Researcher',
    'VP of Finance', 'VP of Operations', 'Web Developer'
]
# --------------------

# Streamlit Input UI
age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", gender_enc.classes_)
edu_level = st.selectbox("Education Level", edu_enc.classes_)
exp = st.slider("Years of Experience", min_value=0, max_value=40, value=5)
job_title = st.selectbox("Job Title", job_titles)

# User Inputs



# --- Predict Button ---
if st.button("Predict Salary"):
    try:
        # Encode inputs
        gender_encoded = gender_enc.transform([gender])[0]
        edu_encoded = edu_enc.transform([edu_level])[0]

        # Prepare input DataFrame
        input_df = pd.DataFrame(columns=feat_cols)
        input_df.loc[0] = [0] * len(feat_cols)
        input_df.at[0, "Age"] = age
        input_df.at[0, "Gender"] = gender_encoded
        input_df.at[0, "Education Level"] = edu_encoded
        input_df.at[0, "Years of Experience"] = exp

        job_title_col = f"JobTitle_{job_title}"
        if job_title_col in input_df.columns:
            input_df.at[0, job_title_col] = 1
        else:
            st.warning(f"⚠️ Job title '{job_title}' not found in training data. Please try another title.")
            st.stop()

        # Scale numerical features
        input_df[["Age", "Years of Experience"]] = scaler.transform(
            input_df[["Age", "Years of Experience"]]
        )

        # Predict
        prediction = model.predict(input_df)[0]
        final_salary = prediction * sal_std + sal_mean

        st.success(f"💰 Predicted Salary: **${final_salary:,.2f}**")

    except Exception as e:
        st.error(f"Error occurred: {e}")
