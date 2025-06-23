import streamlit as st
import pandas as pd

# Load the dataset
try:
    data = pd.read_csv("Data Science Job Salaries.csv")
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Debugging help: show cleaned column names
st.write("Actual Columns:", data.columns.tolist())

# App Title
st.title("Data Science Job Salaries")

# Summary Statistics
st.subheader("Summary Statistics")
st.write(data.describe())

# Salary Trend Chart
st.subheader("Salary Trend (Line Chart)")
if 'salary_in_usd' in data.columns:
    st.line_chart(data['salary_in_usd'])
else:
    st.error("Column 'salary_in_usd' not found in the dataset.")

# Job Title Filter
st.subheader("Filter by Job Title")
if 'job_title' in data.columns and 'salary_in_usd' in data.columns:
    job_filter = st.selectbox('Select Job Title', data['job_title'].dropna().unique())
    filtered_data = data[data['job_title'] == job_filter]

    st.subheader(f"Salaries for '{job_filter}'")
    if not filtered_data.empty:
        st.bar_chart(filtered_data['salary_in_usd'])
    else:
        st.warning("No data found for this job title.")
else:
    st.error("Required columns 'job_title' or 'salary_in_usd' are missing.")





