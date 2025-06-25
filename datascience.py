import streamlit as st
import pandas as pd

# 🔹 Load dataset safely
try:
    data = pd.read_csv("Data Science Job Salaries.csv")
    data.columns = data.columns.str.strip()  # remove spaces, hidden characters
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# 🔹 Show clean column names
st.write("Column names:", data.columns.tolist())

# 🔹 Title
st.title('Data Science Job Salaries')

# 🔹 Summary statistics
if not data.empty:
    st.subheader("Summary Statistics")
    st.write(data.describe())
else:
    st.warning("Dataset is empty.")

# 🔹 Line chart
st.subheader("Salary Trend (Line Chart)")
if 'salary_in_usd' in data.columns:
    st.line_chart(data['salary_in_usd'])
else:
    st.error("Column 'salary_in_usd' not found in the dataset.")

# 🔹 Filter by job title
st.subheader("Filter by Job Title")
if 'job_title' in data.columns and 'salary_in_usd' in data.columns:
    try:
        job_filter = st.selectbox('Select Job Title', data['job_title'].dropna().unique())
        filtered_data = data[data['job_title'] == job_filter]

        st.subheader(f"Salaries for '{job_filter}'")
        if not filtered_data.empty:
            st.bar_chart(filtered_data['salary_in_usd'])
        else:
            st.warning("No salary data available for this job title.")
    except Exception as e:
        st.error(f"An error occurred while filtering job titles: {e}")
else:
    st.error("Required columns 'job_title' or 'salary_in_usd' are missing.")
