import streamlit as st
import pandas as pd

# 🔹 Load dataset
data = pd.read_csv("Data Science Job Salaries.csv")

# 🔹 Clean column names: remove extra spaces, newlines, carriage returns
data.columns = data.columns.str.strip()

# 🔹 Debugging: Show actual column names
st.write("🧾 Column names:", data.columns.tolist())

# 🔹 App Title
st.title('📊 Data Science Job Salaries')

# 🔹 Show summary stats
st.subheader("📈 Summary Statistics")
st.write(data.describe())

# 🔹 Line Chart for Salary
st.subheader("💹 Salary Trend (Line Chart)")
if 'salary_in_usd' in data.columns:
    st.line_chart(data['salary_in_usd'])
else:
    st.error("❌ 'salary_in_usd' column not found in dataset.")

# 🔹 Filter by Job Title
st.subheader("🔍 Filter by Job Title")
if 'job_title' in data.columns and 'salary_in_usd' in data.columns:
    job_filter = st.selectbox('Select Job Title', data['job_title'].unique())
    filtered_data = data[data['job_title'] == job_filter]

    st.subheader(f"📊 Salaries for '{job_filter}'")
    if not filtered_data.empty:
        st.bar_chart(filtered_data['salary_in_usd'])
    else:
        st.warning("⚠️ No data found for this job title.")
else:
    st.error("❌ 'job_title' or 'salary_in_usd' column not found.")




