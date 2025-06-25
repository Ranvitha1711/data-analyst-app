import streamlit as st
import pandas as pd

# ✅ Load and clean the dataset
try:
    data = pd.read_csv("Data Science Job Salaries.csv")
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")  # standardize
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ✅ Debug: Display actual column names
st.write("🧾 Actual Columns:", data.columns.tolist())

# ✅ Title
st.title("📊 Data Science Job Salaries")

# ✅ Summary
st.subheader("📈 Summary Statistics")
st.write(data.describe())

# ✅ Salary Trend Chart
st.subheader("💹 Salary Trend (Line Chart)")
if 'salary_in_usd' in data.columns:
    st.line_chart(data['salary_in_usd'])
else:
    st.error("❌ Column 'salary_in_usd' not found in the dataset.")

# ✅ Filter by Job Title
st.subheader("🔍 Filter by Job Title")
if 'job_title' in data.columns:
    job_filter = st.selectbox("Select Job Title", data['job_title'].dropna().unique())
    filtered_data = data[data['job_title'] == job_filter]

    st.subheader(f"💼 Salary Distribution for '{job_filter}'")
    if not filtered_data.empty:
        st.bar_chart(filtered_data['salary_in_usd'])
    else:
        st.warning("⚠️ No data found for this job title.")
else:
    st.error("❌ Column 'job_title' not found in the dataset.")
