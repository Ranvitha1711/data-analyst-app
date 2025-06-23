import streamlit as st
import pandas as pd

# Load your dataset
data = pd.read_csv('Data Science Job Salaries.csv')
import streamlit as st
import pandas as pd

# ✅ Load dataset
data = pd.read_csv("Data Science Job Salaries.csv")

# ✅ Clean column names (strip whitespaces, hidden characters like \r or \n)
data.columns = data.columns.str.strip()

# ✅ App Title
st.title('📊 Data Science Job Salaries')

# ✅ Show actual column names (for debug/help)
st.write("🧾 Column names:", data.columns.tolist())

# ✅ Show summary statistics
st.subheader("📈 Summary Statistics")
st.write(data.describe())

# ✅ Line chart for salary trend (can be cumulative or index)
st.subheader("💹 Salary Trend (Line Chart)")
if 'salary_in_usd' in data.columns:
    st.line_chart(data['salary_in_usd'])
else:
    st.error("Column 'salary_in_usd' not found in dataset.")

# ✅ Filter by Job Title
st.subheader("🔍 Filter by Job Title")
job_filter = st.selectbox('Select Job Title', data['job_title'].unique())
filtered_data = data[data['job_title'] == job_filter]

st.subheader(f"📊 Salaries for '{job_filter}'")
if not filtered_data.empty:
    st.bar_chart(filtered_data['salary_in_usd'])
else:
    st.warning("No data found for this job title.")



