import streamlit as st
import pandas as pd

# Load your dataset
data = pd.read_csv('Data Science Job Salaries.csv')
import streamlit as st

st.title('Data Science Job Salaries')

# Upload summary statistics
st.write(data.describe())

# Visualization
st.line_chart(data['salary_in_usd'])

# Filter by job title
job_filter = st.selectbox('Select Job Title',
data['job_title'].unique())
filtered_data = data[data['job_title'] == job_filter]

st.bar_chart(filtered_data['salary_in_usd'])






