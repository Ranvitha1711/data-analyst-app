import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("DataAnalyst.csv")
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Set page configuration first (as required)
st.set_page_config(
    page_title="Data Analyst Salary Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Simple prediction function as replacement for ML model
@st.cache_resource
def get_simple_model():
    # This is a simple rule-based prediction function
    # It mimics a model's prediction without requiring sklearn
    def predict(X):
        # X should be a list of [rating, tech_skills, size_encoded]
        predictions = []

        for features in X:
            rating, tech_skills, size = features

            # Base salary
            base_salary = 50000

            # Company rating impact (higher rating = higher salary)
            rating_factor = 5000 * (rating - 1)  # 0 to 20000 increase

            # Tech skills impact (0-4 scale)
            skills_factor = 8000 * tech_skills  # 0 to 32000 increase

            # Company size impact
            size_factors = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
            size_factor = size_factors[int(size)]

            # Calculate final salary prediction with some randomness
            salary = base_salary + rating_factor + skills_factor + size_factor
            salary *= (1 + np.random.normal(0, 0.05))  # Add small random variation

            predictions.append(salary)

        return np.array(predictions)

    # Return an object with a predict method to mimic a model interface
    class SimpleModel:
        def predict(self, X):
            return predict(X)

    return SimpleModel()


# Sidebar for advanced settings
with st.sidebar:
    st.header("About")
    st.info("This app predicts salaries for data analyst positions based on company characteristics.")

    st.header("Advanced Features")
    show_analysis = st.checkbox("Show salary analysis", value=False)
    compare_mode = st.checkbox("Enable comparison mode", value=False)

    st.header("Model Information")
    st.write("Using simple rule-based prediction")
    st.caption("Model features: Company Rating, Tech Skills, Company Size")

# Main content
st.title("📊 Data Analyst Job Salary Predictor")
st.write("Predict the average salary for a data analyst job based on company characteristics.")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Input sliders
    rating = st.slider("🌟 Company Rating", 1.0, 5.0, 3.5, step=0.1)
    tech_skills = st.slider("🧠 Tech Skills Score (0–4)", 0, 4, 2)

with col2:
    size_label = st.selectbox("🏢 Company Size", [
        "Unknown",
        "1 to 50 employees",
        "51 to 200 employees",
        "201 to 500 employees",
        "501 to 1000 employees",
        "1001 to 5000 employees",
        "5001 to 10000 employees",
        "10000+ employees"
    ])

    # Encode size to match model
    size_mapping = {
        "Unknown": 0,
        "1 to 50 employees": 1,
        "51 to 200 employees": 2,
        "201 to 500 employees": 3,
        "501 to 1000 employees": 4,
        "1001 to 5000 employees": 5,
        "5001 to 10000 employees": 6,
        "10000+ employees": 7
    }
    size_encoded = size_mapping[size_label]

    # Year Founded (optional feature)
    founded = st.number_input("🏗️ Year Founded", min_value=1900, max_value=2025, value=2010)

# Comparison mode
if compare_mode:
    st.subheader("Comparison Mode")
    st.write("Compare salaries across different company sizes with the same rating and skills")

    if st.button("Compare Across Company Sizes"):
        model = get_simple_model()
        sizes = list(size_mapping.values())
        size_labels = list(size_mapping.keys())
        predictions = []

        for size in sizes:
            pred = model.predict([[rating, tech_skills, size]])
            predictions.append(pred[0])

        # Create a comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(size_labels, predictions)
        ax.set_xlabel('Company Size')
        ax.set_ylabel('Predicted Salary ($)')
        ax.set_title('Salary Comparison by Company Size')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
else:
    # Button to trigger prediction
    if st.button("🔮 Predict Salary"):
        model = get_simple_model()
        prediction = model.predict([[rating, tech_skills, size_encoded]])

        # Display with animation
        with st.container():
            st.markdown("### Results")
            st.metric(
                label="💰 Predicted Annual Salary",
                value=f"${prediction[0]:,.2f}"
            )

            # Confidence interval (as an example)
            st.caption("Salary range may vary based on location, experience, and other factors")
            lower_bound = prediction[0] * 0.9
            upper_bound = prediction[0] * 1.1
            st.write(f"Estimated range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")

# Additional analysis section
if show_analysis:
    st.subheader("Salary Analysis")

    # Create demonstration data
    model = get_model()

    # Effect of rating on salary
    st.write("### Impact of Company Rating on Salary")
    ratings = np.linspace(1.0, 5.0, 10)
    rating_salaries = []

    for r in ratings:
        pred = model.predict([[r, tech_skills, size_encoded]])
        rating_salaries.append(pred[0])

    fig1, ax1 = plt.subplots()
    ax1.plot(ratings, rating_salaries)
    ax1.set_xlabel('Company Rating')
    ax1.set_ylabel('Predicted Salary ($)')
    st.pyplot(fig1)

    # Effect of tech skills on salary
    st.write("### Impact of Tech Skills on Salary")
    skills = np.array([0, 1, 2, 3, 4])
    skill_salaries = []

    for s in skills:
        pred = model.predict([[rating, s, size_encoded]])
        skill_salaries.append(pred[0])

    fig2, ax2 = plt.subplots()
    ax2.bar(skills, skill_salaries)
    ax2.set_xlabel('Tech Skills Score')
    ax2.set_ylabel('Predicted Salary ($)')
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.caption("By Ranvitha")
