# Import necessary libraries
import streamlit as st
import eda
import prediction

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Navigation sidebar
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🔍 Prediction"])

    if page == "🏠 Home":
        # Sidebar content for Home page
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 About the Model")
        recall = 0.82
        st.sidebar.write("🎯 Model Recall:")
        st.sidebar.progress(recall)
        st.sidebar.write(f"{recall:.2%}")
        st.sidebar.write("**🤔 What is Recall?**")
        st.sidebar.write("Recall measures how well our model identifies people who actually have diabetes.")
        st.sidebar.write("**💡 What does this mean?**")
        st.sidebar.write("Out of all the people who truly have diabetes, our model correctly identifies 82% of them.")
        st.sidebar.write("This helps us catch most cases, *reducing the chance of missing someone who needs attention*")

        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Fun Fact")
        st.sidebar.info("Diabetes affects about 422 million people worldwide, according to the World Health Organization.")

        # Main content for Home page
        st.title("🩺 Welcome to Diabetes Risk Assessment Tool")
        st.write("""
        This application provides functionalities for Exploratory Data Analysis and 
        Prediction regarding diabetes risk. Use the navigation pane on the left to 
        select the module you wish to utilize.
        """)
        
        
        # Dataset information
        st.write("#### 📊 Dataset")
        st.info("""
        The dataset is sourced from the CDC Diabetes Health Indicators. For additional 
        details, visit [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).
        """)
        
        # Problem Statement
        st.write("#### ⚠️ Problem Statement")
        st.warning("""
        Diabetes is a growing public health issue, affecting millions of people worldwide. **Early detection and effective intervention are critical** to managing this condition and reducing associated health risks. The CDC provides a dataset that includes health indicators such as BMI, blood pressure, age, physical activity, and other factors that could be linked to diabetes. By understanding these factors and creating predictive models, public health agencies and individuals can take proactive steps toward better management and prevention of diabetes. **As a data scientist from hospital "XYZ"**, you are tasked to develop a machine learning model that can predict an individual's risk of developing diabetes based on survey information from the Behavioral Risk Factor Surveillance System (BRFSS). 
        
        The prevalence of diabetes has significantly increased, creating a public health burden. 
        The goal is to develop a machine learning model that can predict an individual's risk of 
        developing diabetes with 80% recall within six months, enabling better early intervention strategies.
        """)
        
        # Project Objective
        st.write("#### 🎯 Objective")
        st.success("""
        This project focuses on creating a classification model to predict diabetes by evaluating 
        KNN, SVM, Logistic Regression, Decision Tree, Random Forest, and XGBoost. Model performance 
        will be primarily assessed using Recall to measure effectiveness in identifying diabetes cases 
        to minimize false negatives.
        """)

    elif page == "📊 EDA":
        # Run the EDA module
        eda.run()
    
    elif page == "🔍 Prediction":
        # Run the Prediction module
        prediction.run()

if __name__ == "__main__":
    main()