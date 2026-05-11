# Diabetes Risk Prediction Model 🩺

![Diabetes Prediction](/deployment/images/diabetes-ml.jpg)

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![Plotly](https://img.shields.io/badge/Plotly-Latest-green.svg)

## 📊 Project Overview

This project aims to develop a machine learning model that predicts an individual's risk of developing diabetes. By utilizing the CDC Diabetes Health Indicators dataset, we have created a robust classification model that achieves over 82% recall in identifying potential diabetes cases.

### 🎯 Objective

To build a predictive model that accurately assesses an individual's likelihood of having or developing diabetes, enabling early intervention and better public health strategies.

## 📝 Table of Contents

- [🔍 Key Features](#-key-features)
- [📈 Results](#-results)
- [🛠️ Technologies Used](#️-technologies-used)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🖥️ Web Application](#️-web-application)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🔍 Prediction Module](#-prediction-module)
- [🧪 Model Inference](#-model-inference)
- [📊 Key Insights](#-key-insights)
- [🔮 Future Work](#-future-work)
- [🔗 Links](#-links)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [👤 Author](#-author)

## 🔍 Key Features

- **Data Source**: CDC Diabetes Health Indicators dataset.
- **Primary Metric**: Recall (to minimize false negatives).
- **Target Performance**: Achieve at least 82% recall.
- **Models Evaluated**: KNN, SVM, Logistic Regression, Decision Tree, Random Forest, XGBoost.
- **Best Performing Model**: Support Vector Machine (SVM).
- **Interactive Web Application**: Deployed on **Hugging Face Spaces** for global accessibility.

## 📈 Results

- **Test Set Recall**: 82%.
- **Model Performance**: Consistent across training and test sets, indicating good generalization.

## 🛠️ Technologies Used

- **Programming Language**: 🐍 Python 3.9.
- **Libraries**:
  - 🐼 Pandas & 🔢 NumPy for data manipulation.
  - 🧠 Scikit-learn for model building.
  - 🌊 Streamlit for web application.
  - 📊 Plotly for interactive visualizations.
  - 🔗 Phik for advanced correlation analysis.

## 📁 Project Structure

```
├── data/
│   └── cdc_diabetes_health_indicators.csv
├── deployment/
│   ├── app.py
│   ├── eda.py
│   ├── prediction.py
│   ├── images/
│   │   ├── diabetes-ml.jpg
│   │   ├── homepage.gif
│   │   ├── eda.gif
│   │   └── prediction.gif
│   └── requirements.txt
├── notebooks/
│   └── diabetes-prediction-inference.ipynb
├── models/
│   └── tuned_model.pkl
├── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher.
- pip package manager.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ayudhaamari/diabetes-risk-prediction.git
   cd diabetes-risk-prediction
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required packages**:

   ```bash
   pip install -r deployment/requirements.txt
   ```

4. **Download the dataset**:

   Ensure the dataset `cdc_diabetes_health_indicators.csv` is in the `data/` directory.

### Running the Application

Start the Streamlit web application locally:

```bash
streamlit run deployment/app.py
```

The application will be available at `http://localhost:8501`.

## 🖥️ Web Application

The project includes a web application built with Streamlit and **deployed on Hugging Face Spaces**, making it easily accessible to users worldwide.

**Check out the live application here: [Diabetes Risk Prediction on Hugging Face Spaces](https://huggingface.co/spaces/amariayudha/Diabetes_Prediction)**

### Application Features

1. **Home Page**: Overview of the project and model performance.

   ![Homepage Demo](/images/homepage.gif)

2. **EDA (Exploratory Data Analysis)**: Interactive exploration of the dataset.

   ![EDA Demo](/images/eda.gif)

3. **Prediction**: Input health information to receive a personalized diabetes risk assessment.

   ![Prediction Demo](/images/prediction.gif)

### Application Structure

- **`app.py`**: Main application script that integrates all components.
- **`eda.py`**: Contains the code for the EDA page.
- **`prediction.py`**: Handles user input and displays predictions.

## 📊 Exploratory Data Analysis

The EDA module provides interactive visualizations of the dataset:

- **Distribution Plots**: Understand the distribution of individual features.
- **Correlation Heatmaps**: Analyze relationships between features.
- **Box Plots**: Identify outliers and understand data spread.

For implementation details, see [`deployment/eda.py`](deployment/eda.py).

## 🔍 Prediction Module

The prediction module allows users to input their health information and receive a diabetes risk assessment based on the trained SVM model.

For the prediction implementation, refer to [`deployment/prediction.py`](deployment/prediction.py).

## 🧪 Model Inference

For a detailed example of how to use the trained model for inference, check out the Jupyter notebook [`diabetes-prediction-inference.ipynb`](notebooks/diabetes-prediction-inference.ipynb). This notebook demonstrates how to load the model and make predictions on new data.

## 📊 Key Insights

- **Age Factor**: Individuals aged 60-64 have the highest prevalence of diabetes.
- **BMI Correlation**: Higher BMI is strongly associated with increased diabetes risk.
- **Lifestyle Impact**:
  - Regular physical activity is linked to lower risk.
  - A healthy diet contributes to decreased risk.
- **Comorbid Conditions**: High blood pressure and cholesterol levels are significant risk factors.

## 🔮 Future Work

- **Deep Learning**: Explore neural networks for potentially improved performance.
- **Real-Time Assessment**: Implement a system for real-time risk assessment in clinical settings.
- **Model Interpretability**: Enhance explainability using tools like SHAP or LIME.
- **Feature Engineering**: Incorporate additional features or external datasets for better accuracy.

## 🔗 Links

- **Live Web Application**: [Diabetes Risk Prediction on Hugging Face Spaces](https://huggingface.co/spaces/amariayudha/Diabetes_Prediction)
- **Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the dataset.
- **Hugging Face** for hosting the web application.
- **Open Source Community** for the amazing tools and libraries.

## 👤 Author

**Ayudha Amari Hirtranusi**

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?logo=Linkedin&logoColor=white)]([your_linkedin_profile_link](https://www.linkedin.com/in/ayudhaamari/)) https://www.linkedin.com/in/ayudhaamari/

[![Email](https://img.shields.io/badge/Email-D14836?logo=Gmail&logoColor=white)](mailto:amariayudha@gmail.com) amariayudha@gmail.com

[![GitHub](https://img.shields.io/badge/GitHub-100000?logo=GitHub&logoColor=white)](https://github.com/ayudhaamari) https://github.com/ayudhaamari
