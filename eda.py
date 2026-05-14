# Import necessary libraries
import streamlit as st  # For creating web applications
import pandas as pd  # For data manipulation and analysis
import plotly.express as px  # For interactive data visualization
from phik import phik_matrix  # For calculating correlation between mixed data types
import numpy as np  # For numerical operations

# Cache the data loading function for better performance
@st.cache_data
def load_data():
    # Load the dataset from a CSV file
    data = pd.read_csv('d:\diabetes-risk-assessment-ml-main\deployment\cdc_diabetes_health_indicators.csv')
    
    # Create age_group column based on the 'Age' column
    conditions = [
        (data['Age'] == 1),
        (data['Age'] == 2),
        (data['Age'] == 3),
        (data['Age'] == 4),
        (data['Age'] == 5),
        (data['Age'] == 6),
        (data['Age'] == 7),
        (data['Age'] == 8),
        (data['Age'] == 9),
        (data['Age'] == 10),
        (data['Age'] == 11),
        (data['Age'] == 12),
        (data['Age'] == 13)
    ]

    choices = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']

    # Use numpy.select to create the age_group column
    data['age_group'] = np.select(conditions, choices, default='Unknown')
    
    # Map binary categorical columns to 'No' and 'Yes', except for Diabetes_binary
    binary_columns = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                      'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 
                      'NoDocbcCost', 'DiffWalk']
    
    # Convert binary columns from 0/1 to No/Yes
    for col in binary_columns:
        data[col] = data[col].map({0: 'No', 1: 'Yes'})
    
    # Map GenHlth to categorical values
    data['GenHlth'] = data['GenHlth'].map({1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'})
    
    # Map Income to categorical values
    income_mapping = {
        1: 'Less than $10,000',
        2: '$10,000 to less than $15,000',
        3: '$15,000 to less than $20,000',
        4: '$20,000 to less than $25,000',
        5: '$25,000 to less than $35,000',
        6: '$35,000 to less than $50,000',
        7: '$50,000 to less than $75,000',
        8: '$75,000 or more'
    }
    data['Income'] = data['Income'].map(income_mapping)
    
    # Map Education to categorical values
    education_mapping = {
        1: 'Never / kindergarten',
        2: 'Elementary school',
        3: 'Junior school',
        4: 'High school Graduate',
        5: 'College',
        6: 'College graduate'
    }
    data['Education'] = data['Education'].map(education_mapping)
    
    # Map Sex to Female/Male
    data['Sex'] = data['Sex'].map({0: 'Female', 1: 'Male'})
    
    return data

def run():
    # Set the title of the Streamlit app
    st.title('📊 Exploratory Data Analysis - Diabetes Risk Factors')

    # Load the data using the cached function
    data = load_data()

    # Dataset info section
    if st.checkbox('Show dataset info'):
        st.subheader("Dataset Information")
        
        # Create two columns for layout
        col1, col2 = st.columns([3, 1])
        
        # Display dataset features in the first column
        with col1:
            st.write("Diabetes Dataset Features (First 10 rows):")
            st.dataframe(data.head(10))
        
        # Display target distribution in the second column
        with col2:
            st.write("Target Distribution:")
            target_df = pd.DataFrame(data['Diabetes_binary'].value_counts()).reset_index()
            target_df.columns = ['Diabetes Status', 'Count']
            target_df['Diabetes Status'] = target_df['Diabetes Status'].map({1: 'Diabetes', 0: 'No Diabetes'})
            st.dataframe(target_df)
        
        # Display insight about the dataset
        st.write("The dataset contains various health indicators that may be associated with diabetes. The target variable `Diabetes_binary` shows an imbalanced distribution, which is common in medical datasets. This imbalance is **crucial to consider** in our analysis and modeling approach. It suggests that we may need to employ **techniques such as oversampling, undersampling, or using class weights** to address this imbalance in our predictive models. The presence of multiple health indicators allows for a **comprehensive analysis of potential risk factors**, which can lead to more **robust and insightful predictions** of diabetes risk.")

    # Column info section
    if st.checkbox('Show columns info'):
        st.subheader("Column Information")
        
        # Create a DataFrame with column information
        column_info = pd.DataFrame({
            'Variable': [
                'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth',
                'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
            ],
            'Description': [
                '0 = no diabetes, 1 = prediabetes or diabetes',
                '0 = no high BP, 1 = high BP',
                '0 = no high cholesterol, 1 = high cholesterol',
                '0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years',
                'Body Mass Index',
                'Have you smoked at least 100 cigarettes in your entire life? (0 = no, 1 = yes)',
                '(Ever told) you had a stroke (0 = no, 1 = yes)',
                'Coronary heart disease (CHD) or myocardial infarction (MI) (0 = no, 1 = yes)',
                'Physical activity in past 30 days - not including job (0 = no, 1 = yes)',
                'Consume fruit 1 or more times per day (0 = no, 1 = yes)',
                'Consume vegetables 1 or more times per day (0 = no, 1 = yes)',
                'Heavy drinkers (men > 14 drinks/week, women > 7 drinks/week) (0 = no, 1 = yes)',
                'Have any kind of health care coverage (0 = no, 1 = yes)',
                'Skipped doctor visit due to cost in the past 12 months (0 = no, 1 = yes)',
                'General health rating (1 = excellent, 5 = poor)',
                'Days of poor mental health in past 30 days (scale: 1-30 days)',
                'Days of poor physical health in past 30 days (scale: 1-30 days)',
                'Difficulty walking or climbing stairs (0 = no, 1 = yes)',
                'Sex (Female = 0, Male = 1)',
                'Age category (1 = 18-24, 9 = 60-64, 13 = 80 or older)',
                'Education level (1 = Elementary, 6 = College graduate)',
                'Income level (1 = less than $10,000, 8 = $75,000 or more)'
            ]
        })
        
        # Display the column information
        st.dataframe(column_info)
        
        # Display insight about the columns
        st.write("This table provides a comprehensive overview of all variables in the dataset, including their descriptions and possible values. Understanding these variables is crucial for interpreting the analysis results and making informed decisions in the feature selection process. The **diversity of variables** spans across various aspects of health and lifestyle, including **physiological measurements** (like `BMI` and blood pressure), **behavioral factors** (such as smoking and physical activity), **socioeconomic indicators** (like `Education` and `Income`), and **general health assessments**. This **rich set of features** allows for a **multifaceted analysis** of diabetes risk factors, potentially uncovering **complex interactions** and **hidden patterns** that might not be apparent when considering these factors in isolation. It's important to note that some variables are **binary** (yes/no), while others are **categorical** or **continuous**, which will require **different analytical approaches** and may influence the choice of machine learning models in the prediction phase.")

    # Feature selection in the sidebar
    st.sidebar.subheader("🔍 Feature Selection")
    # Define numerical and categorical features
    numerical_features = ['BMI', 'MentHlth', 'PhysHlth', 'age_group', 'Income']
    categorical_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex', 'Education', 'GenHlth']
    
    # Allow user to select numerical features
    selected_num_features = st.sidebar.multiselect(
        "Select numerical features for analysis",
        options=numerical_features,
        default=[]
    )
    
    # Allow user to select categorical features
    selected_cat_features = st.sidebar.multiselect(
        "Select categorical features for analysis",
        options=categorical_features,
        default=[]
    )

    # Visualization options in the sidebar
    st.sidebar.subheader("📊 Visualization Options")
    show_distribution = st.sidebar.checkbox("📈 Show Distribution Plots", True)
    show_boxplot = st.sidebar.checkbox("📦 Show Box Plots", False, help="This visualization option is only applicable to true numerical features, namely BMI, MentHlth, and PhysHlth.")
    show_correlation = st.sidebar.checkbox("🔗 Show Correlation Heatmap", False)

    # Function to get insights for each feature
    def get_feature_insight(feature):
        if feature == 'HighBP':
            return st.write("**Insight for `HighBP`:** The histogram reveals a **significant difference** in the distribution of high blood pressure between individuals with and without diabetes. Those with diabetes show a **higher prevalence** of high blood pressure, indicating that `HighBP` could be a **valuable risk factor** for predicting diabetes. This aligns with medical knowledge that hypertension often coexists with diabetes due to shared risk factors and physiological mechanisms.")
        elif feature == 'DiffWalk':
            return st.write("**Insight for `DiffWalk`:** The plot demonstrates a **notable disparity** in difficulty walking or climbing stairs between diabetic and non-diabetic groups. Individuals with diabetes appear to have a **higher incidence** of mobility issues, suggesting that `DiffWalk` could be a **useful indicator** in diabetes risk assessment. This relationship might be due to diabetes-related complications affecting the lower extremities or overall physical fitness.")
        elif feature == 'HighChol':
            return st.write("**Insight for `HighChol`:** The distribution shows a **clear distinction** in high cholesterol prevalence between diabetic and non-diabetic individuals. The diabetic group exhibits a **higher proportion** of high cholesterol cases, indicating that `HighChol` could be a **strong predictor** for diabetes risk. This correlation is consistent with the metabolic disturbances often associated with both conditions.")
        elif feature == 'BMI':
            return st.write("**Insight for `BMI`:** The histogram reveals a **marked difference** in BMI distribution between diabetic and non-diabetic groups. Individuals with diabetes tend to have **higher BMI values**, suggesting that `BMI` is likely a **crucial factor** in predicting diabetes risk. This aligns with the well-established link between obesity and type 2 diabetes.")
        elif feature == 'HeartDiseaseorAttack':
            return st.write("**Insight for `HeartDiseaseorAttack`:** The plot shows a **substantial difference** in the prevalence of heart disease or heart attacks between diabetic and non-diabetic individuals. Those with diabetes have a **higher incidence** of these cardiovascular issues, indicating that `HeartDiseaseorAttack` could be a **significant predictor** of diabetes risk. This relationship reflects the shared risk factors and physiological impacts of both conditions.")
        elif feature == 'GenHlth':
            return st.write("**Insight for `GenHlth`:** The distribution reveals a **noticeable disparity** in general health ratings between diabetic and non-diabetic groups. Individuals with diabetes tend to report **poorer general health**, suggesting that `GenHlth` could be a **valuable indicator** in assessing diabetes risk. This relationship likely reflects the overall impact of diabetes on various aspects of health and well-being.")
        elif feature == 'PhysHlth':
            return st.write("**Insight for `PhysHlth`:** The histogram shows a **clear difference** in the number of days of poor physical health between diabetic and non-diabetic individuals. Those with diabetes report **more days** of poor physical health, indicating that `PhysHlth` could be a **useful predictor** of diabetes risk. This relationship may be due to the various physical complications and symptoms associated with diabetes.")
        elif feature == 'Income':
            return st.write("**Insight for `Income`:** The plot reveals a **notable variation** in income distribution between diabetic and non-diabetic groups. There appears to be a **higher prevalence** of diabetes in lower income brackets, suggesting that `Income` could be a **relevant factor** in predicting diabetes risk. This relationship might reflect the impact of socioeconomic status on health behaviors, access to healthcare, and overall lifestyle factors that influence diabetes risk.")
        elif feature == 'age_group':
            return st.write("**Insight for `age_group`:** The distribution shows a **significant difference** in age group composition between diabetic and non-diabetic individuals. Diabetes prevalence appears to **increase with age**, indicating that `age_group` is likely a **crucial predictor** of diabetes risk. This trend aligns with the known association between aging and increased risk of type 2 diabetes due to factors such as decreased insulin sensitivity and changes in body composition.")
        else:
            return st.write(f"**Insight:** The distribution of `{feature}` does not show substantial differences between diabetic and non-diabetic individuals. This suggests that `{feature}` may not be a **relevant indicator** for diabetes risk. The **similarity in distributions** between the two groups indicates that this feature might have **limited predictive power** in distinguishing between diabetic and non-diabetic individuals. However, it's important to note that **lack of univariate correlation doesn't necessarily mean the feature is irrelevant**. It's possible that `{feature}` could still be **valuable in combination with other features** or might have a **non-linear relationship** with diabetes risk that isn't apparent in this distribution plot. In our modeling phase, we might consider **feature engineering** or **interaction terms** involving this feature to potentially uncover hidden relationships.")

    # Function to get insights for boxplots
    def get_boxplot_insight(feature):
        if feature == 'BMI':
            return st.write("**Insight for `BMI` boxplot:** Looking at the box plot for `BMI`, we can observe a **significant difference** between the diabetic and non-diabetic groups. The **median `BMI`** for the diabetic group is **noticeably higher**, and the **interquartile range** (box) is shifted upwards compared to the non-diabetic group. This suggests that individuals with diabetes tend to have **higher `BMI` values**. The **whiskers** of the diabetic group also extend to higher values, indicating a **wider range** of `BMI` in this group. These observations support the idea that `BMI` is a **valuable feature** in determining diabetes risk. The clear separation between the two groups aligns with medical knowledge that **excess body weight** is a **significant risk factor** for type 2 diabetes, likely due to its association with insulin resistance and metabolic dysfunction.")
        elif feature == 'PhysHlth':
            return st.write("**Insight for `PhysHlth` boxplot:** Examining the box plot for `PhysHlth` (number of days of poor physical health), we can see a **notable distinction** between the diabetic and non-diabetic groups. The **median** number of days of poor physical health is **higher** for the diabetic group, and the **box** (interquartile range) extends further up the y-axis. This indicates that individuals with diabetes generally report **more days** of poor physical health. The **upper whisker** for the diabetic group also reaches higher values, suggesting some diabetic individuals experience **prolonged periods** of poor physical health. These differences highlight that `PhysHlth` could be a **significant factor** in determining diabetes risk. The relationship likely stems from the **various physical complications** associated with diabetes, such as fatigue, neuropathy, and cardiovascular issues, which can contribute to more frequent experiences of poor physical health.")
        else:
            return st.write(f"**Insight:** The box plot for `{feature}` shows no significant difference between diabetic and non-diabetic groups. This suggests that `{feature}` is **not a strong indicator** for diabetes risk and **may not be valuable** for predicting diabetes. The **similarity in medians**, **interquartile ranges**, and **whisker extents** between the two groups indicates that this feature has **limited discriminatory power** in distinguishing between diabetic and non-diabetic individuals. However, it's important to note that the **lack of univariate difference doesn't necessarily mean the feature is irrelevant**. It's possible that `{feature}` could still be **valuable in combination with other features** or might have a **non-linear relationship** with diabetes risk that isn't apparent in this box plot. In our modeling phase, we might consider **feature engineering** or **interaction terms** involving this feature to potentially uncover hidden relationships. Further investigation might be needed to determine if this feature has any relevance in diabetes risk assessment.")

    # Main content area
    if len(selected_num_features) > 0 or len(selected_cat_features) > 0:
        # Distribution plots
        if show_distribution:
            st.subheader("Distribution of Variables")
            for feature in selected_num_features + selected_cat_features:
                category_orders = {
                    'Diabetes_binary': [0, 1],
                    'age_group': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
                    'Education': ['Never / kindergarten', 'Elementary school', 'Junior school', 'High school Graduate', 'College', 'College graduate'],
                    'Income': ['Less than $10,000', '$10,000 to less than $15,000', '$15,000 to less than $20,000', '$20,000 to less than $25,000', '$25,000 to less than $35,000', '$35,000 to less than $50,000', '$50,000 to less than $75,000', '$75,000 or more']
                }
                fig = px.histogram(data, x=feature, color='Diabetes_binary', 
                                   title=f'Distribution of {feature}', 
                                   labels={'Diabetes_binary': 'Diabetes Status'},
                                   color_discrete_map={0: 'blue', 1: 'red'},
                                   category_orders=category_orders)
                fig.update_layout(legend_title_text='Diabetes Status')
                fig.for_each_trace(lambda t: t.update(name='No Diabetes' if t.name == '0' else 'Diabetes'))
                st.plotly_chart(fig)
                get_feature_insight(feature)

        # Box plots
        if show_boxplot:
            st.subheader("Box Plots of Variables")
            true_numerical_features = ['BMI', 'MentHlth', 'PhysHlth']
            selected_true_numerical = [f for f in selected_num_features if f in true_numerical_features]
            if len(selected_true_numerical) > 0:
                for feature in selected_true_numerical:
                    fig = px.box(data, x='Diabetes_binary', y=feature, 
                                 title=f'Box Plot of {feature} by Diabetes Status',
                                 labels={'Diabetes_binary': 'Diabetes Status'},
                                 category_orders={'Diabetes_binary': [0, 1]})
                    fig.update_xaxes(ticktext=['No Diabetes', 'Diabetes'], tickvals=[0, 1])
                    st.plotly_chart(fig)
                    get_boxplot_insight(feature)
            else:
                if 'age_group' in selected_num_features or 'Income' in selected_num_features:
                    st.warning("**Note:** To use box plot visualization, you need to select at least one of the true numerical features (BMI, MentHlth, PhysHlth). The features `age_group` and `Income` are actually encoded categorical variables and cannot be used for box plots.")
                else:
                    st.warning("**Note:** To use box plot visualization, you need to select at least one of the numerical features.")

        # Correlation heatmap
        if show_correlation:
            st.subheader("Correlation Analysis")
            all_features = selected_num_features + selected_cat_features + ['Diabetes_binary']
            
            # Create a copy of the data with only selected features
            selected_data = data[all_features].copy()
            
            # Convert categorical variables to numeric
            for col in selected_cat_features:
                if col in ['age_group', 'Income']:
                    selected_data[col] = pd.Categorical(selected_data[col]).codes
                elif selected_data[col].dtype == 'object':
                    selected_data[col] = pd.factorize(selected_data[col])[0]
            
            # Calculate phik correlation matrix
            phik_corr = phik_matrix(selected_data)
            
            fig = px.imshow(phik_corr, text_auto=True, aspect="auto",
                            title="Phik Correlation Heatmap of Selected Features")
            
            # Update x and y axes labels safely
            if fig.layout.xaxis and fig.layout.xaxis.ticktext:
                fig.update_xaxes(ticktext=['No Diabetes' if x == 'Diabetes_binary' else x for x in fig.layout.xaxis.ticktext])
            if fig.layout.yaxis and fig.layout.yaxis.ticktext:
                fig.update_yaxes(ticktext=['No Diabetes' if y == 'Diabetes_binary' else y for y in fig.layout.yaxis.ticktext])
            
            st.plotly_chart(fig)
            st.write("**Insight:** The `phik correlation heatmap` provides a **comprehensive visualization** of the relationships between all selected features and `diabetes status`, encompassing both **numerical and categorical variables**. Features exhibiting **strong positive correlations** with the `target variable` are likely to be **powerful indicators** for predicting diabetes. Conversely, features showing **weak or negligible correlations** may not be as relevant for `diabetes risk assessment`. **Moderately correlated variables** suggest they could also contribute meaningfully to `predictive models`. This correlation analysis is **valuable for feature selection** in `predictive modeling`, as it helps identify the **most informative variables** without the need for `encoding`, thanks to the `phik correlation's` ability to handle both `numerical` and `categorical data`. By focusing on **strongly correlated features**, we can potentially **enhance the accuracy and efficiency** of `diabetes risk prediction models`.")

    else:
        st.warning("Please select at least one feature for analysis.")

    # Add a fun fact in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Did You Know?")
    st.sidebar.info("According to the CDC, more than 37 million Americans have diabetes (about 1 in 10), and approximately 90-95% of them have type 2 diabetes.")

if __name__ == "__main__":
    run()