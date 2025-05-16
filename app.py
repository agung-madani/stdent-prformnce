import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #000c24;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .prediction-card.dropout {
        background-color: #1c0000;
        border-left: 4px solid #EF4444;
    }
    .prediction-card.enrolled {
        background-color: #000f1a;
        border-left: 4px solid #3B82F6;
    }
    .prediction-card.graduate {
        background-color: #002914;
        border-left: 4px solid #10B981;
    }
    .prediction-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .info-text {
        color: #4B5563;
        font-size: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #609bfc;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 500;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1061e6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🎓 Student Dropout Prediction Tool</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
<p>This application predicts student academic outcomes based on various factors.
Enter the student's information below, and the model will predict whether the student is likely to:</p>
<ul>
    <li><strong>Dropout</strong> - Leave education before completion</li>
    <li><strong>Enrolled</strong> - Currently enrolled and continuing studies</li>
    <li><strong>Graduate</strong> - Successfully complete their education</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Try to load the model from the current directory
        model = joblib.load('model/best_model.pkl')
        return model
    except:
        st.error("Model file 'best_model.pkl' not found. Please upload the model file.")
        return None

# Function to create input fields for each feature
def create_feature_input():
    # Define the feature ranges based on the provided data
    features_info = {
        'Marital_status': {'min': 1, 'max': 6, 'type': 'int', 'desc': 'Marital status (1-6)'},
        'Application_mode': {'min': 1, 'max': 57, 'type': 'int', 'desc': 'Application mode (1-57)'},
        'Application_order': {'min': 0, 'max': 9, 'type': 'int', 'desc': 'Order of application (0-9)'},
        'Course': {'min': 33, 'max': 9991, 'type': 'int', 'desc': 'Course code (33-9991)'},
        'Daytime_evening_attendance': {'min': 0, 'max': 1, 'type': 'int', 'desc': 'Attendance type (0=Daytime, 1=Evening)'},
        'Previous_qualification': {'min': 1, 'max': 43, 'type': 'int', 'desc': 'Previous qualification (1-43)'},
        'Previous_qualification_grade': {'min': 95.0, 'max': 190.0, 'type': 'float', 'desc': 'Previous qualification grade (95.0-190.0)'},
        'Nacionality': {'min': 1, 'max': 109, 'type': 'int', 'desc': 'Nationality code (1-109)'},
        'Mothers_qualification': {'min': 1, 'max': 44, 'type': 'int', 'desc': "Mother's qualification (1-44)"},
        'Fathers_qualification': {'min': 1, 'max': 44, 'type': 'int', 'desc': "Father's qualification (1-44)"},
        'Mothers_occupation': {'min': 0, 'max': 194, 'type': 'int', 'desc': "Mother's occupation (0-194)"},
        'Fathers_occupation': {'min': 0, 'max': 195, 'type': 'int', 'desc': "Father's occupation (0-195)"},
        'Admission_grade': {'min': 95.0, 'max': 190.0, 'type': 'float', 'desc': 'Admission grade (95.0-190.0)'},
        'Displaced': {'min': 0, 'max': 1, 'type': 'int', 'desc': 'Displaced student (0=No, 1=Yes)'},
        'Educational_special_needs': {'min': 0, 'max': 1, 'type': 'int', 'desc': 'Special educational needs (0=No, 1=Yes)'},
        'Debtor': {'min': 0, 'max': 1, 'type': 'int', 'desc': 'Student is debtor (0=No, 1=Yes)'},
        'Tuition_fees_up_to_date': {'min': 0, 'max': 1, 'type': 'int', 'desc': 'Tuition fees up to date (0=No, 1=Yes)'},
        'Gender': {'min': 0, 'max': 1, 'type': 'int', 'desc': 'Gender (0=Female, 1=Male)'},
        'Scholarship_holder': {'min': 0, 'max': 1, 'type': 'int', 'desc': 'Scholarship holder (0=No, 1=Yes)'},
        'Age_at_enrollment': {'min': 17, 'max': 70, 'type': 'int', 'desc': 'Age at enrollment (17-70)'},
        'International': {'min': 0, 'max': 1, 'type': 'int', 'desc': 'International student (0=No, 1=Yes)'},
        'Curricular_units_1st_sem_credited': {'min': 0, 'max': 20, 'type': 'int', 'desc': 'Curricular units credited in the 1st semester (0-20)'},
        'Curricular_units_1st_sem_enrolled': {'min': 0, 'max': 26, 'type': 'int', 'desc': 'Curricular units enrolled in the 1st semester (0-26)'},
        'Curricular_units_1st_sem_evaluations': {'min': 0, 'max': 45, 'type': 'int', 'desc': 'Curricular units evaluations in the 1st semester (0-45)'},
        'Curricular_units_1st_sem_approved': {'min': 0, 'max': 26, 'type': 'int', 'desc': 'Curricular units approved in the 1st semester (0-26)'},
        'Curricular_units_1st_sem_grade': {'min': 0.0, 'max': 18.875, 'type': 'float', 'desc': 'Curricular units grade in the 1st semester (0.0-18.875)'},
        'Curricular_units_1st_sem_without_evaluations': {'min': 0, 'max': 12, 'type': 'int', 'desc': 'Curricular units without evaluations in the 1st semester (0-12)'},
        'Curricular_units_2nd_sem_credited': {'min': 0, 'max': 19, 'type': 'int', 'desc': 'Curricular units credited in the 2nd semester (0-19)'},
        'Curricular_units_2nd_sem_enrolled': {'min': 0, 'max': 23, 'type': 'int', 'desc': 'Curricular units enrolled in the 2nd semester (0-23)'},
        'Curricular_units_2nd_sem_evaluations': {'min': 0, 'max': 33, 'type': 'int', 'desc': 'Curricular units evaluations in the 2nd semester (0-33)'},
        'Curricular_units_2nd_sem_approved': {'min': 0, 'max': 20, 'type': 'int', 'desc': 'Curricular units approved in the 2nd semester (0-20)'},
        'Curricular_units_2nd_sem_grade': {'min': 0.0, 'max': 18.57, 'type': 'float', 'desc': 'Curricular units grade in the 2nd semester (0.0-18.57)'},
        'Curricular_units_2nd_sem_without_evaluations': {'min': 0, 'max': 12, 'type': 'int', 'desc': 'Curricular units without evaluations in the 2nd semester (0-12)'},
        'Unemployment_rate': {'min': 7.6, 'max': 16.2, 'type': 'float', 'desc': 'Unemployment rate (7.6-16.2)'},
        'Inflation_rate': {'min': -0.8, 'max': 3.7, 'type': 'float', 'desc': 'Inflation rate (-0.8-3.7)'},
        'GDP': {'min': -4.06, 'max': 3.51, 'type': 'float', 'desc': 'GDP (-4.06-3.51)'}
    }
    
    # Sample default values based on provided example
    default_values = {
        'Marital_status': 1,
        'Application_mode': 17,
        'Application_order': 5,
        'Course': 171,
        'Daytime_evening_attendance': 1,
        'Previous_qualification': 1,
        'Previous_qualification_grade': 122.0,
        'Nacionality': 1,
        'Mothers_qualification': 19,
        'Fathers_qualification': 12,
        'Mothers_occupation': 5,
        'Fathers_occupation': 9,
        'Admission_grade': 127.3,
        'Displaced': 1,
        'Educational_special_needs': 0,
        'Debtor': 0,
        'Tuition_fees_up_to_date': 1,
        'Gender': 1,
        'Scholarship_holder': 0,
        'Age_at_enrollment': 20,
        'International': 0,
        'Curricular_units_1st_sem_credited': 0,
        'Curricular_units_1st_sem_enrolled': 0,
        'Curricular_units_1st_sem_evaluations': 0,
        'Curricular_units_1st_sem_approved': 0,
        'Curricular_units_1st_sem_grade': 0.0,
        'Curricular_units_1st_sem_without_evaluations': 0,
        'Curricular_units_2nd_sem_credited': 0,
        'Curricular_units_2nd_sem_enrolled': 0,
        'Curricular_units_2nd_sem_evaluations': 0,
        'Curricular_units_2nd_sem_approved': 0,
        'Curricular_units_2nd_sem_grade': 0.0,
        'Curricular_units_2nd_sem_without_evaluations': 0,
        'Unemployment_rate': 10.8,
        'Inflation_rate': 1.4,
        'GDP': 1.74,
    }
    
    # Create a container for all inputs
    input_data = {}
    
    # Create categories to organize fields
    categories = {
        "Demographics": [
            "Marital_status", "Age_at_enrollment", "Gender", "Nacionality", 
            "International", "Displaced", "Educational_special_needs"
        ],
        "Application & Course": [
            "Application_mode", "Application_order", "Course", 
            "Daytime_evening_attendance", "Admission_grade"
        ],
        "Prior Education": [
            "Previous_qualification", "Previous_qualification_grade",
            "Mothers_qualification", "Fathers_qualification", 
            "Mothers_occupation", "Fathers_occupation"
        ],
        "Financial Factors": [
            "Debtor", "Tuition_fees_up_to_date", "Scholarship_holder",
            "Unemployment_rate", "Inflation_rate", "GDP"
        ],
        "1st Semester Performance": [
            "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
            "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
            "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations"
        ],
        "2nd Semester Performance": [
            "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
            "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
            "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations"
        ]
    }
    
    # Create tabbed interface for categories
    st.markdown('<h2 class="sub-header">Student Information</h2>', unsafe_allow_html=True)
    tabs = st.tabs(list(categories.keys()))
    
    # Icons for each category
    category_icons = {
        "Demographics": "👤",
        "Application & Course": "📝",
        "Prior Education": "🎓",
        "Financial Factors": "💰",
        "1st Semester Performance": "📊",
        "2nd Semester Performance": "📈"
    }
    
    for i, (category, features) in enumerate(categories.items()):
        with tabs[i]:
            st.markdown(f"### {category_icons[category]} {category}")
            
            # Create columns for better layout
            if len(features) > 3:
                cols = st.columns(2)
            else:
                cols = st.columns(1)
            
            for j, feature in enumerate(features):
                col_idx = j % len(cols)
                with cols[col_idx]:
                    info = features_info[feature]
                    
                    # Create appropriate input widget based on type
                    if feature == 'Gender':
                        value = st.radio(
                            f"{info['desc']}",
                            options=[0, 1],
                            index=default_values[feature],
                            format_func=lambda x: "Female" if x == 0 else "Male",
                            horizontal=True
                        )
                    elif info['max'] - info['min'] == 1:  # Binary/boolean feature
                        value = st.selectbox(
                            f"{info['desc']}",
                            options=[info['min'], info['max']],
                            index=default_values[feature],
                            format_func=lambda x: f"{'Yes' if x == info['max'] else 'No'}"
                        )
                    elif info['type'] == 'int':
                        value = st.number_input(
                            f"{info['desc']}",
                            min_value=info['min'],
                            max_value=info['max'],
                            value=default_values[feature],
                            step=1
                        )
                    else:  # float values
                        value = st.number_input(
                            f"{info['desc']}",
                            min_value=float(info['min']),
                            max_value=float(info['max']),
                            value=float(default_values[feature]),
                            step=0.1,
                            format="%.1f"
                        )
                    
                    input_data[feature] = value
    
    return input_data

# Function to make prediction
def predict(model, input_data):
    # Define the expected feature order to match training data
    expected_feature_order = [
        'Marital_status', 'Application_mode', 'Application_order', 'Course', 
        'Daytime_evening_attendance', 'Previous_qualification', 'Previous_qualification_grade', 
        'Nacionality', 'Mothers_qualification', 'Fathers_qualification', 
        'Mothers_occupation', 'Fathers_occupation', 'Admission_grade', 
        'Displaced', 'Educational_special_needs', 'Debtor', 
        'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 
        'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited', 
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 
        'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited', 
        'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations', 
        'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 
        'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate', 
        'Inflation_rate', 'GDP'
    ]
    
    # Create DataFrame with ordered features
    ordered_data = {}
    for feature in expected_feature_order:
        ordered_data[feature] = input_data[feature]
    
    input_df = pd.DataFrame([ordered_data])
    
    # Ensure the column order is exactly as expected
    input_df = input_df[expected_feature_order]
    
    # Probability predictions
    try:
        probabilities = model.predict_proba(input_df)
        predicted_class = model.predict(input_df)[0]
        
        # Map numeric labels to meaningful labels
        class_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        predicted_label = class_mapping.get(predicted_class, f"Class {predicted_class}")
        
        # Get class labels if possible
        try:
            class_labels = [class_mapping.get(i, f"Class {i}") for i in model.classes_]
        except:
            class_labels = [class_mapping.get(i, f"Class {i}") for i in range(probabilities.shape[1])]
        
        return predicted_class, predicted_label, probabilities, class_labels
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None, None

# Function to create a visual gauge chart for the prediction
def create_gauge_chart(prediction_probs, class_labels):
    # Create a custom color scale for the gauge chart
    colors = ['#EF4444', '#3B82F6', '#10B981']  # Red for Dropout, Blue for Enrolled, Green for Graduate
    
    fig = go.Figure()
    
    # Add one gauge chart for each probability
    for i, (prob, label) in enumerate(zip(prediction_probs[0], class_labels)):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'row': 0, 'column': i},
            title={'text': label},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': colors[i]},
                'bgcolor': "black",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#F3F4F6'},
                    {'range': [50, 75], 'color': '#E5E7EB'},
                    {'range': [75, 100], 'color': '#D1D5DB'}
                ],
            }
        ))
    
    # Update layout
    fig.update_layout(
        grid={'rows': 1, 'columns': len(class_labels), 'pattern': "independent"},
        margin=dict(l=40, r=40, t=40, b=40),
        height=250,
        plot_bgcolor='#000c24',
        paper_bgcolor='#000c24'
    )
    
    return fig

# Function to create a bar chart for the prediction probabilities
def create_prob_chart(prediction_probs, class_labels):
    # Create a DataFrame for the probabilities
    df = pd.DataFrame({
        'Class': class_labels,
        'Probability': prediction_probs[0] * 100
    })
    
    # Create the bar chart
    fig = px.bar(
        df, 
        x='Class', 
        y='Probability',
        color='Class',
        color_discrete_map={
            'Dropout': '#EF4444',
            'Enrolled': '#3B82F6',
            'Graduate': '#10B981'
        },
        text='Probability',
        title='Prediction Probabilities (%)'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=None,
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        plot_bgcolor='#000c24',
        font=dict(size=14),
        paper_bgcolor='#000c24'
    )
    
    # Update text formatting
    fig.update_traces(
        texttemplate='%{y:.1f}%', 
        textposition='auto',
        marker_line_width=1,
        marker_line_color='white'
    )
    
    return fig

# Main app function
def main():
    # Load the model
    model = load_model()
    
    # Create a file uploader widget in case the model needs to be uploaded
    with st.sidebar:
        st.header("Model Settings")
        uploaded_model = st.file_uploader("Upload model file (if not found locally)", type=["pkl"])
        
        if uploaded_model is not None:
            # Save the uploaded model file
            with open('best_model.pkl', 'wb') as f:
                f.write(uploaded_model.getbuffer())
            model = joblib.load('best_model.pkl')
            st.success("Model loaded successfully!")
        
        # Info section
        with st.expander("About This App"):
            st.markdown("""
            <div class="info-text">
            This app predicts student academic outcomes using machine learning:
            
            - **Dropout**: Students who leave before completing their education
            - **Enrolled**: Students who are currently continuing their studies
            - **Graduate**: Students who successfully complete their degree
            
            The prediction is based on various factors including demographics, application details, prior education, financial factors, and academic performance.
            </div>
            """, unsafe_allow_html=True)
        
        # Add debug information
        with st.expander("Model & Data Information", expanded=False):
            if model is not None:
                try:
                    st.write("**Model Type:**", type(model).__name__)
                    if hasattr(model, 'classes_'):
                        st.write("**Classes:**", ', '.join([f"{i}: {'Dropout' if i==0 else 'Enrolled' if i==1 else 'Graduate'}" for i in model.classes_]))
                except:
                    st.write("Could not extract model information")
            else:
                st.write("No model loaded yet")
    
    # Get input features from user
    input_data = create_feature_input()
    
    # Prediction section
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    prediction_container = st.container()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        predict_button = st.button("📊 Generate Prediction", type="primary", use_container_width=True)
        
        # Add some preset examples
        st.markdown("### Try Presets")
        if st.button("🎓 Likely Graduate", use_container_width=True):
            # Update with preset values for a likely graduate
            # This is just an example - customize based on your model
            input_data.update({
                'Curricular_units_1st_sem_grade': 15.5,
                'Curricular_units_2nd_sem_grade': 16.0,
                'Curricular_units_1st_sem_approved': 6,
                'Curricular_units_2nd_sem_approved': 6,
                'Tuition_fees_up_to_date': 1,
                'Scholarship_holder': 1
            })
            predict_button = True
            
        if st.button("⚠️ At-Risk Student", use_container_width=True):
            # Update with preset values for a likely dropout
            input_data.update({
                'Curricular_units_1st_sem_grade': 8.5,
                'Curricular_units_2nd_sem_grade': 7.0,
                'Curricular_units_1st_sem_approved': 2,
                'Curricular_units_2nd_sem_approved': 1,
                'Tuition_fees_up_to_date': 0,
                'Debtor': 1
            })
            predict_button = True
    
    with col2:
        st.markdown("""
        <div class="info-text">
        Click the "Generate Prediction" button to analyze the entered student data and predict their academic outcome.
        
        The system will provide:
        - The most likely outcome (Dropout, Enrolled, or Graduate)
        - The probability for each potential outcome
        - Visual charts to help interpret the results
        </div>
        """, unsafe_allow_html=True)
    
    if predict_button:
        if model is not None:
            # Make prediction
            predicted_class, predicted_label, probabilities, class_labels = predict(model, input_data)
            
            if predicted_class is not None:
                # Display prediction results
                with prediction_container:
                    # Create status card based on prediction
                    status_class = "dropout" if predicted_label == "Dropout" else "enrolled" if predicted_label == "Enrolled" else "graduate"
                    status_icon = "❌" if predicted_label == "Dropout" else "⏳" if predicted_label == "Enrolled" else "✅"
                    
                    st.markdown(f"""
                    <div class="prediction-card {status_class}">
                        <h3>Predicted Outcome</h3>
                        <p class="prediction-value">{status_icon} {predicted_label}</p>
                        <p>The student is most likely to {predicted_label.lower()} based on the provided information.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create visualizations
                    st.markdown("### Prediction Details")
                    
                    tab1, tab2 = st.tabs(["Probability Gauge", "Probability Chart"])
                    
                    with tab1:
                        gauge_fig = create_gauge_chart(probabilities, class_labels)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    with tab2:
                        bar_fig = create_prob_chart(probabilities, class_labels)
                        st.plotly_chart(bar_fig, use_container_width=True)
                    
                    # Show probability values as text
                    col1, col2, col3 = st.columns(3)
                    
                    # Display each probability with appropriate styling
                    for i, (class_label, col) in enumerate(zip(class_labels, [col1, col2, col3])):
                        with col:
                            prob_percent = probabilities[0][i] * 100
                            color = "#EF4444" if class_label == "Dropout" else "#3B82F6" if class_label == "Enrolled" else "#10B981"
                            st.markdown(f"""
                            <div style="padding:10px; border-radius:5px; border-left:4px solid {color}; background-color:#000c24;">
                                <strong>{class_label}</strong>: {prob_percent:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show key factors influencing the prediction
                with st.expander("What factors influenced this prediction?"):
                    st.markdown("""
                    The model considers all input factors to make its prediction, but typically Students who drop out tend to have::
                    
                    1. Lower number of approved courses (especially in the second semester)
                    2. Significantly lower academic grades
                    3. Deteriorating performance from the first to the second semester

                    """)
                
                # Show input data used for prediction
                with st.expander("View input data used for prediction"):
                    st.dataframe(pd.DataFrame([input_data]).T.rename(columns={0: 'Value'}))
        else:
            st.error("Please upload a model file first.")

# Run the app
if __name__ == "__main__":
    main()
