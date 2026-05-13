import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import os
import base64
from fpdf import FPDF
from datetime import datetime

# Local imports
from prediction import load_models, predict_liver_disease, generate_recommendations, FEATURE_COLS
from chatbot import get_chatbot_response
from model_training import train_and_evaluate

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="LiverCare AI - Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #0d6efd;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None

# ----------------- SIDEBAR -----------------
st.sidebar.title("🩺 LiverCare AI")
st.sidebar.markdown("Advanced Liver Disorder Detection")

st.sidebar.subheader("Chatbot Mode")
st.sidebar.info("Running locally (Rule-Based)")

st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "AI Chatbot", "Analytics Dashboard"])

# ----------------- HELPER FUNCTIONS -----------------
@st.cache_resource
def get_or_train_models():
    model, scaler = load_models()
    if not model or not scaler:
        train_and_evaluate()
        model, scaler = load_models()
    return model, scaler

def create_pdf_report(patient_data, result, recommendations):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="LiverCare AI - Health Report", ln=True, align='C')
    pdf.ln(10)
    
    # Date
    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Patient Data
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Vitals:", ln=True)
    pdf.set_font("Arial", '', 11)
    for key, value in patient_data.items():
        pdf.cell(200, 8, txt=f"{key.replace('_', ' ')}: {value}", ln=True)
    pdf.ln(5)
    
    # Prediction Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Prediction Summary:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"Result: {result['prediction']}", ln=True)
    pdf.cell(200, 8, txt=f"Risk Probability: {result['risk_percentage']}%", ln=True)
    pdf.cell(200, 8, txt=f"Model Confidence: {result['confidence_score']}%", ln=True)
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Recommendations:", ln=True)
    pdf.set_font("Arial", '', 11)
    for rec in recommendations:
        pdf.cell(200, 8, txt=f"- {rec}", ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

# ----------------- PAGES -----------------

if page == "Home":
    st.title("Welcome to LiverCare AI")
    st.markdown("""
    <div class="card">
        <h3>AI-Powered Clinical Decision Support System</h3>
        <p>This platform combines advanced Machine Learning with Generative AI to provide a comprehensive healthcare tool for liver disease detection and risk prediction.</p>
        <ul>
            <li><b>Machine Learning Prediction:</b> Utilizes a highly accurate Random Forest model trained on the Indian Liver Patient Dataset.</li>
            <li><b>Explainable AI:</b> Understand <i>why</i> the model made a prediction using SHAP visualizations.</li>
            <li><b>Medical AI Chatbot:</b> A specialized medical assistant powered by Google Gemini to answer your queries.</li>
            <li><b>Downloadable Reports:</b> Generate comprehensive PDF health reports instantly.</li>
        </ul>
        <p><i>Navigate using the sidebar to explore the features.</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a nice banner image (optional placeholder)
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", caption="Empowering Healthcare with AI")

elif page == "Prediction":
    st.title("Liver Disease Prediction")
    st.write("Enter patient vitals to assess the risk of liver disease.")
    
    model, scaler = get_or_train_models()
    if not model or not scaler:
        st.error("⚠️ Failed to train or load the model. Please check the dataset and logs.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=40)
                gender = st.selectbox("Gender", ["Male", "Female"])
                tot_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=50.0, value=0.9, format="%.2f")
                dir_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=20.0, value=0.2, format="%.2f")
                alk_phos = st.number_input("Alkaline Phosphotase", min_value=0, max_value=2500, value=190)
                
            with col2:
                alamine = st.number_input("Alamine Aminotransferase", min_value=0, max_value=2000, value=25)
                aspartate = st.number_input("Aspartate Aminotransferase", min_value=0, max_value=3000, value=25)
                tot_proteins = st.number_input("Total Proteins", min_value=0.0, max_value=15.0, value=6.8, format="%.1f")
                albumin = st.number_input("Albumin", min_value=0.0, max_value=10.0, value=3.3, format="%.1f")
                ag_ratio = st.number_input("A/G Ratio", min_value=0.0, max_value=5.0, value=0.9, format="%.2f")
                
            submit_button = st.form_submit_button(label="Predict Risk")
            
        if submit_button:
            input_data = {
                'Age': age, 'Gender': gender, 'Total_Bilirubin': tot_bilirubin,
                'Direct_Bilirubin': dir_bilirubin, 'Alkaline_Phosphotase': alk_phos,
                'Alamine_Aminotransferase': alamine, 'Aspartate_Aminotransferase': aspartate,
                'Total_Protiens': tot_proteins, 'Albumin': albumin, 'Albumin_and_Globulin_Ratio': ag_ratio
            }
            
            with st.spinner('Analyzing...'):
                result = predict_liver_disease(input_data, model, scaler)
                st.session_state.prediction_result = result
                st.session_state.patient_data = input_data
                
        if st.session_state.prediction_result:
            st.markdown("---")
            res = st.session_state.prediction_result
            
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            if res['is_high_risk']:
                col1.error(f"**{res['prediction']}**")
            else:
                col1.success(f"**{res['prediction']}**")
                
            col2.metric("Risk Probability", f"{res['risk_percentage']}%")
            col3.metric("Model Confidence", f"{res['confidence_score']}%")
            
            st.subheader("Health Recommendations")
            recs = generate_recommendations(res['is_high_risk'])
            for r in recs:
                st.write(f"- {r}")
                
            # PDF Generation
            st.markdown("---")
            pdf_bytes = create_pdf_report(st.session_state.patient_data, res, recs)
            st.download_button(
                label="📄 Download Full PDF Report",
                data=pdf_bytes,
                file_name=f"LiverCare_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

elif page == "AI Chatbot":
    st.title("AI Medical Chatbot")
    st.write("Ask our specialized AI assistant about liver health, symptoms, and lifestyle changes.")
    
    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Accept user input
    if prompt := st.chat_input("Ask a medical question (e.g., 'What are the symptoms of fatty liver?'):"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = get_chatbot_response(prompt, st.session_state.chat_history)
                st.markdown(response_text)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

elif page == "Analytics Dashboard":
    st.title("Explainable AI & Analytics Dashboard")
    
    # Load dataset for EDA
    try:
        df = pd.read_csv("dataset/indian_liver_patient.csv")
        
        st.subheader("Dataset Overview")
        st.write(df.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Target Distribution**")
            # 1 = Liver Disease, 2 = No Liver Disease
            target_counts = df['Dataset'].value_counts().reset_index()
            target_counts.columns = ['Status', 'Count']
            target_counts['Status'] = target_counts['Status'].map({1: 'Disease', 2: 'Healthy'})
            fig = px.pie(target_counts, names='Status', values='Count', hole=0.4, color_discrete_sequence=['#ef476f', '#06d6a0'])
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("**Age Distribution by Gender**")
            fig = px.histogram(df, x='Age', color='Gender', marginal='box', barmode='overlay', color_discrete_sequence=['#118ab2', '#ffd166'])
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Explainable AI (SHAP)")
        st.write("Feature importance explaining how the model makes its decisions.")
        
        model, scaler = get_or_train_models()
        if model and scaler:
            # We use a sample of the data for SHAP to make it fast
            from preprocessing import clean_and_preprocess
            X, y, _ = clean_and_preprocess(df, is_training=True)
            
            # Sample 100 random background samples to initialize the explainer
            background = X.sample(n=100, random_state=42)
            
            # Create a Tree explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(background)
            
            # SHAP Summary Plot
            fig_shap, ax = plt.subplots(figsize=(10, 6))
            # SHAP values for class 1 (Disease)
            shap.summary_plot(shap_values[1], background, show=False)
            st.pyplot(fig_shap)
            
            st.info("💡 **How to read this chart:** Features at the top are most important. Red dots indicate high feature values, and blue dots indicate low feature values. If the dots are on the right side of the center line, they increase the risk of liver disease.")
        else:
            st.error("Failed to train or load the model.")
            
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'dataset/indian_liver_patient.csv' exists.")
