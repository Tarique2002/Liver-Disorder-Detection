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
        try:
            train_and_evaluate()
            model, scaler = load_models()
        except Exception as e:
            import streamlit as st
            st.error(f"Could not train the model. Are you sure 'dataset/indian_liver_patient.csv' is uploaded to your repository? Error: {e}")
            return None, None
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
    st.title("Welcome to LiverCare AI 🩺")
    st.markdown("""
    <h3 style="color: #6c757d; font-weight: 400;">Advanced Clinical Decision Support System for Hepatic Health</h3>
    <hr>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        <div class="card" style="padding: 2rem;">
            <h4 style="color: #0d6efd; margin-top: 0;">Empowering Early Detection</h4>
            <p style="font-size: 1.1rem; line-height: 1.6;">
            LiverCare AI is a state-of-the-art diagnostic assistant designed to bridge the gap between artificial intelligence and clinical hepatology. 
            By analyzing complex biochemical markers, our platform provides instant, highly accurate risk assessments for liver disorders.
            </p>
            <hr>
            <h5>Core Capabilities:</h5>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li>🔬 <b>Predictive Analytics:</b> High-accuracy risk stratification using Random Forest algorithms.</li>
                <li>🧠 <b>Explainable AI (SHAP):</b> Transparent decision-making visualizing clinical feature importance.</li>
                <li>💬 <b>Virtual AI Assistant:</b> Rule-based conversational agent for instant clinical guidance.</li>
                <li>📄 <b>Automated Reporting:</b> Instant generation of comprehensive PDF medical reports.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", use_column_width=True)
        st.info("💡 **Getting Started:** Use the sidebar on the left to navigate to the **Prediction** dashboard to run a risk assessment, or visit the **AI Chatbot** for medical guidance.")

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
    st.title("AI Medical Chatbot 🤖")
    st.write("Ask our specialized AI assistant about liver health, symptoms, and lifestyle changes.")
    
    chat_col, faq_col = st.columns([2, 1])
    
    with faq_col:
        st.markdown("### 💡 Frequently Asked Questions")
        with st.container(height=650):
            with st.expander("1. What is fatty liver?"):
                st.write("Fatty liver disease is a condition caused by excess fat build-up in the liver. It's often linked to obesity, type 2 diabetes, and excessive alcohol consumption.")
            with st.expander("2. What does high Bilirubin mean?"):
                st.write("Bilirubin is a yellowish pigment. High levels can indicate liver damage, causing jaundice (yellowing of eyes and skin).")
            with st.expander("3. What are ALT and AST?"):
                st.write("ALT and AST are enzymes found in the liver. When liver cells are damaged or inflamed, they leak these enzymes into the bloodstream, causing levels to rise in blood tests.")
            with st.expander("4. Can liver damage be reversed?"):
                st.write("The liver is unique in its ability to regenerate. Early-stage damage (like fatty liver or mild inflammation) can often be reversed with diet and lifestyle changes. However, late-stage scarring (cirrhosis) is usually permanent.")
            with st.expander("5. What causes liver cirrhosis?"):
                st.write("Cirrhosis is severe scarring of the liver. The most common causes are chronic alcohol abuse, chronic viral hepatitis (B and C), and non-alcoholic fatty liver disease.")
            with st.expander("6. What foods protect the liver?"):
                st.write("A liver-friendly diet includes coffee, tea, berries, cruciferous vegetables (like broccoli), and lean proteins. Avoid fried and processed foods.")
            with st.expander("7. What are early symptoms of liver disease?"):
                st.write("Early symptoms are often subtle, including chronic fatigue, unexplained weight loss, mild abdominal pain, and nausea.")
            with st.expander("8. What is Alkaline Phosphatase (ALP)?"):
                st.write("ALP is an enzyme found in your blood that helps break down proteins. High levels can indicate liver disease, blocked bile ducts, or bone disorders.")
            with st.expander("9. What is Albumin?"):
                st.write("Albumin is a protein made by your liver that keeps fluid from leaking out of blood vessels. Low levels indicate poor liver function.")
            with st.expander("10. What does the A/G ratio indicate?"):
                st.write("The Albumin to Globulin (A/G) ratio compares the amount of albumin to globulin in your blood. A low ratio can indicate liver disease, kidney disease, or autoimmune issues.")
            with st.expander("11. How much alcohol is safe for the liver?"):
                st.write("There is no 'completely safe' level, but guidelines suggest limiting intake to 1 drink per day for women and 2 for men. For those with liver disease, zero alcohol is the safest.")
            with st.expander("12. Is coffee good for the liver?"):
                st.write("Yes! Studies show drinking coffee can protect against liver disease, reduce inflammation, and lower the risk of cirrhosis and liver cancer.")
            with st.expander("13. Can medications cause liver damage?"):
                st.write("Yes, many medications (including over-the-counter painkillers like Acetaminophen) can cause liver damage if taken in excessive amounts or combined with alcohol.")
            with st.expander("14. What is Hepatitis A?"):
                st.write("Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus, usually transmitted through contaminated food or water. It does not cause chronic liver disease.")
            with st.expander("15. What is Hepatitis B?"):
                st.write("Hepatitis B is a viral infection transmitted through bodily fluids. It can be acute or chronic, potentially leading to cirrhosis or liver cancer. A vaccine is available.")
            with st.expander("16. What is Hepatitis C?"):
                st.write("Hepatitis C is a blood-borne virus that often leads to chronic liver disease. While there is no vaccine, it is highly curable with modern antiviral medications.")
            with st.expander("17. How does obesity affect the liver?"):
                st.write("Excess body weight causes fat to accumulate in liver cells, leading to Non-Alcoholic Fatty Liver Disease (NAFLD), which can progress to inflammation and scarring.")
            with st.expander("18. What is a liver biopsy?"):
                st.write("A procedure where a small needle is used to extract a tiny piece of liver tissue for microscopic examination to diagnose the severity of liver disease.")
            with st.expander("19. Can drinking water detox the liver?"):
                st.write("Water doesn't 'detox' the liver, but staying hydrated helps the liver function optimally to filter toxins from your blood.")
            with st.expander("20. Are liver detox supplements safe?"):
                st.write("Many 'detox' teas and supplements are not FDA-regulated and can actually harm the liver. Always consult a doctor before taking them.")
            with st.expander("21. What is jaundice?"):
                st.write("Jaundice is the yellowing of the skin and the whites of the eyes, caused by an excessive buildup of bilirubin in the blood due to liver dysfunction.")
            with st.expander("22. Why does liver disease cause itchy skin?"):
                st.write("Itchy skin (pruritus) in liver disease is often caused by the buildup of bile salts under the skin when the liver's bile ducts are blocked or damaged.")
            with st.expander("23. Does smoking affect the liver?"):
                st.write("Yes, smoking yields toxic chemicals that cause inflammation, accelerate liver scarring (fibrosis), and increase the risk of liver cancer.")
            with st.expander("24. What is liver cancer?"):
                st.write("The most common type of primary liver cancer mostly occurs in people with chronic liver diseases like cirrhosis or hepatitis B/C.")
            with st.expander("25. Can diabetes lead to liver disease?"):
                st.write("Yes, type 2 diabetes increases the risk of NAFLD, as insulin resistance promotes fat storage in the liver.")
            with st.expander("26. What happens if the liver stops working?"):
                st.write("Acute or chronic liver failure causes toxins to build up in the brain, internal bleeding, and fluid accumulation, requiring immediate medical care or a transplant.")
            with st.expander("27. Is liver disease hereditary?"):
                st.write("Some liver diseases are genetic, such as Hemochromatosis (iron buildup), Wilson's disease (copper buildup), and Alpha-1 antitrypsin deficiency.")
            with st.expander("28. What is an enlarged liver?"):
                st.write("Swelling of the liver beyond its normal size, often a symptom of underlying issues like fatty liver, heart failure, or infections.")
            with st.expander("29. How often should I test my liver function?"):
                st.write("Healthy adults usually get it checked during an annual physical. Those with risk factors (obesity, diabetes, heavy drinking) may need it more frequently.")
            with st.expander("30. Can stress affect the liver?"):
                st.write("Chronic psychological stress can alter blood flow to the liver and exacerbate inflammation, worsening existing liver diseases.")
            with st.expander("31. What is the role of the liver in digestion?"):
                st.write("The liver produces bile, which breaks down fats in the small intestine, and processes nutrients absorbed from the digestive tract.")
            with st.expander("32. How does the liver process toxins?"):
                st.write("It uses enzymes to break down harmful substances (like alcohol and drugs) into harmless byproducts that are excreted in bile or urine.")
            
    with chat_col:
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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "dataset", "indian_liver_patient.csv")
        df = pd.read_csv(csv_path)
        
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
            
            # Handle different SHAP versions (list vs 3D array)
            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif len(np.shape(shap_values)) == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values
            
            # SHAP Summary Plot
            fig_shap, ax = plt.subplots(figsize=(10, 6))
            # SHAP values for class 1 (Disease)
            shap.summary_plot(sv, background, show=False)
            st.pyplot(fig_shap)
            
            st.info("💡 **How to read this chart:** Features at the top are most important. Red dots indicate high feature values, and blue dots indicate low feature values. If the dots are on the right side of the center line, they increase the risk of liver disease.")
        else:
            st.error("Failed to train or load the model.")
            
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'dataset/indian_liver_patient.csv' exists.")
