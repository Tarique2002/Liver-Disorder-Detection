# AI-Powered Clinical Decision Support Chatbot for Advanced Liver Disorder Detection

This project is a comprehensive Machine Learning and AI healthcare application designed for the early detection and risk prediction of liver diseases. It combines a highly accurate Random Forest classification model with a Medical AI Chatbot powered by Google Gemini, all accessible through an interactive Streamlit Web Dashboard.

## Features

- **Machine Learning Prediction:** Input patient vitals (Age, Bilirubin, ALT, AST, etc.) to get an instant prediction on the risk of liver disease.
- **AI Medical Assistant:** An integrated chatbot using the Gemini API to answer medical queries regarding liver health, symptoms, and lifestyle changes.
- **Explainable AI (SHAP):** Visualizations that explain how the model makes predictions by highlighting feature importance.
- **Report Generation:** Downloadable PDF health reports summarizing predictions and recommendations.
- **Interactive Dashboard:** Data visualizations (EDA) of the Indian Liver Patient Dataset.

## Project Structure

```
liver-ai-project/
│
├── app.py                 # Streamlit application entry point
├── chatbot.py             # Gemini AI chatbot integration
├── prediction.py          # Helper functions for model prediction
├── preprocessing.py       # Data cleaning and feature scaling
├── model_training.py      # Script to train and save the Random Forest model
│
├── models/                # Saved models (liver_model.pkl, scaler.pkl)
├── dataset/               # Indian Liver Patient Dataset
├── assets/                # Static assets (images, charts)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## How to Run Locally

1. **Install Requirements:**
   Make sure you have Python 3.8+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   Generate the `.pkl` files by running the training script:
   ```bash
   python model_training.py
   ```

3. **Start the Web App:**
   Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Access the Chatbot:**
   Enter your Google Gemini API Key in the sidebar to interact with the AI assistant.

## Deployment to Streamlit Cloud

1. Upload this entire `liver-ai-project` directory to a GitHub repository.
2. Ensure `models/liver_model.pkl` and `models/scaler.pkl` are generated and pushed to the repo (or train it on the cloud by adding a training hook).
3. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
4. Click "New app", connect your GitHub repo, and select `app.py` as the main file.
5. Provide your `GEMINI_API_KEY` in the Streamlit App Settings -> Secrets.
6. Click "Deploy".
