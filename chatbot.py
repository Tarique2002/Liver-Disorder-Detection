import google.generativeai as genai
import os

SYSTEM_PROMPT = """
You are an advanced AI Clinical Decision Support Assistant specialized in liver diseases.
Your role is to provide medically accurate, helpful, and clear information regarding liver health, diseases (like fatty liver, cirrhosis, hepatitis), symptoms, causes, and prevention.

Guidelines:
1. Always base your answers on established medical knowledge.
2. If asked about liver disease risk factors, explain how features like Bilirubin, ALT, AST, Albumin, and Protein affect liver health.
3. Keep your answers concise, empathetic, and structured (use bullet points where appropriate).
4. ALWAYS include a disclaimer that you are an AI assistant and not a substitute for professional medical advice, diagnosis, or treatment.
5. If the user shares their prediction results, help them understand what 'High Risk' or 'Low Risk' means and suggest general lifestyle changes.
6. Do NOT diagnose the user. Instead, explain the possible meanings of their symptoms or test results.
"""

def initialize_chatbot(api_key):
    """Initializes the Gemini AI model."""
    try:
        genai.configure(api_key=api_key)
        # Use gemini-1.5-flash or gemini-pro
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
        return model
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        return None

def get_chatbot_response(model, user_input, chat_history=None):
    """
    Sends a message to the Gemini model and returns the response.
    """
    if model is None:
        return "Error: Chatbot is not initialized. Please provide a valid Gemini API Key."
    
    try:
        # Start a chat session if history is needed, or just generate content
        # For simplicity in Streamlit, we can just send the chat history + new message
        
        # Build prompt from history
        prompt = ""
        if chat_history:
            for msg in chat_history:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += f"User: {user_input}\nAssistant:"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I'm sorry, I encountered an error while processing your request. Please ensure your API key is valid. Error: {str(e)}"
