import re
import requests
import json
# pyrefly: ignore [missing-import]
import streamlit as st

# Fallback Medical Knowledge Database
MEDICAL_KNOWLEDGE_DB = {
    "fatty liver": "Fatty liver disease (steatosis) is a common condition caused by having too much fat build up in your liver. It can be alcoholic (AFLD) or non-alcoholic (NAFLD). Treatment usually involves diet and lifestyle changes.",
    "bilirubin": "Bilirubin is a yellowish pigment made during the normal breakdown of red blood cells. High levels can cause jaundice (yellowing of skin/eyes) and may indicate liver damage or disease.",
    "alt": "ALT (Alanine transaminase) is an enzyme found mostly in the liver. When liver cells are damaged, they release ALT into the bloodstream. High ALT levels are a sign of liver injury.",
    "ast": "AST (Aspartate transaminase) is an enzyme found in the liver, heart, and muscles. Like ALT, high levels of AST in the blood can indicate liver damage.",
    "albumin": "Albumin is a protein made by your liver. It helps keep fluid in your bloodstream so it doesn't leak into other tissues. Low albumin levels can indicate that your liver isn't functioning properly.",
    "protein": "Total protein measures the total amount of two classes of proteins found in the fluid portion of your blood: albumin and globulin. Abnormal levels can indicate liver or kidney issues.",
    "alkaline": "Alkaline Phosphatase (ALP) is an enzyme found in your blood that helps break down proteins. High levels can indicate liver disease, blocked bile ducts, or bone disorders.",
    "cirrhosis": "Cirrhosis is a late stage of scarring (fibrosis) of the liver caused by many forms of liver diseases and conditions, such as hepatitis and chronic alcoholism. It is typically irreversible.",
    "diet": "For a healthy liver, maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins. Drink plenty of water. Limit foods high in saturated fats, sugars, and salt.",
    "foods": "Foods good for the liver include coffee, tea, grapefruit, blueberries, grapes, and cruciferous vegetables. You should avoid processed foods, fatty meats, and excessive alcohol.",
    "avoid": "To protect your liver, you should strictly limit or avoid alcohol, fried foods, foods high in salt, red meat, and processed foods with added sugars.",
    "symptoms": "Common symptoms of liver disease include jaundice (yellowing of skin and eyes), abdominal pain and swelling, swelling in the legs and ankles, itchy skin, dark urine color, and chronic fatigue.",
    "improve": "You can improve liver health by maintaining a healthy weight, eating a balanced diet, exercising regularly, avoiding toxins (including excessive alcohol), and not sharing personal hygiene items.",
    "hello": "Hello! I am your LiverCare Assistant. How can I help you understand liver health today?",
    "hi": "Hi there! I am your LiverCare Assistant. How can I help you understand liver health today?",
    "risk": "If your prediction shows a High Risk, it means your blood markers align with patterns seen in liver patients. You should consult a doctor immediately and consider lifestyle changes."
}

class LocalLLMConnector:
    ENDPOINTS = [
        {"name": "Ollama", "url": "http://localhost:11434/api/chat", "type": "ollama"},
        {"name": "LM Studio", "url": "http://localhost:1234/v1/chat/completions", "type": "openai"},
        {"name": "Local OpenAI", "url": "http://localhost:8000/v1/chat/completions", "type": "openai"}
    ]
    
    @staticmethod
    def check_connection(url, timeout=0.3):
        try:
            # Isolate hostname and port for quick check
            root_url = url.split("/api/")[0] if "/api/" in url else url.split("/v1/")[0]
            r = requests.get(root_url, timeout=timeout)
            return r.status_code in [200, 404, 405]
        except Exception:
            return False

    @classmethod
    def get_active_endpoint(cls):
        for ep in cls.ENDPOINTS:
            if cls.check_connection(ep["url"]):
                return ep
        return None

    @classmethod
    def is_available(cls):
        return cls.get_active_endpoint() is not None

    @classmethod
    def get_ollama_model(cls):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=0.3)
            if r.status_code == 200:
                models = r.json().get("models", [])
                names = [m["name"] for m in models]
                # Prioritize faster/smaller model
                for preferred in ["qwen2.5:1.5b", "qwen2.5:latest", "gemma3:latest", "llama3:latest"]:
                    if preferred in names:
                        return preferred
                if names:
                    return names[0]
        except Exception:
            pass
        return "qwen2.5:1.5b"

    @classmethod
    def stream_query(cls, messages, system_prompt=None):
        endpoint = cls.get_active_endpoint()
        if not endpoint:
            return None
            
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
            
        try:
            if endpoint["type"] == "ollama":
                model_name = cls.get_ollama_model()
                payload = {
                    "model": model_name,
                    "messages": formatted_messages,
                    "stream": True
                }
                r = requests.post(endpoint["url"], json=payload, stream=True, timeout=30.0)
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if line:
                            data = json.loads(line.decode("utf-8"))
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
            else:
                # OpenAI-compatible API stream
                payload = {
                    "model": "default",
                    "messages": formatted_messages,
                    "stream": True
                }
                r = requests.post(endpoint["url"], json=payload, stream=True, timeout=30.0)
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if line:
                            decoded = line.decode("utf-8").strip()
                            if decoded.startswith("data: "):
                                data_str = decoded[6:]
                                if data_str == "[DONE]":
                                    break
                                data = json.loads(data_str)
                                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
        except Exception as e:
            yield f"\n\n*[Connection lost or error reading from Local LLM ({endpoint['name']}): {e}. Gracefully reverting to rule-based diagnostics database]*\n\n"
            return

def get_chatbot_response(user_input, chat_history=None, system_context=None):
    """
    Specialized conversational assistant that prioritizes local LLMs and falls
    back to rule-based keywords search.
    """
    system_prompt = (
        "You are an experienced clinical AI assistant named LiverCare AI. "
        "Your role is to help medical staff interpret hepatic lab metrics, explain blood biomarkers, "
        "and suggest general care guidelines based on gastroenterology standards. "
        "Write in a helpful, warm, yet highly professional tone. "
        "Support markdown, bullets, tables, and bold tags. Never sound robotic. "
    )
    if system_context:
        system_prompt += f"\n[Active Patient Clinical Context: {system_context}]"
        
    recent_history = chat_history[-10:] if (chat_history and len(chat_history) > 10) else (chat_history or [])
    
    # Try querying the active local LLM generator
    if LocalLLMConnector.is_available():
        return LocalLLMConnector.stream_query(recent_history, system_prompt)
        
    # Fallback to local rule-based database matching
    user_input_clean = user_input.lower()
    found_responses = []
    
    for keyword, response in MEDICAL_KNOWLEDGE_DB.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_input_clean):
            found_responses.append(response)
            
    if found_responses:
        return "\n\n".join(found_responses[:2]) + "\n\n*[Local LLM connection offline. Responding from rule-based clinical database core]*"
    else:
        return (
            "I'm currently running in Rule-Based Diagnostics mode because no local LLM endpoint (Ollama or LM Studio) was detected. "
            "Try asking about 'fatty liver', 'cirrhosis', 'symptoms', 'diet', or medical markers like 'bilirubin', 'ALT', and 'AST'.\n\n"
            "*(Tip: Run Ollama with 'qwen2.5' or LM Studio locally to enable the full intelligent conversational assistant)*"
        )
