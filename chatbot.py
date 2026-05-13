import re

# Medical Knowledge Database
MEDICAL_KNOWLEDGE_DB = {
    "fatty liver": "Fatty liver disease (steatosis) is a common condition caused by having too much fat build up in your liver. It can be caused by alcohol (AFLD) or other factors like obesity and diabetes (NAFLD). Treatment usually involves diet and lifestyle changes.",
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
    "hello": "Hello! I am your Rule-Based LiverCare Assistant. You can ask me questions about liver diseases, symptoms, blood test markers (like ALT, AST, Bilirubin), or how to maintain a healthy liver.",
    "hi": "Hi there! I am your Rule-Based LiverCare Assistant. How can I help you understand liver health today?",
    "risk": "If your prediction shows a High Risk, it means your blood markers align with patterns seen in liver patients. You should consult a doctor immediately and consider lifestyle changes."
}

def get_chatbot_response(user_input, chat_history=None):
    """
    Rule-based chatbot that matches user queries to an internal Medical Knowledge DB.
    Does NOT require any external API key.
    """
    user_input = user_input.lower()
    
    found_responses = []
    
    # Check for keyword matches in the DB
    for keyword, response in MEDICAL_KNOWLEDGE_DB.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_input):
            found_responses.append(response)
            
    if found_responses:
        # Join multiple responses if multiple keywords are found
        return "\n\n".join(found_responses[:2]) # Limit to 2 paragraphs for readability
    else:
        return "I'm a rule-based AI assistant. I don't have information on that specific query. Try asking about 'fatty liver', 'cirrhosis', 'symptoms', 'diet', or medical markers like 'bilirubin', 'ALT', and 'AST'."
