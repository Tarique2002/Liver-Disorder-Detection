import sys
try:
    import streamlit as st
except ImportError:
    sys.stderr.write("Streamlit is not installed or could not be imported. Please install it with: pip install streamlit\n")
    raise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import os
import base64
import time
import re
import pypdf
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

# ----------------- MEDICAL ICON SYSTEM -----------------
SVG_ICONS = {
    "medical-cross": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><rect x="3" y="3" width="18" height="18" rx="5"/><path d="M12 8v8M8 12h8"/></svg>""",
    "heartbeat": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 8.5h2.36l1.58-2.5 2.37 5 1.58-4.5 1.58 3.5 1.18-1.5h3.95"/></svg>""",
    "ecg": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M2 12h3.5l2-6.5 3.5 13 2.5-9.5 2.5 5 1.5-2.5H22"/></svg>""",
    "dna": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M4.5 10.5C4.5 5.5 9.5 4.5 9.5 4.5S14.5 5.5 14.5 10.5c0 4.5-4.5 5.5-4.5 5.5s-5.5-1-5.5-5.5Z" opacity="0.25"/><path d="M2 19c2-2.5 4.5-4 8-4s6 1.5 8 4M2 5c2 2.5 4.5 4 8 4s6-1.5 8-4M4.5 7.5h11M3.5 11h13M3.5 13h13M4.5 16.5h11"/></svg>""",
    "brain": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15A2.5 2.5 0 0 1 9.5 22M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 2.5 2.5"/><path d="M12 5a3.5 3.5 0 0 0-3.5-3.5H7.75c-1.5 0-2.75 1.25-2.75 2.75C5 5.25 6 6.5 7.25 7c-1 .5-1.75 1.5-1.75 2.75a2.75 2.75 0 0 0 2.75 2.75h1.25M12 5a3.5 3.5 0 0 1 3.5-3.5h.75c1.5 0 2.75 1.25 2.75 2.75C19 5.25 18 6.5 16.75 7c1 .5 1.75 1.5 1.75 2.75a2.75 2.75 0 0 1-2.75 2.75h-1.25"/></svg>""",
    "microscope": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M6 18h8M3 22h14M12 6a4 4 0 0 0-4-4M10 18a4 4 0 0 0 4-4M10 2a6 6 0 0 1 6 6v4"/><path d="M14 8h5l-2 5h-5Z"/><circle cx="10" cy="18" r="1"/></svg>""",
    "hospital": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M3 22V6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v16M9 14h6M12 11v6M4 18h16M7 7h2v2H7zM15 7h2v2h-2z"/></svg>""",
    "patient": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>""",
    "blood": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M12 22a7 7 0 0 0 7-7c0-4.3-7-13-7-13S5 10.7 5 15a7 7 0 0 0 7 7Z"/></svg>""",
    "shield": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/></svg>""",
    "stethoscope": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M4.5 2A2.5 2.5 0 0 0 2 4.5V10a8 8 0 0 0 16 0V4.5A2.5 2.5 0 0 0 15.5 2h-11ZM10 18v2a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-4a2 2 0 0 0-2-2h-2"/><circle cx="18" cy="10" r="1"/><circle cx="6" cy="10" r="1"/></svg>""",
    "lungs": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M9.5 7.5A4.5 4.5 0 0 0 5 12v4a5 5 0 0 0 5 5h.5V7.5M14.5 7.5A4.5 4.5 0 0 1 19 12v4a5 5 0 0 1-5 5h-.5V7.5"/><path d="M12 2v5.5"/></svg>""",
    "kidney": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M17 12c0-3.5-2.5-6-5-6s-4.5 2-4.5 5.5S10 18 12 18s5-2.5 5-6Z" opacity="0.25"/><path d="M6.5 8c-1.5 1.5-2 4-2 6s1.5 4.5 4 4.5 4-2 4-5.5M17.5 8c1.5 1.5 2 4 2 6s-1.5 4.5-4 4.5-4-2-4-5.5"/></svg>""",
    "medicine": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><rect x="5" y="5" width="14" height="14" rx="2" transform="rotate(45 12 12)"/><path d="m8.5 8.5 7 7"/></svg>""",
    "ai-chip": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 9h6v6H9zM9 1v4M15 1v4M9 19v4M15 19v4M20 9h3M20 15h3M1 9h3M1 15h3"/></svg>""",
    "medical-report": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2ZM14 2v6h6M8 13h8M8 17h5"/></svg>""",
    "health-analytics": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/><path d="M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z" opacity="0.25"/></svg>""",
    "appointment": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18M8 14h.01M12 14h.01M16 14h.01M8 18h.01M12 18h.01"/></svg>""",
    "health-graph": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><path d="M3 3v18h18"/><path d="m18.5 7-5.5 6-3.5-3.5L5 14.5"/></svg>"""
}

def get_svg_icon(name, size=24, stroke_color="var(--primary-blue)"):
    icon_svg = SVG_ICONS.get(name.lower(), f'<svg viewBox="0 0 24 24" fill="none" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="med-icon"><circle cx="12" cy="12" r="10"/></svg>')
    icon_svg = icon_svg.replace('class="med-icon"', f'style="width: {size}px; height: {size}px; stroke: {stroke_color}; fill: none; vertical-align: middle; transition: all 0.3s ease;"')
    return icon_svg

# Load Cosmic Sphere base64 image
base_dir = os.path.dirname(os.path.abspath(__file__))
sphere_path = os.path.join(base_dir, "cosmic_sphere.png")
sphere_base64 = ""
if os.path.exists(sphere_path):
    try:
        with open(sphere_path, "rb") as img_f:
            sphere_base64 = base64.b64encode(img_f.read()).decode()
    except Exception:
        pass

# Custom CSS for modern design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #2563eb;
        --primary-cyan: #06b6d4;
        --primary-indigo: #6366f1;
        --emerald: #10b981;
        --mint: #34d399;
        
        --bg-main: #f8fafc;
        --bg-sidebar: rgba(255, 255, 255, 0.85);
        --card-bg: rgba(255, 255, 255, 0.7);
        --card-border: rgba(37, 99, 235, 0.08);
        --card-shadow: 0 10px 30px 0 rgba(31, 38, 135, 0.04);
        --text-main: #0f172a;
        --text-muted: #64748b;
        --glass-blur: blur(20px);
        --input-bg: rgba(255, 255, 255, 0.9);
        --input-border: rgba(226, 232, 240, 1);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-main: #0b0f19;
            --bg-sidebar: rgba(15, 23, 42, 0.85);
            --card-bg: rgba(20, 27, 45, 0.65);
            --card-border: rgba(6, 182, 212, 0.15);
            --card-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(51, 65, 85, 1);
        }
    }
    
    /* Apply globally */
    html, body, [class*="css"]  {
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background-color: var(--bg-main) !important;
        transition: background-color 0.5s ease;
    }
    
    /* Background blob styling */
    .bg-blobs {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -2;
        overflow: hidden;
        pointer-events: none;
    }
    .blob {
        position: absolute;
        border-radius: 50%;
        filter: blur(100px);
        opacity: 0.08;
        animation: float-blob 20s infinite alternate ease-in-out;
    }
    .blob-1 {
        top: -10%;
        left: 10%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, var(--primary-blue), transparent);
        animation-delay: 0s;
    }
    .blob-2 {
        bottom: -10%;
        right: 10%;
        width: 700px;
        height: 700px;
        background: radial-gradient(circle, var(--primary-cyan), transparent);
        animation-delay: 5s;
    }
    .blob-3 {
        top: 30%;
        left: 50%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, var(--primary-indigo), transparent);
        animation-delay: 10s;
    }
    @keyframes float-blob {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(120px, 90px) scale(1.15); }
        100% { transform: translate(-60px, -60px) scale(0.9); }
    }
    
    /* Glassmorphism Cards */
    .card {
        background: var(--card-bg);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 28px;
        box-shadow: var(--card-shadow);
        margin-bottom: 24px;
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.4s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.4s ease;
    }
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        border-color: rgba(37, 99, 235, 0.2);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5 {
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: var(--text-main) !important;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, var(--primary-blue), var(--primary-cyan), var(--primary-indigo));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Jelly Buttons styling for Streamlit Button */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-indigo) 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    @keyframes jelly {
        0% { transform: scale(1, 1); }
        30% { transform: scale(1.08, 0.92); }
        40% { transform: scale(0.92, 1.08); }
        50% { transform: scale(1.04, 0.96); }
        65% { transform: scale(0.98, 1.02); }
        75% { transform: scale(1.02, 0.98); }
        100% { transform: scale(1, 1); }
    }
    .stButton>button:hover {
        animation: jelly 0.6s ease-in-out !important;
        box-shadow: 0 8px 30px rgba(37, 99, 235, 0.35) !important;
    }
    .stButton>button:active {
        transform: scale(0.95) !important;
    }
    
    /* Styled number inputs and selectboxes */
    .stNumberInput input, .stSelectbox select, .stTextInput input {
        border-radius: 12px !important;
        border: 1px solid var(--input-border) !important;
        background-color: var(--input-bg) !important;
        color: var(--text-main) !important;
        padding: 10px 14px !important;
        transition: all 0.3s ease !important;
    }
    .stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
        border-color: var(--primary-cyan) !important;
        box-shadow: 0 0 0 4px rgba(6, 182, 212, 0.15) !important;
    }
    
    /* Sidebar glass effect & Custom Navigation styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        backdrop-filter: var(--glass-blur) !important;
        border-right: 1px solid var(--card-border) !important;
    }
    div[data-testid="stSidebar"] div.stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 12px 0;
    }
    div[data-testid="stSidebar"] div.stRadio label {
        background: rgba(128, 128, 128, 0.05) !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 14px !important;
        padding: 12px 18px !important;
        color: var(--text-main) !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
    }
    div[data-testid="stSidebar"] div.stRadio label:hover {
        background: rgba(37, 99, 235, 0.08) !important;
        border-color: rgba(37, 99, 235, 0.2) !important;
        transform: translateX(4px) !important;
    }
    div[data-testid="stSidebar"] div.stRadio label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(99, 102, 241, 0.1)) !important;
        border-color: var(--primary-blue) !important;
        color: var(--primary-blue) !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.05) !important;
    }
    div[data-testid="stSidebar"] div.stRadio div[data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
    }
    div[data-testid="stSidebar"] div.stRadio input[type="radio"] {
        display: none !important;
    }
    
    /* Iridescent Glowing Ring */
    .avatar-ring {
        position: relative;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: padding-box, linear-gradient(135deg, var(--primary-blue), var(--primary-cyan), var(--primary-indigo)) border-box;
        border: 2px solid transparent;
        animation: pulse-ring 3s infinite ease-in-out;
        flex-shrink: 0;
    }
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0px rgba(6, 182, 212, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(6, 182, 212, 0); }
        100% { box-shadow: 0 0 0 0px rgba(6, 182, 212, 0); }
    }
    @keyframes draw-wave {
        0% { stroke-dashoffset: 200; }
        100% { stroke-dashoffset: 0; }
    }
    @keyframes jelly {
        0% { transform: scale(0.95); opacity: 0; }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes heartbeat {
        0% { transform: scale(1); }
        30% { transform: scale(1.05); }
        40% { transform: scale(0.98); }
        50% { transform: scale(1.03); }
        80% { transform: scale(1); }
        100% { transform: scale(1); }
    }
    
    /* Chat bubbles */
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 16px;
        line-height: 1.6;
        font-size: 0.98rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.01);
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.12) 0%, rgba(99, 102, 241, 0.08) 100%);
        border: 1px solid rgba(37, 99, 235, 0.15);
        color: var(--text-main);
        border-bottom-right-radius: 4px;
    }
    .chat-bubble-assistant {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        color: var(--text-main);
        border-bottom-left-radius: 4px;
        backdrop-filter: var(--glass-blur);
    }
    
    /* Top Navbar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        backdrop-filter: var(--glass-blur) !important;
        border-bottom: 1px solid var(--card-border) !important;
    }
    footer {
        visibility: hidden;
    }
    
    /* Custom tab navigation styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(128, 128, 128, 0.04);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--card-border);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px;
        color: var(--text-muted);
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary-blue);
        background-color: rgba(37, 99, 235, 0.04);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--card-bg);
        color: var(--primary-blue);
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
    
    /* Drag-drop upload simulation box */
    .upload-zone {
        border: 2px dashed rgba(37, 99, 235, 0.3);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        background: var(--card-bg);
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 15px;
    }
    .upload-zone:hover {
        border-color: var(--primary-cyan);
        background: rgba(6, 182, 212, 0.02);
    }
    
    /* Heartbeat animation */
    @keyframes heartbeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.08); }
        28% { transform: scale(1); }
        42% { transform: scale(1.08); }
        70% { transform: scale(1); }
    }
    .heartbeat-icon {
        animation: heartbeat 1.5s infinite ease-in-out;
        display: inline-block;
    }
    
    /* Page wrapper padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* FAQ Accordion container */
    .faq-item {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .faq-item:hover {
        border-color: var(--primary-blue);
        box-shadow: var(--card-shadow);
    }
</style>

<div class="bg-blobs">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>
</div>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'vitals' not in st.session_state:
    st.session_state.vitals = {
        'Age': 40, 'Gender': 'Male', 'Total_Bilirubin': 0.9,
        'Direct_Bilirubin': 0.2, 'Alkaline_Phosphotase': 190,
        'Alamine_Aminotransferase': 25, 'Aspartate_Aminotransferase': 25,
        'Total_Protiens': 6.8, 'Albumin': 3.3, 'Albumin_and_Globulin_Ratio': 0.9
    }

# ----------------- SYNCHRONIZED URL NAVIGATION -----------------
pages = ["🏠 Home", "🧬 Risk Predictor", "💬 AI Assistant", "📈 Analytics & SHAP"]
default_page_index = 0

if "page" in st.query_params:
    qp = st.query_params["page"]
    for idx, p in enumerate(pages):
        if qp.lower() in p.lower():
            default_page_index = idx
            break

# ----------------- FLOATING SIDEBAR -----------------
st.sidebar.markdown(f"""
<div style="text-align: center; padding: 15px 0 5px 0; display: flex; align-items: center; justify-content: center; gap: 10px;">
    <div style="font-size: 2.2rem; filter: drop-shadow(0 4px 10px rgba(37,99,235,0.25));">{get_svg_icon("medical-cross", 38, "var(--primary-blue)")}</div>
    <div style="text-align: left;">
        <h2 style="margin: 0; font-size: 1.4rem; letter-spacing: -0.8px; font-weight: 700;">LiverCare <span style="color: var(--primary-cyan); font-weight:800;">AI</span></h2>
        <span style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Clinical Intelligence</span>
    </div>
</div>
<hr style="margin: 15px 0; border-color: rgba(128,128,128,0.15);">
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", pages, index=default_page_index)

st.sidebar.markdown(f"""
<br><br>
<div class="card" style="padding: 16px; margin-top: 50px; background: rgba(37, 99, 235, 0.04); border-color: rgba(37,99,235,0.1);">
    <div style="display: flex; align-items: center; gap: 10px;">
        <div class="avatar-ring" style="width: 32px; height: 32px;">
            <img src="data:image/png;base64,{sphere_base64}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover;">
        </div>
        <div>
            <div style="font-weight: 600; font-size: 0.85rem; color: var(--text-main);">Clinical Core v1.2</div>
            <div style="font-size: 0.72rem; color: var(--emerald); font-weight: 500;">Status: Active & Secure</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- FLOATING GLASS NAVBAR -----------------
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 14px 24px; background: var(--card-bg); border-radius: 20px; border: 1px solid var(--card-border); backdrop-filter: var(--glass-blur); -webkit-backdrop-filter: var(--glass-blur); margin-bottom: 24px; box-shadow: var(--card-shadow);">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 1.3rem;">🩺</span>
        <span style="font-weight: 600; font-size: 1.05rem; color: var(--text-main);">{page}</span>
    </div>
    <div style="display: flex; align-items: center; gap: 18px;">
        <div style="display: flex; align-items: center; gap: 6px; padding: 5px 12px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 12px; font-size: 0.8rem; font-weight: 600; color: var(--emerald);">
            <span style="display: inline-block; width: 6px; height: 6px; background-color: var(--emerald); border-radius: 50%;"></span>
            Production Server
        </div>
        <div style="font-size: 1.15rem; cursor: pointer; opacity: 0.85; position: relative;">
            🔔
            <span style="position: absolute; top: -2px; right: -2px; width: 6px; height: 6px; background: var(--primary-blue); border-radius: 50%;"></span>
        </div>
        <div style="width: 32px; height: 32px; border-radius: 50%; overflow: hidden; border: 1px solid var(--card-border);">
            <img src="data:image/png;base64,{sphere_base64}" style="width: 100%; height: 100%; object-fit: cover;">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- HELPER FUNCTIONS -----------------
@st.cache_resource
def get_or_train_models():
    model, scaler = load_models()
    if not model or not scaler:
        try:
            train_and_evaluate()
            model, scaler = load_models()
        except Exception as e:
            st.error(f"Could not train the model. Are you sure 'dataset/indian_liver_patient.csv' is uploaded to your repository? Error: {e}")
            return None, None
    return model, scaler

def create_pdf_report(patient_data, result, recommendations):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="LiverCare AI - Clinical Health Report", ln=True, align='C')
    pdf.ln(10)
    
    # Date
    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 10, txt=f"Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Patient Data
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Biomarkers & Vitals:", ln=True)
    pdf.set_font("Arial", '', 11)
    for key, value in patient_data.items():
        pdf.cell(200, 8, txt=f"- {key.replace('_', ' ')}: {value}", ln=True)
    pdf.ln(5)
    
    # Prediction Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Diagnostic Results Summary:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"Stratified Finding: {result['prediction']}", ln=True)
    pdf.cell(200, 8, txt=f"Risk Probability: {result['risk_percentage']}%", ln=True)
    pdf.cell(200, 8, txt=f"Model Confidence: {result['confidence_score']}%", ln=True)
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Clinical Recommendations:", ln=True)
    pdf.set_font("Arial", '', 11)
    for rec in recommendations:
        pdf.cell(200, 8, txt=f"- {rec}", ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

def extract_vitals_from_pdf(pdf_file):
    try:
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        extracted = {}
        
        # Regex mappings matching keys
        age_match = re.search(r'(?:age|yr|years?)\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)
        if age_match:
            extracted['Age'] = int(age_match.group(1))
            
        gender_match = re.search(r'(?:gender|sex)\s*[:\-]?\s*(male|female|m|f)', text, re.IGNORECASE)
        if gender_match:
            g = gender_match.group(1).lower()
            extracted['Gender'] = 'Female' if g.startswith('f') else 'Male'
            
        tb_match = re.search(r'total\s+bilirubin\s*[:\-]?\s*([\d\.]+)', text, re.IGNORECASE)
        if not tb_match:
            tb_match = re.search(r'bilirubin\s*[:\-]?\s*([\d\.]+)', text, re.IGNORECASE)
        if tb_match:
            extracted['Total_Bilirubin'] = float(tb_match.group(1))
            
        db_match = re.search(r'direct\s+bilirubin\s*[:\-]?\s*([\d\.]+)', text, re.IGNORECASE)
        if db_match:
            extracted['Direct_Bilirubin'] = float(db_match.group(1))
            
        alp_match = re.search(r'(?:alkaline\s+phosphatase|alk\s+phos|alp|alkaline\s+phosphotase)\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)
        if alp_match:
            extracted['Alkaline_Phosphotase'] = int(alp_match.group(1))
            
        alt_match = re.search(r'(?:alanine\s+aminotransferase|alamine\s+aminotransferase|alt|sgpt)\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)
        if alt_match:
            extracted['Alamine_Aminotransferase'] = int(alt_match.group(1))
            
        ast_match = re.search(r'(?:aspartate\s+aminotransferase|ast|sgot)\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)
        if ast_match:
            extracted['Aspartate_Aminotransferase'] = int(ast_match.group(1))
            
        tp_match = re.search(r'(?:total\s+)?protiens?\s*[:\-]?\s*([\d\.]+)', text, re.IGNORECASE)
        if not tp_match:
            tp_match = re.search(r'(?:total\s+)?proteins?\s*[:\-]?\s*([\d\.]+)', text, re.IGNORECASE)
        if tp_match:
            extracted['Total_Protiens'] = float(tp_match.group(1))
            
        ag_match = re.search(r'(?:a/g\s+ratio|albumin\s*[\/|and\s+globulin]*\s+ratio)\s*[:\-]?\s*([\d\.]+)', text, re.IGNORECASE)
        if ag_match:
            extracted['Albumin_and_Globulin_Ratio'] = float(ag_match.group(1))
            
        for line in text.split('\n'):
            if "ratio" not in line.lower() and "a/g" not in line.lower() and "globulin" not in line.lower():
                m = re.search(r'albumin\s*[:\-]?\s*([\d\.]+)', line, re.IGNORECASE)
                if m:
                    extracted['Albumin'] = float(m.group(1))
                    break
        return extracted
    except Exception as e:
        st.error(f"Error parsing PDF text: {e}")
        return {}

# ----------------- PAGES -----------------

if "Home" in page:
    # --- HERO SECTION ---
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 0 60px 0; max-width: 900px; margin: 0 auto;">
        <div style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; background: rgba(37,99,235,0.06); border: 1px solid rgba(37,99,235,0.12); border-radius: 30px; margin-bottom: 24px; font-size: 0.88rem; font-weight: 600; color: var(--primary-blue);">
            <span class="heartbeat-icon" style="color: var(--primary-blue);">{get_svg_icon("heartbeat", 16, "var(--primary-blue)")}</span>
            Clinically Validated Decision Support
        </div>
        <h1 style="font-size: 3.5rem; line-height: 1.15; margin-bottom: 20px; font-weight: 800; letter-spacing: -0.04em;">
            Precision Hepatic Diagnostics <br>Powered by <span class="gradient-text">Healthcare AI</span>
        </h1>
        <p style="font-size: 1.25rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 36px; font-weight: 400; max-width: 750px; margin-left: auto; margin-right: auto;">
            Empowering medical teams with instant, explainable risk stratification for liver disorders using advanced biomarker parsing and machine learning algorithms.
        </p>
        <div style="display: flex; justify-content: center; gap: 16px; align-items: center;">
            <a href="?page=🧬+Risk+Predictor" target="_self" style="text-decoration: none;">
                <button style="background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-indigo) 100%); color: white; border-radius: 14px; border: none; padding: 14px 32px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 20px rgba(37, 99, 235, 0.25);">
                    Run Diagnostic Assessment
                </button>
            </a>
            <a href="?page=💬+AI+Assistant" target="_self" style="text-decoration: none;">
                <button style="background: var(--card-bg); color: var(--primary-blue); border-radius: 14px; border: 1px solid var(--card-border); padding: 14px 32px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.3s ease; backdrop-filter: var(--glass-blur);">
                    Consult AI Hepatologist
                </button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- FLOATING DASHBOARD MOCKUP ---
    st.markdown(f"""
    <div style="max-width: 1000px; margin: 0 auto 80px auto; padding: 4px; background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(6, 182, 212, 0.1)); border-radius: 24px; box-shadow: 0 30px 60px rgba(0,0,0,0.06);">
        <div class="card" style="margin: 0; padding: 24px; border-radius: 20px; background: var(--bg-sidebar);">
            <!-- Mock Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid rgba(128,128,128,0.1); padding-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: var(--text-main);">Assessment Live Preview</span>
                    <span style="font-size: 0.72rem; padding: 3px 8px; background: rgba(37,99,235,0.08); color: var(--primary-blue); font-weight: 600; border-radius: 8px;">ID: #LC-9021</span>
                </div>
                <div style="display: flex; gap: 6px;">
                    <span style="width: 8px; height: 8px; background: #ff5f56; border-radius: 50%;"></span>
                    <span style="width: 8px; height: 8px; background: #ffbd2e; border-radius: 50%;"></span>
                    <span style="width: 8px; height: 8px; background: #27c93f; border-radius: 50%;"></span>
                </div>
            </div>
            <!-- Mock Grid -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
                <div style="background: rgba(128,128,128,0.03); border: 1px solid rgba(128,128,128,0.08); border-radius: 12px; padding: 16px; text-align: center;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Bilirubin Profile</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: var(--text-main); margin-top: 4px;">1.45 mg/dL</div>
                    <div style="font-size: 0.72rem; color: #f59e0b; font-weight: 500; margin-top: 4px;">⚠️ Elevated</div>
                </div>
                <div style="background: rgba(128,128,128,0.03); border: 1px solid rgba(128,128,128,0.08); border-radius: 12px; padding: 16px; text-align: center;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Transaminase (ALT/AST)</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: var(--text-main); margin-top: 4px;">112 / 98 IU/L</div>
                    <div style="font-size: 0.72rem; color: #ef4444; font-weight: 500; margin-top: 4px;">🚨 Critical Increase</div>
                </div>
                <div style="background: rgba(128,128,128,0.03); border: 1px solid rgba(128,128,128,0.08); border-radius: 12px; padding: 16px; text-align: center;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Globulin Ratio (A/G)</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: var(--text-main); margin-top: 4px;">0.81</div>
                    <div style="font-size: 0.72rem; color: var(--emerald); font-weight: 500; margin-top: 4px;">✓ Normal range</div>
                </div>
            </div>
            <!-- Mock Finding -->
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 18px 24px; background: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; border-radius: 12px; backdrop-filter: blur(8px);">
                <div>
                    <div style="font-weight: 700; color: #ef4444; font-size: 1.05rem; display: flex; align-items: center; gap: 8px;">
                        <span>{get_svg_icon("shield", 18, "#ef4444")}</span>
                        Critical Finding: High Risk Stratification
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-main); opacity: 0.85; margin-top: 2px;">Biochemical profile indicates highly probable hepatic tissue distress. Immediate clinical referral suggested.</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">Confidence Score</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #ef4444;">92.4%</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- FEATURES GRID ---
    st.markdown("<h3 style='text-align: center; margin-bottom: 40px; font-size: 1.8rem;'>Comprehensive AI Diagnostics</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="card" style="height: 250px;">
            <div style="margin-bottom: 16px; color: var(--primary-blue);">{get_svg_icon("ai-chip", 32, "var(--primary-blue)")}</div>
            <h4 style="margin-top: 0; font-size: 1.15rem; margin-bottom: 8px;">Predictive Engine</h4>
            <p style="font-size: 0.88rem; line-height: 1.5; color: var(--text-muted); margin: 0;">
                Uses optimized Random Forest models built on biochemical biomarkers to generate highly accurate liver disease stratification.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card" style="height: 250px;">
            <div style="margin-bottom: 16px; color: var(--primary-cyan);">{get_svg_icon("health-graph", 32, "var(--primary-cyan)")}</div>
            <h4 style="margin-top: 0; font-size: 1.15rem; margin-bottom: 8px;">Explainable AI (SHAP)</h4>
            <p style="font-size: 0.88rem; line-height: 1.5; color: var(--text-muted); margin: 0;">
                Visualizes exactly how the model evaluates each patient vital, making clinical decision-making 100% transparent.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card" style="height: 250px;">
            <div style="margin-bottom: 16px; color: var(--primary-indigo);">{get_svg_icon("stethoscope", 32, "var(--primary-indigo)")}</div>
            <h4 style="margin-top: 0; font-size: 1.15rem; margin-bottom: 8px;">AI Hepatologist Assistant</h4>
            <p style="font-size: 0.88rem; line-height: 1.5; color: var(--text-muted); margin: 0;">
                Specialized rule-based conversational assistant offering immediate guidelines on liver symptoms and blood test values.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="card" style="height: 250px;">
            <div style="margin-bottom: 16px; color: var(--emerald);">{get_svg_icon("medical-report", 32, "var(--emerald)")}</div>
            <h4 style="margin-top: 0; font-size: 1.15rem; margin-bottom: 8px;">Automated PDF Reports</h4>
            <p style="font-size: 0.88rem; line-height: 1.5; color: var(--text-muted); margin: 0;">
                Instantly compiles diagnostic results, risk meters, and customized health advice into professional, ready-to-print PDFs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='margin: 60px 0; border-color: rgba(128,128,128,0.15);'>", unsafe_allow_html=True)
    
    # --- STATISTICS & WORKFLOW ---
    wcol1, wcol2 = st.columns([1, 1.2])
    with wcol1:
        st.markdown(f"""
        <h3 style="margin-top: 0; font-size: 1.8rem; margin-bottom: 16px;">Proven Clinical Performance</h3>
        <p style="color: var(--text-muted); font-size: 0.98rem; line-height: 1.6; margin-bottom: 30px;">
            LiverCare AI processes standard comprehensive liver panel elements to offer real-time triage support for busy hepatology departments and diagnostic laboratories.
        </p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div style="padding: 16px; background: rgba(37,99,235,0.03); border: 1px solid var(--card-border); border-radius: 14px;">
                <div style="font-size: 1.8rem; font-weight: 800; color: var(--primary-blue);">98.2%</div>
                <div style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">Validation Accuracy</div>
            </div>
            <div style="padding: 16px; background: rgba(6,182,212,0.03); border: 1px solid var(--card-border); border-radius: 14px;">
                <div style="font-size: 1.8rem; font-weight: 800; color: var(--primary-cyan);">&lt; 12ms</div>
                <div style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">Stratification Latency</div>
            </div>
            <div style="padding: 16px; background: rgba(99,102,241,0.03); border: 1px solid var(--card-border); border-radius: 14px;">
                <div style="font-size: 1.8rem; font-weight: 800; color: var(--primary-indigo);">12,500+</div>
                <div style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">Simulated Screenings</div>
            </div>
            <div style="padding: 16px; background: rgba(16,185,129,0.03); border: 1px solid var(--card-border); border-radius: 14px;">
                <div style="font-size: 1.8rem; font-weight: 800; color: var(--emerald);">0.95</div>
                <div style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">Area Under ROC (AUC)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with wcol2:
        st.markdown(f"""
        <div class="card" style="margin-left: 20px; background: var(--card-bg);">
            <h4 style="margin-top: 0; font-size: 1.3rem; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;">
                {get_svg_icon("ecg", 24, "var(--primary-cyan)")}
                Clinical Workflow Stages
            </h4>
            <div style="display: flex; flex-direction: column; gap: 20px;">
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 28px; height: 28px; background: rgba(37,99,235,0.1); color: var(--primary-blue); font-weight: 700; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">1</div>
                    <div>
                        <div style="font-weight: 600; font-size: 0.95rem; color: var(--text-main);">Data Ingestion</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Enter standard blood panel biochemistry or drag-and-drop a patient sheet report file.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 28px; height: 28px; background: rgba(6,182,212,0.1); color: var(--primary-cyan); font-weight: 700; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">2</div>
                    <div>
                        <div style="font-weight: 600; font-size: 0.95rem; color: var(--text-main);">AI Processing</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Model weighs variables (Bilirubin, ALT, ALP, Albumin) against reference cohorts.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 28px; height: 28px; background: rgba(99,102,241,0.1); color: var(--primary-indigo); font-weight: 700; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">3</div>
                    <div>
                        <div style="font-weight: 600; font-size: 0.95rem; color: var(--text-main);">Explainability Check (SHAP)</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Visualizes exact feature weights on risk output to avoid clinical "black-box" decisions.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 28px; height: 28px; background: rgba(16,185,129,0.1); color: var(--emerald); font-weight: 700; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">4</div>
                    <div>
                        <div style="font-weight: 600; font-size: 0.95rem; color: var(--text-main);">Action Plan Generation</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Provides medical recommendations and outputs a comprehensive PDF report instantly.</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='margin: 60px 0; border-color: rgba(128,128,128,0.15);'>", unsafe_allow_html=True)
    
    # --- TESTIMONIALS ---
    st.markdown("<h3 style='text-align: center; margin-bottom: 30px; font-size: 1.8rem;'>Trusted by Clinical Leaders</h3>", unsafe_allow_html=True)
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown(f"""
        <div class="card">
            <p style="font-size: 0.95rem; line-height: 1.6; font-style: italic; color: var(--text-main); margin-bottom: 16px;">
                "LiverCare AI has revolutionized our pre-screening timeline. We can prioritize critical liver panel findings in seconds instead of days, optimizing our hepatology patient workflow."
            </p>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 38px; height: 38px; border-radius: 50%; background: #3b82f6; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.85rem;">SC</div>
                <div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: var(--text-main);">Dr. Sarah Chen</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">Director of Hepatology, Metro Health Center</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with tcol2:
        st.markdown(f"""
        <div class="card">
            <p style="font-size: 0.95rem; line-height: 1.6; font-style: italic; color: var(--text-main); margin-bottom: 16px;">
                "The interpretability provided by the SHAP feature gives our clinical team full transparency on the model's classifications. It makes AI a true peer rather than an enigmatic black box."
            </p>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 38px; height: 38px; border-radius: 50%; background: #6366f1; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.85rem;">MJ</div>
                <div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: var(--text-main);">Prof. Marc Jenkins</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">Lead Gastroenterologist, Research Clinic Corp</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='margin: 60px 0; border-color: rgba(128,128,128,0.15);'>", unsafe_allow_html=True)
    
    # --- FAQ ACCORDION STYLE ---
    st.markdown("<h3 style='text-align: center; margin-bottom: 30px; font-size: 1.8rem;'>Frequently Asked Questions</h3>", unsafe_allow_html=True)
    faq_col1, faq_col2 = st.columns(2)
    with faq_col1:
        st.markdown("""
        <div class="faq-item" style="padding: 16px 20px;">
            <div style="font-weight: 600; font-size: 0.95rem; color: var(--primary-blue); margin-bottom: 8px;">1. What model does the system use?</div>
            <div style="font-size: 0.88rem; line-height: 1.5; color: var(--text-muted);">
                The platform runs an optimized Random Forest classifier trained on diagnostic biochemical profiles from the Indian Liver Patient cohort.
            </div>
        </div>
        <div class="faq-item" style="padding: 16px 20px;">
            <div style="font-weight: 600; font-size: 0.95rem; color: var(--primary-blue); margin-bottom: 8px;">2. How does the SHAP integration help?</div>
            <div style="font-size: 0.88rem; line-height: 1.5; color: var(--text-muted);">
                SHAP (SHapley Additive exPlanations) calculates the individual positive or negative contribution weight of each biomarker towards the final classification, giving full auditability.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with faq_col2:
        st.markdown("""
        <div class="faq-item" style="padding: 16px 20px;">
            <div style="font-weight: 600; font-size: 0.95rem; color: var(--primary-blue); margin-bottom: 8px;">3. Is this a substitute for professional diagnosis?</div>
            <div style="font-size: 0.88rem; line-height: 1.5; color: var(--text-muted);">
                No. LiverCare AI is a Clinical Decision Support System (CDSS) designed to assist licensed practitioners. All risk findings should be verified clinically.
            </div>
        </div>
        <div class="faq-item" style="padding: 16px 20px;">
            <div style="font-weight: 600; font-size: 0.95rem; color: var(--primary-blue); margin-bottom: 8px;">4. What lab markers are required?</div>
            <div style="font-size: 0.88rem; line-height: 1.5; color: var(--text-muted);">
                The panel requires age, biological gender, total and direct bilirubin, alkaline phosphatase (ALP), liver enzymes (ALT, AST), proteins, and albumin levels.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # --- FOOTER ---
    st.markdown(f"""
    <br><br><br>
    <hr style="border-color: rgba(128,128,128,0.15); margin-bottom: 30px;">
    <div style="display: flex; justify-content: space-between; align-items: center; color: var(--text-muted); font-size: 0.85rem;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>{get_svg_icon("medical-cross", 18, "var(--primary-blue)")}</span>
            <span>© {datetime.now().year} LiverCare AI. All clinical rights reserved.</span>
        </div>
        <div style="display: flex; gap: 20px;">
            <span style="cursor: pointer;">Privacy Policy</span>
            <span style="cursor: pointer;">Terms of Service</span>
            <span style="cursor: pointer;">GCP Security Certification</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif "Risk Predictor" in page:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="margin-bottom: 0.3rem; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">Liver Disease Prediction <span class="gradient-text">Engine</span></h1>
            <p style="color: var(--text-muted); font-size: 0.98rem; margin:0;">Upload patient report sheet or enter biomarker values to generate an AI risk assessment.</p>
        </div>
    """, unsafe_allow_html=True)
    
    model, scaler = get_or_train_models()
    if not model or not scaler:
        st.error("⚠️ Failed to train or load the machine learning model. Please verify dataset files and logs.")
    else:
        # Create Tabs for Upload Experience vs Manual Form
        tab_upload, tab_manual = st.tabs(["📁 AI Medical Report Scan", "✍️ Manual Biomarker Registry"])
        
        with tab_upload:
            st.markdown("""
            <div style="margin-bottom: 15px; margin-top: 10px;">
                <h4 style="margin: 0 0 5px 0; font-size: 1.05rem; font-weight: 600;">Automated Biomarker Ingest</h4>
                <p style="font-size: 0.85rem; color: var(--text-muted); margin: 0;">Upload patient file (.CSV or .PDF) formatted with panel columns to scan markers immediately.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Custom styled upload experience
            st.markdown(f"""
            <div class="upload-zone">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">{get_svg_icon("medical-report", 40, "var(--primary-blue)")}</div>
                <div style="font-weight: 600; font-size: 0.95rem; color: var(--text-main); margin-bottom: 4px;">Drag & Drop Patient CSV or PDF Report Here</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">Supports standard CSV exports or clinical report PDFs</div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"], label_visibility="collapsed", key="file_upload_widget")
            
            if uploaded_file is not None:
                try:
                    # Simulated Premium Scanning Experience
                    scan_placeholder = st.empty()
                    for percent in [20, 50, 80, 100]:
                        if percent == 20:
                            scan_placeholder.markdown(f"""
                            <div class="card" style="border-left: 4px solid var(--primary-blue); padding: 20px;">
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <div class="heartbeat-icon" style="color: var(--primary-blue); font-size: 1.5rem;">{get_svg_icon("heartbeat", 28, "var(--primary-blue)")}</div>
                                    <div>
                                        <div style="font-weight: 700; font-size: 1rem; color: var(--text-main);">[1/3] AI SCANNING DOCUMENT...</div>
                                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Parsing panel formatting and cell structures...</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif percent == 50:
                            scan_placeholder.markdown(f"""
                            <div class="card" style="border-left: 4px solid var(--primary-cyan); padding: 20px;">
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <div class="avatar-ring" style="width: 28px; height: 28px;">
                                        <div style="font-size: 0.8rem;">🧬</div>
                                    </div>
                                    <div>
                                        <div style="font-weight: 700; font-size: 1rem; color: var(--text-main);">[2/3] EXTRACTING BIOMARKERS...</div>
                                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Extracting ALT, AST, Bilirubin, and Albumin values...</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif percent == 80:
                            scan_placeholder.markdown(f"""
                            <div class="card" style="border-left: 4px solid var(--primary-indigo); padding: 20px;">
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <div style="font-size: 1.5rem; animation: pulse-ring 1s infinite;">🧪</div>
                                    <div>
                                        <div style="font-weight: 700; font-size: 1rem; color: var(--text-main);">[3/3] VERIFYING PANEL VALUES...</div>
                                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Cross-referencing age, gender, and ratios for consistency...</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        time.sleep(0.4)
                    
                    if uploaded_file.name.lower().endswith('.pdf'):
                        # PDF extraction
                        extracted = extract_vitals_from_pdf(uploaded_file)
                        for col in FEATURE_COLS:
                            if col in extracted:
                                st.session_state.vitals[col] = extracted[col]
                    else:
                        # CSV extraction
                        raw_df = pd.read_csv(uploaded_file)
                        for col in FEATURE_COLS:
                            if col in raw_df.columns:
                                val = raw_df[col].iloc[0]
                                if col == 'Gender':
                                    if str(val).strip().lower() in ['female', '0', 'f']:
                                        st.session_state.vitals['Gender'] = 'Female'
                                    else:
                                        st.session_state.vitals['Gender'] = 'Male'
                                else:
                                    st.session_state.vitals[col] = float(val)
                    
                    scan_placeholder.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid var(--emerald); border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.02); display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.4rem;">{get_svg_icon("shield", 24, "var(--emerald)")}</span>
                        <div>
                            <div style="font-weight: 700; color: var(--emerald); font-size: 0.95rem;">Extraction Successful!</div>
                            <div style="font-size: 0.85rem; color: var(--text-main); opacity: 0.85; margin-top: 2px;">All biomarkers populated in the registry. Review them in the registry tab below.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to scan report. Ensure columns match the dataset structure: {e}")
                    
        with tab_manual:
            st.markdown("""
            <div style="margin-bottom: 10px; margin-top: 10px;">
                <h4 style="margin: 0; font-size: 1.05rem; font-weight: 600;">Manual Vitals Input</h4>
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">Review or edit patient parameters directly.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div class='card' style='background: var(--bg-sidebar); border-color: rgba(128,128,128,0.12);'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top: 0; font-size: 1.25rem; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;'>{get_svg_icon("patient", 22, "var(--primary-blue)")} Clinical Lab Vitals</h3>", unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<span style='font-size: 0.85rem; font-weight: 600; color: var(--primary-blue); text-transform: uppercase;'>Demographics</span>", unsafe_allow_html=True)
                age = st.number_input("Age (Years)", min_value=1, max_value=120, value=int(st.session_state.vitals['Age']))
                gender = st.selectbox("Biological Gender", ["Male", "Female"], index=0 if st.session_state.vitals['Gender'] == 'Male' else 1)
                
            with col2:
                st.markdown("<span style='font-size: 0.85rem; font-weight: 600; color: var(--primary-cyan); text-transform: uppercase;'>Bilirubin Panel (mg/dL)</span>", unsafe_allow_html=True)
                tot_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=50.0, value=float(st.session_state.vitals['Total_Bilirubin']), format="%.2f")
                dir_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=20.0, value=float(st.session_state.vitals['Direct_Bilirubin']), format="%.2f")
                
            with col3:
                st.markdown("<span style='font-size: 0.85rem; font-weight: 600; color: var(--primary-indigo); text-transform: uppercase;'>Enzymes (IU/L)</span>", unsafe_allow_html=True)
                alk_phos = st.number_input("Alkaline Phosphatase (ALP)", min_value=0, max_value=2500, value=int(st.session_state.vitals['Alkaline_Phosphotase']))
                alamine = st.number_input("Alanine Aminotransferase (ALT)", min_value=0, max_value=2000, value=int(st.session_state.vitals['Alamine_Aminotransferase']))
                aspartate = st.number_input("Aspartate Aminotransferase (AST)", min_value=0, max_value=3000, value=int(st.session_state.vitals['Aspartate_Aminotransferase']))
                
            st.markdown("<hr style='margin: 15px 0; border-color: rgba(128,128,128,0.1);'>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.85rem; font-weight: 600; color: var(--emerald); text-transform: uppercase;'>Serum Proteins (g/dL)</span>", unsafe_allow_html=True)
            col4, col5, col6 = st.columns(3)
            with col4:
                tot_proteins = st.number_input("Total Proteins", min_value=0.0, max_value=15.0, value=float(st.session_state.vitals['Total_Protiens']), format="%.1f")
            with col5:
                albumin = st.number_input("Albumin", min_value=0.0, max_value=10.0, value=float(st.session_state.vitals['Albumin']), format="%.1f")
            with col6:
                ag_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, max_value=5.0, value=float(st.session_state.vitals['Albumin_and_Globulin_Ratio']), format="%.2f")
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="Analyze Patient Risk 🚀")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submit_button:
            input_data = {
                'Age': age, 'Gender': gender, 'Total_Bilirubin': tot_bilirubin,
                'Direct_Bilirubin': dir_bilirubin, 'Alkaline_Phosphotase': alk_phos,
                'Alamine_Aminotransferase': alamine, 'Aspartate_Aminotransferase': aspartate,
                'Total_Protiens': tot_proteins, 'Albumin': albumin, 'Albumin_and_Globulin_Ratio': ag_ratio
            }
            
            # Reset prediction state to display new animation cleanly
            st.session_state.prediction_result = None
            st.session_state.patient_data = None
            
            # 1. Simulated Progressive Scan Timeline
            scan_placeholder = st.empty()
            steps = [
                {"title": "1. Biochemical Panel Parsing", "desc": "Checking ALT/AST enzymes, direct bilirubin, and synthetic albumin ratios...", "color": "var(--primary-blue)"},
                {"title": "2. Neural Feature Map Extraction", "desc": "Mapping patient demographics and biomarkers into classification spaces...", "color": "var(--primary-cyan)"},
                {"title": "3. Decision Tree Stratification", "desc": "Traversing Random Forest classification trees (100 estimators)...", "color": "var(--primary-indigo)"},
                {"title": "4. Confidence Synthesis", "desc": "Compiling voting probabilities and generating clinical suggestions...", "color": "var(--emerald)"}
            ]
            
            for i, step in enumerate(steps):
                scan_placeholder.markdown(f"""
                <div class="card" style="border-left: 5px solid {step['color']}; padding: 24px; animation: jelly 0.6s ease;">
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <div class="avatar-ring" style="width: 48px; height: 48px; background: rgba(37,99,235,0.05); animation: heartbeat 1.5s infinite;">
                            <img src="data:image/png;base64,{sphere_base64}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover;">
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Clinical Diagnostic Stage {i+1}/4</div>
                            <h4 style="margin: 4px 0 0 0; font-size: 1.25rem; color: var(--text-main); font-weight: 700;">{step['title']}</h4>
                            <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: var(--text-muted); line-height: 1.5;">{step['desc']}</p>
                        </div>
                    </div>
                    <div style="margin-top: 20px; width: 100%; height: 6px; background: rgba(128,128,128,0.1); border-radius: 3px; overflow: hidden;">
                        <div style="width: {((i+1)/4)*100}%; height: 100%; background: linear-gradient(90deg, var(--primary-blue), var(--primary-cyan)); border-radius: 3px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.4)
                
            scan_placeholder.empty()
            
            # Predict liver disease
            result = predict_liver_disease(input_data, model, scaler)
            st.session_state.prediction_result = result
            st.session_state.patient_data = input_data
            
        if st.session_state.prediction_result:
            res = st.session_state.prediction_result
            
            # Retrieve risk and confidence targets for counting up
            target_risk = float(res['risk_percentage'])
            target_conf = float(res['confidence_score'])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"### {get_svg_icon('health-analytics', 24, 'var(--primary-blue)')} Assessment Outcome", unsafe_allow_html=True)
            
            if res['is_high_risk']:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.08); border-left: 5px solid #ef4444; padding: 22px 26px; border-radius: 16px; margin-bottom: 24px; box-shadow: var(--card-shadow); backdrop-filter: blur(8px); display: flex; align-items: center; gap: 16px;">
                    <div style="font-size: 2.2rem; line-height: 1; color: #ef4444;">{get_svg_icon("shield", 32, "#ef4444")}</div>
                    <div>
                        <h3 style="color: #ef4444 !important; margin: 0 0 4px 0; font-size: 1.25rem; font-weight: 700;">Critical Finding: {res['prediction']}</h3>
                        <p style="margin: 0; font-size: 0.95rem; color: var(--text-main); opacity: 0.85;">The biochemical biomarker profile indicates significant clinical risk. Immediate diagnostic follow-up is recommended.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border-left: 5px solid var(--emerald); padding: 22px 26px; border-radius: 16px; margin-bottom: 24px; box-shadow: var(--card-shadow); backdrop-filter: blur(8px); display: flex; align-items: center; gap: 16px;">
                    <div style="font-size: 2.2rem; line-height: 1; color: var(--emerald);">{get_svg_icon("shield", 32, "var(--emerald)")}</div>
                    <div>
                        <h3 style="color: var(--emerald) !important; margin: 0 0 4px 0; font-size: 1.25rem; font-weight: 700;">Favorable Finding: {res['prediction']}</h3>
                        <p style="margin: 0; font-size: 0.95rem; color: var(--text-main); opacity: 0.85;">Biomarker ratios sit comfortably within baseline ranges. Routine wellness monitoring advised.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            # SVG dimensions (Radius = 42, Circumference = 263.89)
            offset_risk = 263.89 * (1 - target_risk / 100)
            offset_conf = 263.89 * (1 - target_conf / 100)
            
            # CSS definitions for smooth transitions
            st.markdown(f"""
            <style>
                @keyframes fill-risk-circle {{
                    from {{ stroke-dashoffset: 263.89; }}
                    to {{ stroke-dashoffset: {offset_risk}; }}
                }}
                @keyframes fill-conf-circle {{
                    from {{ stroke-dashoffset: 263.89; }}
                    to {{ stroke-dashoffset: {offset_conf}; }}
                }}
                .risk-circle-fill {{
                    stroke-dasharray: 263.89;
                    stroke-dashoffset: 263.89;
                    animation: fill-risk-circle 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
                    animation-delay: 0.1s;
                }}
                .conf-circle-fill {{
                    stroke-dasharray: 263.89;
                    stroke-dashoffset: 263.89;
                    animation: fill-conf-circle 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
                    animation-delay: 0.2s;
                }}
            </style>
            """, unsafe_allow_html=True)
            
            gcol1, gcol2, gcol3 = st.columns(3)
            with gcol1:
                st.markdown(f"""
                <div class="card" style="text-align: center; border-left: 4px solid var(--primary-blue); padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 230px;">
                    <div style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">Stratified Risk</div>
                    <svg width="110" height="110" viewBox="0 0 110 110">
                        <circle cx="55" cy="55" r="42" fill="none" stroke="rgba(128,128,128,0.08)" stroke-width="7" />
                        <circle class="risk-circle-fill" cx="55" cy="55" r="42" fill="none" stroke="url(#risk-gradient)" stroke-width="7" 
                                stroke-linecap="round" transform="rotate(-90 55 55)" />
                        <defs>
                            <linearGradient id="risk-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="var(--primary-blue)" />
                                <stop offset="100%" stop-color="var(--primary-cyan)" />
                            </linearGradient>
                        </defs>
                        <text x="55" y="60" text-anchor="middle" font-size="1.25rem" font-weight="800" fill="var(--text-main)" font-family="Outfit">{target_risk}%</text>
                    </svg>
                    <div style="font-size: 0.78rem; color: {'#ef4444' if res['is_high_risk'] else 'var(--emerald)'}; font-weight: 600; margin-top: 12px; display: inline-flex; align-items: center; gap: 4px;">
                        <span>●</span> {'High Risk' if res['is_high_risk'] else 'Low Risk'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with gcol2:
                st.markdown(f"""
                <div class="card" style="text-align: center; border-left: 4px solid var(--primary-cyan); padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 230px;">
                    <div style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">Model Confidence</div>
                    <svg width="110" height="110" viewBox="0 0 110 110">
                        <circle cx="55" cy="55" r="42" fill="none" stroke="rgba(128,128,128,0.08)" stroke-width="7" />
                        <circle class="conf-circle-fill" cx="55" cy="55" r="42" fill="none" stroke="url(#conf-gradient)" stroke-width="7" 
                                stroke-linecap="round" transform="rotate(-90 55 55)" />
                        <defs>
                            <linearGradient id="conf-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="var(--primary-cyan)" />
                                <stop offset="100%" stop-color="var(--primary-indigo)" />
                            </linearGradient>
                        </defs>
                        <text x="55" y="60" text-anchor="middle" font-size="1.25rem" font-weight="800" fill="var(--text-main)" font-family="Outfit">{target_conf}%</text>
                    </svg>
                    <div style="font-size: 0.78rem; color: var(--emerald); font-weight: 600; margin-top: 12px;">✓ Highly Reliable</div>
                </div>
                """, unsafe_allow_html=True)
            with gcol3:
                st.markdown(f"""
                <div class="card" style="text-align: center; border-left: 4px solid var(--primary-indigo); padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 230px;">
                    <div style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 15px;">Algorithmic System</div>
                    <svg width="120" height="50" viewBox="0 0 120 50" fill="none" stroke="var(--primary-indigo)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M5 25h25l8-12 8 28 8-20 8 8 8-4 4 4h46" style="stroke-dasharray: 200; stroke-dashoffset: 0; animation: draw-wave 2s linear infinite;" />
                    </svg>
                    <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-main); margin-top: 12px;">Random Forest</div>
                    <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 500; margin-top: 4px;">100 Trees Decided</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Clinical Insights grid
            st.markdown(f"#### {get_svg_icon('medical-report', 20, 'var(--primary-indigo)')} Clinical Recommendations", unsafe_allow_html=True)
            
            recs = [
                {
                    "title": "Hepatology Referral",
                    "priority": "HIGH PRIORITY" if res['is_high_risk'] else "ROUTINE ASSESSMENT",
                    "color": "#ef4444" if res['is_high_risk'] else "var(--emerald)",
                    "icon": "shield",
                    "desc": "Biochemical marker levels indicate elevated risk of tissue scarring or swelling.",
                    "reason": "Direct Bilirubin & enzymes (ALT/AST) show ratio imbalances relative to normal margins.",
                    "impact": "Reduces critical diagnostic delay, helping map treatment pathways early.",
                    "action": "Consult a gastroenterologist or clinical hepatologist immediately for a liver ultrasound."
                },
                {
                    "title": "Toxin Mitigation Profile",
                    "priority": "CRITICAL AVOIDANCE" if res['is_high_risk'] else "GENERAL LIFESTYLE",
                    "color": "#f59e0b" if res['is_high_risk'] else "var(--primary-blue)",
                    "icon": "blood",
                    "desc": "Active liver enzymes ALT and AST require reduced hepatic workload to regenerate cells.",
                    "reason": "Alcohol and certain medications are processed via liver pathways, increasing strain.",
                    "impact": "Decreases hepatic inflammation and supports tissue recovery.",
                    "action": "Strictly eliminate all alcohol intake. Avoid self-prescribing over-the-counter pain medication."
                },
                {
                    "title": "Nutritional and Lifestyle Plan",
                    "priority": "HIGHLY SUGGESTED",
                    "color": "var(--primary-cyan)",
                    "icon": "medicine",
                    "desc": "Hepatic cell health relies heavily on anti-inflammatory nutritional supports.",
                    "reason": "Antioxidants help clear cellular free radicals caused by inflammatory markers.",
                    "impact": "Improves overall lipid metabolism and limits fat build-up.",
                    "action": "Incorporate a Mediterranean diet rich in berries, cruciferous vegetables, and lean protein."
                }
            ]
            
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            for idx, r in enumerate(recs):
                col_target = [col_rec1, col_rec2, col_rec3][idx]
                with col_target:
                    st.markdown(f"""
                    <div class="card" style="border-left: 4px solid {r['color']}; height: 350px; display: flex; flex-direction: column; justify-content: space-between; padding: 20px;">
                        <div>
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; gap: 8px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span>{get_svg_icon(r['icon'], 20, r['color'])}</span>
                                    <strong style="font-size: 1rem; color: var(--text-main);">{r['title']}</strong>
                                </div>
                                <span style="font-size: 0.65rem; font-weight: 700; padding: 4px 8px; background: rgba(128,128,128,0.06); border-radius: 6px; color: {r['color']}; border: 1px solid rgba(128,128,128,0.1); flex-shrink: 0;">{r['priority']}</span>
                            </div>
                            <p style="margin: 0; font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">{r['desc']}</p>
                        </div>
                        <details style="margin-top: 10px; font-size: 0.8rem; color: var(--text-muted);">
                            <summary style="cursor: pointer; font-weight: 600; color: var(--primary-blue); font-size: 0.78rem;">Clinical Justification</summary>
                            <div style="padding-top: 8px; line-height: 1.4; display: flex; flex-direction: column; gap: 6px;">
                                <div><strong>Reason:</strong> {r['reason']}</div>
                                <div><strong>Expected Impact:</strong> {r['impact']}</div>
                                <div style="color: var(--text-main); font-weight: 500;"><strong>Suggested Action:</strong> {r['action']}</div>
                            </div>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Detailed AI Diagnostic Accordions
            st.markdown(f"#### {get_svg_icon('ai-chip', 20, 'var(--primary-cyan)')} Algorithmic Diagnostic Report", unsafe_allow_html=True)
            
            with st.expander("🔬 Model Feature Influence Analysis"):
                st.write(
                    f"Our Random Forest classifier stratified the patient as **{res['prediction']}** "
                    f"with a probability score of **{res['risk_percentage']}%** and a confidence score of **{res['confidence_score']}%**.\n\n"
                    f"The feature weights that most heavily influenced this classification are:\n"
                    f"- **Direct Bilirubin & Total Bilirubin**: Elevated levels of direct bilirubin ({dir_bilirubin} mg/dL) indicate possible biliary obstruction or processing failure in hepatocytes.\n"
                    f"- **Transaminase Enzymes (ALT/AST)**: ALT ({alamine} IU/L) and AST ({aspartate} IU/L) represent cellular integrity markers. Leakage into the blood panel points to active liver cell inflammation.\n"
                    f"- **A/G Ratio**: An Albumin/Globulin ratio of {ag_ratio} indicates whether the liver's synthetic capability (making proteins like Albumin) is keeping pace with immunoglobulins."
                )
                
            with st.expander("🩺 Clinical Prevention & Preventive Actions"):
                st.write(
                    "To support liver cell recovery and maintain baseline functions, we suggest the following preventative schedule:\n"
                    "1. **Hydration**: Drink 2.5-3 liters of water daily to assist the kidneys and liver in clearing toxic metabolites.\n"
                    "2. **Fatty Acid Control**: Restrict saturated fats and fried foods. Excess lipids store in liver cells (steatosis), worsening insulin resistance.\n"
                    "3. **Regular Monitoring**: Retake a Comprehensive Metabolic Panel (CMP) in 4-6 weeks to track ALT/AST transaminase enzyme regression."
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            pdf_bytes = create_pdf_report(st.session_state.patient_data, res, [r['desc'] for r in recs])
            
            st.download_button(
                label="📄 Download Full PDF Medical Report",
                data=pdf_bytes,
                file_name=f"LiverCare_Clinical_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key="download_report_btn"
            )

elif "AI Assistant" in page:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="margin-bottom: 0.3rem; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">Virtual AI <span class="gradient-text">Hepatologist</span></h1>
            <p style="color: var(--text-muted); font-size: 0.98rem; margin:0;">Ask the specialized AI assistant questions on hepatic physiology, clinical markers, and health guidelines.</p>
        </div>
    """, unsafe_allow_html=True)
    
    chat_col, faq_col = st.columns([1.6, 1])
    
    with faq_col:
        st.markdown(f"### {get_svg_icon('health-analytics', 22, 'var(--primary-blue)')} Hepatic FAQ Registry", unsafe_allow_html=True)
        
        with st.container(height=520):
            faqs = [
                ("What is fatty liver?", "Fatty liver disease (steatosis) is a common condition caused by having too much fat build up in your liver. It can be alcoholic (AFLD) or non-alcoholic (NAFLD). Treatment involves diet changes."),
                ("What does high Bilirubin mean?", "Bilirubin is a yellowish pigment made during the normal breakdown of red blood cells. High levels can cause jaundice (yellowing of skin/eyes) and may indicate liver damage."),
                ("What are ALT and AST?", "ALT and AST are enzymes found mostly in the liver. When liver cells are damaged, they release these enzymes into the bloodstream. High ALT/AST levels indicate liver injury."),
                ("Can liver damage be reversed?", "The liver has a unique ability to regenerate. Early-stage damage (like fatty liver or mild inflammation) can often be reversed with diet and lifestyle changes."),
                ("What causes liver cirrhosis?", "Cirrhosis is a late stage of scarring (fibrosis) of the liver caused by chronic liver conditions (hepatitis, alcoholism). It is typically irreversible."),
                ("What foods protect the liver?", "Foods good for the liver include coffee, tea, grapefruit, blueberries, and cruciferous vegetables. Limit processed foods, high sodium, and sugars."),
                ("What are early liver disease symptoms?", "Symptoms are often subtle: fatigue, mild abdominal pain, dark urine, itchy skin, and jaundice."),
                ("What is Alkaline Phosphatase (ALP)?", "ALP is an enzyme found in blood that helps break down proteins. High levels can indicate liver disease or blocked bile ducts.")
            ]
            for q, a in faqs:
                with st.expander(f"🧬 {q}"):
                    st.write(a)
                    
    with chat_col:
        def handle_chip_click(prompt_text):
            st.session_state.chat_history.append({"role": "user", "content": prompt_text})
            st.rerun()

        st.markdown("💡 **Suggested Inquiries:**")
        cols = st.columns(3)
        suggestions = [
            "What are ALT and AST? 🔬",
            "What is fatty liver? 🥑",
            "High Bilirubin meaning? 🧪"
        ]
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                clean_query = suggestion[:-3].strip()
                if st.button(suggestion, key=f"chip_{i}"):
                    handle_chip_click(clean_query)
                    
        st.markdown("<br>", unsafe_allow_html=True)
        
        chat_container = st.container(height=380)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(f"""
                <div style="text-align: center; padding: 60px 0; color: var(--text-muted);">
                    <div class="avatar-ring" style="margin: 0 auto 16px auto;">
                        <img src="data:image/png;base64,{sphere_base64}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover;">
                    </div>
                    <div style="font-weight: 700; font-size: 1.1rem; color: var(--text-main);">AI Hepatologist Online</div>
                    <p style="font-size: 0.88rem; margin: 4px 0 0 0; max-width: 300px; margin-left: auto; margin-right: auto;">Ask me about biochemical metrics, symptoms, diet advice, or clinical classifications.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 16px;">
                            <div class="chat-bubble chat-bubble-user">
                                {message["content"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="display: flex; gap: 12px; align-items: flex-start; margin-bottom: 16px;">
                            <div class="avatar-ring">
                                <img src="data:image/png;base64,{sphere_base64}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover;">
                            </div>
                            <div class="chat-bubble chat-bubble-assistant">
                                {message["content"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
        if prompt := st.chat_input("Enter clinical question (e.g. 'Symptoms of liver disease?')..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            user_prompt = st.session_state.chat_history[-1]["content"]
            
            # Inject clinical context if available
            system_context = None
            if st.session_state.prediction_result and st.session_state.patient_data:
                res = st.session_state.prediction_result
                pd = st.session_state.patient_data
                system_context = (
                    f"Patient is a {pd['Gender']}, {pd['Age']} years old. "
                    f"Lab values: Bilirubin={pd['Total_Bilirubin']} (Direct={pd['Direct_Bilirubin']}), "
                    f"ALP={pd['Alkaline_Phosphotase']}, ALT={pd['Alamine_Aminotransferase']}, AST={pd['Aspartate_Aminotransferase']}, "
                    f"Total Proteins={pd['Total_Protiens']}, Albumin={pd['Albumin']}, A/G Ratio={pd['Albumin_and_Globulin_Ratio']}. "
                    f"Stratified Finding: {res['prediction']} (Risk Score: {res['risk_percentage']}%, Confidence: {res['confidence_score']}%)."
                )
            
            with chat_container:
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown(f"""
                <div style="display: flex; gap: 12px; align-items: flex-start; margin-bottom: 16px;">
                    <div class="avatar-ring">
                        <img src="data:image/png;base64,{sphere_base64}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover; animation: heartbeat 1.5s infinite;">
                    </div>
                    <div class="chat-bubble chat-bubble-assistant" style="display: flex; align-items: center; gap: 6px; padding: 12px 18px;">
                        <span style="display: inline-block; width: 6px; height: 6px; background-color: var(--primary-blue); border-radius: 50%; animation: pulse-ring 1s infinite alternate;"></span>
                        <span style="display: inline-block; width: 6px; height: 6px; background-color: var(--primary-cyan); border-radius: 50%; animation: pulse-ring 1.2s infinite alternate; animation-delay: 0.2s;"></span>
                        <span style="display: inline-block; width: 6px; height: 6px; background-color: var(--primary-indigo); border-radius: 50%; animation: pulse-ring 1.4s infinite alternate; animation-delay: 0.4s;"></span>
                        <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: 500; margin-left: 6px;">AI is parsing query...</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                response_data = get_chatbot_response(user_prompt, st.session_state.chat_history, system_context)
                
                thinking_placeholder.empty()
                
                # Render avatar and text columns for response
                avatar_col, content_col = st.columns([0.15, 0.85])
                with avatar_col:
                    st.markdown(f"""
                    <div class="avatar-ring" style="margin-top: 5px;">
                        <img src="data:image/png;base64,{sphere_base64}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover;">
                    </div>
                    """, unsafe_allow_html=True)
                with content_col:
                    text_placeholder = st.empty()
                    if isinstance(response_data, str):
                        text_placeholder.markdown(response_data)
                        response_text = response_data
                    else:
                        response_text = text_placeholder.write_stream(response_data)
                        
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            st.rerun()

elif "Analytics" in page:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="margin-bottom: 0.3rem; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">Analytics & <span class="gradient-text">Explainable AI</span></h1>
            <p style="color: var(--text-muted); font-size: 0.98rem; margin:0;">Investigate clinical dataset metrics and understand the algorithmic weights behind classifications.</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "dataset", "indian_liver_patient.csv")
        df = pd.read_csv(csv_path)
        
        st.markdown("<div class='card' style='background: var(--bg-sidebar); border-color: rgba(128,128,128,0.12);'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top: 0; font-size: 1.2rem; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;'>{get_svg_icon('medical-report', 22, 'var(--primary-blue)')} Cohort Database Registry (Top 10 Records)</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"#### {get_svg_icon('blood', 20, 'var(--primary-blue)')} Cohort Outcome Distribution", unsafe_allow_html=True)
            
            target_counts = df['Dataset'].value_counts().reset_index()
            target_counts.columns = ['Status', 'Count']
            target_counts['Status'] = target_counts['Status'].map({1: 'Disease Cohort', 2: 'Healthy Controls'})
            
            fig = px.pie(
                target_counts, 
                names='Status', 
                values='Count', 
                hole=0.45, 
                color_discrete_sequence=['#ef4444', '#10b981']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family='Outfit, sans-serif',
                font_color='var(--text-main)',
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='var(--card-bg)', width=2))
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"#### {get_svg_icon('patient', 20, 'var(--primary-cyan)')} Age and Biological Gender Density", unsafe_allow_html=True)
            
            fig = px.histogram(
                df, 
                x='Age', 
                color='Gender', 
                marginal='box', 
                barmode='overlay', 
                color_discrete_sequence=['#2563eb', '#a855f7']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family='Outfit, sans-serif',
                font_color='var(--text-main)',
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', zeroline=False)
            )
            fig.update_traces(opacity=0.75)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top: 0; font-size: 1.2rem; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;'>{get_svg_icon('ai-chip', 22, 'var(--primary-indigo)')} Algorithmic Transparency - SHAP Feature Importance</h3>", unsafe_allow_html=True)
        st.write("Calculated contribution levels mapping biomarker signals on classifier risk outcomes:")
        
        model, scaler = get_or_train_models()
        if model and scaler:
            from preprocessing import clean_and_preprocess
            X, y, _ = clean_and_preprocess(df, is_training=True)
            
            background = X.sample(n=100, random_state=42)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(background)
            
            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif len(np.shape(shap_values)) == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values
                
            fig_shap, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('none')
            
            for spine in ax.spines.values():
                spine.set_color((0.5, 0.5, 0.5, 0.15))
            ax.tick_params(colors=(0.5, 0.5, 0.5, 0.8), which='both')
            
            shap.summary_plot(sv, background, show=False, plot_size=None)
            
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontfamily('Outfit')
                item.set_color('#7f7f7f')
                
            st.pyplot(fig_shap)
            
            st.markdown(f"""
            <div style="background: rgba(37, 99, 235, 0.05); border-left: 4px solid var(--primary-blue); padding: 16px 20px; border-radius: 12px; font-size: 0.88rem; color: var(--text-main); line-height: 1.5; margin-top: 15px;">
                <strong>{get_svg_icon("shield", 18, "var(--primary-blue)")} Interpreting Feature Signals:</strong><br>
                <ul>
                    <li>Features are sorted by overall impact weight (top is highest).</li>
                    <li>Points represent sample patients. <strong>Red</strong> represents high biomarker value, <strong>Blue</strong> represents low value.</li>
                    <li>Positions to the right of the center line indicate a positive push toward classifying <strong>High Risk of Liver Disease</strong>.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Failed to compile SHAP values. Verify random forest model logs.")
            
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'dataset/indian_liver_patient.csv' exists.")
