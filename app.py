import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

st.set_page_config(
    page_title="Breast Cancer Guardian | AI Vision",
    page_icon="🎗️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #fdfafb;
    }
    .main-title {
        font-family: 'Avenir', sans-serif;
        color: #d81b60;
        text-align: center;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: -10px;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    .quote-box {
        background-color: #fce4ec;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #d81b60;
        margin-bottom: 20px;
        color: #495057;
        font-style: italic;
    }
    .quote-author {
        text-align: right;
        font-weight: bold;
        font-size: 0.9rem;
        color: #d81b60;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3364/3364307.png", width=80) 
    st.title("🎗️ Guardian AI")
    st.markdown("### A gentle reminder...")
    
    st.markdown("""
        <div class="quote-box">
            "Courage doesn't always roar. Sometimes courage is the quiet voice at the end of the day saying, 'I will try again tomorrow.'"
            <div class="quote-author">- Mary Anne Radmacher</div>
        </div>
    """, unsafe_allow_html=True) 
    
    st.markdown("""
        <div class="quote-box">
            "There is a crack in everything. That's how the light gets in."
            <div class="quote-author">- Leonard Cohen</div>
        </div>
    """, unsafe_allow_html=True) 

    st.markdown("""
        <div class="quote-box">
            "The human spirit is stronger than anything that can happen to it."
            <div class="quote-author">- C.C. Scott</div>
        </div>
    """, unsafe_allow_html=True) 

st.markdown('<div class="main-title">Breast Cancer Guardian</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by MobileNetV2 & Deep Learning</div>', unsafe_allow_html=True)

st.image("https://images.unsplash.com/photo-1579154204601-01588f351e67?q=80&w=2000&auto=format&fit=crop", use_container_width=True)

with st.expander("ℹ️ Important: What kind of images should I upload?"):
    st.write("""
        **✅ Please upload: Microscopic Tissue Slides (Histopathology)** This AI is specially trained to detect cancer at the cellular level using **H&E stained tissue slides** (which typically feature pink and purple cellular patterns).
        
        **❌ Please DO NOT upload: Mammograms or X-Rays** Mammograms are macro-level, black-and-white X-rays. Because this AI specifically looks for microscopic cellular structures, uploading an X-ray or ultrasound will confuse the model and provide inaccurate results.
    """)

@st.cache_resource
def load_guardian_model():
    return tf.keras.models.load_model("breast_cancer_final_model.keras")

try:
    model = load_guardian_model()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load the model. Error: {e}")
    model_loaded = False

st.markdown("### 🔬 Upload Histopathology Image")
uploaded_file = st.file_uploader("Choose a cell image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True, clamp=True)
    
    st.markdown("---")
    
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        analyze_button = st.button("🚀 Analyze Image", use_container_width=True)
    
    if analyze_button:
        with st.spinner("Analyzing cellular structures..."):
            time.sleep(1)
            
            IMG_SIZE = 128
            img_resized = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0)
            
            prediction = model.predict(img_array)[0][0]
            
            is_malignant = prediction > 0.5
            confidence = prediction if is_malignant else (1 - prediction)
            
            if is_malignant:
                st.error("### 🛑 Analysis Complete: Malignant Detected")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.markdown("**What this means:** The AI detected cellular patterns that are **cancerous**.")
                st.info("💡 Please remember this is an AI tool, not a doctor. Consult with an oncologist or medical professional for a clinical diagnosis.")
            else:
                st.success("### ✅ Analysis Complete: Benign Detected")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.markdown("**What this means:** The AI detected normal cellular patterns. **No cancer was detected** in this specific image.")
                st.balloons()

st.markdown("<br><hr><center>Built with ❤️ and Streamlit</center>", unsafe_allow_html=True)