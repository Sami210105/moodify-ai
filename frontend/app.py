import streamlit as st
import requests
import base64
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from io import BytesIO
import soundfile as sf

# Page configuration
st.set_page_config(
    page_title="Moodify AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* App background */
.stApp{
    background-color: #ffcfd2;
}

/* Main content */
section.main > div {
    background: #ffe5ec;
    padding: 2rem;
    border-radius: 20px 20px 0 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f8ad9d;
}

/* Fonts */
h1, h2, h3, p{
    font-family: 'Poppins', sans-serif;
    color: black;
}

/* Layout spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 0rem !important;
    min-height: auto !important;
}

/* Buttons */
.stButton>button {
    width: 100%;
    background: #fcf6bd;
    color: black;
    border: none;
    padding: 15px;
    font-size: 18px;
    border-radius: 10px;
    font-weight: bold;
}
.stButton>button:hover {
    color: #f08080;
    transform: scale(1.05);
    transition: 0.3s;
}

a[href^="#"] {
    display: none !important;
}

*:focus {
    outline: none !important;
    box-shadow: none !important;
}

footer {
    visibility: hidden !important;
    height: 0 !important;
}

section.main > div:empty {
    display: none !important;
}

.custom-footer {
    text-align: center;
    margin-top: 2rem;
    padding-bottom: 1rem;
    color: #f08080;
}
            
</style>
""", unsafe_allow_html=True)

# API Base URL
API_URL = "http://localhost:8000"

# Header
st.markdown("<h1 style='text-align: center; color: #f08080;'>Moodify AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #f08080;'>AI-Based Music Mood & Remix Generator</h3>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    mode = st.radio(
        "Select Mode",
        ["Generate New Music", "Remix Existing Audio", "Analyze Audio"],
        help="Choose what you want to do with Moodify AI"
    )

# Main content area
if mode == "Generate New Music":
    st.header("Generate New Music")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select Mood")
        mood = st.selectbox(
            "Choose a mood for your music",
            ["", "Happy", "Calm", "Energetic", "Romantic", "Focused", "Party"]
        )

    with col2:
        st.subheader("Select Genre")
        genre = st.selectbox(
            "Choose a genre",
            ["", "Pop", "Rock", "Electronic", "Classical", "Jazz", "Hip Hop", "Ambient", "Lo-fi"]
        )

    st.subheader("Describe Your Music")
    prompt = st.text_area(
        "Describe your music",
        placeholder="e.g., 'A peaceful morning with birds chirping'",
        height=100
    )

    duration = st.slider("Duration (seconds)", 5, 30, 10)

    if st.button("Generate Music"):
        if not mood and not genre and not prompt:
            st.error("Please select a mood, genre, or enter a text prompt!")
        else:
            with st.spinner("Generating your music..."):
                response = requests.post(
                    f"{API_URL}/generate",
                    json={"mood": mood, "genre": genre, "prompt": prompt, "duration": duration}
                )

                if response.status_code == 200:
                    result = response.json()
                    audio_bytes = base64.b64decode(result["audio_base64"])
                    st.audio(audio_bytes, format="audio/wav")
                    st.download_button(
                        "Download Audio",
                        audio_bytes,
                        file_name="moodify.wav",
                        mime="audio/wav"
                    )

elif mode == "Remix Existing Audio":
    st.header("Remix Existing Audio")

    uploaded_file = st.file_uploader(
        "Upload audio file (WAV, MP3, OGG)",
        type=["wav", "mp3", "ogg"]
    )

    if uploaded_file:
        st.audio(uploaded_file)

        target_mood = st.selectbox(
            "Select Target Mood for Remix",
            ["Happy", "Calm", "Energetic", "Romantic", "Focused", "Party"]
        )

        if st.button("Remix Audio"):
            with st.spinner("Remixing your audio..."):
                audio_bytes = uploaded_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
                response = requests.post(
                    f"{API_URL}/remix",
                    json={"audio_base64": audio_base64, "target_mood": target_mood}
                )

                if response.status_code == 200:
                    result = response.json()
                    remixed_bytes = base64.b64decode(result["remixed_audio_base64"])
                    st.audio(remixed_bytes)
                    st.download_button(
                        "Download Remixed Audio",
                        remixed_bytes,
                        file_name="moodify_remix.wav",
                        mime="audio/wav"
                    )

elif mode == "Analyze Audio":
    st.header("Analyze Audio Features")

    uploaded_file = st.file_uploader(
        "Upload audio file for analysis",
        type=["wav", "mp3", "ogg"]
    )

    if uploaded_file and st.button("Analyze Audio"):
        audio_bytes = uploaded_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        response = requests.post(
            f"{API_URL}/analyze",
            json={"audio_base64": audio_base64}
        )

        if response.status_code == 200:
            st.success("Analysis completed!")

# Footer
st.markdown("---")
st.markdown("""
<div class="custom-footer">
    <p>Built with ♡ | Sam</p>
</div>
""", unsafe_allow_html=True)

# Force layout refresh (prevents bottom notch)
st.write("")
