import streamlit as st
import base64
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from io import BytesIO
import soundfile as sf

@st.cache_resource
def load_musicgen():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        "facebook/musicgen-small",
        trust_remote_code=True
    )

    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    model.to(device)
    model.eval()
    return processor, model, device


processor, model, device_type = load_musicgen()

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
API_URL = "http://127.0.0.1:8000"

# Timeout settings based on duration
def get_timeout(duration):
    """Calculate appropriate timeout based on duration"""
    # CPU: ~6-10 seconds per 1 second of audio
    # Add 30s buffer for processing
    return max(duration * 10 + 30, 180)  # Minimum 3 minutes


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
    
    # Performance tips
    with st.expander("⚡ Performance Tips"):
        st.markdown("""
        **On CPU:**
        - Start with 5-8 second durations
        - Generation takes ~30-120 seconds
        - Be patient - it's working!
        
        **To Speed Up:**
        - Use shorter durations (5-10s)
        - Close other heavy applications
        - Use a GPU if available
        """)

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

    # Duration with warning
    duration = st.slider("Duration (seconds)", 5, 30, 8)
    
    if device_type == 'cpu' and duration > 15:
        st.warning(f"⚠️ {duration}s on CPU will take ~{duration * 6}-{duration * 10} seconds to generate")

    if st.button("Generate Music"):
        if not mood and not genre and not prompt:
            st.error("Please select a mood, genre, or enter a text prompt!")
        else:
            timeout = get_timeout(duration)
            
            # Show estimated time
            if device_type == 'cpu':
                estimated_time = duration * 8
                st.info(f"⏱️ Estimated time: ~{estimated_time} seconds. Please wait...")
            
            with st.spinner(f"🎵 Generating {duration}s of music... This may take a while on CPU..."):
                try:
                    response = requests.post(
                        f"{API_URL}/generate",
                        json={"mood": mood, "genre": genre, "prompt": prompt, "duration": duration},
                        timeout=timeout
                    )

                    if response.status_code == 200:
                        result = response.json()
                        audio_bytes = base64.b64decode(result["audio_base64"])
                        
                        st.success("✅ Music generated successfully!")
                        st.audio(audio_bytes, format="audio/wav")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Download Audio",
                                audio_bytes,
                                file_name="moodify_generated.wav",
                                mime="audio/wav"
                            )
                        with col2:
                            st.metric("Duration", f"{result.get('duration', duration)}s")
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Generation failed: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Is it running on port 8000?")
                except requests.exceptions.Timeout:
                    st.error(f"⏱️ Generation timed out after {timeout} seconds.")
                    st.info("""
                    **Troubleshooting:**
                    - Try a shorter duration (5-8 seconds)
                    - Backend is still working - check terminal
                    - On CPU, 10s takes ~60-90 seconds
                    - Wait a bit and check if file appears
                    """)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif mode == "Remix Existing Audio":
    st.header("Remix Existing Audio")

    uploaded_file = st.file_uploader(
        "Upload audio file (WAV, MP3, OGG)",
        type=["wav", "mp3", "ogg"]
    )

    if uploaded_file:
        st.subheader("Original Audio")
        st.audio(uploaded_file)

        target_mood = st.selectbox(
            "Select Target Mood for Remix",
            ["Happy", "Calm", "Energetic", "Romantic", "Focused", "Party"]
        )

        if st.button("Remix Audio"):
            with st.spinner("🎚️ Remixing your audio... This takes 10-30 seconds..."):
                try:
                    audio_bytes = uploaded_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    
                    response = requests.post(
                        f"{API_URL}/remix",
                        json={"audio_base64": audio_base64, "target_mood": target_mood},
                        timeout=120  # 2 minutes for remix
                    )

                    if response.status_code == 200:
                        result = response.json()
                        remixed_bytes = base64.b64decode(result["remixed_audio_base64"])
                        
                        st.success("✅ Audio remixed successfully!")
                        st.subheader("Remixed Audio")
                        st.audio(remixed_bytes, format="audio/wav")
                        
                        # Show transformation details if available
                        if "original_tempo" in result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Tempo", f"{result['original_tempo']:.1f} BPM")
                            with col2:
                                st.metric("Target Tempo", f"{result['target_tempo']:.1f} BPM")
                            with col3:
                                st.metric("Stretch Factor", f"{result['stretch_factor']:.2f}x")
                        
                        st.download_button(
                            "📥 Download Remixed Audio",
                            remixed_bytes,
                            file_name=f"moodify_remix_{target_mood.lower()}.wav",
                            mime="audio/wav"
                        )
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Remix failed: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Is it running on port 8000?")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Remix timed out. Try a shorter audio file (< 30 seconds).")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif mode == "Analyze Audio":
    st.header("Analyze Audio Features")

    uploaded_file = st.file_uploader(
        "Upload audio file for analysis",
        type=["wav", "mp3", "ogg"]
    )

    if uploaded_file:
        st.subheader("Uploaded Audio")
        st.audio(uploaded_file)
        
        if st.button("Analyze Audio"):
            with st.spinner("🔍 Analyzing audio features..."):
                try:
                    audio_bytes = uploaded_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    
                    response = requests.post(
                        f"{API_URL}/analyze",
                        json={"audio_base64": audio_base64},
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Analysis completed!")
                        
                        # Display results in a nice layout
                        st.subheader("Audio Features")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tempo", f"{result.get('tempo', 0):.1f} BPM")
                            st.metric("Duration", f"{result.get('duration', 0):.2f}s")
                        with col2:
                            st.metric("Energy", f"{result.get('energy', 0):.3f}")
                            st.metric("Estimated Key", result.get('estimated_key', 'N/A'))
                        with col3:
                            st.metric("Spectral Centroid", f"{result.get('spectral_centroid', 0):.1f} Hz")
                            st.metric("Zero Crossing Rate", f"{result.get('zero_crossing_rate', 0):.4f}")
                        
                        # Additional details in expander
                        with st.expander("🔬 Technical Details"):
                            st.json(result)
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Analysis failed: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Is it running on port 8000?")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Analysis timed out. Try a shorter audio file.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="custom-footer">
    <p>Built with ♡ | Sam</p>
</div>
""", unsafe_allow_html=True)

# Force layout refresh (prevents bottom notch)
st.write("")