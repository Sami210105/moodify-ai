from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import librosa
import numpy as np
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import io
import base64
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Moodify AI API")

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = "facebook/musicgen-small"

print("🎵 Loading Moodify AI...")
print(f"📦 Model: {MODEL_NAME}")
print("⏳ This may take a few minutes on first run (downloading model)...")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}")

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load model with CPU-compatible settings
if device == "cuda":
    # GPU: Use float16 and low memory mode
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
else:
    # CPU: Use float32 WITHOUT low_cpu_mem_usage to avoid Meta backend issues
    print("💾 Loading model (this will use ~3-4GB RAM)...")
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
        # NOT using low_cpu_mem_usage on CPU to avoid Meta backend error
    )

model = model.to(device)
model.eval()

print(f"✅ Model loaded successfully! (dtype: {model.dtype})")

# Mood mapping
MOOD_MAPPING = {
    "Happy": {"tempo_range": (120, 140), "energy": "high", "valence": "positive"},
    "Calm": {"tempo_range": (60, 80), "energy": "low", "valence": "neutral"},
    "Energetic": {"tempo_range": (140, 180), "energy": "very_high", "valence": "positive"},
    "Romantic": {"tempo_range": (80, 100), "energy": "medium", "valence": "positive"},
    "Focused": {"tempo_range": (90, 110), "energy": "medium", "valence": "neutral"},
    "Party": {"tempo_range": (120, 130), "energy": "high", "valence": "positive"}
}

class MusicGenerationRequest(BaseModel):
    mood: Optional[str] = None
    genre: Optional[str] = None
    prompt: Optional[str] = None
    duration: int = 10

class RemixRequest(BaseModel):
    audio_base64: str
    target_mood: str

class AnalyzeRequest(BaseModel):
    audio_base64: str


def preprocess_audio(audio_array: np.ndarray, sr: int, max_duration: int = 60, preserve_quality: bool = False):
    """Preprocess audio to prevent memory issues"""
    # Convert to mono
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # Cap duration
    audio_array = audio_array[: sr * max_duration]

    # Downsample if needed
    if not preserve_quality and sr > 22050:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=22050)
        sr = 22050

    return audio_array, sr


def apply_mood_effects(audio, sr, mood):
    """Apply mood-specific audio effects"""
    if mood["energy"] in ["high", "very_high"]:
        audio = librosa.effects.preemphasis(audio)
    elif mood["energy"] == "low":
        win = int(sr * 0.001)
        if win > 1:
            audio = np.convolve(audio, np.ones(win) / win, mode="same")
    return audio


def build_prompt(mood, genre, prompt):
    """Build a comprehensive prompt from user inputs"""
    parts = []
    if mood and mood in MOOD_MAPPING:
        tmin, tmax = MOOD_MAPPING[mood]["tempo_range"]
        parts += [f"{mood.lower()} mood", f"{tmin}-{tmax} BPM"]
    if genre:
        parts.append(f"{genre.lower()} genre")
    if prompt:
        parts.append(prompt)
    return ", ".join(parts)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": MODEL_NAME,
        "device": device,
        "dtype": str(model.dtype),
        "message": "Moodify AI Backend is running!"
    }


@app.post("/generate")
async def generate_music(request: MusicGenerationRequest):
    """Generate new music from mood, genre, and prompt"""
    try:
        text_prompt = build_prompt(request.mood, request.genre, request.prompt)
        if not text_prompt:
            raise HTTPException(status_code=400, detail="Please provide mood, genre, or prompt")

        print(f"🎼 Generating music with prompt: {text_prompt}")
        print(f"⏱️  Duration: {request.duration}s")
        if device == "cpu":
            print(f"⚠️  CPU mode: This will take 30-90 seconds...")

        # Process inputs
        inputs = processor(text=[text_prompt], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate audio
        with torch.no_grad():
            audio = model.generate(
                **inputs,
                max_new_tokens=request.duration * 50,
                do_sample=True,
                guidance_scale=3.0
            )

        # Convert to numpy and normalize
        audio = audio[0, 0].cpu().numpy().astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Save to buffer
        buffer = io.BytesIO()
        sf.write(
            buffer,
            audio,
            model.config.audio_encoder.sampling_rate,
            format="WAV",
            subtype="PCM_24"
        )
        buffer.seek(0)

        # Clean up GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Music generation complete!")

        return {
            "success": True,
            "audio_base64": base64.b64encode(buffer.read()).decode(),
            "sampling_rate": model.config.audio_encoder.sampling_rate,
            "duration": request.duration
        }

    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/remix")
async def remix_audio(request: RemixRequest):
    """Remix existing audio to match target mood"""
    try:
        print(f"🎚️ Remixing audio to {request.target_mood} mood")

        # Decode audio
        audio, sr = sf.read(io.BytesIO(base64.b64decode(request.audio_base64)))
        audio = audio.astype(np.float32)
        audio, sr = preprocess_audio(audio, sr, preserve_quality=True)

        # Get mood parameters
        if request.target_mood not in MOOD_MAPPING:
            raise HTTPException(status_code=400, detail=f"Invalid mood: {request.target_mood}")
        
        mood = MOOD_MAPPING[request.target_mood]

        # Analyze current tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        target = sum(mood["tempo_range"]) / 2
        stretch = np.clip(tempo / target if tempo > 0 else 1.0, 0.7, 1.5)

        # Apply transformations
        audio = librosa.effects.time_stretch(audio, rate=stretch)
        audio = apply_mood_effects(audio, sr, mood)
        audio = audio * 0.9 / (np.max(np.abs(audio)) + 1e-8)

        # Save to buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV", subtype="PCM_24")
        buffer.seek(0)

        print("✅ Remix complete!")

        return {
            "success": True,
            "remixed_audio_base64": base64.b64encode(buffer.read()).decode(),
            "original_tempo": float(tempo),
            "target_tempo": target,
            "stretch_factor": float(stretch)
        }

    except Exception as e:
        print(f"❌ Remix error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Remix failed: {str(e)}")


@app.post("/analyze")
async def analyze_audio(request: AnalyzeRequest):
    """Analyze audio features and characteristics"""
    try:
        print("🔍 Analyzing audio...")

        # Decode audio
        audio, sr = sf.read(io.BytesIO(base64.b64decode(request.audio_base64)))
        audio = audio.astype(np.float32)
        audio, sr = preprocess_audio(audio, sr)

        # Extract features
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        energy = np.mean(librosa.feature.rms(y=audio))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        
        # Estimate key (simplified)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key = keys[key_idx]

        print("✅ Analysis complete!")

        return {
            "success": True,
            "tempo": float(tempo),
            "spectral_centroid": float(spectral_centroid),
            "energy": float(energy),
            "zero_crossing_rate": float(zero_crossing_rate),
            "estimated_key": estimated_key,
            "mfcc_mean": float(np.mean(mfcc)),
            "duration": len(audio) / sr,
            "sample_rate": sr,
            "num_beats": len(beats)
        }

    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/moods")
def get_moods():
    """Get list of available moods"""
    return {
        "moods": list(MOOD_MAPPING.keys()),
        "mood_details": MOOD_MAPPING
    }


@app.on_event("startup")
async def startup_event():
    """Startup message"""
    print("\n" + "="*50)
    print("🎵 Moodify AI Backend Ready!")
    print(f"📡 Server: http://0.0.0.0:8000")
    print(f"📚 Docs: http://0.0.0.0:8000/docs")
    print(f"🔧 Device: {device}")
    if device == "cpu":
        print("⚠️  Running on CPU - generation will be slower (30-90s)")
        print("💡 For faster generation, use a GPU-enabled system")
    print("="*50 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)