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

app = FastAPI(title="Moodify AI API")

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MusicGen model
MODEL_NAME = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Mood mapping 
MOOD_MAPPING = {
    "Happy": {
        "tempo_range": (120, 140),
        "key_preference": ["C", "G", "D", "A"],
        "mode": "major",
        "energy": "high",
        "valence": "positive"
    },
    "Calm": {
        "tempo_range": (60, 80),
        "key_preference": ["Am", "Em", "Dm"],
        "mode": "minor",
        "energy": "low",
        "valence": "neutral"
    },
    "Energetic": {
        "tempo_range": (140, 180),
        "key_preference": ["E", "B", "F#"],
        "mode": "major",
        "energy": "very_high",
        "valence": "positive"
    },
    "Romantic": {
        "tempo_range": (80, 100),
        "key_preference": ["F", "Bb", "Eb"],
        "mode": "major",
        "energy": "medium",
        "valence": "positive"
    },
    "Focused": {
        "tempo_range": (90, 110),
        "key_preference": ["C", "Am", "G"],
        "mode": "neutral",
        "energy": "medium",
        "valence": "neutral"
    },
    "Party": {
        "tempo_range": (120, 130),
        "key_preference": ["C", "G", "D"],
        "mode": "major",
        "energy": "high",
        "valence": "positive"
    }
}

class MusicGenerationRequest(BaseModel):
    mood: Optional[str] = None
    genre: Optional[str] = None
    prompt: Optional[str] = None
    duration: int = 10

class AudioFeaturesResponse(BaseModel):
    tempo: float
    key: str
    timbre: str
    rhythm: str
    harmonic_complexity: str
    spectral_contrast: Dict
    mel_spectrogram_shape: tuple
    pitch_features: Dict

class RemixRequest(BaseModel):
    audio_base64: str
    target_mood: str

class AnalyzeRequest(BaseModel):
    audio_base64: str


def preprocess_audio(audio_array: np.ndarray, sr: int, max_duration: int = 60, preserve_quality: bool = False) -> np.ndarray:
    """
    Preprocess audio to prevent memory issues:
    1. Convert to mono (high quality averaging)
    2. Limit duration
    3. Optionally downsample for analysis (not for remix output)
    """
    # Convert to mono if stereo - use proper weighted average for quality
    if audio_array.ndim > 1:
        audio_array = 0.5 * audio_array[:, 0] + 0.5 * audio_array[:, 1]
    
    # Cap length to prevent massive arrays
    max_samples = sr * max_duration
    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]
    
    # Only downsample for analysis, not for remix output
    if not preserve_quality and sr > 22050:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=22050)
        sr = 22050
    
    return audio_array, sr


def extract_audio_features(audio_array: np.ndarray, sr: int) -> Dict:
    """
    Extract audio features using Librosa with memory-safe preprocessing
    """
    # Preprocess audio to prevent memory issues
    audio_array, sr = preprocess_audio(audio_array, sr)
    
    # Ensure audio is 1D (mono)
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()
    
    # Extract tempo and beats
    tempo, beats = librosa.beat.beat_track(y=audio_array, sr=sr)

    # Mel spectrogram with reduced resolution for memory efficiency
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array, 
        sr=sr, 
        n_mels=128,  # Reduced from default
        fmax=8000    # Limit frequency range
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pitch tracking with reduced bins
    pitches, magnitudes = librosa.piptrack(
        y=audio_array, 
        sr=sr,
        fmin=80,
        fmax=400
    )
    
    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_array, sr=sr)
    
    # Chroma features for key detection
    chroma = librosa.feature.chroma_stft(y=audio_array, sr=sr)
    key_index = np.argmax(np.sum(chroma, axis=1))
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    detected_key = keys[key_index]

    # Harmonic complexity
    harmonic = librosa.effects.harmonic(y=audio_array)
    harmonic_energy = np.sum(harmonic ** 2)
    total_energy = np.sum(audio_array ** 2)
    harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0

    if harmonic_ratio > 0.7:
        complexity = "High"
    elif harmonic_ratio > 0.4:
        complexity = "Medium"
    else:
        complexity = "Low"

    # Rhythm analysis
    onset_env = librosa.onset.onset_strength(y=audio_array, sr=sr)
    rhythm_regularity = np.std(onset_env)

    if rhythm_regularity < 5:
        rhythm_pattern = "Steady and regular"
    elif rhythm_regularity < 10:
        rhythm_pattern = "Moderately varied"
    else:
        rhythm_pattern = "Highly syncopated"

    # Timbre analysis
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr)
    avg_centroid = np.mean(spectral_centroid)

    if avg_centroid > 3000:
        timbre_desc = "Bright and crisp"
    elif avg_centroid > 1500:
        timbre_desc = "Balanced and warm"
    else:
        timbre_desc = "Dark and mellow"

    return {
        "tempo": float(tempo),
        "key": detected_key,
        "timbre": timbre_desc,
        "rhythm": rhythm_pattern,
        "harmonic_complexity": complexity,
        "spectral_contrast": {
            "mean": float(np.mean(spectral_contrast)),
            "std": float(np.std(spectral_contrast))
        },
        "mel_spectrogram_shape": mel_spec_db.shape,
        "pitch_features": {
            "mean_pitch": float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0,
            "pitch_range": float(np.ptp(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
        }
    }


def apply_mood_effects(audio: np.ndarray, sr: int, mood_data: Dict) -> np.ndarray:
    """
    Apply mood-specific audio effects for better remix quality
    """
    # Apply subtle filtering based on energy level
    energy = mood_data["energy"]
    
    if energy in ["very_high", "high"]:
        # Boost high frequencies slightly for energetic moods
        # Simple high-pass emphasis
        audio = librosa.effects.preemphasis(audio, coef=0.95)
    elif energy == "low":
        # Gentle low-pass for calm moods using smoothing
        # Apply a simple moving average for smoothing
        window_size = int(sr * 0.001)  # 1ms window
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            audio = np.convolve(audio, kernel, mode='same')
    
    return audio


def build_prompt_from_mood_genre(mood: str, genre: str, user_prompt: str) -> str:
    prompt_parts = []

    if mood and mood in MOOD_MAPPING:
        mood_data = MOOD_MAPPING[mood]
        tempo_min, tempo_max = mood_data["tempo_range"]
        prompt_parts.append(f"{mood.lower()} mood")
        prompt_parts.append(f"{tempo_min}-{tempo_max} BPM")
        prompt_parts.append(f"{mood_data['energy']} energy")

    if genre:
        prompt_parts.append(f"{genre.lower()} genre")

    if user_prompt:
        prompt_parts.append(user_prompt)

    return ", ".join(prompt_parts)


@app.get("/")
def read_root():
    return {"message": "Moodify AI - AI-Based Music Mood & Remix Generator"}


@app.post("/generate")
async def generate_music(request: MusicGenerationRequest):
    try:
        generation_prompt = build_prompt_from_mood_genre(
            request.mood,
            request.genre,
            request.prompt
        )

        if not generation_prompt:
            raise HTTPException(status_code=400, detail="Please provide mood, genre, or text prompt")

        inputs = processor(
            text=[generation_prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)

        max_new_tokens = request.duration * 50

        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0
        )

        audio_array = audio_values[0, 0].cpu().numpy()
        sampling_rate = model.config.audio_encoder.sampling_rate

        features = extract_audio_features(audio_array, sampling_rate)

        audio_array = audio_array / np.max(np.abs(audio_array))

        buffer = io.BytesIO()
        wavfile.write(buffer, sampling_rate, (audio_array * 32767).astype(np.int16))
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return {
            "success": True,
            "audio_base64": audio_base64,
            "sampling_rate": sampling_rate,
            "features": features,
            "generation_params": {
                "mood": request.mood,
                "genre": request.genre,
                "prompt": generation_prompt,
                "duration": request.duration
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/analyze")
async def analyze_audio(request: AnalyzeRequest):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Preprocess before analysis
        audio_array, sr = preprocess_audio(audio_array, sr, preserve_quality=False)
        
        features = extract_audio_features(audio_array, sr)
        return {"success": True, "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/remix")
async def remix_audio(request: RemixRequest):
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Store original sample rate for high-quality output
        original_sr = sr

        # CRITICAL: Preprocess audio to prevent memory issues
        # For remix, we preserve quality (no downsampling)
        audio_array, sr = preprocess_audio(audio_array, sr, max_duration=60, preserve_quality=True)
        
        if request.target_mood not in MOOD_MAPPING:
            raise HTTPException(status_code=400, detail="Invalid mood")

        mood_data = MOOD_MAPPING[request.target_mood]
        target_tempo = sum(mood_data["tempo_range"]) / 2

        # Get current tempo
        current_tempo, beats = librosa.beat.beat_track(y=audio_array, sr=sr)
        
        # Calculate stretch factor with better bounds
        if current_tempo > 0:
            stretch_factor = current_tempo / target_tempo
            # More reasonable bounds to preserve quality
            stretch_factor = np.clip(stretch_factor, 0.7, 1.5)
        else:
            stretch_factor = 1.0

        # Apply high-quality time stretching
        remixed_audio = librosa.effects.time_stretch(audio_array, rate=stretch_factor)

        # Apply more subtle pitch shifting with better quality
        pitch_shift_steps = 0
        if mood_data["valence"] == "positive" and mood_data["energy"] in ["high", "very_high"]:
            pitch_shift_steps = 1  # More subtle shift
        elif mood_data["energy"] == "low":
            pitch_shift_steps = -1
        
        if pitch_shift_steps != 0:
            remixed_audio = librosa.effects.pitch_shift(
                remixed_audio, 
                sr=sr, 
                n_steps=pitch_shift_steps,
                bins_per_octave=24  # Higher resolution for better quality
            )

        # Apply dynamic range compression for more professional sound
        remixed_audio = librosa.effects.preemphasis(remixed_audio)
        
        # Apply mood-specific EQ and effects
        remixed_audio = apply_mood_effects(remixed_audio, sr, mood_data)
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(remixed_audio))
        if max_val > 0:
            remixed_audio = remixed_audio * 0.9 / max_val

        # Extract features from remixed audio
        features = extract_audio_features(remixed_audio, sr)

        # Write to buffer with high quality settings
        buffer = io.BytesIO()
        sf.write(buffer, remixed_audio, sr, format="WAV", subtype='PCM_24')
        buffer.seek(0)

        remixed_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return {
            "success": True,
            "remixed_audio_base64": remixed_base64,
            "features": features,
            "remix_params": {
                "target_mood": request.target_mood,
                "tempo_adjustment": f"{current_tempo:.1f} -> {target_tempo:.1f} BPM",
                "stretch_factor": f"{stretch_factor:.2f}x",
                "pitch_shift": f"{pitch_shift_steps:+d} semitones"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Remix failed: {str(e)}")


@app.get("/moods")
def get_moods():
    return {"moods": MOOD_MAPPING}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)