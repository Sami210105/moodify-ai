# Moodify AI - Technical Documentation

## Architecture Overview

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Streamlit  │  HTTP   │   FastAPI   │  AI/DSP │  MusicGen   │
│   Frontend  │◄───────►│   Backend   │◄───────►│  + Librosa  │
│   :8501     │         │   :8000     │         │             │
└─────────────┘         └─────────────┘         └─────────────┘
```

**Flow:** User Input → Streamlit → FastAPI → AI Models → Audio Processing → Response

---

## Backend (FastAPI)

### Core Components

**main.py** - FastAPI application with endpoints:
- `/generate` - Create music from mood/genre/prompt
- `/remix` - Transform audio to target mood  
- `/analyze` - Extract audio features
- `/moods` - Return available moods

### Request Models

```python
class GenerateRequest(BaseModel):
    mood: str = ""
    genre: str = ""
    prompt: str = ""
    duration: int = 10

class RemixRequest(BaseModel):
    audio_base64: str
    target_mood: str

class AnalyzeRequest(BaseModel):
    audio_base64: str
```

### Mood Mapping

```python
MOOD_PARAMETERS = {
    "Happy": {"tempo": (120, 140), "energy": 0.7},
    "Calm": {"tempo": (60, 80), "energy": 0.3},
    "Energetic": {"tempo": (140, 180), "energy": 0.9}
}
```

---

## Audio Processing Pipeline

### 1. Input Sanitization (Critical)

```python
# Force mono
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Cap duration (prevent memory overflow)
audio = audio[:sr * 30]

# Normalize
audio = audio / np.max(np.abs(audio) + 1e-8)
```

**Why:** Prevents 138 GB memory crashes from stereo/long audio

### 2. Feature Extraction

```python
# Temporal
tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

# Spectral
spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# Harmonic
chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
```

### 3. Mood Transformation

```python
# Tempo adjustment (clamped to prevent distortion)
stretch_factor = np.clip(target_tempo / current_tempo, 0.5, 2.0)
audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)

# Pitch shift
n_steps = 2 if mood_params['key'] == 'major' else -2
audio_shifted = librosa.effects.pitch_shift(audio_stretched, sr=sr, n_steps=n_steps)

# Energy adjustment
audio_final = audio_shifted * mood_params['energy']
```

---

## Frontend (Streamlit)

### UI Structure

```python
# Sidebar - Mode selection
mode = st.radio("Select Mode", [
    "Generate New Music",
    "Remix Existing Audio", 
    "Analyze Audio"
])

# Main area - Mode-specific controls
if mode == "Generate New Music":
    mood = st.selectbox("Mood", moods)
    genre = st.selectbox("Genre", genres)
    prompt = st.text_area("Description")
    duration = st.slider("Duration", 5, 30, 10)
```

### API Communication

```python
response = requests.post(
    f"{API_URL}/generate",
    json={"mood": mood, "genre": genre, "prompt": prompt, "duration": duration}
)

if response.status_code == 200:
    result = response.json()
    audio_bytes = base64.b64decode(result["audio_base64"])
    st.audio(audio_bytes, format="audio/wav")
```

---

## AI Models

### MusicGen (facebook/musicgen-small)

- **Size:** ~2.5 GB
- **Cache:** `~/.cache/huggingface/`
- **Architecture:** Transformer-based audio generation
- **Input:** Text prompts + conditioning
- **Output:** 32kHz mono audio

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small"
)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

inputs = processor(text=[prompt], return_tensors="pt")
audio = model.generate(**inputs, max_new_tokens=duration*50)
```

---

## Key Bug Fixes

### 1. Memory Overflow (138 GB)

**Problem:** Stereo + long audio → exponential mel-spectrogram size

**Solution:**
```python
# Convert to mono
audio = np.mean(audio, axis=1) if audio.ndim > 1 else audio

# Cap at 30 seconds
audio = audio[:sr * 30]
```

### 2. Tempo Explosion

**Problem:** Extreme tempo ratios (e.g., 60→180 BPM = 3x stretch)

**Solution:**
```python
stretch_factor = np.clip(stretch_factor, 0.5, 2.0)
```

### 3. JSON Parsing

**Problem:** FastAPI expected Pydantic models, not raw params

**Solution:**
```python
# Use BaseModel for all requests
class AnalyzeRequest(BaseModel):
    audio_base64: str
```

---

## API Reference

### POST /generate

**Request:**
```json
{
  "mood": "Happy",
  "genre": "Pop", 
  "prompt": "Upbeat summer vibes",
  "duration": 10
}
```

**Response:**
```json
{
  "audio_base64": "UklGRiQAAABXQVZF...",
  "metadata": {
    "sample_rate": 32000,
    "duration": 10.0
  }
}
```

### POST /remix

**Request:**
```json
{
  "audio_base64": "UklGRiQAAABXQVZF...",
  "target_mood": "Energetic"
}
```

**Response:**
```json
{
  "remixed_audio_base64": "UklGRiQAAABXQVZF...",
  "applied_effects": ["tempo_stretch", "pitch_shift", "energy_boost"]
}
```

### POST /analyze

**Response:**
```json
{
  "tempo": 120.5,
  "key": "C major",
  "energy": 0.75,
  "spectral_centroid": 2045.3,
  "duration": 180.5
}
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000 8501

CMD uvicorn backend.main:app --host 0.0.0.0 & streamlit run frontend/app.py
```

### Environment Variables

```env
API_HOST=localhost
API_PORT=8000
MODEL_CACHE_DIR=./models
MAX_AUDIO_DURATION=30
```

---

## Performance Tips

### GPU Acceleration
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

### Caching
```python
@st.cache_data
def load_model():
    return MusicgenForConditionalGeneration.from_pretrained(...)
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate(...):
    pass
```

---

## Security

### Input Validation
```python
@validator('duration')
def validate_duration(cls, v):
    if not 5 <= v <= 30:
        raise ValueError('Duration must be 5-30 seconds')
    return v
```

### File Upload Limits
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
```

---

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Author:** Sam