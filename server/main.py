from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import requests
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# only the model.pkl — no torch, no sentence-transformers
model = joblib.load("emotion_song_model.pkl")

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  # add this in Render env vars

HF_EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

mood_to_tag = {
    "happy": "happy",
    "sad": "sad",
    "angry": "metal",
    "anxious": "anxiety",
    "calm": "chill",
    "romantic": "romance",
}

class TextInput(BaseModel):
    text: str


def get_embedding(text: str) -> np.ndarray:
    """Call HuggingFace Inference API to get sentence embedding."""
    response = requests.post(
        HF_EMBEDDING_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text, "options": {"wait_for_model": True}},
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HF API error: {response.text}")
    
    embedding = response.json()
    # HF returns [[...]] for single input, flatten to 1D
    embedding = np.array(embedding)
    if embedding.ndim == 2:
        embedding = embedding[0]            # shape: (384,)
    elif embedding.ndim == 3:
        embedding = embedding[0].mean(0)    # token-level fallback, mean pool

    return embedding.reshape(1, -1)         # shape: (1, 384) for sklearn


@app.post("/detect-mood")
def detect_mood(input: TextInput):
    vector = get_embedding(input.text)
    mood = model.predict(vector)[0]
    return {"mood": mood}


@app.post("/recommendations")
def get_recommendations(input: TextInput):
    vector = get_embedding(input.text)
    mood = model.predict(vector)[0]
    tag = mood_to_tag[mood]

    response = requests.get(
        "http://ws.audioscrobbler.com/2.0/",
        params={
            "method": "tag.getTopTracks",
            "tag": tag,
            "api_key": LASTFM_API_KEY,
            "format": "json",
            "limit": 10,
        }
    )

    tracks = response.json()["tracks"]["track"]
    songs = [
        {
            "name": track["name"],
            "artist": track["artist"]["name"],
            "url": track["url"],
        }
        for track in tracks
    ]

    return {"mood": mood, "songs": songs}