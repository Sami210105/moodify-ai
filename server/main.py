from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# allow react to talk to fastapi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model = joblib.load("emotion_song_model.pkl")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

class TextInput(BaseModel):
    text: str

# mood to lastfm tag mapping
mood_to_tag = {
    "happy": "happy",
    "sad": "sad",
    "angry": "metal",
    "anxious": "anxiety",
    "calm": "chill",
    "romantic": "romance",
}

@app.post("/detect-mood")
def detect_mood(input: TextInput):
    vector = encoder.encode([input.text])
    mood = model.predict(vector)[0]
    return {"mood": mood}

@app.post("/recommendations")
def get_recommendations(input: TextInput):
    vector = encoder.encode([input.text])
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