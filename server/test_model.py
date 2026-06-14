from sentence_transformers import SentenceTransformer
import joblib

model = joblib.load("emotion_song_model.pkl")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def predict_mood(text):
    vector = encoder.encode([text])
    mood = model.predict(vector)[0]
    return mood

test_inputs = [
    "I am so happy today",
    "everything is falling apart",
    "I want to punch someone",
    "I feel so peaceful and content",
    "my heart is full of love",
    "I am so nervous about tomorrow",
    "I am dead tired",
    "I am at so much serenity",
    "Weather is so tranquil",
]

for text in test_inputs:
    print(f"{text} -> {predict_mood(text)}")
    