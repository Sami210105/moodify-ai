import joblib

model = joblib.load('emotion_song_model.pkl')
vectorizer = joblib.load('emotion_song_vectorizer.pkl')

def predict_mood(text):
    vector = vectorizer.transform([text])
    mood = model.predict(vector)[0]
    return mood

test_inputs = [
    "I am so happy today",
    "everything is falling apart",
    "I want to punch someone",
    "I feel so peaceful and content",
    "my heart is full of love",
    "I am so nervous about tomorrow"
]

for text in test_inputs:
    print(f"{text} -> {predict_mood(text)}")
    