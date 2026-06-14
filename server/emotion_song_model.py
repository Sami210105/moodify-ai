from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#load dataset
dataset = load_dataset("go_emotions")
train_data = dataset['train']

#get emotion names
emotion_names = train_data.features['labels'].feature.names

#mood to emotion mapping
emotion_to_mood = {
    "joy": "happy", "amusement": "happy", "excitement": "happy", "gratitude": "happy",
    "sadness": "sad", "grief": "sad", "disappointment": "sad", "remorse": "sad",
    "anger": "angry", "annoyance": "angry", "disgust": "angry",
    "fear": "anxious", "nervousness": "anxious", "surprise": "anxious",
    "relief": "calm", "realization": "calm", "optimism": "calm","approval": "calm",
    "love": "romantic", "admiration": "romantic", "caring": "romantic",
}

#final data
X = []
Y = []

with open("calm_sentences.txt", "r", encoding="utf-8") as f:
    calm_sentences = [line.strip() for line in f if line.strip()]
    
for sentence in calm_sentences:
    X.append(sentence)
    Y.append("calm")
    
#convert emotions to moods
for i in range(len(train_data)):

    text = train_data[i]['text']
    label_ids = train_data[i]['labels']

    mood = None

    # convert label id → emotion name
    for label_id in label_ids:

        emotion_name = emotion_names[label_id]

        # map emotion → mood
        if emotion_name in emotion_to_mood:
            mood = emotion_to_mood[emotion_name]
            break

    # skip samples with no mapped mood
    if mood is not None:
        X.append(text)
        Y.append(mood)

#encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')
X_vectors = encoder.encode(X, show_progress_bar=True)

print(X_vectors.shape)
print(len(Y))

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_vectors,
    Y,
    test_size=0.2,
    random_state=42
)

#model
model = LogisticRegression(max_iter=1000)

#train
model.fit(X_train, Y_train)

#predict
predictions = model.predict(X_test)

#results
print(classification_report(Y_test, predictions))
print("Accuracy:", accuracy_score(Y_test, predictions))

import joblib
joblib.dump(model, 'emotion_song_model.pkl')

# save encoder name instead of the object
with open("encoder_name.txt", "w") as f:
    f.write("all-MiniLM-L6-v2")

print("model saved!")