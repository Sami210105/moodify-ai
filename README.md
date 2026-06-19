# Moodify AI 🎵

**Feel it. Find it. Play it.**

Moodify AI reads how you're feeling from a few sentences of plain text and recommends a playlist to match — powered by a custom-trained emotion classifier and Last.fm's music catalog.

🔗 **Live demo:** [moodifyai-feel-it-find-it.vercel.app](https://moodifyai-feel-it-find-it.vercel.app/)

---

## How it works

1. You type how you're feeling into the Detector.
2. The text is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`) and classified into one of six moods by a `LogisticRegression` model trained on the GoEmotions dataset.
3. The detected mood is mapped to genre tags, which are sent to the Last.fm API to pull matching tracks.
4. Duplicate tracks are filtered out, the list is shuffled, and the top 10 songs are returned and rendered as a playlist.

**Moods detected:** Happy, Sad, Angry, Calm, Romantic, Anxious

---

## Tech stack

**Frontend**
- React (Vite)
- Custom scroll-driven scene architecture — a shared `ScrollContext` + scroll-progress hook drives section transitions with plain JS math (no GSAP/Lenis/animation libraries)
- Vanilla CSS

**Backend**
- FastAPI
- `sentence-transformers` for text embeddings
- `scikit-learn` (LogisticRegression) for mood classification, persisted with `joblib`
- Last.fm API for track recommendations
- Deployed on Hugging Face Spaces (Docker SDK)

---

## Project structure

```
moodify-ai/
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Scene layout + nav dots
│   │   ├── ScrollContext.jsx    # Shared scroll progress provider
│   │   ├── useScrollProgress.js # clamp / remap / lerp helpers
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Hero.jsx
│   │   │   ├── MoodsSection.jsx # Auto-playing mood carousel
│   │   │   ├── Detector.jsx     # Text input → mood + playlist
│   │   │   └── AboutFooter.jsx
│   │   ├── assets/images/       # Mood GIFs
│   │   ├── index.css
│   │   └── main.jsx
│   └── package.json
│
└── backend/
    ├── main.py                  # FastAPI app (/detect-mood, /recommendations)
    ├── emotion_song_model.pkl   # Trained classifier
    ├── calm_sentences.txt       # Extra training data (class imbalance fix)
    ├── requirements.txt
    ├── Dockerfile
    └── README.md                # Hugging Face Spaces config
```

---

## Running locally

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file:

```
LASTFM_API_KEY=your_lastfm_api_key
```

Run the server:

```bash
uvicorn main:app --reload --port 7860
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Update the backend URL in `Detector.jsx` if you're not using the deployed Hugging Face endpoint.

---

## API

### `POST /recommendations`

**Request**
```json
{ "text": "I can't stop smiling today" }
```

**Response**
```json
{
  "mood": "happy",
  "songs": [
    { "name": "...", "artist": "...", "url": "..." }
  ]
}
```

### `POST /detect-mood`

Returns just the predicted mood for a given text, without song recommendations.

---

## Deployment

- **Backend** — Hugging Face Spaces (Docker SDK), exposed on port `7860`. See `backend/Dockerfile`.
- **Frontend** — Vercel.

---

Built by [Samidha](https://github.com/Sami210105) 

I believe music should meet you where you are, not where an algorithm thinks you should be.
