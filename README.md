# 🎵 Moodify AI

AI-powered music generation and remixing platform that creates and transforms audio based on moods and genres.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red)

## ✨ Features

- **🎼 Generate Music** - Create original tracks from mood, genre, and text prompts
- **🔄 Remix Audio** - Transform existing audio to match different moods
- **📊 Analyze Audio** - Extract detailed audio features and characteristics
- **🎨 Beautiful UI** - Intuitive Streamlit interface with custom styling

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/moodify-ai.git
cd moodify-ai
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Terminal 1 - Start Backend
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 - Start Frontend
cd frontend
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 📁 Project Structure

```
moodify-ai/
├── backend/
│   └── main.py           # FastAPI server
│   └── requirements.txt        
├── frontend/
│   └── app.py            # Streamlit UI
├── requirements.txt      # Dependencies
└── venv/                # Virtual environment
```

## 🔧 Tech Stack

- **Backend**: FastAPI, PyTorch, Transformers (MusicGen), Librosa
- **Frontend**: Streamlit, Matplotlib
- **Audio**: Librosa, SoundFile, NumPy

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate new music |
| `/remix` | POST | Remix existing audio |
| `/analyze` | POST | Analyze audio features |
| `/moods` | GET | List available moods |

API docs: `http://localhost:8000/docs`

## 🎮 Usage

**Generate Music:**
1. Select mood and genre
2. Add text description (optional)
3. Set duration (5-30 seconds)
4. Click "Generate Music"

**Remix Audio:**
1. Upload audio file (WAV/MP3/OGG)
2. Choose target mood
3. Click "Remix Audio"

**Analyze Audio:**
1. Upload audio file
2. Click "Analyze Audio"
3. View extracted features

## ⚠️ Important Notes

- First run downloads MusicGen model (~2.5 GB) - cached for reuse
- Backend must run before frontend
- FFmpeg required for audio processing
- GPU recommended for faster generation

## 🐛 Troubleshooting

**Connection refused:** Ensure backend is running on port 8000

**Module not found:** Activate venv and reinstall: `pip install -r requirements.txt`

**Slow generation:** Normal on CPU (2-5 min). Use GPU for faster results (10-30s)

## Requirements Structure

The project uses separate dependency files for frontend and backend.

- Root `requirements.txt` → Frontend (Streamlit Cloud)
- `backend/requirements.txt` → Backend (FastAPI + MusicGen)

This separation avoids cloud deployment limitations and follows standard ML system design.


For technical documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)