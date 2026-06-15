import joy from "../assets/images/joy.gif";
import sadness from "../assets/images/sadness.gif";
import anger from "../assets/images/anger.gif";
import fear from "../assets/images/fear.gif";
import romance from "../assets/images/romantic.gif";
import anxietyGif from "../assets/images/anxiety.gif";
import calm from "../assets/images/calm.gif";
import { useState } from "react";

const MOOD_CONFIG = {
  happy: {
    color: "#FFD93D",
    gif: joy,
    label: "Happy!",
    desc: "Pure joy detected. Time to dance.",
  },
  sad: {
    color: "#5ba8ff",
    gif: sadness,
    label: "Sad.",
    desc: "Sitting with the feeling. Songs that hold.",
  },
  angry: {
    color: "#ff4d6d",
    gif: anger,
    label: "Angry!",
    desc: "That fire in you. Let it out.",
  },
  calm: {
    color: "#8fcca5",
    gif: calm,
    label: "Calm.",
    desc: "Breathing slow. Soft music incoming.",
  },
  romantic: {
    color: "#FF6FB7",
    gif: romance,
    label: "Romantic♡",
    desc: "Heart is full. Love songs only.",
  },
  anxious: {
    color: "#fd8662",
    gif: anxietyGif,
    label: "Anxious.",
    desc: "We got you. Grounding tunes on the way.",
  },
};

function SongRow({ song, index, color, onClick }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "12px 18px",
        borderRadius: 14,
        border: `2px solid ${color}22`,
        background: `${color}08`,
        transition: "all 0.2s ease",
        animation: `fadeUp 0.4s ease ${index * 0.07}s both`,
        cursor: "pointer",
      }}
      onClick={onClick}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = `${color}18`;
        e.currentTarget.style.borderColor = `${color}55`;
        e.currentTarget.style.transform = "translateX(6px)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = `${color}08`;
        e.currentTarget.style.borderColor = `${color}22`;
        e.currentTarget.style.transform = "none";
      }}
    >
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: 10,
          background: `${color}22`,
          border: `2px solid ${color}44`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "'Fredoka One',cursive",
          fontSize: 15,
          color,
          flexShrink: 0,
        }}
      >
        {String(index + 1).padStart(2, "0")}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontSize: 15,
            fontWeight: 700,
            color: color,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {song.name}
        </div>
        <div
          style={{
            fontSize: 12,
            color: "#1a1630",
            marginTop: 2,
            fontWeight: 600,
          }}
        >
          {song.artist}
        </div>
      </div>
    </div>
  );
}

export default function Detector() {
  const [text, setText] = useState("");
  const [mood, setMood] = useState(null);
  const [songs, setSongs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [charVisible, setCharVisible] = useState(false);

  const cfg = mood ? MOOD_CONFIG[mood] : null;

  const detect = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setMood(null);
    setSongs([]);
    setError(null);
    setCharVisible(false);

    try {
      const res = await fetch("https://samidha21-moodify-ai-backend.hf.space/recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error("Backend error");
      const data = await res.json();
      setMood(data.mood);
      setSongs(data.songs);
      setTimeout(() => setCharVisible(true), 200);
    } catch {
      setError("Could not reach the server. Please try again later.");
    }
    setLoading(false);
  };

  return (
    <section
      id="try-it"
      style={{
        background: "#f5f0e8",
        padding: "120px 40px 160px",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Top wave */}
      <svg
        viewBox="0 0 1440 80"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          position: "absolute",
          top: -2,
          left: 0,
          width: "100%",
          pointerEvents: "none",
        }}
      >
        <path
          d="M0 0 L0 40 Q180 80 360 40 Q540 0 720 40 Q900 80 1080 40 Q1260 0 1440 40 L1440 0 Z"
          fill="#0d0b14"
        />
      </svg>

      {/* Decorative floating elements */}
      <div
        style={{
          position: "absolute",
          top: "10%",
          right: "5%",
          fontSize: 64,
          opacity: 0.12,
          userSelect: "none",
          animation: "wobble 3s ease-in-out infinite",
        }}
      >
        🎵
      </div>
      <div
        style={{
          position: "absolute",
          bottom: "15%",
          left: "5%",
          fontSize: 48,
          opacity: 0.12,
          userSelect: "none",
          animation: "wobble 4s ease-in-out 1s infinite",
        }}
      >
        🎶
      </div>
      
      <div
        style={{
          maxWidth: 680,
          margin: "0 auto",
          position: "relative",
          zIndex: 1,
        }}
      >
        <div style={{ textAlign: "center", marginBottom: 56 }}>
          <h2
            className="groovy-title"
            style={{
              fontSize: "clamp(38px,5vw,62px)",
              color: "#1a1630",
              textShadow: "4px 4px 0 #FF6FB744",
              marginBottom: 16,
            }}
          >
            Come with me to a land
            <br />
            <span style={{ color: "#b06dff" }}>of perfect music ✦</span>
          </h2>
          <p style={{ fontSize: 16, color: "#5a5270", fontWeight: 600 }}>
            Tell us how you're feeling and we'll find you the soundtrack.
          </p>
        </div>

        {/* Character popup */}
        {charVisible && cfg && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 16,
              marginBottom: 36,
              animation: "popIn 0.5s cubic-bezier(0.34,1.56,0.64,1)",
            }}
          >
            <img
              src={cfg.gif}
              alt={cfg.label}
              style={{
                width: 200,
                height: 200,
                objectFit: "cover",
                borderRadius: 28,
                animation: "slideInChar 0.6s cubic-bezier(0.34,1.56,0.64,1)",
                filter: `drop-shadow(0 4px 20px ${cfg.color}99)`,
              }}
            />
            <div style={{ textAlign: "center" }}>
              <div
                style={{
                  fontFamily: "'Fredoka One',cursive",
                  fontSize: 32,
                  color: cfg.color,
                  textShadow: `3px 3px 0 ${cfg.color}33`,
                }}
              >
                {cfg.label}
              </div>
              <div
                style={{
                  fontSize: 15,
                  color: "#5a5270",
                  fontWeight: 600,
                  marginTop: 4,
                }}
              >
                {cfg.desc}
              </div>
            </div>
          </div>
        )}

        {/* Input card */}
        <div
          className="retro-card"
          style={{
            borderColor: cfg ? cfg.color : "#1a1630",
            background: "#fff",
            transition: "border-color 0.4s ease",
          }}
        >
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) detect();
            }}
            placeholder="Tell me how you're feeling right now... I'm all ears 👂"
            rows={5}
            style={{
              width: "100%",
              padding: "24px 28px",
              background: "transparent",
              border: "none",
              resize: "none",
              fontFamily: "'Nunito',sans-serif",
              fontSize: 17,
              lineHeight: 1.7,
              color: "#1a1630",
              outline: "none",
            }}
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "14px 20px",
              borderTop: "2px solid #1a163014",
              background: "#f5f0e811",
            }}
          >
            <span
              style={{
                fontFamily: "'Space Mono',monospace",
                fontSize: 11,
                color: "#9a9ab0",
              }}
            >
              ⌘↵ to submit
            </span>
            <button
              onClick={detect}
              disabled={loading || !text.trim()}
              className="retro-btn"
              style={{
                background: cfg ? cfg.color : "#fadf72",
                color: "#1a1630",
                fontSize: 15,
                padding: "11px 28px",
                boxShadow: `3px 3px 0 #1a1630`,
                opacity: !text.trim() ? 0.5 : 1,
                transition: "all 0.2s",
              }}
            >
              {loading ? "Reading you..." : "Detect my mood"}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div
            style={{
              marginTop: 16,
              padding: "14px 20px",
              borderRadius: 12,
              background: "#ff4d6d11",
              border: "2px solid #ff4d6d44",
              fontSize: 14,
              color: "#ff4d6d",
              fontWeight: 700,
              animation: "fadeUp 0.3s ease",
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* Songs */}
        {songs.length > 0 && cfg && (
          <div style={{ marginTop: 40, animation: "fadeUp 0.5s ease" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                marginBottom: 20,
              }}
            >
              <div className="section-label" style={{ color: cfg.color }}>
                Your {cfg.label.replace(/[^a-zA-Z]/g, "")} playlist
              </div>
              <div
                style={{
                  flex: 1,
                  height: 2,
                  background: `${cfg.color}33`,
                  borderRadius: 1,
                }}
              />
              <div
                style={{
                  fontFamily: "'Space Mono',monospace",
                  fontSize: 11,
                  color: `${cfg.color}88`,
                }}
              >
                via Last.fm
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {songs.map((song, i) => (
                <SongRow
                  key={i}
                  song={song}
                  index={i}
                  color={cfg.color}
                  onClick={() =>
                    window.open(
                      `https://www.youtube.com/results?search_query=${encodeURIComponent(song.name + " " + song.artist)}`,
                      "_blank",
                    )
                  }
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Bottom wave to dark */}
      <svg
        viewBox="0 0 1440 80"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          position: "absolute",
          bottom: -2,
          left: 0,
          width: "100%",
          pointerEvents: "none",
        }}
      >
        <path
          d="M0 80 L0 40 Q180 0 360 40 Q540 80 720 40 Q900 0 1080 40 Q1260 80 1440 40 L1440 80 Z"
          fill="#0d0b14"
        />
      </svg>
    </section>
  );
}