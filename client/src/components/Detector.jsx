import { useState } from "react";

const MOOD_CONFIG = {
  happy: {
    color: "#FFD93D",
    emoji: "😊",
    label: "Happy!",
    desc: "Pure joy detected. Time to dance.",
    char: "🎉",
  },
  sad: {
    color: "#5ba8ff",
    emoji: "😢",
    label: "Sad.",
    desc: "Sitting with the feeling. Songs that hold.",
    char: "🌧️",
  },
  angry: {
    color: "#ff4d6d",
    emoji: "😤",
    label: "Angry!",
    desc: "That fire in you. Let it out.",
    char: "🔥",
  },
  calm: {
    color: "#3de8c8",
    emoji: "😌",
    label: "Calm~",
    desc: "Breathing slow. Soft music incoming.",
    char: "🌿",
  },
  romantic: {
    color: "#FF6FB7",
    emoji: "💖",
    label: "Romantic♡",
    desc: "Heart is full. Love songs only.",
    char: "🌹",
  },
  anxious: {
    color: "#b06dff",
    emoji: "⚡",
    label: "Anxious.",
    desc: "We got you. Grounding tunes on the way.",
    char: "😰",
  },
};

const DEMO_SONGS = {
  happy: [
    { name: "Happy", artist: "Pharrell Williams" },
    { name: "Good as Hell", artist: "Lizzo" },
    { name: "Shake It Off", artist: "Taylor Swift" },
    { name: "Uptown Funk", artist: "Mark Ronson" },
    { name: "Walking on Sunshine", artist: "Katrina & The Waves" },
  ],
  sad: [
    { name: "The Night We Met", artist: "Lord Huron" },
    { name: "Skinny Love", artist: "Bon Iver" },
    { name: "Someone Like You", artist: "Adele" },
    { name: "River", artist: "Joni Mitchell" },
    { name: "Hurt", artist: "Johnny Cash" },
  ],
  angry: [
    { name: "Break Stuff", artist: "Limp Bizkit" },
    { name: "Given Up", artist: "Linkin Park" },
    { name: "Killing in the Name", artist: "Rage Against Machine" },
    { name: "Bodies", artist: "Drowning Pool" },
    { name: "Last Resort", artist: "Papa Roach" },
  ],
  calm: [
    { name: "Weightless", artist: "Marconi Union" },
    { name: "Holocene", artist: "Bon Iver" },
    { name: "Comptine d'un autre été", artist: "Yann Tiersen" },
    { name: "Experience", artist: "Ludovico Einaudi" },
    { name: "Clair de lune", artist: "Debussy" },
  ],
  romantic: [
    { name: "At Last", artist: "Etta James" },
    { name: "La Vie en Rose", artist: "Édith Piaf" },
    { name: "Perfect", artist: "Ed Sheeran" },
    { name: "Make You Feel My Love", artist: "Adele" },
    { name: "Can't Help Falling in Love", artist: "Elvis" },
  ],
  anxious: [
    { name: "Breathe (2 AM)", artist: "Anna Nalick" },
    { name: "The Sound of Silence", artist: "Simon & Garfunkel" },
    { name: "Heavy", artist: "Birdtalker" },
    { name: "Keep Your Head Up", artist: "Ben Howard" },
    { name: "It'll All Work Out", artist: "Tom Petty" },
  ],
};

function LoadingDots({ color }) {
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: color,
            animation: `loadingDots 1.2s ease-in-out ${i * 0.15}s infinite`,
          }}
        />
      ))}
    </div>
  );
}

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
            color: "#fff",
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
            color: "#ffffffaa",
            marginTop: 2,
            fontWeight: 600,
          }}
        >
          {song.artist}
        </div>
      </div>
      <div style={{ fontSize: 18, flexShrink: 0 }}>🎵</div>
    </div>
  );
}

export default function Detector() {
  const [text, setText] = useState("");
  const [mood, setMood] = useState(null);
  const [songs, setSongs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [demoMode, setDemoMode] = useState(true);
  const [charVisible, setCharVisible] = useState(false);

  const cfg = mood ? MOOD_CONFIG[mood] : null;

  const detect = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setMood(null);
    setSongs([]);
    setError(null);
    setCharVisible(false);
    if (demoMode) {
      await new Promise((r) => setTimeout(r, 1400));
      const kw = {
        happy: [
          "happy",
          "joy",
          "great",
          "amazing",
          "excited",
          "laugh",
          "smile",
          "wonderful",
          "good",
          "love",
          "fantastic",
        ],
        sad: [
          "sad",
          "cry",
          "miss",
          "alone",
          "hurt",
          "broken",
          "tired",
          "lost",
          "grief",
          "down",
          "empty",
        ],
        angry: [
          "angry",
          "mad",
          "furious",
          "hate",
          "rage",
          "annoyed",
          "frustrated",
          "pissed",
          "upset",
        ],
        anxious: [
          "nervous",
          "anxious",
          "worried",
          "scared",
          "fear",
          "stress",
          "panic",
          "overwhelm",
          "tense",
        ],
        calm: [
          "calm",
          "peaceful",
          "quiet",
          "relax",
          "serene",
          "gentle",
          "still",
          "tranquil",
          "breathe",
          "chill",
        ],
        romantic: [
          "love",
          "romance",
          "heart",
          "beautiful",
          "crush",
          "sweet",
          "tender",
          "darling",
          "affection",
        ],
      };
      const lower = text.toLowerCase();
      let detected = null;
      for (const [m, words] of Object.entries(kw)) {
        if (words.some((w) => lower.includes(w))) {
          detected = m;
          break;
        }
      }
      if (!detected) {
        const all = Object.keys(MOOD_CONFIG);
        detected = all[Math.floor(Math.random() * all.length)];
      }
      setMood(detected);
      setSongs(DEMO_SONGS[detected]);
      setLoading(false);
      setTimeout(() => setCharVisible(true), 200);
      return;
    }
    try {
      const res = await fetch("http://localhost:8000/recommendations", {
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
      setError("Could not reach backend. Enable demo mode above!");
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
          position: "absolute",
          top: "40%",
          right: "3%",
          fontSize: 32,
          opacity: 0.1,
          userSelect: "none",
          animation: "floatY 3s ease-in-out infinite",
        }}
      >
        ✨
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
          <div
            className="section-label"
            style={{ color: "#b06dff", marginBottom: 12 }}
          >
            Mood detector
          </div>
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
            Tell us how you're feeling — we'll find the soundtrack.
          </p>
        </div>

        {/* Demo toggle */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            gap: 10,
            marginBottom: 24,
          }}
        >
          <span
            style={{
              fontSize: 13,
              color: "#5a5270",
              fontWeight: 700,
              fontFamily: "'Space Mono',monospace",
            }}
          >
            demo mode
          </span>
          <div
            onClick={() => setDemoMode(!demoMode)}
            style={{
              width: 48,
              height: 26,
              borderRadius: 13,
              background: demoMode ? "#b06dff" : "#ccc",
              cursor: "pointer",
              position: "relative",
              border: "2px solid #1a1630",
              transition: "background 0.25s",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 3,
                left: demoMode ? 24 : 3,
                width: 16,
                height: 16,
                borderRadius: "50%",
                background: "#fff",
                transition: "left 0.25s",
              }}
            />
          </div>
        </div>

        {/* Orb indicator */}
        {(mood || loading) && (
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
            <div
              style={{
                width: 120,
                height: 120,
                borderRadius: "50%",
                border: `4px solid ${cfg?.color || "#b06dff"}`,
                background: `${cfg?.color || "#b06dff"}18`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 52,
                animation: loading
                  ? "orbPulse 1s ease-in-out infinite"
                  : "floatY 3s ease-in-out infinite",
                "--orb-color": cfg?.color || "#b06dff",
                boxShadow: `0 0 30px ${cfg?.color || "#b06dff"}44`,
              }}
            >
              {loading ? (
                <LoadingDots color={cfg?.color || "#b06dff"} />
              ) : (
                cfg?.emoji
              )}
            </div>
            {cfg && !loading && (
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
            )}
            {/* Character popup */}
            {charVisible && cfg && (
              <div
                style={{
                  fontSize: 72,
                  animation: "slideInChar 0.6s cubic-bezier(0.34,1.56,0.64,1)",
                  filter: `drop-shadow(0 4px 16px ${cfg.color}88)`,
                }}
              >
                {cfg.char}
              </div>
            )}
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
              {demoMode ? "✦ demo mode active" : "⌘↵ to submit"}
            </span>
            <button
              onClick={detect}
              disabled={loading || !text.trim()}
              className="retro-btn"
              style={{
                background: cfg ? cfg.color : "#FFD93D",
                color: "#1a1630",
                fontSize: 15,
                padding: "11px 28px",
                boxShadow: `3px 3px 0 #1a1630`,
                opacity: !text.trim() ? 0.5 : 1,
                transition: "all 0.2s",
              }}
            >
              {loading ? "Reading you..." : "Detect my mood 🎯"}
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
                <SongRow key={i} song={song} index={i} color={cfg.color} onClick={() => window.open(`https://www.youtube.com/results?search_query=${encodeURIComponent(song.name + ' ' + song.artist)}`, '_blank')}/>
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
