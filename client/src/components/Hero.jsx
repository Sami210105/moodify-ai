import { useScroll } from "../ScrollContext";
import { clamp, remap, SCENES } from "../usescrollprogress";
import { memo } from "react";

const STARS = Array.from({ length: 80 }, (_, i) => ({
  id: i,
  left: `${Math.random() * 100}%`,
  top: `${Math.random() * 100}%`,
  size: Math.random() * 3 + 1,
  dur: `${Math.random() * 3 + 1.5}s`,
  del: `${Math.random() * 3}s`,
}));

function RainbowRibbon() {
  return (
    <svg
      viewBox="0 0 1440 520"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        opacity: 0.65,
      }}
    >
      {[
        { color: "#FF4D6D", offset: 0 },
        { color: "#FF8C42", offset: 12 },
        { color: "#FFD93D", offset: 24 },
        { color: "#7bc67e", offset: 36 },
        { color: "#5ba8ff", offset: 48 },
        { color: "#b06dff", offset: 60 },
        { color: "#FF6FB7", offset: 64 },
      ].map(({ color, offset }) => (
        <path
          key={color}
          d={`M${-60 + offset} ${40 + offset} C 200 ${-20 + offset}, 500 ${300 + offset}, 700 ${200 + offset} S 1100 ${-10 + offset}, 1500 ${180 + offset}`}
          stroke={color}
          strokeWidth="18"
          strokeLinecap="round"
          fill="none"
          opacity="0.9"
        />
      ))}
    </svg>
  );
}

function Hero() {
  const { progress, scrollRef } = useScroll();

  // scene 0: 0 → 1/SCENES of total scroll
  const p = remap(progress, 0, 1 / SCENES);
  const starsOp = clamp(1 - p * 1.4);
  const contentOp = clamp(1 - p * 3);
  const hintOp = clamp(1 - p * 6);

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        background: "#0d0b14",
        opacity: clamp(1 - p * 2),
        transform: `translate3d(0,${-p * 40}vh,0)`,
        willChange: "transform,opacity",
        backfaceVisibility: "hidden",
      }}
    >
      {/* stars */}
      <div style={{ position: "absolute", inset: 0, opacity: starsOp }}>
        {STARS.map((s) => (
          <div
            key={s.id}
            className="star"
            style={{
              left: s.left,
              top: s.top,
              width: s.size,
              height: s.size,
              "--dur": s.dur,
              "--del": s.del,
            }}
          />
        ))}
      </div>

      <RainbowRibbon />

      {/* text content */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 5,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          padding: "80px 32px 0",
          transform: `translateY(${-p * 20}vh)`,
          opacity: contentOp,
          willChange: "transform,opacity",
        }}
      >
        <div
          className="section-label"
          style={{ color: "#FFD93D", marginBottom: 20 }}
        >
          ♪ &nbsp; Music for your soul &nbsp; ♪
        </div>
        <h1
          className="groovy-title"
          style={{
            fontSize: "clamp(64px,12vw,120px)",
            color: "#fff",
            textShadow: "6px 6px 0 #1a1630, 8px 8px 0 #b06dff66",
            marginBottom: 4,
          }}
        >
          Feel the
        </h1>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "16px",
            marginBottom: "24px",
          }}
        >
          <h1
            className="groovy-title"
            style={{
              fontSize: "clamp(64px,12vw,120px)",
              background: "linear-gradient(135deg,#FFD93D,#FF6FB7,#b06dff)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              margin: 5,
              filter: "drop-shadow(8px 8px 0 #1a163044)",
            }}
          >
            Music
          </h1>

          <img
            src="/icons/logo.png"
            alt="heart"
            style={{
              width: 120,
              height: 120,
              paddingTop: 10,
            }}
          />
        </div>
        <p
          style={{
            fontSize: "clamp(24px,2.2vw,20px)",
            color: "#ffffffcc",
            lineHeight: 1.7,
            maxWidth: 560,
            margin: "0 auto 40px",
            fontWeight: 600,
          }}
        >
          Tell us how you're feeling in any messy, honest, beautiful way and
          we'll match you with songs that <em>get it</em>.
        </p>
        <div
          style={{
            display: "flex",
            gap: 16,
            justifyContent: "center",
            flexWrap: "wrap",
            pointerEvents: "all",
          }}
        >
          <button
            onClick={() => {
              const max =
                scrollRef.current.scrollHeight - scrollRef.current.clientHeight;
              scrollRef.current.scrollTo({
                top: max * (2 / SCENES + 0.01),
                behavior: "smooth",
              });
            }}
            className="retro-btn"
            style={{
              background: "#FFD93D",
              color: "#0d0b14",
              fontSize: 18,
              padding: "16px 38px",
            }}
          >
            Recommend me songs
          </button>
          <button
            onClick={() => {
              const max =
                scrollRef.current.scrollHeight - scrollRef.current.clientHeight;
              scrollRef.current.scrollTo({
                top: max * (1 / SCENES + 0.01),
                behavior: "smooth",
              });
            }}
            className="retro-btn"
            style={{
              background: "transparent",
              color: "#fff",
              borderColor: "#ffffff55",
              boxShadow: "4px 4px 0 #ffffff22",
            }}
          >
            Meet your moods ↓
          </button>
        </div>
      </div>

      {/* scroll hint */}
      <div
        style={{
          position: "absolute",
          bottom: 32,
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 6,
          color: "#ffffff55",
          fontSize: 12,
          fontFamily: "'Space Mono',monospace",
          animation: "shimmer 2s infinite",
          opacity: hintOp,
          pointerEvents: "none",
        }}
      >
        <span>scroll to explore</span>
        <div
          style={{
            width: 1,
            height: 40,
            background: "linear-gradient(to bottom,#fff5,transparent)",
          }}
        />
      </div>
    </div>
  );
}

export default memo(Hero);
