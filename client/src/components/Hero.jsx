import { useEffect, useRef } from 'react'

const STARS = Array.from({length:80},(_,i)=>({
  id:i,
  left: `${Math.random()*100}%`,
  top: `${Math.random()*100}%`,
  size: Math.random()*3+1,
  dur: `${Math.random()*3+1.5}s`,
  del: `${Math.random()*3}s`,
}))

function RainbowRibbon() {
  return (
    <svg viewBox="0 0 1440 520" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%', pointerEvents:'none', opacity:0.95 }}>
      {[
        { color:'#FF4D6D', offset:0 },
        { color:'#FF8C42', offset:12 },
        { color:'#FFD93D', offset:24 },
        { color:'#7bc67e', offset:36 },
        { color:'#5ba8ff', offset:48 },
        { color:'#b06dff', offset:60 },
        { color:'#FF6FB7', offset:72 },
      ].map(({ color, offset }) => (
        <path key={color}
          d={`M${-60+offset} ${40+offset} C 200 ${-20+offset}, 500 ${300+offset}, 700 ${200+offset} S 1100 ${-10+offset}, 1500 ${180+offset}`}
          stroke={color} strokeWidth="18" strokeLinecap="round" fill="none" opacity="0.9"
        />
      ))}
    </svg>
  )
}

function FloatingNote({ children, style }) {
  return (
    <div style={{
      position:'absolute',
      fontSize: 28,
      animation: `floatY ${2+Math.random()*2}s ease-in-out infinite`,
      animationDelay: `${Math.random()*2}s`,
      userSelect:'none', pointerEvents:'none',
      filter:'drop-shadow(0 2px 8px #0008)',
      ...style
    }}>{children}</div>
  )
}

export default function Hero() {
  return (
    <section id="hero" style={{
      minHeight:'100vh', position:'relative', overflow:'hidden',
      background:'#0d0b14',
      display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
      paddingTop:80,
    }}>
      {STARS.map(s => (
        <div key={s.id} className="star" style={{
          left:s.left, top:s.top,
          width:s.size, height:s.size,
          '--dur':s.dur, '--del':s.del,
        }}/>
      ))}

      <RainbowRibbon />

      {/* Groovy sun/moon decoration top-right */}
      <div style={{
        position:'absolute', top:'8%', right:'14%',
        width:90, height:90, borderRadius:'50%',
        background:'#FFD93D',
        border:'4px solid #1a1630',
        display:'flex', alignItems:'center', justifyContent:'center',
        fontSize:40,
        animation:'floatYslow 3s ease-in-out infinite',
        boxShadow:'5px 5px 0 #1a1630',
      }}>🌕</div>

      {/* Main content */}
      <div style={{
        position:'relative', zIndex:5,
        textAlign:'center', maxWidth:800, padding:'0 32px',
        animation:'fadeUp 0.9s ease both',
      }}>
        <div className="section-label" style={{ color:'#FFD93D', marginBottom:20 }}>
          ♪ &nbsp; AI-powered music for your soul &nbsp; ♪
        </div>

        <h1 className="groovy-title" style={{
          fontSize:'clamp(64px,12vw,120px)',
          color:'#fff',
          textShadow:'6px 6px 0 #1a1630, 8px 8px 0 #b06dff66',
          marginBottom:4,
        }}>
          Feel the
        </h1>
        <h1 className="groovy-title" style={{
          fontSize:'clamp(64px,12vw,120px)',
          background:'linear-gradient(135deg,#FFD93D,#FF6FB7,#b06dff)',
          WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent',
          textShadow:'none',
          marginBottom:24,
          filter:'drop-shadow(4px 4px 0 #1a163044)',
        }}>
          Music ✦
        </h1>

        <p style={{
          fontSize:'clamp(16px,2.2vw,20px)', color:'#ffffffcc',
          lineHeight:1.7, maxWidth:560, margin:'0 auto 40px',
          fontWeight:600,
        }}>
          Tell us how you're feeling — in any messy, honest, beautiful way —
          and we'll match you with songs that <em>get it</em>.
        </p>

        <div style={{ display:'flex', gap:16, justifyContent:'center', flexWrap:'wrap' }}>
          <a href="#try-it" className="retro-btn" style={{
            background:'#FFD93D', color:'#0d0b14',
            fontSize:18, padding:'16px 38px',
          }}>
            Detect my mood
          </a>
          <a href="#how-it-works" className="retro-btn" style={{
            background:'transparent', color:'#fff', borderColor:'#ffffff55',
            boxShadow:'4px 4px 0 #ffffff22',
          }}>
            How it works ↓
          </a>
        </div>
      </div>

      {/* Characters bottom */}
      <div style={{
        position:'absolute', bottom:0, left:0, right:0,
        display:'flex', justifyContent:'space-between', alignItems:'flex-end',
        padding:'0 60px', pointerEvents:'none',
      }}>
        {/* Left character - happy yellow girl */}
        <div style={{
          width:160, height:200,
          background:'linear-gradient(180deg,transparent,#0d0b14)',
          borderRadius:'20px 20px 0 0',
          display:'flex', alignItems:'flex-end', justifyContent:'center',
          overflow:'hidden',
          animation:'floatY 3s ease-in-out infinite',
        }}>
          <div style={{ fontSize:100, lineHeight:1, filter:'drop-shadow(0 4px 16px #FFD93D66)' }}>😊</div>
        </div>

        {/* Scroll hint */}
        <div style={{
          display:'flex', flexDirection:'column', alignItems:'center', gap:6,
          color:'#ffffff44', fontSize:12, fontFamily:"'Space Mono',monospace",
          marginBottom:32, animation:'shimmer 2s infinite',
        }}>
          <span>scroll down</span>
          <div style={{ width:1, height:40, background:'linear-gradient(to bottom,#fff4,transparent)' }}/>
        </div>

        {/* Right character - fear purple guy placeholder */}
        <div style={{
          width:160, height:200,
          borderRadius:'20px 20px 0 0',
          display:'flex', alignItems:'flex-end', justifyContent:'center',
          overflow:'hidden',
          animation:'floatY 3.5s ease-in-out 0.5s infinite',
        }}>
          <div style={{ fontSize:100, lineHeight:1, filter:'drop-shadow(0 4px 16px #b06dff66)' }}>😰</div>
        </div>
      </div>

      {/* Wave transition to cream */}
      <svg viewBox="0 0 1440 80" fill="none" xmlns="http://www.w3.org/2000/svg"
        style={{ position:'absolute', bottom:-2, left:0, width:'100%', pointerEvents:'none' }}>
        <path d="M0 80 L0 40 Q180 0 360 40 Q540 80 720 40 Q900 0 1080 40 Q1260 80 1440 40 L1440 80 Z"
          fill="#f5f0e8"/>
      </svg>
    </section>
  )
}
