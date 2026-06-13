function Step({ number, icon, title, desc, color, delay }) {
  return (
    <div style={{
      display:'flex', flexDirection:'column', alignItems:'center', gap:16,
      animation:`fadeUp 0.7s ease ${delay}s both`,
    }}>
      <div style={{
        width:100, height:100, borderRadius:'50%',
        background:color,
        border:'4px solid #1a1630',
        boxShadow:`6px 6px 0 #1a1630`,
        display:'flex', alignItems:'center', justifyContent:'center',
        fontSize:44,
        animation:'floatY 3s ease-in-out infinite',
        animationDelay:`${delay}s`,
      }}>
        {icon}
      </div>
      <div style={{
        width:36, height:36, borderRadius:'50%',
        background:'#1a1630', color:'#fff',
        fontFamily:"'Fredoka One',cursive", fontSize:18,
        display:'flex', alignItems:'center', justifyContent:'center',
        border:`3px solid ${color}`,
        marginTop:-10,
      }}>{number}</div>
      <h3 style={{
        fontFamily:"'Fredoka One',cursive", fontSize:22,
        color:'#1a1630', textAlign:'center',
      }}>{title}</h3>
      <p style={{ fontSize:15, color:'#5a5270', textAlign:'center', lineHeight:1.6, maxWidth:200 }}>{desc}</p>
    </div>
  )
}

function PipeConnector({ color = '#1a1630' }) {
  return (
    <div style={{ display:'flex', alignItems:'center', paddingBottom:60, flexShrink:0 }}>
      <svg width="80" height="20" viewBox="0 0 80 20">
        <path d="M0 10 Q20 0 40 10 Q60 20 80 10" stroke={color} strokeWidth="4" fill="none" strokeLinecap="round"/>
        <circle cx="20" cy="7" r="4" fill={color}/>
        <circle cx="60" cy="13" r="4" fill={color}/>
      </svg>
    </div>
  )
}

export default function HowItWorks() {
  return (
    <section id="how-it-works" style={{
      background:'#f5f0e8',
      padding:'100px 40px 120px',
      position:'relative', overflow:'hidden',
    }}>
      {/* Decorative gears */}
      <div style={{
        position:'absolute', top:40, right:60,
        fontSize:80, opacity:0.08,
        animation:'spinSlow 12s linear infinite',
        color:'#1a1630',
        userSelect:'none',
      }}>⚙️</div>
      <div style={{
        position:'absolute', bottom:60, left:40,
        fontSize:60, opacity:0.08,
        animation:'spinSlow 8s linear infinite reverse',
        color:'#1a1630',
        userSelect:'none',
      }}>⚙️</div>

      <div style={{ maxWidth:1100, margin:'0 auto' }}>
        <div style={{ textAlign:'center', marginBottom:72 }}>
          <div className="section-label" style={{ color:'#b06dff', marginBottom:12 }}>The process</div>
          <h2 className="groovy-title" style={{
            fontSize:'clamp(40px,6vw,68px)',
            color:'#1a1630',
            textShadow:'4px 4px 0 #b06dff44',
          }}>
            We are the music makers<br/>
            <span style={{ color:'#b06dff' }}>& dreamers of dreams</span>
          </h2>
          <p style={{ fontSize:17, color:'#5a5270', marginTop:20, fontWeight:600 }}>
            Three steps from feeling to playlist.
          </p>
        </div>

        <div style={{
          display:'flex', alignItems:'center', justifyContent:'center',
          gap:0, flexWrap:'wrap', rowGap:48,
        }}>
          <Step
            number="1" icon="💬" color="#FFD93D"
            title="You talk to us"
            desc="Type exactly how you feel — rambling, poetic, raw. No filter needed."
            delay={0}
          />
          <PipeConnector color="#FFD93D"/>

          <Step
            number="2" icon="🧠" color="#b06dff"
            title="AI reads the vibe"
            desc="GoEmotions-trained model decodes your emotional signature in milliseconds."
            delay={0.15}
          />
          <PipeConnector color="#b06dff"/>

          <Step
            number="3" icon="🎵" color="#FF6FB7"
            title="Songs start flowing"
            desc="Last.fm delivers curated real tracks matched perfectly to your mood."
            delay={0.3}
          />
        </div>

        {/* Illustrated machine row */}
        <div style={{
          marginTop:80, display:'flex', justifyContent:'center', gap:24, flexWrap:'wrap',
        }}>
          {[
            { icon:'🎭', label:'Emotion Engine', color:'#FFD93D' },
            { icon:'⚗️', label:'Mood Lab', color:'#3de8c8' },
            { icon:'📻', label:'Last.fm Radio', color:'#FF6FB7' },
            { icon:'🎁', label:'Your Playlist', color:'#b06dff' },
          ].map((item, i) => (
            <div key={item.label} style={{
              display:'flex', flexDirection:'column', alignItems:'center', gap:8,
              padding:'20px 24px',
              background:'#fff',
              border:'3px solid #1a1630',
              borderRadius:16,
              boxShadow:'5px 5px 0 #1a1630',
              minWidth:140,
              animation:`fadeUp 0.6s ease ${i*0.1}s both`,
            }}>
              <div style={{ fontSize:36, filter:`drop-shadow(2px 2px 0 ${item.color})` }}>{item.icon}</div>
              <div style={{
                fontFamily:"'Space Mono',monospace",
                fontSize:11, color:'#5a5270', textAlign:'center',
                letterSpacing:1, textTransform:'uppercase',
              }}>{item.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom wave back to dark */}
      <svg viewBox="0 0 1440 80" fill="none" xmlns="http://www.w3.org/2000/svg"
        style={{ position:'absolute', bottom:-2, left:0, width:'100%', pointerEvents:'none' }}>
        <path d="M0 80 L0 40 Q180 80 360 40 Q540 0 720 40 Q900 80 1080 40 Q1260 0 1440 40 L1440 80 Z"
          fill="#0d0b14"/>
      </svg>
    </section>
  )
}
