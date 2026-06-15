function TechPill({ icon, label, color }) {
  return (
    <div style={{
      display:'flex', alignItems:'center', gap:8,
      padding:'10px 20px', borderRadius:40,
      border:`2px solid ${color}44`,
      background:`${color}11`,
      fontSize:14, fontWeight:700,
      color,
      transition:'transform 0.2s, border-color 0.2s',
    }}
    onMouseEnter={e=>{ e.currentTarget.style.transform='scale(1.05)'; e.currentTarget.style.borderColor=color+'99' }}
    onMouseLeave={e=>{ e.currentTarget.style.transform='none'; e.currentTarget.style.borderColor=color+'44' }}
    >
      <span style={{ fontSize:18 }}>{icon}</span>
      <span>{label}</span>
    </div>
  )
}

export default function AboutFooter() {
  return (
    <>
      <section id="about" style={{
        background:'#0d0b14', padding:'120px 40px',
        position:'relative', overflow:'hidden',
      }}>
        <div style={{
          position:'absolute', top:0, left:0, right:0, bottom:0,
          background:'radial-gradient(ellipse at top, #0d0b14 0%, transparent 60%)',
          pointerEvents:'none',
        }}/>

        <div style={{ maxWidth:1100, margin:'0 auto', position:'relative', zIndex:1 }}>
            <div>
              <div className="section-label" style={{ color:'#3de8c8', marginBottom:16 }}>About Moodify AI</div>
              <h2 className="groovy-title" style={{
                fontSize:'clamp(36px,5vw,58px)', color:'#fff',
                textShadow:'4px 4px 0 #3de8c844',
                marginBottom:24,
              }}>
                Music that<br/>
                <span style={{ color:'#3de8c8' }}>truly understands</span><br/>
                you ✦
              </h2>
              <p style={{
                fontSize:17, color:'#ffffffcc', lineHeight:1.8, fontWeight:600, marginBottom:24,
              }}>
                Moodify AI uses a GoEmotions-trained emotion classifier combined
                with sentence transformers to decode exactly how you're feeling — then
                connects to Last.fm's massive music library to find songs that match.
              </p>
              <p style={{ fontSize:15, color:'#ffffff88', lineHeight:1.7, fontWeight:600 }}>
                Built with love by a student who believed music should meet you where you are,
                not where an algorithm thinks you should be.
              </p>
            </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{
        background:'#08060f',
        borderTop:'3px solid #1a1630',
        padding:'60px 40px 40px',
      }}>
        <div style={{ maxWidth:1100, margin:'0 auto' }}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', flexWrap:'wrap', gap:40, marginBottom:60 }}>
            <div>
              <div style={{ fontFamily:"'Fredoka One',cursive", fontSize:32, color:'#fff', marginBottom:12 }}>
                Moodify<span style={{ color:'#FFD93D' }}>AI</span>
              </div>
              <p style={{ fontSize:15, color:'#ffffff66', fontWeight:600, maxWidth:280, lineHeight:1.7 }}>
                Feel it. Find it. Play it.<br/>
              </p>
            </div>
            <div style={{ display:'flex', gap:60, flexWrap:'wrap' }}>
              {[
                { head:'Navigate', links:['Meet your moods','Mood detector','About','GitHub'] },
                { head:'Moods', links:['Happy','Sad','Angry','Calm','Romantic','Anxious'] },
              ].map(col=>(
                <div key={col.head}>
                  <div style={{
                    fontFamily:"'Space Mono',monospace", fontSize:11, letterSpacing:2,
                    textTransform:'uppercase', color:'#ffffff44', marginBottom:16,
                  }}>{col.head}</div>
                  <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
                    {col.links.map(l=>(
                      <a key={l} href="#" style={{
                        fontSize:15, color:'#ffffff88', textDecoration:'none', fontWeight:600,
                        transition:'color 0.2s',
                      }}
                      onMouseEnter={e=>e.target.style.color='#FFD93D'}
                      onMouseLeave={e=>e.target.style.color='#ffffff88'}
                      >{l}</a>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{
            borderTop:'1px solid #ffffff11', paddingTop:28,
            display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:16,
          }}>
            <span style={{ fontFamily:"'Space Mono',monospace", fontSize:12, color:'#ffffff33' }}>
              © 2025 Moodify AI — built with 🎵 
            </span>
          </div>
        </div>
      </footer>
    </>
  )
}
