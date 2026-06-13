export default function StatsSection() {
  const stats = [
    { num:'6', unit:'moods', icon:'🎭', color:'#FFD93D', desc:'precisely mapped emotion categories' },
    { num:'54K+', unit:'training samples', icon:'📚', color:'#b06dff', desc:'GoEmotions dataset entries' },
    { num:'10', unit:'songs per vibe', icon:'🎵', color:'#FF6FB7', desc:'curated from Last.fm every time' },
    { num:'<0.5s', unit:'detection speed', icon:'⚡', color:'#3de8c8', desc:'milliseconds from text to mood' },
  ]
  return (
    <section style={{
      background:'#f5f0e8', padding:'100px 40px',
      position:'relative', overflow:'hidden',
    }}>
      {/* Top wave */}
      <svg viewBox="0 0 1440 80" fill="none"
        style={{ position:'absolute', top:-2, left:0, width:'100%', pointerEvents:'none' }}>
        <path d="M0 0 L0 40 Q180 80 360 40 Q540 0 720 40 Q900 80 1080 40 Q1260 0 1440 40 L1440 0 Z"
          fill="#0d0b14"/>
      </svg>

      <div style={{ maxWidth:1100, margin:'0 auto' }}>
        <div style={{ textAlign:'center', marginBottom:64 }}>
          <div className="section-label" style={{ color:'#FF8C42', marginBottom:12 }}>By the numbers</div>
          <h2 className="groovy-title" style={{
            fontSize:'clamp(36px,5vw,58px)', color:'#1a1630',
            textShadow:'4px 4px 0 #FF8C4244',
          }}>
            Music science,<br/>
            <span style={{ color:'#FF8C42' }}>human soul</span>
          </h2>
        </div>

        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))', gap:24 }}>
          {stats.map((s,i)=>(
            <div key={s.num} style={{
              padding:'32px 24px',
              background:'#fff',
              border:'3px solid #1a1630',
              borderRadius:20,
              boxShadow:`6px 6px 0 #1a1630`,
              display:'flex', flexDirection:'column', gap:12,
              animation:`fadeUp 0.6s ease ${i*0.1}s both`,
              transition:'transform 0.2s',
            }}
            onMouseEnter={e=>{ e.currentTarget.style.transform='translateY(-4px)' }}
            onMouseLeave={e=>{ e.currentTarget.style.transform='none' }}
            >
              <div style={{ fontSize:40, filter:`drop-shadow(2px 2px 0 ${s.color})` }}>{s.icon}</div>
              <div>
                <div style={{
                  fontFamily:"'Fredoka One',cursive",
                  fontSize:44, color:s.color,
                  lineHeight:1,
                  textShadow:`3px 3px 0 ${s.color}33`,
                }}>{s.num}</div>
                <div style={{
                  fontFamily:"'Space Mono',monospace",
                  fontSize:11, color:'#1a1630', letterSpacing:1, textTransform:'uppercase', marginTop:4,
                }}>{s.unit}</div>
              </div>
              <p style={{ fontSize:13, color:'#5a5270', lineHeight:1.5, fontWeight:600 }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom wave */}
      <svg viewBox="0 0 1440 80" fill="none"
        style={{ position:'absolute', bottom:-2, left:0, width:'100%', pointerEvents:'none' }}>
        <path d="M0 80 L0 40 Q180 0 360 40 Q540 80 720 40 Q900 0 1080 40 Q1260 80 1440 40 L1440 80 Z"
          fill="#0d0b14"/>
      </svg>
    </section>
  )
}
