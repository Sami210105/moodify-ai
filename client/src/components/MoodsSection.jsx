const MOODS = [
  { key:'happy',    emoji:'😊', color:'#FFD93D', bg:'#FFD93D22', border:'#FFD93D', label:'Happy',    tagline:'Bright & bubbly energy',  tag:'happy' },
  { key:'sad',      emoji:'😢', color:'#5ba8ff', bg:'#5ba8ff22', border:'#5ba8ff', label:'Sad',      tagline:'Soulful & introspective',  tag:'sad' },
  { key:'angry',    emoji:'😤', color:'#ff4d6d', bg:'#ff4d6d22', border:'#ff4d6d', label:'Angry',    tagline:'Raw & electric fire',      tag:'metal' },
  { key:'calm',     emoji:'😌', color:'#3de8c8', bg:'#3de8c822', border:'#3de8c8', label:'Calm',     tagline:'Soft & flowing stillness', tag:'chill' },
  { key:'romantic', emoji:'💖', color:'#FF6FB7', bg:'#FF6FB722', border:'#FF6FB7', label:'Romantic', tagline:'Tender & heart-open',      tag:'romance' },
  { key:'anxious',  emoji:'⚡', color:'#b06dff', bg:'#b06dff22', border:'#b06dff', label:'Anxious',  tagline:'Tense & searching',        tag:'anxiety' },
]

function MoodCard({ mood, index }) {
  return (
    <div style={{
      padding:'32px 24px',
      border:`3px solid ${mood.border}44`,
      borderRadius:24,
      background:mood.bg,
      display:'flex', flexDirection:'column', gap:14,
      cursor:'default',
      transition:'transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease',
      animation:`fadeUp 0.6s ease ${index*0.08}s both`,
    }}
    onMouseEnter={e=>{
      e.currentTarget.style.transform='translateY(-6px) rotate(-1deg)'
      e.currentTarget.style.borderColor=mood.border
      e.currentTarget.style.boxShadow=`0 12px 40px ${mood.color}33`
    }}
    onMouseLeave={e=>{
      e.currentTarget.style.transform='none'
      e.currentTarget.style.borderColor=mood.border+'44'
      e.currentTarget.style.boxShadow='none'
    }}
    >
      <div style={{
        fontSize:52,
        filter:`drop-shadow(0 3px 10px ${mood.color}88)`,
        animation:'floatY 3s ease-in-out infinite',
        animationDelay:`${index*0.4}s`,
        lineHeight:1,
      }}>{mood.emoji}</div>

      <div>
        <h3 style={{
          fontFamily:"'Fredoka One',cursive",
          fontSize:28, color:'#fff',
          marginBottom:4,
          textShadow:`2px 2px 0 ${mood.color}55`,
        }}>{mood.label}</h3>
        <p style={{ fontSize:13, color:'#ffffffaa', fontWeight:600 }}>{mood.tagline}</p>
      </div>

      <div style={{
        display:'inline-flex', alignItems:'center', gap:6,
        padding:'5px 14px', borderRadius:40,
        background:mood.color+'22', border:`1px solid ${mood.color}55`,
        width:'fit-content',
      }}>
        <span style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:mood.color, letterSpacing:1 }}>
          #{mood.tag}
        </span>
      </div>
    </div>
  )
}

export default function MoodsSection() {
  return (
    <section id="moods" style={{
      background:'#0d0b14', padding:'120px 40px',
      position:'relative', overflow:'hidden',
    }}>
      {/* Background decoration */}
      <div style={{
        position:'absolute', top:'50%', left:'50%',
        transform:'translate(-50%,-50%)',
        width:600, height:600, borderRadius:'50%',
        background:'radial-gradient(circle, #b06dff0a 0%, transparent 70%)',
        pointerEvents:'none',
      }}/>

      <div style={{ maxWidth:1100, margin:'0 auto' }}>
        <div style={{ textAlign:'center', marginBottom:72 }}>
          <div className="section-label" style={{ color:'#FF6FB7', marginBottom:12 }}>Six moods. Infinite songs.</div>
          <h2 className="groovy-title" style={{
            fontSize:'clamp(40px,6vw,68px)', color:'#fff',
            textShadow:'4px 4px 0 #FF6FB744',
          }}>
            Every feeling deserves<br/>
            <span style={{ color:'#FFD93D' }}>its own soundtrack ✦</span>
          </h2>
        </div>

        <div style={{
          display:'grid',
          gridTemplateColumns:'repeat(auto-fit,minmax(240px,1fr))',
          gap:20,
        }}>
          {MOODS.map((m,i) => <MoodCard key={m.key} mood={m} index={i}/>)}
        </div>
      </div>
    </section>
  )
}
