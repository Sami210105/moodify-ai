import joy from '../assets/images/joy.gif'
import sadness from '../assets/images/sadness.gif'
import anger from '../assets/images/anger.gif'
import fear from '../assets/images/fear.gif'
import romance from '../assets/images/romantic.gif'
import anxiety from '../assets/images/anxiety.gif'
import calm from '../assets/images/calm.gif'

const MOODS = [
  { key:'happy',    img:joy, color:'#FFD93D', bg:'#FFD93D22', border:'#FFD93D', label:'Joy',    tagline:'Bright & bubbly energy',  tag:'happy' },
  { key:'sad',      img:sadness, color:'#5ba8ff', bg:'#5ba8ff22', border:'#5ba8ff', label:'Sadness',      tagline:'Soulful & introspective',  tag:'sad' },
  { key:'angry',    img:anger, color:'#ff4d6d', bg:'#ff4d6d22', border:'#ff4d6d', label:'Anger',    tagline:'Raw & electric fire',      tag:'metal' },
  { key:'calm',     img:calm, color:'#8fcca5', bg:'#8fcca522', border:'#8fcca5', label:'Calm',     tagline:'Soft & flowing stillness', tag:'chill' },
  { key:'romantic', img:romance, color:'#fd7ebe', bg:'#ff6fb72f', border:'#fd7ebe', label:'Romance', tagline:'Tender & heart-open',      tag:'romance' },
  { key:'anxious',  img:anxiety, color:'#fd8662', bg:'#fd866222', border:'#fd8662', label:'Anxiety',  tagline:'Tense & searching',        tag:'anxiety' },
]

function MoodImagePlaceholder({ mood, index }) {
  return (
    <div style={{
      width: '100%',
      height: 180,
      borderRadius: 16,
      background: `linear-gradient(135deg, ${mood.color}22, ${mood.color}44)`,
      border: `2px dashed ${mood.color}88`,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 4,
      animation: 'floatY 3s ease-in-out infinite',
      animationDelay: `${index * 0.4}s`,
      flexShrink: 0,
    }}>
      {/* Image icon SVG */}
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="3" width="18" height="18" rx="3" stroke={mood.color} strokeWidth="1.5" strokeDasharray="3 2"/>
        <circle cx="8.5" cy="8.5" r="1.5" fill={mood.color} opacity="0.7"/>
        <path d="M3 15l5-5 4 4 3-3 6 6" stroke={mood.color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.7"/>
      </svg>
      <span style={{
        fontFamily: "'Space Mono', monospace",
        fontSize: 8,
        color: mood.color,
        opacity: 0.7,
        letterSpacing: 0.5,
        textTransform: 'uppercase',
      }}>img</span>
    </div>
  )
}

function MoodCard({ mood, index }) {
  return (
    <div style={{
      padding: '32px 24px',
      border: `3px solid ${mood.border}44`,
      borderRadius: 24,
      background: mood.bg,
      display: 'flex', flexDirection: 'column', gap: 14,
      cursor: 'default',
      transition: 'transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease',
      animation: `fadeUp 0.6s ease ${index * 0.08}s both`,
    }}
    onMouseEnter={e => {
      e.currentTarget.style.transform = 'translateY(-6px) rotate(-1deg)'
      e.currentTarget.style.borderColor = mood.border
      e.currentTarget.style.boxShadow = `0 12px 40px ${mood.color}33`
    }}
    onMouseLeave={e => {
      e.currentTarget.style.transform = 'none'
      e.currentTarget.style.borderColor = mood.border + '44'
      e.currentTarget.style.boxShadow = 'none'
    }}
    >
      {/* Image placeholder — swap src in once you have assets */}
      {mood.img
        ? <img
            src={mood.img}
            alt={mood.label}
            style={{
              width: '100%', height: 180,
              borderRadius: 16,
              objectFit: 'cover',
              filter: `drop-shadow(0 3px 10px ${mood.color}88)`,
            }}
          />
        : <MoodImagePlaceholder mood={mood} index={index} />
      }

      <div>
        <h3 style={{
          fontFamily: "'Fredoka One', cursive",
          fontSize: 28, color: '#fff',
          marginBottom: 4,
          textShadow: `2px 2px 0 ${mood.color}55`,
        }}>{mood.label}</h3>
        <p style={{ fontSize: 13, color: '#ffffffaa', fontWeight: 600 }}>{mood.tagline}</p>
      </div>

      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '5px 14px', borderRadius: 40,
        background: mood.color + '22', border: `1px solid ${mood.color}55`,
        width: 'fit-content',
      }}>
        <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 11, color: mood.color, letterSpacing: 1 }}>
          #{mood.tag}
        </span>
      </div>
    </div>
  )
}

export default function MoodsSection() {
  return (
    <section id="moods" style={{
      background: '#0d0b14', padding: '120px 40px',
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: '50%', left: '50%',
        transform: 'translate(-50%,-50%)',
        width: 600, height: 600, borderRadius: '50%',
        background: 'radial-gradient(circle, #b06dff0a 0%, transparent 70%)',
        pointerEvents: 'none',
      }}/>

      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: 72 }}>
          <h2 className="groovy-title" style={{
            fontSize: 'clamp(40px,6vw,68px)', color: '#fff',
            textShadow: '4px 4px 0 #FF6FB744',
          }}>
            Every feeling deserves<br/>
            <span style={{ color: '#FFD93D' }}>its own soundtrack ✦</span>
          </h2>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))',
          gap: 20,
        }}>
          {MOODS.map((m, i) => <MoodCard key={m.key} mood={m} index={i}/>)}
        </div>
      </div>
    </section>
  )
}