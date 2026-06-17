import { useScroll } from '../ScrollContext'
import { clamp, remap, lerp, SCENES } from '../useScrollProgress'
import joy      from '../assets/images/joy.gif'
import sadness  from '../assets/images/sadness.gif'
import anger    from '../assets/images/anger.gif'
import romance  from '../assets/images/romantic.gif'
import anxietyG from '../assets/images/anxiety.gif'
import calm     from '../assets/images/calm.gif'

const MOODS = [
  { key:'happy',    img:joy,      color:'#FFD93D', label:'Joy',     tagline:'Bright & bubbly energy',   tag:'happy'   },
  { key:'sad',      img:sadness,  color:'#5ba8ff', label:'Sadness', tagline:'Soulful & introspective',  tag:'sad'     },
  { key:'angry',    img:anger,    color:'#ff4d6d', label:'Anger',   tagline:'Raw & electric fire',      tag:'metal'   },
  { key:'calm',     img:calm,     color:'#8fcca5', label:'Calm',    tagline:'Soft & flowing stillness', tag:'chill'   },
  { key:'romantic', img:romance,  color:'#fd7ebe', label:'Romance', tagline:'Tender & heart-open',      tag:'romance' },
  { key:'anxious',  img:anxietyG, color:'#fd8662', label:'Anxiety', tagline:'Tense & searching',        tag:'anxiety' },
]

export default function MoodsSection() {
  const { progress } = useScroll()

  // scene 1: enters during 0.5–1.5 / SCENES, exits during 1–2 / SCENES
  const pIn  = remap(progress, 0.5/SCENES, 1.5/SCENES)
  const pOut = remap(progress, 1/SCENES,   2/SCENES)

  const sceneOp = clamp(pIn) * clamp(1 - pOut * 1.5)
  const sceneY  = lerp(8, 0, clamp(pIn))

  return (
    <div style={{
      position:'absolute', inset:0,
      background:'#0d0b14',
      opacity: sceneOp,
      transform:`translateY(${sceneY}vh)`,
      willChange:'transform,opacity',
      overflow:'hidden',
    }}>
      {/* glow */}
      <div style={{
        position:'absolute', top:'50%', left:'50%', transform:'translate(-50%,-50%)',
        width:700, height:700, borderRadius:'50%',
        background:'radial-gradient(circle,#b06dff0d 0%,transparent 70%)',
        pointerEvents:'none',
      }}/>

      <div style={{ position:'absolute', inset:0, overflowY:'auto', padding:'80px 40px 40px' }}>
        <div style={{ maxWidth:1100, margin:'0 auto' }}>
          {/* heading */}
          <div style={{ textAlign:'center', marginBottom:48 }}>
            <h2 className="groovy-title" style={{
              fontSize:'clamp(36px,5vw,62px)', color:'#fff',
              textShadow:'4px 4px 0 #FF6FB744',
            }}>
              Every feeling deserves<br/>
              <span style={{ color:'#FFD93D' }}>its own soundtrack ✦</span>
            </h2>
          </div>

          {/* grid */}
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))', gap:18 }}>
            {MOODS.map((m, i) => {
              const isLeft  = i % 2 === 0
              const exitX   = pOut * (isLeft ? -60 : 60)
              const cardOp  = clamp(pIn * 3 - i * 0.2) * clamp(1 - pOut * 2)
              const cardY   = lerp(30, 0, clamp(pIn - i * 0.03))
              const active  = clamp(pIn) > 0.5 && clamp(pOut) < 0.4

              return (
                <div key={m.key}
                  style={{
                    padding:'24px 20px',
                    border:`3px solid ${m.color}44`,
                    borderRadius:20,
                    background:`${m.color}15`,
                    display:'flex', flexDirection:'column', gap:12,
                    transform:`translateX(${exitX}vw) translateY(${cardY}px)`,
                    opacity: cardOp,
                    transition:'border-color 0.2s, box-shadow 0.2s',
                    willChange:'transform,opacity',
                    pointerEvents: active ? 'all' : 'none',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor=m.color; e.currentTarget.style.boxShadow=`0 10px 32px ${m.color}33` }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor=m.color+'44'; e.currentTarget.style.boxShadow='none' }}
                >
                  <img src={m.img} alt={m.label} style={{
                    width:'100%', height:160, borderRadius:14, objectFit:'cover',
                    filter:`drop-shadow(0 3px 10px ${m.color}88)`,
                  }}/>
                  <div>
                    <h3 style={{ fontFamily:"'Fredoka One',cursive", fontSize:26, color:'#fff', marginBottom:3, textShadow:`2px 2px 0 ${m.color}55` }}>
                      {m.label}
                    </h3>
                    <p style={{ fontSize:12, color:'#ffffffaa', fontWeight:600 }}>{m.tagline}</p>
                  </div>
                  <div style={{
                    display:'inline-flex', alignItems:'center', gap:5,
                    padding:'4px 12px', borderRadius:40,
                    background:m.color+'22', border:`1px solid ${m.color}55`, width:'fit-content',
                  }}>
                    <span style={{ fontFamily:"'Space Mono',monospace", fontSize:10, color:m.color, letterSpacing:1 }}>#{m.tag}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}