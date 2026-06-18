import { useScroll } from '../ScrollContext'
import { clamp, remap, lerp, SCENES } from '../useScrollProgress'
import joy      from '../assets/images/joy.gif'
import sadness  from '../assets/images/sadness.gif'
import anger    from '../assets/images/anger.gif'
import romance  from '../assets/images/romantic.gif'
import anxietyG from '../assets/images/anxiety.gif'
import calm     from '../assets/images/calm.gif'
import { memo, useRef, useState, useEffect } from 'react'

const MOODS = [
  { key:'happy',    img:joy,      color:'#FFD93D', label:'Joy',     tagline:'Bright & bubbly energy',   tag:'happy'   },
  { key:'sad',      img:sadness,  color:'#5ba8ff', label:'Sadness', tagline:'Soulful & introspective',  tag:'sad'     },
  { key:'angry',    img:anger,    color:'#ff4d6d', label:'Anger',   tagline:'Raw & electric fire',      tag:'metal'   },
  { key:'calm',     img:calm,     color:'#8fcca5', label:'Calm',    tagline:'Soft & flowing stillness', tag:'chill'   },
  { key:'romantic', img:romance,  color:'#fd7ebe', label:'Romance', tagline:'Tender & heart-open',      tag:'romance' },
  { key:'anxious',  img:anxietyG, color:'#fd8662', label:'Anxiety', tagline:'Tense & searching',        tag:'anxiety' },
]

// card footprint in px — width + the flex gap between cards (kept in one place
// so the wrap math and the rendered card width can never drift out of sync)
const CARD_W   = 320
const CARD_GAP = 20
const STEP     = CARD_W + CARD_GAP
const TRACK_W  = STEP * MOODS.length // width of one full loop of the original set

// duplicate the set once so there's always a second lap rendered ahead of the
// first — when the offset wraps past TRACK_W we snap it back to 0 with modulo,
// and because lap two is visually identical to lap one, the snap is invisible
const LOOP_MOODS = [...MOODS, ...MOODS]

// px/second the track auto-advances. This is now the ONLY thing tuning speed —
// no more relationship to scroll distance, card width, or SCENES.
const PLAY_SPEED = 40

function MoodsSection() {
  const { progress } = useScroll()

  const pIn  = remap(progress, 0.5/SCENES, 1.0/SCENES)
  const pOut = remap(progress, 1.5/SCENES, 2.0/SCENES)

  const sceneOp = clamp(pIn) * clamp(1 - pOut * 1.5)
  const sceneY  = lerp(8, 0, clamp(pIn))
  const active  = clamp(pIn) > 0.3 && clamp(pOut) < 0.6

  // ── time-based auto-play, fully decoupled from scroll progress ──
  // offsetRef holds the live px value (avoids a re-render every frame);
  // trackX is what we actually render, synced via rAF at display rate.
  const offsetRef    = useRef(0)
  const [trackX, setTrackX] = useState(0)
  const [paused, setPaused] = useState(false)
  const lastTsRef     = useRef(null)
  const rafRef         = useRef(null)

  useEffect(() => {
    function tick(ts) {
      if (lastTsRef.current == null) lastTsRef.current = ts
      const dt = (ts - lastTsRef.current) / 1000
      lastTsRef.current = ts

      // only advance while the scene is actually on screen AND not paused —
      // saves cycles when scrolled away, and respects hover/touch pause
      if (active && !paused) {
        offsetRef.current += PLAY_SPEED * dt
        const wrapped = ((offsetRef.current % TRACK_W) + TRACK_W) % TRACK_W
        setTrackX(-wrapped)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      lastTsRef.current = null
    }
  }, [active, paused])

  return (
    <div style={{
      position:'absolute', inset:0,
      background:'#0d0b14',
      opacity: sceneOp,
      transform:`translateY(${sceneY}vh)`,
      willChange:'transform,opacity',
      // overflow:clip on inner wrapper handles containment — outer div needs no overflow set
    }}>
      {/* glow */}
      <div style={{
        position:'absolute', top:'50%', left:'50%', transform:'translate(-50%,-50%)',
        width:700, height:700, borderRadius:'50%',
        background:'radial-gradient(circle,#b06dff0d 0%,transparent 70%)',
        pointerEvents:'none',
      }}/>

      <div style={{
        position:'absolute', inset:0,
        // overflow:clip = hard clip with no scrollbar whatsoever (unlike overflow:hidden
        // which reserves scrollbar space and causes layout shifts / flicker on animated children)
        overflow:'clip',
        padding:'80px 0 60px',
      }}>
        <div style={{ maxWidth:1100, margin:'0 auto', padding:'0 40px' }}>
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
        </div>

        {/* carousel viewport — full-bleed so cards can clip cleanly at the
            screen edge instead of the maxWidth container edge.
            Pausing on hover/touch lives here so it covers the whole strip,
            not just whichever card the pointer happens to be over. */}
        <div
          onMouseEnter={() => setPaused(true)}
          onMouseLeave={() => setPaused(false)}
          onTouchStart={() => setPaused(true)}
          onTouchEnd={() => setPaused(false)}
          style={{
            position:'relative',
            overflow:'clip',
            // soft fade at both edges so cards don't hard-cut against the bg
            maskImage:'linear-gradient(90deg, transparent 0%, #000 8%, #000 92%, transparent 100%)',
            WebkitMaskImage:'linear-gradient(90deg, transparent 0%, #000 8%, #000 92%, transparent 100%)',
            pointerEvents: active ? 'all' : 'none',
          }}
        >
          <div style={{
            display:'flex',
            gap:CARD_GAP,
            width:'max-content',
            transform:`translate3d(${trackX}px,0,0)`,
            willChange:'transform',
          }}>
            {LOOP_MOODS.map((m, i) => {
              // gentle stagger fade only on first entrance (pIn), so cards
              // already mid-loop don't keep re-fading as they cycle around
              const cardOp = clamp(pIn * 3 - (i % MOODS.length) * 0.15) * clamp(1 - pOut * 2)

              return (
                <div key={`${m.key}-${i}`}
                  style={{
                    flex:`0 0 ${CARD_W}px`,
                    padding:'24px 20px',
                    border:`3px solid ${m.color}44`,
                    borderRadius:20,
                    background:`${m.color}15`,
                    display:'flex', flexDirection:'column', gap:12,
                    opacity: cardOp,
                    transition:'border-color 0.2s, box-shadow 0.2s',
                    willChange:'opacity',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor=m.color; e.currentTarget.style.boxShadow=`0 10px 32px ${m.color}33` }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor=m.color+'44'; e.currentTarget.style.boxShadow='none' }}
                >
                  <img src={m.img} alt={m.label} style={{
                    width:'100%', height:220, borderRadius:14, objectFit:'cover',
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

export default memo(MoodsSection)