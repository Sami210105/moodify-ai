import { useScroll } from '../ScrollContext'
import { clamp, remap, lerp, SCENES } from '../usescrollprogress'
import { memo } from 'react'

// scene index of each link target, out of SCENES total — matches the order
// App.jsx mounts components in and what Hero.jsx / NavDots already scroll to
const SCENE_INDEX = { hero:0, moods:1, detector:2, about:3, footer:4 }

function AboutFooter() {
  const { progress, scrollRef } = useScroll()

  // FIX: tighter enter ramp, exit pushed later — gives full-opacity plateau
  // scene 3 (about): enters 2.5→3.0/SCENES, exits 3.5→4.0/SCENES
  const p3In  = remap(progress, 2.5/SCENES, 3.0/SCENES)
  const p3Out = remap(progress, 3.5/SCENES, 4.0/SCENES)
  const aboutOp = clamp(p3In * 1.5) * clamp(1 - p3Out * 1.5)
  const aboutY  = lerp(6, 0, clamp(p3In))

  // scene 4 (footer): enters 3.5–4.5/SCENES (unchanged — no exit needed)
  const p4In     = remap(progress, 3.5/SCENES, 4.5/SCENES)
  const footerOp = clamp(p4In * 1.5)
  const footerY  = lerp(5, 0, clamp(p4In))

  // same scroll-to-scene approach as Hero.jsx's buttons and App.jsx's NavDots —
  // kept identical so all three stay in sync if scene timing ever changes
  const goToScene = (sceneKey) => {
    const i   = SCENE_INDEX[sceneKey]
    const el  = scrollRef.current
    if (!el) return
    const max = el.scrollHeight - el.clientHeight
    el.scrollTo({ top: max * (i / SCENES + 0.01), behavior:'smooth' })
  }

  return (
    <>
      {/* ── ABOUT ── */}
      <div style={{
        position:'absolute', inset:0,
        background:'#0d0b14',
        opacity: aboutOp,
        transform:`translateY(${aboutY}vh)`,
        willChange:'transform,opacity',
        display:'flex', alignItems:'center',
        overflow:'hidden',
      }}>
        <div style={{
          position:'absolute', top:'30%', left:'50%', transform:'translate(-50%,-50%)',
          width:600, height:600, borderRadius:'50%',
          background:'radial-gradient(circle,#3de8c80a 0%,transparent 70%)',
          pointerEvents:'none',
        }}/>

        <div style={{ maxWidth:800, margin:'0 auto', padding:'80px 48px', position:'relative', zIndex:1 }}>
          <div className="section-label" style={{ color:'#3de8c8', marginBottom:16 }}>About Moodify AI</div>

          <h2 className="groovy-title" style={{
            fontSize:'clamp(36px,5vw,58px)', color:'#fff',
            textShadow:'4px 4px 0 #3de8c844', marginBottom:28,
            transform:`translateY(${lerp(30,0,clamp(p3In))}px)`,
            opacity: clamp(p3In * 2),
          }}>
            Music that<br/>
            <span style={{ color:'#3de8c8' }}>truly understands</span><br/>
            you ✦
          </h2>

          <p style={{
            fontSize:20, color:'#ffffffcc', lineHeight:1.8, fontWeight:600, marginBottom:20,
            transform:`translateY(${lerp(30,0,clamp(p3In - 0.1))}px)`,
            opacity: clamp((p3In - 0.1) * 3),
          }}>
            Moodify AI uses a GoEmotions-trained emotion classifier combined
            with sentence transformers to decode exactly how you're feeling — then
            connects to Last.fm's massive music library to find songs that match.
          </p>

          <p style={{
            fontSize:18, color:'#ffffff88', lineHeight:1.7, fontWeight:600,
            transform:`translateY(${lerp(30,0,clamp(p3In - 0.2))}px)`,
            opacity: clamp((p3In - 0.2) * 4),
          }}>
            Built with love by a student who believed music should meet you where you are,
            not where an algorithm thinks you should be.
          </p>
        </div>
      </div>

      {/* ── FOOTER ── */}
      <div style={{
        position:'absolute', inset:0,
        background:'#08060f',
        borderTop:'3px solid #1a1630',
        opacity: footerOp,
        transform:`translateY(${footerY}vh)`,
        willChange:'transform,opacity',
        display:'flex', flexDirection:'column', justifyContent:'center',
      }}>
        <div style={{ maxWidth:1100, margin:'0 auto', padding:'0 40px', width:'100%' }}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', flexWrap:'wrap', gap:40, marginBottom:48 }}>
            <div>
              <div style={{ fontFamily:"'Fredoka One',cursive", fontSize:36, color:'#fff', marginBottom:12 }}>
                Moodify<span style={{ color:'#FFD93D' }}>AI</span>
              </div>
              <p style={{ fontSize:15, color:'#ffffff66', fontWeight:600, maxWidth:280, lineHeight:1.7 }}>
                Feel it. Find it. Play it.
              </p>
            </div>
            <div style={{ display:'flex', gap:60, flexWrap:'wrap' }}>
              {[
                { head:'Navigate', links:[
                  { label:'Meet your moods', scene:'moods'    },
                  { label:'Mood detector',   scene:'detector' },
                  { label:'About',           scene:'about'    },
                  { label:'GitHub',          href:'https://github.com/Sami210105/moodify-ai' },
                ]},
                { head:'Moods', links:[
                  { label:'Happy',    scene:'moods' },
                  { label:'Sad',      scene:'moods' },
                  { label:'Angry',    scene:'moods' },
                  { label:'Calm',     scene:'moods' },
                  { label:'Romantic', scene:'moods' },
                  { label:'Anxious',  scene:'moods' },
                ]},
              ].map(col => (
                <div key={col.head}>
                  <div style={{ fontFamily:"'Space Mono',monospace", fontSize:11, letterSpacing:2, textTransform:'uppercase', color:'#ffffff44', marginBottom:16 }}>
                    {col.head}
                  </div>
                  <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
                    {col.links.map(l => (
                      <a key={l.label}
                        href={l.href ?? '#'}
                        target={l.href ? '_blank' : undefined}
                        rel={l.href ? 'noopener noreferrer' : undefined}
                        onClick={e => {
                          // internal links scroll-to-scene instead of following href="#"
                          // (which would otherwise just jump to the top of the page)
                          if (l.scene) { e.preventDefault(); goToScene(l.scene) }
                        }}
                        style={{ fontSize:14, color:'#ffffff88', textDecoration:'none', fontWeight:600, transition:'color 0.2s', pointerEvents:'all' }}
                        onMouseEnter={e => e.target.style.color='#FFD93D'}
                        onMouseLeave={e => e.target.style.color='#ffffff88'}
                      >{l.label}</a>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ borderTop:'1px solid #ffffff11', paddingTop:24, display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:16 }}>
            <span style={{ fontFamily:"'Space Mono',monospace", fontSize:12, color:'#ffffff33' }}>
              © 2025 Moodify AI — built with 🎵
            </span>
          </div>
        </div>
      </div>
    </>
  )
}

export default memo(AboutFooter)