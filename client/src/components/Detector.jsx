import { useState, useEffect, useRef } from 'react'
import { useScroll } from '../ScrollContext'
import { clamp, remap, lerp, SCENES } from '../useScrollProgress'
import joy      from '../assets/images/joy.gif'
import sadness  from '../assets/images/sadness.gif'
import anger    from '../assets/images/anger.gif'
import romance  from '../assets/images/romantic.gif'
import anxietyG from '../assets/images/anxiety.gif'
import calm     from '../assets/images/calm.gif'
import { memo } from 'react'

const MOOD_CONFIG = {
  happy:    { color:'#FFD93D', gif:joy,      label:'Happy!',    desc:'Pure joy detected. Time to dance.'          },
  sad:      { color:'#5ba8ff', gif:sadness,  label:'Sad.',      desc:'Sitting with the feeling. Songs that hold.' },
  angry:    { color:'#ff4d6d', gif:anger,    label:'Angry!',    desc:'That fire in you. Let it out.'              },
  calm:     { color:'#8fcca5', gif:calm,     label:'Calm.',     desc:'Breathing slow. Soft music incoming.'       },
  romantic: { color:'#FF6FB7', gif:romance,  label:'Romantic♡', desc:'Heart is full. Love songs only.'           },
  anxious:  { color:'#fd8662', gif:anxietyG, label:'Anxious.',  desc:'We got you. Grounding tunes on the way.'   },
}

function SongRow({ song, index, color, onClick }) {
  return (
    <div onClick={onClick} style={{
      display:'flex', alignItems:'center', gap:14,
      padding:'10px 16px', borderRadius:12,
      border:`2px solid ${color}22`, background:`${color}08`,
      cursor:'pointer', transition:'all 0.2s',
    }}
    onMouseEnter={e => { e.currentTarget.style.background=`${color}18`; e.currentTarget.style.transform='translateX(5px)' }}
    onMouseLeave={e => { e.currentTarget.style.background=`${color}08`; e.currentTarget.style.transform='none' }}
    >
      <div style={{
        width:32, height:32, borderRadius:8,
        background:`${color}22`, border:`2px solid ${color}44`,
        display:'flex', alignItems:'center', justifyContent:'center',
        fontFamily:"'Fredoka One',cursive", fontSize:13, color, flexShrink:0,
      }}>{String(index+1).padStart(2,'0')}</div>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontSize:14, fontWeight:700, color, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{song.name}</div>
        <div style={{ fontSize:11, color:'#1a1630', fontWeight:600 }}>{song.artist}</div>
      </div>
    </div>
  )
}

function Detector() {
  const { progress } = useScroll()

  const [text, setText]               = useState('')
  const [mood, setMood]               = useState(null)
  const [songs, setSongs]             = useState([])
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [charVisible, setCharVisible] = useState(false)

  // FIX (songs not in viewport): ref to the inner scrollable container
  // and ref to the playlist section so we can scroll to it after results land
  const innerScrollRef = useRef(null)
  const playlistRef    = useRef(null)

  const pIn  = remap(progress, 1.5/SCENES, 2.0/SCENES)
  const pOut = remap(progress, 2.5/SCENES, 3.0/SCENES)

  const sceneOp = clamp(pIn * 1.5) * clamp(1 - pOut * 1.5)
  const sceneY  = lerp(6, 0, clamp(pIn))
  const active  = clamp(pIn) > 0.3 && clamp(pOut) < 0.6

  const t  = clamp(pIn)
  const bg = `rgb(${Math.round(lerp(13,245,t))},${Math.round(lerp(11,240,t))},${Math.round(lerp(20,232,t))})`

  const cfg = mood ? MOOD_CONFIG[mood] : null

  // FIX (songs not in viewport): auto-scroll inner container to playlist section
  // whenever songs arrive so they're immediately visible
  useEffect(() => {
    if (songs.length > 0 && playlistRef.current && innerScrollRef.current) {
      setTimeout(() => {
        playlistRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }, 300) // slight delay so character animation plays first
    }
  }, [songs])

  const detect = async () => {
    if (!text.trim()) return
    setLoading(true); setMood(null); setSongs([]); setError(null); setCharVisible(false)
    try {
      const res = await fetch('https://samidha21-moodify-ai-backend.hf.space/recommendations', {
        method:'POST', headers:{ 'Content-Type':'application/json' }, body:JSON.stringify({ text }),
      })
      if (!res.ok) throw new Error()
      const data = await res.json()
      setMood(data.mood); setSongs(data.songs)
      setTimeout(() => setCharVisible(true), 200)
    } catch { setError('Could not reach the server. Please try again later.') }
    setLoading(false)
  }

  return (
    <div style={{
      position:'absolute', inset:0,
      background: bg,
      opacity: sceneOp,
      transform:`translateY(${sceneY}vh)`,
      willChange:'transform,opacity',
      overflow:'hidden',
    }}>
      {/* deco */}
      <div style={{ position:'absolute', top:'8%', right:'4%',  fontSize:64, opacity:0.1, userSelect:'none', animation:'wobble 3s ease-in-out infinite' }}>🎵</div>
      <div style={{ position:'absolute', bottom:'12%', left:'4%', fontSize:48, opacity:0.1, userSelect:'none', animation:'wobble 4s ease-in-out 1s infinite' }}>🎶</div>

      {/* FIX (songs not in viewport): changed from overflow:hidden to overflowY:auto
          so songs can be scrolled to within this scene's viewport */}
      <div
        ref={innerScrollRef}
        style={{
          position:'absolute', inset:0,
          overflowY:'auto',
          overflowX:'hidden',
          padding:'100px 40px 80px',
          display:'flex', flexDirection:'column', alignItems:'center',
          // hide the scrollbar visually — the scene itself handles paging
          scrollbarWidth:'none',
        }}
      >
        <style>{`div::-webkit-scrollbar { display: none; }`}</style>
        <div style={{ width:'100%', maxWidth:640 }}>
          {/* heading */}
          <div style={{
            textAlign:'center', marginBottom:36,
            transform:`translateY(${lerp(40,0,clamp(pIn))}px)`,
            opacity: clamp(pIn * 2),
          }}>
            <h2 className="groovy-title" style={{
              fontSize:'clamp(34px,5vw,58px)', color:'#1a1630',
              textShadow:'4px 4px 0 #FF6FB744', marginBottom:12,
            }}>
              Come with me to a land<br/>
              <span style={{ color:'#b06dff' }}>of perfect music ✦</span>
            </h2>
            <p style={{ fontSize:20, color:'#5a5270', fontWeight:600 }}>
              Tell us how you're feeling and we'll find you the soundtrack.
            </p>
          </div>

          {/* character */}
          {charVisible && cfg && (
            <div style={{
              display:'flex', flexDirection:'column', alignItems:'center', gap:12, marginBottom:24,
              animation:'popIn 0.5s cubic-bezier(0.34,1.56,0.64,1)',
            }}>
              <img src={cfg.gif} alt={cfg.label} style={{
                width:160, height:160, objectFit:'cover', borderRadius:24,
                filter:`drop-shadow(0 4px 20px ${cfg.color}99)`,
              }}/>
              <div style={{ textAlign:'center' }}>
                <div style={{ fontFamily:"'Fredoka One',cursive", fontSize:28, color:cfg.color }}>{cfg.label}</div>
                <div style={{ fontSize:14, color:'#5a5270', fontWeight:600, marginTop:3 }}>{cfg.desc}</div>
              </div>
            </div>
          )}

          {/* input card */}
          <div style={{
            transform:`translateY(${lerp(60,0,clamp(pIn*1.5))}px)`,
            opacity: clamp(pIn * 2 - 0.3),
            pointerEvents: active ? 'all' : 'none',
          }}>
            <div className="retro-card" style={{ borderColor: cfg ? cfg.color : '#1a1630', background:'#fff', transition:'border-color 0.4s' }}>
              <textarea
                value={text} onChange={e => setText(e.target.value)}
                onKeyDown={e => { if (e.key==='Enter' && (e.metaKey||e.ctrlKey)) detect() }}
                placeholder="Tell me how you're feeling right now... I'm all ears"
                rows={4}
                style={{
                  width:'100%', padding:'22px 26px', background:'transparent',
                  border:'none', resize:'none', fontFamily:"'Nunito',sans-serif",
                  fontSize:16, lineHeight:1.7, color:'#1a1630', outline:'none',
                }}
              />
              <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'12px 18px', borderTop:'2px solid #1a163014' }}>
                <span style={{ fontFamily:"'Space Mono',monospace", fontSize:11, color:'#9a9ab0' }}>⌘↵ to submit</span>
                <button onClick={detect} disabled={loading || !text.trim()} className="retro-btn"
                  style={{ background: cfg ? cfg.color : '#fadf72', color:'#1a1630', fontSize:14, padding:'10px 24px', boxShadow:'3px 3px 0 #1a1630', opacity: !text.trim() ? 0.5 : 1 }}>
                  {loading ? 'Reading you…' : 'Detect my mood'}
                </button>
              </div>
            </div>

            {error && (
              <div style={{ marginTop:14, padding:'12px 18px', borderRadius:12, background:'#ff4d6d11', border:'2px solid #ff4d6d44', fontSize:13, color:'#ff4d6d', fontWeight:700 }}>
                ⚠️ {error}
              </div>
            )}

            {songs.length > 0 && cfg && (
              <div ref={playlistRef} style={{ marginTop:28 }}>
                <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:14 }}>
                  <div className="section-label" style={{ color:cfg.color }}>Your {cfg.label.replace(/[^a-zA-Z]/g,'')} playlist</div>
                  <div style={{ flex:1, height:2, background:`${cfg.color}33`, borderRadius:1 }}/>
                  <span style={{ fontFamily:"'Space Mono',monospace", fontSize:10, color:`${cfg.color}88` }}>via Last.fm</span>
                </div>
                <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
                  {songs.map((song, i) => (
                    <SongRow key={i} song={song} index={i} color={cfg.color}
                      onClick={() => window.open(`https://www.youtube.com/results?search_query=${encodeURIComponent(song.name+' '+song.artist)}`,'_blank')}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default memo(Detector)