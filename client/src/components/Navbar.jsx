import { useState } from 'react'

export default function Navbar() {
  const [open, setOpen] = useState(false)
  const links = ['Meet your Moods', 'About']
  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 999,
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '14px 40px',
      background: 'rgba(13,11,20,0.85)',
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid #ffffff14',
    }}>
      <a href="#hero" style={{ display:'flex', alignItems:'center', gap:10, textDecoration:'none' }}>
        <div style={{
          width: 36, height: 36, borderRadius: '50%',
          background: 'linear-gradient(135deg, #FFD93D, #FF6FB7)',
          border: '2px solid #1a1630',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 18,
        }}>🎵</div>
        <span style={{ fontFamily:"'Fredoka One',cursive", fontSize:22, color:'#fff', letterSpacing:0.5 }}>
          Moodify<span style={{ color:'#FFD93D' }}>AI</span>
        </span>
      </a>

      <div style={{ display:'flex', gap:32, alignItems:'center' }}>
        {links.map(l => (
          <a key={l} href={`#${l.toLowerCase().replace(/\s+/g,'-')}`} style={{
            color:'#ffffffaa', fontSize:15, fontWeight:600, textDecoration:'none',
            transition:'color 0.2s',
          }}
          onMouseEnter={e=>e.target.style.color='#FFD93D'}
          onMouseLeave={e=>e.target.style.color='#ffffffaa'}
          >{l}</a>
        ))}
        <a href="#try-it" className="retro-btn" style={{
          background:'#FFD93D', color:'#0d0b14', fontSize:14, padding:'10px 22px',
          boxShadow:'3px 3px 0 #1a1630',
        }}>
          Try for free ✨
        </a>
      </div>
    </nav>
  )
}
