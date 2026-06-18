import { memo } from 'react'

function Navbar() {
  return (
    <nav style={{
      position:'absolute', top:0, left:0, right:0, zIndex:999,
      display:'flex', alignItems:'center', justifyContent:'space-between',
      padding:'14px 40px',
      background:'rgba(13,11,20,0.6)',
      backdropFilter:'blur(12px)',
      borderBottom:'1px solid #ffffff14',
      pointerEvents:'all',
    }}>
      <div style={{ display:'flex', alignItems:'center', gap:10 }}>
        <span style={{ fontFamily:"'Fredoka One',cursive", fontSize:22, color:'#fff', letterSpacing:0.5 }}>
          Moodify<span style={{ color:'#FFD93D' }}>AI</span>
        </span>
      </div>
    </nav>
  )
}
export default memo(Navbar)