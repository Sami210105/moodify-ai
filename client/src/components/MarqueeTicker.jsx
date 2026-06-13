export default function MarqueeTicker({ items, bg = '#FFD93D', color = '#0d0b14' }) {
  const repeated = [...items, ...items, ...items, ...items]
  return (
    <div style={{
      background: bg,
      borderTop: '3px solid #1a1630',
      borderBottom: '3px solid #1a1630',
      padding: '12px 0',
      overflow: 'hidden',
      whiteSpace: 'nowrap',
    }}>
      <div style={{
        display: 'inline-flex',
        gap: 0,
        animation: 'waveMarquee 18s linear infinite',
        willChange: 'transform',
      }}>
        {repeated.map((item, i) => (
          <span key={i} style={{
            fontFamily: "'Fredoka One', cursive",
            fontSize: 18,
            color,
            padding: '0 24px',
            display: 'inline-flex', alignItems: 'center', gap: 12,
          }}>
            {item} <span style={{ color: color + '88', fontSize: 12 }}>✦</span>
          </span>
        ))}
      </div>
    </div>
  )
}
