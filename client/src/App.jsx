import { useScroll, ScrollProvider } from './ScrollContext'
import { SCENES, clamp, remap } from './useScrollProgress'
import Navbar       from './components/Navbar'
import Hero         from './components/Hero'
import MoodsSection from './components/MoodsSection'
import Detector     from './components/Detector'
import AboutFooter  from './components/AboutFooter'

function NavDots() {
  const { progress, scrollRef } = useScroll()
  const labels = ['Hero','Moods','Detect','About','Footer']
  return (
    <div style={{
      position:'absolute', right:24, top:'50%', transform:'translateY(-50%)',
      display:'flex', flexDirection:'column', gap:10, zIndex:100,
    }}>
      {labels.map((label, i) => {
        const sp = clamp((progress - i/SCENES) / (1/SCENES))
        const active = sp > 0.3 && sp < 0.9
        return (
          <button key={label} title={label}
            onClick={() => {
              const max = scrollRef.current.scrollHeight - scrollRef.current.clientHeight
              scrollRef.current.scrollTo({ top: max * (i/SCENES + 0.01), behavior:'smooth' })
            }}
            style={{
              width: active ? 10 : 7, height: active ? 10 : 7,
              borderRadius:'50%', border:'none', cursor:'pointer', padding:0,
              background: active ? '#FFD93D' : '#ffffff33',
              transition:'all 0.3s ease',
              boxShadow: active ? '0 0 8px #FFD93D88' : 'none',
            }}
          />
        )
      })}
    </div>
  )
}

function Layout() {
  const { scrollRef } = useScroll()
  return (
    <div style={{ width:'100vw', height:'100vh', overflow:'hidden', position:'relative' }}>
      {/* invisible scroll canvas */}
      <div ref={scrollRef} style={{ position:'absolute', inset:0, overflowY:'scroll', overflowX:'hidden' }}>
        <div style={{ height:`${SCENES * 1.5 * 100}vh` }} />
      </div>

      {/* fixed render layer */}
      <div style={{ position:'absolute', inset:0, pointerEvents:'none', overflow:'hidden' }}>
        <Navbar />
        <Hero />
        <MoodsSection />
        <Detector />
        <AboutFooter />
        <NavDots />
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ScrollProvider>
      <Layout />
    </ScrollProvider>
  )
}