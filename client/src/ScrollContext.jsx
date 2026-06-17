import { createContext, useContext, useRef, useState, useEffect } from 'react'
import { SCENES } from './useScrollProgress'

const ScrollCtx = createContext({ progress: 0, scrollRef: null })

export function ScrollProvider({ children }) {
  const scrollRef = useRef(null)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = () => {
      const max = el.scrollHeight - el.clientHeight
      setProgress(max > 0 ? el.scrollTop / max : 0)
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => el.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <ScrollCtx.Provider value={{ progress, scrollRef }}>
      {children}
    </ScrollCtx.Provider>
  )
}

export const useScroll = () => useContext(ScrollCtx)