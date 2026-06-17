import { useEffect, useState } from 'react'

// Shared scroll listener — pass the scrollRef from App
export function useScrollProgress(scrollRef) {
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
  }, [scrollRef])
  return progress
}

// Math helpers
export const clamp  = (v, lo=0, hi=1) => Math.max(lo, Math.min(hi, v))
export const remap  = (v, inLo, inHi, outLo=0, outHi=1) =>
  clamp((v - inLo) / (inHi - inLo)) * (outHi - outLo) + outLo
export const lerp   = (a, b, t) => a + (b - a) * t

// 5 scenes total — each gets 1.5×vh of scroll room
export const SCENES = 5