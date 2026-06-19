// Math helpers
export const clamp  = (v, lo=0, hi=1) => Math.max(lo, Math.min(hi, v))
export const remap  = (v, inLo, inHi, outLo=0, outHi=1) =>
  clamp((v - inLo) / (inHi - inLo)) * (outHi - outLo) + outLo
export const lerp   = (a, b, t) => a + (b - a) * t

// 5 scenes total — each gets 1.5×vh of scroll room
export const SCENES = 5
