import { createContext, useContext, useRef, useState, useEffect, useMemo } from "react";

const ScrollCtx = createContext({ progress: 0, scrollRef: null });

export function ScrollProvider({ children }) {
  const scrollRef = useRef(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    let ticking = false;

    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          const max = el.scrollHeight - el.clientHeight;
          setProgress(max > 0 ? el.scrollTop / max : 0);
          ticking = false;
        });
        ticking = true;
      }
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // FIX: memoize context value so its reference only changes when progress changes
  // Without this, every scroll event creates a new { progress, scrollRef } object,
  // bypassing memo() on every child component and causing full re-renders each frame
  const value = useMemo(() => ({ progress, scrollRef }), [progress]);

  return (
    <ScrollCtx.Provider value={value}>
      {children}
    </ScrollCtx.Provider>
  );
}

export const useScroll = () => useContext(ScrollCtx);