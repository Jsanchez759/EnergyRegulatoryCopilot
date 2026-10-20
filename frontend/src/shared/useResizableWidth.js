import { useCallback, useEffect, useState } from "react";

function readStoredWidth(storageKey, fallback) {
  try {
    const value = Number(localStorage.getItem(storageKey));
    return Number.isFinite(value) && value > 0 ? value : fallback;
  } catch {
    return fallback;
  }
}

export default function useResizableWidth({ storageKey, defaultWidth, minWidth, maxWidth }) {
  const clamp = useCallback(
    (value) => Math.min(maxWidth, Math.max(minWidth, Math.round(value))),
    [minWidth, maxWidth],
  );
  const [width, setWidth] = useState(() => clamp(readStoredWidth(storageKey, defaultWidth)));
  const [isResizing, setIsResizing] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, String(width));
    } catch {
      // Persisting the width is a convenience; ignore blocked storage.
    }
  }, [storageKey, width]);

  const onPointerDown = useCallback((event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = width;
    setIsResizing(true);

    function onPointerMove(moveEvent) {
      setWidth(clamp(startWidth + moveEvent.clientX - startX));
    }
    function onPointerUp() {
      setIsResizing(false);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    }
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }, [clamp, width]);

  const onKeyDown = useCallback((event) => {
    const step = event.shiftKey ? 48 : 16;
    if (event.key === "ArrowLeft") setWidth((value) => clamp(value - step));
    else if (event.key === "ArrowRight") setWidth((value) => clamp(value + step));
    else if (event.key === "Home") setWidth(minWidth);
    else if (event.key === "End") setWidth(maxWidth);
    else return;
    event.preventDefault();
  }, [clamp, minWidth, maxWidth]);

  const reset = useCallback(() => setWidth(clamp(defaultWidth)), [clamp, defaultWidth]);

  const handleProps = {
    role: "separator",
    "aria-orientation": "vertical",
    "aria-valuenow": width,
    "aria-valuemin": minWidth,
    "aria-valuemax": maxWidth,
    tabIndex: 0,
    onPointerDown,
    onKeyDown,
    onDoubleClick: reset,
  };

  return { width, isResizing, handleProps };
}
