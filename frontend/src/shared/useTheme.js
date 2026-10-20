import { useEffect, useState } from "react";

const THEME_STORAGE_KEY = "enercol_theme_v1";

function storedTheme() {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) || "light";
  } catch {
    return "light";
  }
}

export default function useTheme() {
  const [theme, setTheme] = useState(storedTheme);

  useEffect(() => {
    document.body.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // The theme still applies for this visit.
    }
  }, [theme]);

  return [theme, setTheme];
}
