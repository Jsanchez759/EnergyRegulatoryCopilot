export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

export async function requestJson(url, options = {}) {
  const response = await fetch(url, { credentials: "include", ...options });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    const detail = Array.isArray(data?.detail)
      ? data.detail.map((item) => item.msg).join(" ")
      : data?.detail;
    const error = new Error(detail || `Error del servidor (${response.status}).`);
    error.status = response.status;
    const isAuthEntryPoint = url.includes("/auth/login") || url.includes("/auth/register");
    if (response.status === 401 && !isAuthEntryPoint) {
      window.dispatchEvent(new Event("enercol:unauthorized"));
    }
    throw error;
  }
  return data;
}
