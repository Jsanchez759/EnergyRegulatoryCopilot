import { API_BASE_URL, requestJson } from "../../lib/api";

export const authQueryKey = ["auth", "me"];

export const authApi = {
  me: () => requestJson(`${API_BASE_URL}/auth/me`),
  login: (payload) => requestJson(`${API_BASE_URL}/auth/login`, jsonOptions("POST", payload)),
  register: (payload) => requestJson(`${API_BASE_URL}/auth/register`, jsonOptions("POST", payload)),
  logout: () => requestJson(`${API_BASE_URL}/auth/logout`, { method: "POST" }),
  updateProfile: (payload) => requestJson(`${API_BASE_URL}/auth/me`, jsonOptions("PATCH", payload)),
  changePassword: (payload) => requestJson(`${API_BASE_URL}/auth/password`, jsonOptions("POST", payload)),
  deleteAccount: (password) => requestJson(`${API_BASE_URL}/auth/me`, jsonOptions("DELETE", { password })),
};

function jsonOptions(method, body) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}
