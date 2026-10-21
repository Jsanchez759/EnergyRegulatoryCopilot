import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { API_BASE_URL, requestJson } from "../../lib/api";

const adminKey = ["admin"];

function adminUrl(path, params = {}) {
  const search = new URLSearchParams(Object.entries(params).filter(([, value]) => value !== "" && value !== null && value !== undefined));
  const query = search.toString();
  return `${API_BASE_URL}/admin${path}${query ? `?${query}` : ""}`;
}

export function useAdminOverview() {
  return useQuery({
    queryKey: [...adminKey, "overview"],
    queryFn: ({ signal }) => requestJson(adminUrl("/overview"), { signal }),
  });
}

export function useAdminUsers(params) {
  return useQuery({
    queryKey: [...adminKey, "users", params],
    queryFn: ({ signal }) => requestJson(adminUrl("/users", params), { signal }),
    placeholderData: keepPreviousData,
  });
}

export function useAdminUser(userId) {
  return useQuery({
    queryKey: [...adminKey, "user", userId],
    queryFn: ({ signal }) => requestJson(adminUrl(`/users/${encodeURIComponent(userId)}`), { signal }),
    enabled: Boolean(userId),
  });
}

export function useAuditLog() {
  return useQuery({
    queryKey: [...adminKey, "audit"],
    queryFn: ({ signal }) => requestJson(adminUrl("/audit", { limit: 200 }), { signal }),
  });
}

function send(method, path, body) {
  return requestJson(adminUrl(path), {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
}

export function useAdminAction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, action, changes }) => {
      const path = `/users/${encodeURIComponent(userId)}`;
      if (action === "delete") return send("DELETE", path);
      if (action === "revoke") return send("POST", `${path}/revoke-sessions`);
      return send("PATCH", path, changes);
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: adminKey }),
  });
}
