import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { API_BASE_URL, requestJson } from "../../lib/api";

export const conversationsQueryKey = ["conversations"];

export function useConversations(search = "") {
  return useQuery({
    queryKey: [...conversationsQueryKey, search],
    queryFn: ({ signal }) => {
      const query = search ? `?search=${encodeURIComponent(search)}` : "";
      return requestJson(`${API_BASE_URL}/conversations${query}`, { signal });
    },
  });
}

export function useConversationMessages(conversationId) {
  return useQuery({
    queryKey: [...conversationsQueryKey, conversationId, "messages"],
    queryFn: ({ signal }) => requestJson(`${API_BASE_URL}/conversations/${conversationId}/messages`, { signal }),
    enabled: Boolean(conversationId),
  });
}

export function useCreateConversation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => requestJson(`${API_BASE_URL}/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "Nueva consulta" }),
    }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: conversationsQueryKey }),
  });
}

export function useDeleteConversation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (conversationId) => requestJson(`${API_BASE_URL}/conversations/${conversationId}`, { method: "DELETE" }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: conversationsQueryKey }),
  });
}
