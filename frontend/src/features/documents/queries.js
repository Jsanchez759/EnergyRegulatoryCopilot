import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { API_BASE_URL, requestJson } from "../../lib/api";

export const documentsQueryKey = ["documents"];

async function fetchDocuments({ signal }) {
  return requestJson(`${API_BASE_URL}/sources`, { signal });
}

async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  return requestJson(`${API_BASE_URL}/sources`, {
    method: "POST",
    body: formData,
  });
}

export function useDocuments() {
  return useQuery({
    queryKey: documentsQueryKey,
    queryFn: fetchDocuments,
    staleTime: 30_000,
    retry: 1,
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: uploadDocument,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: documentsQueryKey }),
  });
}
