import { useQuery } from "@tanstack/react-query";

import { API_BASE_URL, requestJson } from "../../lib/api";

export function usePublicOverview() {
  return useQuery({
    queryKey: ["public", "overview"],
    queryFn: ({ signal }) => requestJson(`${API_BASE_URL}/public/overview`, { signal }),
    staleTime: 5 * 60_000,
    retry: 1,
  });
}
