import { keepPreviousData, useInfiniteQuery, useQuery } from "@tanstack/react-query";

import { API_BASE_URL, requestJson } from "../../lib/api";

const corpusKey = ["corpus"];
const FEED_PAGE_SIZE = 30;

function withParams(path, params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  });
  const query = search.toString();
  return `${API_BASE_URL}${path}${query ? `?${query}` : ""}`;
}

export function fileUrl(fileId) {
  return `${API_BASE_URL}/corpus/files/${encodeURIComponent(fileId)}`;
}

export function useSourceCatalog() {
  return useQuery({
    queryKey: [...corpusKey, "sources"],
    queryFn: ({ signal }) => requestJson(withParams("/corpus/sources", {}), { signal }),
    staleTime: Infinity,
  });
}

export function useCorpusStatus() {
  return useQuery({
    queryKey: [...corpusKey, "status"],
    queryFn: ({ signal }) => requestJson(withParams("/corpus/status", {}), { signal }),
    refetchInterval: 60_000,
  });
}

export function useCorpusFeed(filters) {
  return useInfiniteQuery({
    queryKey: [...corpusKey, "events", filters],
    queryFn: ({ signal, pageParam }) => requestJson(
      withParams("/corpus/events", { ...filters, offset: pageParam, limit: FEED_PAGE_SIZE }),
      { signal },
    ),
    initialPageParam: 0,
    getNextPageParam: (lastPage, pages) => (
      lastPage.events.length === FEED_PAGE_SIZE ? pages.length * FEED_PAGE_SIZE : undefined
    ),
  });
}

export function useNorms(filters) {
  return useQuery({
    queryKey: [...corpusKey, "norms", filters],
    queryFn: ({ signal }) => requestJson(withParams("/corpus/norms", filters), { signal }),
    placeholderData: keepPreviousData,
  });
}

export function useCorpusSearch(query) {
  return useQuery({
    queryKey: [...corpusKey, "search", query],
    queryFn: ({ signal }) => requestJson(withParams("/corpus/search", { q: query, limit: 20 }), { signal }),
    enabled: query.trim().length >= 2,
    placeholderData: keepPreviousData,
  });
}

export function useNorm(normId) {
  return useQuery({
    queryKey: [...corpusKey, "norm", normId],
    queryFn: ({ signal }) => requestJson(withParams(`/corpus/norms/${encodeURIComponent(normId)}`, {}), { signal }),
    enabled: Boolean(normId),
  });
}
