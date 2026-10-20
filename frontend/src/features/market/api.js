import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { API_BASE_URL, requestJson } from "../../lib/api";
import { consumeSseResponse } from "../chat/stream";

const marketKey = ["market"];

function url(path, params = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  });
  const query = search.toString();
  return `${API_BASE_URL}/market${path}${query ? `?${query}` : ""}`;
}

export function useMarketCatalog() {
  return useQuery({
    queryKey: [...marketKey, "catalog"],
    queryFn: ({ signal }) => requestJson(url("/metrics"), { signal }),
    staleTime: 60 * 60_000,
  });
}

export function useMetricSearch(query) {
  return useQuery({
    queryKey: [...marketKey, "search", query],
    queryFn: ({ signal }) => requestJson(url("/metrics", { q: query, limit: 8 }), { signal }),
    enabled: query.trim().length >= 2,
    placeholderData: keepPreviousData,
    staleTime: 10 * 60_000,
  });
}

export function useMarketCodes(entity, query) {
  return useQuery({
    queryKey: [...marketKey, "codes", entity, query],
    queryFn: ({ signal }) => requestJson(url("/codes", { entity, q: query }), { signal }),
    enabled: Boolean(entity) && query.trim().length >= 2,
    placeholderData: keepPreviousData,
    staleTime: 60 * 60_000,
  });
}

export function useMarketOverview() {
  return useQuery({
    queryKey: [...marketKey, "overview"],
    queryFn: ({ signal }) => requestJson(url("/overview"), { signal }),
    staleTime: 10 * 60_000,
  });
}

export function seriesQueryOptions(query) {
  return {
    queryKey: [...marketKey, "series", query],
    queryFn: ({ signal }) => requestJson(url("/series", {
      metric_id: query.metric_id,
      entity: query.entity,
      start: query.start,
      end: query.end,
      codes: query.codes.join(","),
      group_by_technology: query.group_by_technology || undefined,
    }), { signal }),
    staleTime: 30 * 60_000,
    retry: (count, error) => error.status >= 500 && count < 1,
  };
}

export async function streamAssistant(body, onEvent) {
  const response = await fetch(`${API_BASE_URL}/market/assistant`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `Error del servidor (${response.status}).`;
    try { message = (await response.json()).detail || message; } catch { /* Keep the status message. */ }
    throw new Error(message);
  }
  await consumeSseResponse(response, onEvent);
}
