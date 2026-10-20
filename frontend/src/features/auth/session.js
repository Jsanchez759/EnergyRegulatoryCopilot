import { authQueryKey } from "./api";

export const AUTH_SYNC_KEY = "enercol_auth_changed";

export function clearSession(queryClient, { notifyOtherTabs = true } = {}) {
  if (notifyOtherTabs) {
    try {
      localStorage.setItem(AUTH_SYNC_KEY, String(Date.now()));
    } catch {
      // Other tabs will notice on their next request.
    }
  }
  queryClient.setQueryData(authQueryKey, null);
  queryClient.removeQueries({ predicate: (query) => query.queryKey[0] !== "auth" });
}
