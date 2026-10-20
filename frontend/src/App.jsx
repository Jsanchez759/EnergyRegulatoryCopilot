import { lazy, Suspense, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Navigate, Route, Routes } from "react-router-dom";

import AuthScreen from "./features/auth/AuthScreen";
import { authApi, authQueryKey } from "./features/auth/api";
import { AUTH_SYNC_KEY, clearSession } from "./features/auth/session";
import ResearchWorkspace from "./features/research/ResearchWorkspace";

const CorpusLayout = lazy(() => import("./features/corpus/CorpusLayout"));
const MonitorPage = lazy(() => import("./features/corpus/MonitorPage"));
const NormsPage = lazy(() => import("./features/corpus/NormsPage"));
const NormPage = lazy(() => import("./features/corpus/NormPage"));

function AppLoading() {
  return <main className="app-loading"><span className="brand-mark"><span /><span /><span /></span><span>Preparando EnerCol…</span></main>;
}

function AppUnavailable({ onRetry }) {
  return <main className="app-unavailable"><span className="brand-mark"><span /><span /><span /></span><h1>No pudimos conectar.</h1><p>Comprueba que el servidor esté activo e inténtalo de nuevo.</p><button type="button" onClick={onRetry}>Reintentar</button></main>;
}

function AuthenticatedApp({ user }) {
  return (
    <Suspense fallback={<AppLoading />}>
      <Routes>
        <Route path="/" element={<ResearchWorkspace user={user} />} />
        <Route element={<CorpusLayout user={user} />}>
          <Route path="/monitor" element={<MonitorPage />} />
          <Route path="/normativa" element={<NormsPage />} />
          <Route path="/normativa/:normId" element={<NormPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}

export default function App() {
  const userQuery = useQuery({
    queryKey: authQueryKey,
    queryFn: authApi.me,
    retry: (failureCount, error) => error.status !== 401 && failureCount < 2,
  });
  const queryClient = useQueryClient();
  useEffect(() => {
    const onUnauthorized = () => clearSession(queryClient, { notifyOtherTabs: false });
    const onStorage = (event) => {
      if (event.key === AUTH_SYNC_KEY) void queryClient.invalidateQueries({ queryKey: authQueryKey });
    };
    window.addEventListener("enercol:unauthorized", onUnauthorized);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener("enercol:unauthorized", onUnauthorized);
      window.removeEventListener("storage", onStorage);
    };
  }, [queryClient]);
  if (userQuery.isLoading) return <AppLoading />;
  if (userQuery.data) return <AuthenticatedApp user={userQuery.data} />;
  if (userQuery.error?.status === 401 || userQuery.data === null) return <AuthScreen />;
  return <AppUnavailable onRetry={() => userQuery.refetch()} />;
}
