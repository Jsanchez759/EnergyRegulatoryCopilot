import { lazy, Suspense, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import AuthScreen from "./features/auth/AuthScreen";
import { authApi, authQueryKey } from "./features/auth/api";
import { AUTH_SYNC_KEY, clearSession } from "./features/auth/session";
import LandingPage from "./features/landing/LandingPage";
import ResearchWorkspace from "./features/research/ResearchWorkspace";

const CorpusLayout = lazy(() => import("./features/corpus/CorpusLayout"));
const MonitorPage = lazy(() => import("./features/corpus/MonitorPage"));
const NormsPage = lazy(() => import("./features/corpus/NormsPage"));
const NormPage = lazy(() => import("./features/corpus/NormPage"));
const MarketPage = lazy(() => import("./features/market/MarketPage"));
const AdminOverview = lazy(() => import("./features/admin/AdminOverview"));
const AdminUsers = lazy(() => import("./features/admin/AdminUsers"));
const AdminActivity = lazy(() => import("./features/admin/AdminActivity"));

function AppLoading() {
  return <main className="app-loading"><span className="brand-mark"><span /><span /><span /></span><span>Preparando EnerCol…</span></main>;
}

function AppUnavailable({ onRetry }) {
  return <main className="app-unavailable"><span className="brand-mark"><span /><span /><span /></span><h1>No pudimos conectar.</h1><p>Comprueba que el servidor esté activo e inténtalo de nuevo.</p><button type="button" onClick={onRetry}>Reintentar</button></main>;
}

// Signing in on /ingresar lands here; go back to the page that asked for the session.
function ReturnAfterSignIn() {
  const location = useLocation();
  return <Navigate to={location.state?.from || "/"} replace />;
}

function AuthenticatedApp({ user }) {
  const isAdmin = user.role === "admin";
  return (
    <Suspense fallback={<AppLoading />}>
      <Routes>
        <Route path="/" element={<ResearchWorkspace user={user} />} />
        <Route element={<CorpusLayout user={user} />}>
          <Route path="/monitor" element={<MonitorPage />} />
          <Route path="/normativa" element={<NormsPage />} />
          <Route path="/normativa/:normId" element={<NormPage />} />
          <Route path="/mercado" element={<MarketPage />} />
          {isAdmin && <>
            <Route path="/admin" element={<AdminOverview />} />
            <Route path="/admin/usuarios" element={<AdminUsers currentUserId={user.id} />} />
            <Route path="/admin/actividad" element={<AdminActivity />} />
          </>}
        </Route>
        <Route path="/ingresar" element={<ReturnAfterSignIn />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}

function PublicApp() {
  const location = useLocation();
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/ingresar" element={<AuthScreen mode="login" />} />
      <Route path="/registro" element={<AuthScreen mode="register" />} />
      {/* A signed-out deep link goes to sign-in and comes back after it. */}
      <Route path="*" element={<Navigate to="/ingresar" replace state={{ from: location.pathname + location.search }} />} />
    </Routes>
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
  if (userQuery.error?.status === 401 || userQuery.data === null) return <PublicApp />;
  return <AppUnavailable onRetry={() => userQuery.refetch()} />;
}
