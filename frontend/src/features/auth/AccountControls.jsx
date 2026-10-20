import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import AccountPanel from "./AccountPanel";
import { authApi } from "./api";
import { clearSession } from "./session";

export default function AccountControls({ user }) {
  const queryClient = useQueryClient();
  const [showAccount, setShowAccount] = useState(false);
  const logoutMutation = useMutation({
    mutationFn: authApi.logout,
    onSuccess: () => clearSession(queryClient),
  });

  return (
    <>
      <button className="account-trigger" type="button" onClick={() => setShowAccount(true)} aria-label="Gestionar cuenta" title="Gestionar cuenta"><span>{user.display_name.slice(0, 1).toUpperCase()}</span></button>
      <button className="topbar-signout" type="button" onClick={() => logoutMutation.mutate()} disabled={logoutMutation.isPending}>Salir</button>
      {showAccount && <AccountPanel user={user} onClose={() => setShowAccount(false)} onDeleted={() => clearSession(queryClient)} />}
    </>
  );
}
