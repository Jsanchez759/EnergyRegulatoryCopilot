import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { authApi, authQueryKey } from "./api";
import Icon from "../../shared/Icon";

export default function AccountPanel({ user, onClose, onDeleted }) {
  const queryClient = useQueryClient();
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const profileMutation = useMutation({
    mutationFn: authApi.updateProfile,
    onSuccess: (updatedUser) => {
      queryClient.setQueryData(authQueryKey, updatedUser);
      setNotice("Datos de cuenta actualizados.");
      setError("");
    },
    onError: (mutationError) => { setError(mutationError.message); setNotice(""); },
  });
  const passwordMutation = useMutation({
    mutationFn: authApi.changePassword,
    onSuccess: () => { setNotice("Contraseña actualizada. Cerramos las demás sesiones."); setError(""); },
    onError: (mutationError) => { setError(mutationError.message); setNotice(""); },
  });
  const deleteMutation = useMutation({
    mutationFn: authApi.deleteAccount,
    onSuccess: onDeleted,
    onError: (mutationError) => { setError(mutationError.message); setNotice(""); },
  });

  function submitProfile(event) {
    event.preventDefault();
    setNotice(""); setError("");
    const values = new FormData(event.currentTarget);
    const email = String(values.get("email") || "").trim();
    const payload = { display_name: String(values.get("display_name") || "").trim() };
    if (email.toLowerCase() !== user.email.toLowerCase()) {
      payload.email = email;
      payload.current_password = String(values.get("profile_password") || "");
    }
    profileMutation.mutate(payload);
  }

  function submitPassword(event) {
    event.preventDefault();
    setNotice(""); setError("");
    const values = new FormData(event.currentTarget);
    const newPassword = String(values.get("new_password") || "");
    if (newPassword !== String(values.get("confirm_password") || "")) {
      setError("Las contraseñas nuevas no coinciden.");
      return;
    }
    passwordMutation.mutate({ current_password: String(values.get("current_password") || ""), new_password: newPassword });
    event.currentTarget.reset();
  }

  function submitDelete(event) {
    event.preventDefault();
    if (!window.confirm("Se eliminarán tu cuenta, tus documentos y tus conversaciones. Esta acción no se puede deshacer. ¿Continuar?")) return;
    const values = new FormData(event.currentTarget);
    deleteMutation.mutate(String(values.get("delete_password") || ""));
  }

  return (
    <div className="account-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="account-panel" role="dialog" aria-modal="true" aria-labelledby="account-title">
        <header className="account-header">
          <div><span className="eyebrow">CUENTA</span><h2 id="account-title">Tu perfil</h2></div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Cerrar"><Icon name="close" size={18} /></button>
        </header>
        <div className="account-identity"><span className="account-avatar">{user.display_name.slice(0, 1).toUpperCase()}</span><div><strong>{user.display_name}</strong><small>{user.email}</small></div></div>
        {notice && <p className="account-notice" role="status">{notice}</p>}
        {error && <p className="form-error" role="alert">{error}</p>}

        <form key={`${user.display_name}:${user.email}`} className="account-form" onSubmit={submitProfile}>
          <h3>Datos personales</h3>
          <label>Nombre<input name="display_name" defaultValue={user.display_name} minLength="2" maxLength="100" required /></label>
          <label>Correo electrónico<input name="email" type="email" defaultValue={user.email} required /></label>
          <label>Contraseña actual <span className="optional-field">solo para cambiar el correo</span>
            <input name="profile_password" type="password" autoComplete="current-password" maxLength="128" />
          </label>
          <button className="account-save" type="submit" disabled={profileMutation.isPending}>{profileMutation.isPending ? "Guardando…" : "Guardar cambios"}</button>
        </form>

        <form className="account-form" onSubmit={submitPassword}>
          <h3>Seguridad</h3>
          <label>Contraseña actual<input name="current_password" type="password" autoComplete="current-password" required /></label>
          <label>Nueva contraseña<input name="new_password" type="password" autoComplete="new-password" minLength="12" maxLength="128" required /></label>
          <label>Repite la nueva contraseña<input name="confirm_password" type="password" autoComplete="new-password" minLength="12" maxLength="128" required /></label>
          <button className="account-save is-secondary" type="submit" disabled={passwordMutation.isPending}>{passwordMutation.isPending ? "Actualizando…" : "Cambiar contraseña"}</button>
        </form>

        <form className="account-delete" onSubmit={submitDelete}>
          <div><strong>Eliminar cuenta</strong><p>Se borrarán tus conversaciones y documentos privados.</p></div>
          <label>Confirma tu contraseña<input name="delete_password" type="password" autoComplete="current-password" required /></label>
          <button type="submit" disabled={deleteMutation.isPending}>{deleteMutation.isPending ? "Eliminando…" : "Eliminar cuenta"}</button>
        </form>
      </section>
    </div>
  );
}
