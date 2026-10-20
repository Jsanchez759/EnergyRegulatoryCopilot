import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { API_BASE_URL, requestJson } from "../../lib/api";
import Icon from "../../shared/Icon";
import { authQueryKey } from "./api";

export default function AuthScreen() {
  const queryClient = useQueryClient();
  const [mode, setMode] = useState("login");
  const [error, setError] = useState("");
  const mutation = useMutation({
    mutationFn: (payload) => requestJson(`${API_BASE_URL}/auth/${mode === "login" ? "login" : "register"}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
    onSuccess: (user) => {
      localStorage.setItem("enercol_auth_changed", String(Date.now()));
      queryClient.setQueryData(authQueryKey, user);
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (mutationError) => setError(mutationError.message),
  });

  function onSubmit(event) {
    event.preventDefault();
    setError("");
    const form = new FormData(event.currentTarget);
    const payload = {
      email: String(form.get("email") || "").trim(),
      password: String(form.get("password") || ""),
    };
    if (mode === "register") payload.display_name = String(form.get("display_name") || "").trim();
    mutation.mutate(payload);
  }

  return (
    <main className="auth-screen">
      <section className="auth-story" aria-label="EnerCol">
        <a className="brand auth-brand" href="#inicio" aria-label="EnerCol">
          <span className="brand-mark"><span /><span /><span /></span>
          <span>enercol<span className="brand-period">.</span></span>
        </a>
        <div className="auth-story-copy">
          <div className="welcome-rule"><span /> INVESTIGACIÓN REGULATORIA</div>
          <h1>Investiga<br />con <em>evidencia.</em></h1>
          <p>Regulación energética colombiana, documentos propios y fuentes consultables en un solo espacio.</p>
        </div>
        <div className="auth-workflow" aria-label="Cómo funciona EnerCol">
          <div><span>01</span><p><strong>Busca en tus documentos</strong><small>PDFs indexados con página de origen</small></p></div>
          <div><span>02</span><p><strong>Contrasta fuentes vigentes</strong><small>Búsqueda web cuando hace falta</small></p></div>
          <div><span>03</span><p><strong>Revisa cómo respondió</strong><small>Método, evidencia y referencias visibles</small></p></div>
        </div>
        <div className="auth-story-foot"><span>COLOMBIA</span><span>ENERGÍA · REGULACIÓN · ANÁLISIS</span></div>
      </section>

      <section className="auth-form-side">
        <div className="auth-form-wrap">
          <span className="auth-overline">TU ESPACIO DE INVESTIGACIÓN</span>
          <h2>{mode === "login" ? "Qué bueno verte." : "Crea tu espacio."}</h2>
          <p className="auth-intro">{mode === "login" ? "Inicia sesión para continuar donde lo dejaste." : "Guarda tus documentos y consultas en un solo lugar."}</p>

          <form className="auth-form" onSubmit={onSubmit}>
            {mode === "register" && <label>Nombre
              <input name="display_name" type="text" autoComplete="name" minLength="2" maxLength="100" required placeholder="Tu nombre" />
            </label>}
            <label>Correo electrónico
              <input name="email" type="email" autoComplete="email" required placeholder="nombre@empresa.com" />
            </label>
            <label>Contraseña
              <input name="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} minLength={mode === "login" ? 1 : 12} maxLength="128" required placeholder={mode === "login" ? "Tu contraseña" : "Mínimo 12 caracteres"} />
            </label>
            {error && <p className="form-error" role="alert">{error}</p>}
            <button className="auth-submit" type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? "Un momento…" : mode === "login" ? "Iniciar sesión" : "Crear cuenta"}
              <Icon name="arrow" size={16} />
            </button>
          </form>

          <p className="auth-mode-switch">
            {mode === "login" ? "¿Primera vez en EnerCol?" : "¿Ya tienes una cuenta?"}
            <button type="button" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
              {mode === "login" ? "Crear cuenta" : "Iniciar sesión"}
            </button>
          </p>
          {mode === "register" && <p className="auth-terms">Al crear una cuenta, aceptas que tus consultas y documentos se almacenen de forma privada en este espacio.</p>}
        </div>
        <div className="auth-secure-note"><Icon name="lock" size={14} /> Acceso privado · Sesión protegida</div>
      </section>
    </main>
  );
}
