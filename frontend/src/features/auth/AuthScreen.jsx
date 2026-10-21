import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useLocation } from "react-router-dom";

import Icon from "../../shared/Icon";
import { usePublicOverview } from "../landing/api";
import { authApi, authQueryKey } from "./api";
import { PROFILES } from "./profiles";

function WaitlistConfirmation({ user }) {
  return (
    <div className="auth-form-wrap waitlist-done" role="status">
      <span className="waitlist-check"><Icon name="check" size={22} /></span>
      <span className="auth-overline">LISTA DE ESPERA</span>
      <h2>Ya estás en la lista.</h2>
      <p className="auth-intro">Gracias, {user.display_name.split(" ")[0]}. Abrimos cupos de la beta por orden de llegada; cuando aprobemos tu acceso podrás entrar con <strong>{user.email}</strong> y la contraseña que acabas de crear.</p>
      <Link className="auth-submit" to="/">Volver al inicio <Icon name="arrow" size={16} /></Link>
    </div>
  );
}

export default function AuthScreen({ mode }) {
  const queryClient = useQueryClient();
  const location = useLocation();
  const overview = usePublicOverview();
  const waitlist = (overview.data?.registration_mode || "waitlist") === "waitlist";
  const [error, setError] = useState("");
  const [waiting, setWaiting] = useState(null);
  const register = mode === "register";
  const mutation = useMutation({
    mutationFn: (payload) => (register ? authApi.register(payload) : authApi.login(payload)),
    onSuccess: (user) => {
      if (user.status === "waitlist") {
        setWaiting(user);
        return;
      }
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
    const value = (name) => String(form.get(name) || "").trim();
    const payload = { email: value("email"), password: String(form.get("password") || "") };
    if (register) {
      Object.assign(payload, {
        display_name: value("display_name"),
        company: value("company") || null,
        profile: value("profile") || null,
        use_case: value("use_case") || null,
      });
    }
    mutation.mutate(payload);
  }

  const heading = register ? (waitlist ? "Únete a la beta." : "Crea tu espacio.") : "Qué bueno verte.";
  const intro = register
    ? (waitlist ? "La beta es gratuita. Déjanos tus datos y te damos acceso por orden de llegada." : "Guarda tus documentos y consultas en un solo lugar.")
    : "Inicia sesión para continuar donde lo dejaste.";

  return (
    <main className="auth-screen">
      <section className="auth-story" aria-label="EnerCol">
        <Link className="brand auth-brand" to="/" aria-label="EnerCol inicio">
          <span className="brand-mark"><span /><span /><span /></span>
          <span>enercol<span className="brand-period">.</span></span>
        </Link>
        <div className="auth-story-copy">
          <div className="welcome-rule"><span /> INVESTIGACIÓN REGULATORIA</div>
          <h1>Investiga<br />con <em>evidencia.</em></h1>
          <p>Regulación de la CREG y MinEnergía, datos del mercado de XM y tus propios documentos en un solo espacio.</p>
        </div>
        <div className="auth-workflow" aria-label="Cómo funciona EnerCol">
          <div><span>01</span><p><strong>Pregunta en lenguaje natural</strong><small>Sobre normas, consultas abiertas o el mercado</small></p></div>
          <div><span>02</span><p><strong>Recibe la cita exacta</strong><small>Resolución, artículo y vigencia</small></p></div>
          <div><span>03</span><p><strong>Revisa cómo respondió</strong><small>Método, evidencia y datos visibles</small></p></div>
        </div>
        <div className="auth-story-foot"><span>COLOMBIA</span><span>ENERGÍA · REGULACIÓN · ANÁLISIS</span></div>
      </section>

      <section className="auth-form-side">
        {waiting ? <WaitlistConfirmation user={waiting} /> : <div className="auth-form-wrap">
          <span className="auth-overline">{register && waitlist ? "BETA GRATUITA" : "TU ESPACIO DE INVESTIGACIÓN"}</span>
          <h2>{heading}</h2>
          <p className="auth-intro">{intro}</p>

          <form className="auth-form" onSubmit={onSubmit}>
            {register && <label>Nombre
              <input name="display_name" type="text" autoComplete="name" minLength="2" maxLength="100" required placeholder="Tu nombre" />
            </label>}
            <label>Correo electrónico
              <input name="email" type="email" autoComplete="email" required placeholder="nombre@empresa.com" />
            </label>
            <label>Contraseña
              <input name="password" type="password" autoComplete={register ? "new-password" : "current-password"} minLength={register ? 12 : 1} maxLength="128" required placeholder={register ? "Mínimo 12 caracteres" : "Tu contraseña"} />
            </label>
            {register && <>
              <div className="auth-form-row">
                <label>Empresa <span className="auth-optional">opcional</span>
                  <input name="company" type="text" autoComplete="organization" maxLength="150" placeholder="Dónde trabajas" />
                </label>
                <label>Perfil <span className="auth-optional">opcional</span>
                  <select name="profile" defaultValue="">
                    <option value="">Elige uno</option>
                    {PROFILES.map((profile) => <option key={profile.id} value={profile.id}>{profile.label}</option>)}
                  </select>
                </label>
              </div>
              <label>¿Para qué lo usarías? <span className="auth-optional">opcional</span>
                <textarea name="use_case" rows="2" maxLength="1000" placeholder="Ej. seguir las consultas de la CREG sobre comercialización" />
              </label>
            </>}
            {error && <p className="form-error" role="alert">{error}</p>}
            <button className="auth-submit" type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? "Un momento…" : register ? (waitlist ? "Unirme a la lista de espera" : "Crear cuenta") : "Iniciar sesión"}
              <Icon name="arrow" size={16} />
            </button>
          </form>

          <p className="auth-mode-switch">
            {register ? "¿Ya tienes acceso?" : "¿Primera vez en EnerCol?"}
            <Link to={register ? "/ingresar" : "/registro"} state={location.state} onClick={() => setError("")}>
              {register ? "Iniciar sesión" : waitlist ? "Unirme a la beta" : "Crear cuenta"}
            </Link>
          </p>
          {register && <p className="auth-terms">Al registrarte, aceptas que tus consultas y documentos se almacenen de forma privada en este espacio.</p>}
        </div>}
        <div className="auth-secure-note"><Icon name="lock" size={14} /> Acceso privado · Sesión protegida</div>
      </section>
    </main>
  );
}
