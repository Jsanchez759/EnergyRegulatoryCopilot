import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Investigación", end: true },
  { to: "/monitor", label: "Monitor regulatorio" },
  { to: "/normativa", label: "Normativa" },
  { to: "/mercado", label: "Mercado" },
];

const ADMIN_LINK = { to: "/admin", label: "Administración" };

export default function PrimaryNav({ user }) {
  const links = user?.role === "admin" ? [...LINKS, ADMIN_LINK] : LINKS;
  return (
    <nav className="primary-nav" aria-label="Secciones">
      {links.map((link) => <NavLink key={link.to} to={link.to} end={link.end} className={({ isActive }) => `primary-nav-link ${isActive ? "is-active" : ""}`}>
        {link.label}
      </NavLink>)}
    </nav>
  );
}

export function BrandLink() {
  return (
    <NavLink className="brand" to="/" aria-label="EnerCol inicio">
      <span className="brand-mark"><span /><span /><span /></span><span>enercol<span className="brand-period">.</span></span>
    </NavLink>
  );
}
