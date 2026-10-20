import { consultation } from "./labels";

export default function Deadline({ norm }) {
  const window = consultation(norm);
  return window && <span className={`deadline ${window.open ? "is-open" : ""}`}>{window.label}</span>;
}
