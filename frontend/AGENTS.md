# Frontend conventions

For React and Vite work, consult `.agents/skills/vite-react-best-practices/SKILL.md` and its relevant rules. Apply these project conventions:

- Group feature-specific components, API calls, and hooks under `src/features/<feature>/`; keep only genuinely shared UI and API utilities in shared folders.
- Keep server state in TanStack Query and transient interface state in React. Mutations should invalidate affected query keys.
- Keep code identifiers and comments in English while presenting user-facing EnerCol copy in Spanish.
- Show actual agent searches, citations, page and URL details, and evidence excerpts in the conversation so users can audit answers.
- Use only `VITE_` variables for public build-time configuration. Never put provider credentials in Vite variables or browser storage.
- Add memoization only when it protects measured expensive work or a memoized child from meaningful rerenders.
- Avoid route splitting until the app has real routes. When routes are introduced, lazy-load them and configure static-host rewrites.
- Run `make check` for the production build and `make preview` to inspect the built SPA before deployment.
- Keep imports pointed at source files, not generated `dist` output. Give any future images explicit dimensions and loading behavior.
