# EnerCol

Espacio de investigación para consultar regulación energética colombiana y buscar dentro de documentos
PDF propios. Cada respuesta muestra las búsquedas realizadas, las citas y los fragmentos de evidencia
para que se pueda auditar.

- **Backend:** FastAPI, SQLAlchemy async, Alembic, Postgres 17 + pgvector, agente LlamaIndex sobre
  OpenRouter, búsqueda web con Tavily.
- **Frontend:** React 18 + Vite + TanStack Query.

## Requisitos

- Python 3.11–3.13 y [`uv`](https://docs.astral.sh/uv/)
- Node.js 18+ y npm
- Docker (Postgres local y stack completo)
- API key de OpenRouter (solo en el backend) y, opcionalmente, de Tavily para búsqueda web

## Inicio rápido

```bash
cp backend/.env.example backend/.env   # añade OPENROUTER_API_KEY
make setup                             # dependencias Python y Node
make dev                               # levanta Postgres, migra y arranca API + frontend
```

`make dev` abre el frontend en http://127.0.0.1:5173 y la API en http://127.0.0.1:8000
(documentación interactiva en `/docs` fuera de producción). Los puertos se pueden cambiar:
`make dev API_PORT=8010 WEB_PORT=5180`. `make help` lista todos los comandos.

| Comando | Qué hace |
| --- | --- |
| `make db` | Levanta Postgres + pgvector en Docker (`localhost:5432`) |
| `make migrate` | Aplica migraciones de Alembic |
| `make test` | pytest (contra la base `enercol_test`) + Vitest |
| `make check` | Lockfile, lint, formato, pruebas y build de producción |
| `make up` / `make down` | Stack completo en Docker: web `:8080`, API `:8000` |
| `make e2e` | Playwright contra el stack levantado, con modelo y embeddings reales |

La primera vez que corras e2e: `npx --prefix frontend playwright install chromium`.
CI (`.github/workflows/ci.yml`) ejecuta lint, `alembic check`, pruebas con Postgres, build y las
imágenes Docker en cada PR.

## Datos

- Cuentas, sesiones, conversaciones, documentos y fragmentos con embeddings viven en Postgres.
  Borrar una cuenta o un documento elimina en cascada sus fragmentos.
- Los embeddings se guardan como `halfvec(1024)` con un índice HNSW de distancia coseno. La búsqueda
  filtra por usuario con `hnsw.iterative_scan` para no perder resultados en bibliotecas pequeñas.
- Cambiar a un modelo de embeddings con otra dimensión requiere una migración nueva y reindexar.
- Cada PDF guarda su SHA-256; subir el mismo archivo dos veces devuelve 409.

## Despliegue

1. **Migraciones como paso de despliegue.** La app nunca migra al arrancar. Ejecuta
   `alembic upgrade head` (en la imagen de la API) antes de iniciar réplicas nuevas;
   `docker-compose.yml` lo modela con el servicio `migrate`.
2. **Probes.** `GET /api/v1/health` es liveness. `GET /api/v1/health/ready` responde 503 si la base no
   responde o si el esquema no está en la última migración.
3. **Entorno.** `ENVIRONMENT=production` exige `SESSION_COOKIE_SECURE=true`, `OPENROUTER_API_KEY` y
   orígenes sin `localhost`, desactiva `/docs` y emite logs JSON. `staging` usa logs JSON sin esas
   validaciones (lo usa el stack local de Docker).
4. **Cookies entre dominios.** Si el frontend y la API están en sitios distintos (p. ej. Vercel y otro
   dominio), usa `SESSION_COOKIE_SAMESITE=none` con cookies seguras, o mejor sirve ambos bajo el mismo
   dominio (`app.` y `api.`).
5. **Proxy.** Detrás de un balanceador, define `FORWARDED_ALLOW_IPS` con la IP del proxy para que el
   rate limiting por IP vea la IP real del cliente.
6. **Rate limiting.** `memory://` vale para una réplica. Con varias, usa
   `RATE_LIMIT_STORAGE_URI=redis://…`.

## Observabilidad

- Cada request recibe un `X-Request-ID` (o reutiliza uno válido que venga del proxy). El ID viaja en
  la cabecera de respuesta y en **todas** las líneas de log emitidas durante el request.
- Logs JSON en una línea (`LOG_FORMAT=json`) con `request_id`, método, ruta, estado y duración; en
  streams SSE la duración cubre el stream completo.
- Sentry se activa con `SENTRY_DSN`. No envía PII ni cuerpos de requests (las contraseñas viajan en
  ellos). `SENTRY_TRACES_SAMPLE_RATE` controla el tracing.

## Configuración del backend

Las opciones se definen en `backend/app/core/config.py` y se sobrescriben desde `backend/.env`
(ver `backend/.env.example`).

| Variable | Valor inicial | Uso |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | `development`, `test`, `staging` o `production` |
| `DATABASE_URL` | `postgresql+asyncpg://enercol:enercol@localhost:5432/enercol` | Conexión a Postgres |
| `DATABASE_POOL_SIZE` / `DATABASE_MAX_OVERFLOW` | `5` / `10` | Pool de conexiones por proceso |
| `OPENROUTER_API_KEY` | — | Credencial privada del proveedor |
| `OPENROUTER_CHAT_MODEL` | `amazon/nova-micro-v1` | Modelo de conversación (debe admitir herramientas) |
| `OPENROUTER_EMBEDDING_MODEL` | `perplexity/pplx-embed-v1-0.6b` | Modelo de embeddings (multilingüe) |
| `EMBEDDING_DIMENSIONS` | `1024` | Debe coincidir con la columna creada por las migraciones |
| `TAVILY_API_KEY` | — | Búsqueda web; sin ella el agente indica que no pudo verificar |
| `ALLOWED_ORIGINS` | `localhost:5173` | Orígenes CORS explícitos |
| `ALLOWED_ORIGIN_REGEX` | localhost en dev/test | Orígenes por patrón |
| `SESSION_TTL_DAYS` | `30` | Duración máxima de una sesión |
| `SESSION_COOKIE_SECURE` / `SESSION_COOKIE_SAMESITE` | `false` / `lax` | Atributos de la cookie |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | Almacén de contadores (`redis://…` con varias réplicas) |
| `RATE_LIMIT_AUTH` | `10/minute;50/hour` | Login, registro y cambio de contraseña, por IP |
| `RATE_LIMIT_AGENT` | `15/minute;300/day` | Consultas al agente, por usuario |
| `RATE_LIMIT_UPLOAD` | `20/hour` | Cargas de PDF, por usuario |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `900` / `160` | Fragmentación en caracteres |
| `EMBEDDING_BATCH_SIZE` | `32` | Fragmentos por llamada de embeddings |
| `EMBEDDING_TIMEOUT_SECONDS` / `EMBEDDING_MAX_RETRIES` | `15` / `4` | Tolerancia a bloqueos del proveedor |
| `AGENT_MAX_OUTPUT_TOKENS` | `4096` | Límite de tokens de la respuesta |
| `MAX_UPLOAD_BYTES` | `26214400` | Máximo de carga (25 MiB) |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / según entorno | Nivel y formato (`json` o `console`) |
| `SENTRY_DSN` / `SENTRY_TRACES_SAMPLE_RATE` | — / `0.0` | Errores y tracing |

## Frontend

`frontend/.env.local` puede indicar la URL del backend:

```dotenv
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

Para desplegar, define `VITE_API_BASE_URL` con la URL pública de la API y añade el dominio del
frontend a `ALLOWED_ORIGINS`.

## API

- `GET /api/v1/health`, `GET /api/v1/health/ready`
- `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `POST /api/v1/auth/logout`
- `GET|PATCH|DELETE /api/v1/auth/me`, `POST /api/v1/auth/password`
- `GET|POST /api/v1/conversations`, `GET /api/v1/conversations/{id}/messages`,
  `DELETE /api/v1/conversations/{id}`
- `POST /api/v1/agent/stream`: investigación con actividad, respuesta y auditoría por SSE
- `GET|POST /api/v1/sources`, `DELETE /api/v1/sources/{id}`: biblioteca de PDFs

Las respuestas 429 incluyen `Retry-After`.

## Limitaciones conocidas

- La ingesta de PDFs es síncrona dentro del request; un worker en segundo plano llegará con la feature
  de scraping e ingesta del corpus regulatorio.
- Los modelos por defecto son los más baratos que respondieron con fuentes y citas en todas las
  preguntas de prueba (unos 0,0003 USD por consulta). Los modelos gratuitos de OpenRouter no sirven:
  bloquean llamadas al azar y limitan a 20 solicitudes por minuto.
- No hay verificación de correo ni recuperación de contraseña todavía.

## Estructura

```text
backend/
  app/
    api/              Rutas y dependencias FastAPI (auth, rate limiting)
    core/             Configuración, logging, request id, Sentry, rate limiting
    schemas/          Modelos de request y respuesta
    services/         Agente, documentos/pgvector, autenticación
  alembic/            Migraciones
  tests/              Pruebas pytest (Postgres real)
  Dockerfile
frontend/
  src/                Aplicación React y estilos
  e2e/                Pruebas Playwright
  Dockerfile          Build de Vite servido con nginx
infra/postgres/       Script de inicialización de Postgres local
docker-compose.yml    db, migrate, api y web
```
