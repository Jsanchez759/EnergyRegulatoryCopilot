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

## Corpus regulatorio

EnerCol mantiene su propio corpus de normativa oficial del sector eléctrico desde 2015:

- **CREG:** resoluciones y proyectos de resolución del Gestor Normativo.
- **MinEnergía:** resoluciones, decretos y leyes del normograma (sector Energía, y del General y
  Regulatorio solo lo eléctrico), y los proyectos publicados para comentarios con sus fechas de
  apertura y cierre. Los decretos y leyes usan una clave nacional (`co:decreto:1073:2015`), así que
  se enlazan con las referencias que hacen otras fuentes.

El análisis de fuentes y las próximas a integrar están en
[docs/spikes/fuentes/](docs/spikes/fuentes/README.md).

- **Qué guarda por norma:** metadatos (número, fechas de expedición y publicación, Diario
  Oficial, plazo de comentarios), texto dividido por artículos con búsqueda léxica y semántica
  (del HTML consolidado, o del PDF/DOCX original con OCR si está escaneado), cada versión del
  texto, el archivo original, las relaciones con otras normas (modifica, adiciona, deroga…) y el
  estado (vigente, derogada o proyecto según la fuente). Las relaciones salen de las notas de
  vigencia de la CREG y de los títulos de MinEnergía ("Por la cual se modifica la Resolución…").
- **Cómo se mantiene al día:** un worker sobre Postgres (`pgqueuer`, sin Redis) revisa los
  listados de la CREG cada 3 horas y los de MinEnergía cada 6 (los últimos 120 días del normograma,
  porque publica con retraso, y los foros más recientes). Cada semana vuelve a consultar todas las
  normas conocidas de cada fuente para detectar modificaciones y derogatorias. Cada cambio genera
  un evento en el feed.
- **Cortesía con las fuentes:** una petición por segundo en la CREG y una cada 4 segundos en
  MinEnergía (su firewall corta conexiones ante ráfagas), User-Agent identificable,
  `ETag`/`Last-Modified` donde existen y pausas automáticas ante cortes de conexión.
- **Uso en el agente:** el chat busca primero en el corpus y cita por artículo, indicando si la
  norma está derogada o es un proyecto.

| Comando | Qué hace |
| --- | --- |
| `make worker` | Arranca solo el worker (`make dev` y `make up` ya lo incluyen) |
| `make backfill` | Encola el rastreo completo desde `CORPUS_START_YEAR` (2015); `SOURCE=minenergia` limita a una fuente |
| `make crawl` | Encola un rastreo incremental inmediato; también acepta `SOURCE=` |
| `uv run --directory backend python -m app.corpus.cli process <URL>` | Indexa una página sin la cola, para depurar el parser (de la CREG, o de otra fuente si ya fue descubierta) |

Los PDF originales y las páginas descargadas se guardan en `CORPUS_STORAGE_DIR`
(`backend/storage/corpus`), direccionados por su hash para poder moverlos a S3/R2 sin cambiar las
claves. `make dev` y `make up` comparten la misma base, la misma cola y ese mismo directorio, así
que sus workers pueden correr a la vez. Tras cambiar el código del worker reinicia `make dev`: la API
se recarga sola, el worker no.

Si mejoras un parser, `python -m app.corpus.cli reparse [--source creg]` vuelve a procesar las
normas desde las páginas guardadas, sin pedir nada a las fuentes ni generar eventos en el feed.

Para agregar una fuente: implementa el protocolo `Source` de `app/corpus/sources/base.py`
(`discover` recorre los listados y `parse` lee la página de una norma), regístrala en
`app/corpus/sources/registry.py` y añade su entrada en `app/corpus/catalog.py`. El pipeline, el
worker (con sus horarios), el CLI, la API y la interfaz la toman sin más cambios.

## Datos de mercado (XM)

El agente consulta en vivo la API pública de XM (Sinergox, sin credenciales) para responder con
cifras del mercado mayorista: precio de bolsa y de escasez, demanda, generación por planta o por
tecnología, embalses, aportes, contratos, intercambios y emisiones.

- **Herramientas del agente:** `find_market_metrics` busca entre las ~190 métricas del catálogo de
  XM (léxico + embeddings, con métricas de referencia para las preguntas comunes),
  `find_market_codes` traduce nombres de plantas, agentes, embalses o ríos a sus códigos, y
  `get_market_data` trae la serie.
- **Las cuentas las hace el código, no el modelo:** la serie se parte en ventanas de 30 días, se
  convierte de unidad (kWh a GWh, fracciones a %) y se resume en totales o promedios según la
  unidad, extremos por día y por hora, variación y como mucho 62 filas.
- **Citas y gráficos:** cada serie es una fuente `[M1]` con la consulta exacta a XM, y la respuesta
  la muestra como gráfico con la tabla de datos. Se guarda con la conversación.
- **Caché en memoria:** el catálogo y las listas de códigos se refrescan a diario; las series viven
  entre 1 y 12 horas según su granularidad. No hay tablas nuevas en la base de datos.
- SIMEM y un histórico propio de series clave quedan para fases siguientes.

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
| `CORPUS_STORAGE_DIR` | `backend/storage/corpus` | Archivos originales del corpus |
| `CORPUS_START_YEAR` | `2015` | Primer año del rastreo completo |
| `CRAWLER_USER_AGENT` | `EnerColBot/0.1 (+repo)` | Identificación ante las fuentes |
| `OCR_LANGUAGE` / `OCR_MIN_CHARS_PER_PAGE` | `spa` / `200` | OCR con Tesseract para PDF escaneados |
| `OCR_MAX_PAGES` | `40` | Páginas máximas por documento con OCR |
| `XM_API_URL` / `XM_TIMEOUT_SECONDS` | `https://servapibi.xm.com.co` / `30` | API de datos de XM |
| `MARKET_MAX_RANGE_DAYS` | `366` | Rango máximo de una consulta horaria o diaria (×3 para mensuales) |
| `AGENT_MAX_ITERATIONS` | `6` | Pasos máximos del agente por pregunta |

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
- `GET /api/v1/corpus/norms` (filtros `q`, `source`, `doc_type`, `status`, `year`,
  `open_consultation`) y
  `GET /api/v1/corpus/norms/{id}`: normas con artículos, archivos y relaciones
- `GET /api/v1/corpus/search?q=`: búsqueda híbrida por artículo
- `GET /api/v1/corpus/events`: feed de cambios; `GET /api/v1/corpus/status`: salud por fuente
- `GET /api/v1/corpus/files/{id}`: archivo original (PDF o DOCX)

Las respuestas 429 incluyen `Retry-After`.

## Limitaciones conocidas

- La ingesta de los PDF que suben los usuarios sigue siendo síncrona dentro del request.
- El corpus aún no incluye circulares ni autos de la CREG, ni UPME, CNO o SUIN.
- La CREG no publica el plazo de comentarios como dato: está en el texto del proyecto y todavía no
  se extrae. MinEnergía sí lo publica y se muestra.
- El normograma de MinEnergía no indica qué norma deroga a otra, solo el estado; las relaciones
  de MinEnergía salen de los títulos y no cubren modificaciones mencionadas solo en el articulado.
- Los modelos por defecto son los más baratos que respondieron con fuentes y citas en todas las
  preguntas de prueba (unos 0,0003 USD por consulta). Los modelos gratuitos de OpenRouter no sirven:
  bloquean llamadas al azar y limitan a 20 solicitudes por minuto.
- No hay verificación de correo ni recuperación de contraseña todavía.
- Con el modelo por defecto, las preguntas de mercado a veces fallan: el modelo omite un filtro o
  inventa un id de métrica. El servicio corrige los ids aproximados y nombra cada serie, pero un
  modelo más capaz reduce estos errores.

## Estructura

```text
backend/
  app/
    api/              Rutas y dependencias FastAPI (auth, rate limiting)
    core/             Configuración, logging, request id, Sentry, rate limiting
    schemas/          Modelos de request y respuesta
    services/         Agente, documentos/pgvector, corpus, autenticación
    corpus/           Worker, crawler por fuente, pipeline de indexación
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
