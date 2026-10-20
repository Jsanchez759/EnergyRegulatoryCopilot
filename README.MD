# EnerCol

Espacio de investigación para consultar regulación energética colombiana y buscar dentro de documentos PDF propios. Cuentas y conversaciones se guardan en SQLite; el índice documental, en `backend/storage/chroma`.

## Requisitos

- Python 3.11–3.13
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 18 o posterior y npm
- Una API key de OpenRouter configurada solo en el backend
- Una API key de Tavily para habilitar búsqueda web (opcional mientras se desarrolla)

## Comandos rápidos

Desde la raíz del repositorio:

Primero configura `OPENROUTER_API_KEY` en `backend/.env` (ver abajo).

```bash
make setup   # instala dependencias Python y Node
make dev     # levanta backend + frontend; Ctrl-C detiene ambos
make help    # lista comandos disponibles
```

## Pruebas y Docker

```bash
make test    # pytest (backend) + Vitest (frontend)
make check   # lockfile, lint, formato, pruebas y build de producción
make up      # construye y levanta web (http://localhost:8080) y API (:8000) con Docker
make e2e     # Playwright contra el stack levantado; usa el modelo y embeddings reales
make down
```

La primera vez, instala el navegador de Playwright con `npx --prefix frontend playwright install chromium`.
CI (`.github/workflows/ci.yml`) ejecuta lint, pruebas, build y las imágenes Docker en cada PR.

También están disponibles `make backend`, `make frontend`, `make health`, `make lint`,
`make format`, `make build` y `make check`. Los puertos se pueden cambiar, por ejemplo:

```bash
make dev API_PORT=8010 WEB_PORT=5180
```

## Configurar el backend

El archivo local `backend/.env` contiene la configuración del servidor. No copies la API key al frontend ni la publiques en el repositorio.

```dotenv
OPENROUTER_API_KEY=tu_api_key
OPENROUTER_CHAT_MODEL=openrouter/free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
TAVILY_API_KEY=
DATABASE_PATH=storage/enercol.db
SESSION_TTL_DAYS=30
SESSION_COOKIE_SECURE=false
```

El modelo configurado debe admitir llamadas a herramientas. Sin `TAVILY_API_KEY`, la búsqueda web
queda deshabilitada; el agente debe indicar cuándo no pudo verificar un dato y no inventar detalles.

Las cuentas, sesiones y conversaciones se guardan en SQLite en `backend/storage/enercol.db`.
Alembic aplica migraciones al iniciar el backend. Cada sesión usa una cookie `HttpOnly`; cada usuario
solo puede consultar sus conversaciones y sus documentos. En despliegues HTTPS configura
`SESSION_COOKIE_SECURE=true` y `ALLOWED_ORIGINS` con el dominio exacto del frontend.

El prototipo incluye registro, inicio/cierre de sesión, edición de nombre y correo, cambio de
contraseña y eliminación de cuenta con sus datos. No incluye todavía verificación de correo ni
recuperación de contraseña: ambas funciones requieren un proveedor de correo transaccional. Antes de
abrir el registro al público, añade límites de intentos en el borde de red y define los términos,
política de privacidad y canal de soporte del producto.

Desde la carpeta `backend`:

```bash
uv sync --group dev
uv run fastapi dev --host 127.0.0.1 --port 8000
```

La documentación interactiva de la API está en `http://127.0.0.1:8000/docs`.

## Configurar el frontend

`frontend/.env.local` puede indicar la URL del backend:

```dotenv
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

Desde `frontend`:

```bash
npm install
npm run dev
```

Vite muestra la dirección local al iniciar. Para desplegar, define `VITE_API_BASE_URL` con la URL pública de la API y configura en el backend los orígenes permitidos en `ALLOWED_ORIGINS`.

## Configuración del backend

Las opciones se definen en `backend/app/core/config.py` y pueden sobrescribirse desde `backend/.env`:

| Variable | Valor inicial | Uso |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | Credencial privada del proveedor |
| `OPENROUTER_CHAT_MODEL` | `openrouter/free` | Modelo de conversación |
| `OPENROUTER_EMBEDDING_MODEL` | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | Modelo de embeddings |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API compatible con OpenAI |
| `TAVILY_API_KEY` | — | Clave privada opcional para búsqueda web |
| `DATABASE_PATH` | `backend/storage/enercol.db` | Base de datos local de cuentas y conversaciones |
| `SESSION_TTL_DAYS` | `30` | Duración máxima de una sesión |
| `SESSION_COOKIE_SECURE` | `false` | Requiere HTTPS para enviar la cookie |
| `VECTOR_DB_PATH` | `backend/storage/chroma` | Índice persistente de documentos |
| `VECTOR_COLLECTION_NAME` | `regulatory_documents` | Colección vectorial |
| `RAG_CHUNK_SIZE` | `900` | Tamaño de fragmentos en caracteres |
| `RAG_CHUNK_OVERLAP` | `160` | Solapamiento entre fragmentos |
| `EMBEDDING_BATCH_SIZE` | `32` | Fragmentos por llamada de embeddings |
| `EMBEDDING_TIMEOUT_SECONDS` | `15` | Timeout por llamada de embeddings |
| `EMBEDDING_MAX_RETRIES` | `4` | Reintentos ante timeouts del proveedor |
| `AGENT_MAX_OUTPUT_TOKENS` | `4096` | Límite de tokens de la respuesta |
| `MAX_UPLOAD_BYTES` | `26214400` | Máximo de carga (25 MiB) |
| `ALLOWED_ORIGINS` | `http://localhost:5173`, `http://127.0.0.1:5173` | Orígenes CORS explícitos |
| `LOG_LEVEL` | `INFO` | Nivel de logs |

## API

- `GET /api/v1/health`
- `POST /api/v1/auth/register` y `POST /api/v1/auth/login` — crear cuenta e iniciar sesión
- `POST /api/v1/auth/logout`, `GET /api/v1/auth/me`, `PATCH /api/v1/auth/me`
- `POST /api/v1/auth/password`, `DELETE /api/v1/auth/me`
- `GET /api/v1/conversations`, `POST /api/v1/conversations`, `GET /api/v1/conversations/{id}/messages`
- `POST /api/v1/agent/stream` — investigación unificada con actividad, respuesta y auditoría SSE
- `GET /api/v1/sources` — lista de PDFs indexados
- `POST /api/v1/sources` — carga e indexación de un PDF

Las cargas aceptan PDFs de hasta 25 MiB con texto seleccionable. El índice Chroma actual parte vacío;
vuelve a cargar los PDFs para incorporarlos. La recuperación y orquestación usan LlamaIndex; cada
respuesta incluye las búsquedas realizadas, citas y fragmentos de evidencia. Los PDFs llevan un hash
SHA-256 para verificar el archivo indexado. La búsqueda web usa Tavily cuando se configura su clave.

## Estructura

```text
backend/
  app/
    api/              Rutas y dependencias FastAPI
    core/             Configuración y logging
    schemas/           Request, source, answer and audit models
    services/          Agent orchestration, document indexing and retrieval
  tests/              Pruebas pytest
  Dockerfile      Dependencias, configuración de uv y Ruff
  uv.lock             Resolución reproducible
frontend/
  src/                Aplicación React y estilos
  e2e/                Pruebas Playwright
  Dockerfile          Build de Vite servido con nginx
docker-compose.yml    Stack local completo
```
