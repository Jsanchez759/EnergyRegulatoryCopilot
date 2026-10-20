# Spike de fuentes regulatorias (electricidad, 2015+)

Fecha: 2026-10-20. Alcance: regulación del sector eléctrico colombiano desde 2015. Cada fuente se
revisó con peticiones reales, espaciadas, con User-Agent descriptivo. Los informes detallados, con
URLs, estructuras y muestras, están en este mismo directorio:

- [creg.md](creg.md): CREG (Gestor Normativo y portal)
- [minenergia.md](minenergia.md): Ministerio de Minas y Energía (normograma y foros)
- [vigencia.md](vigencia.md): SUIN-Juriscol, Gestor Normativo de Función Pública, Diario Oficial
- [secondary.md](secondary.md): UPME, Superservicios, XM, CNO

## Resumen por fuente

| Fuente | Acceso | Texto | Detección de cambios | Volumen 2015+ | Release 1 |
| --- | --- | --- | --- | --- | --- |
| **CREG** | HTML estático por año, sin bloqueos | HTML consolidado con notas de vigencia; PDF/DOCX originales | `ETag`/`Last-Modified` con 304; numeración por sector desde 2022 | ~2.000–3.000 | Sí |
| **MinEnergía normas** | Buscador PHP con filtros en la URL | PDF; muchos escaneados | IDs crecientes (no en orden de fecha); sin cabeceras de caché | ~1.000–1.200 | Sí |
| **MinEnergía proyectos** | API JSON abierta (Wagtail) | PDF adjuntos en el HTML | IDs y `start_date`/`end_date` del periodo de comentarios | ~300 | Sí |
| **SUIN-Juriscol** | Elasticsearch y API JSON internos, sin autenticación | HTML consolidado por artículo | `updateTimestamp` por documento | Leyes y decretos del sector | Sí (leyes, decretos, vigencia) |
| **CNO** | HTML paginado, sin bloqueos | HTML reciente; PDF en S3 (2015 escaneados) | Número de acuerdo creciente en la página 0 | ~1.500 acuerdos | Sí |
| **UPME** | Biblioteca jurídica HTML con filtro `energia_electrica` | PDF limpio desde 2020; escaneados antes | `ETag`/`Last-Modified` en PDFs | ~500 (2020+) | Sí, desde 2020 |
| Diario Oficial | JSF con sesión, un PDF por edición | PDF de 2 columnas | Últimas 100 ediciones | n/a | Fase 2 (detección temprana de decretos) |
| XM | Sin índice central; ~40 páginas sueltas | PDF | Sitemap con fechas y RSS | Bajo | Diferido |
| Superservicios | Firewall Imperva bloquea los listados | PDFs y RSS accesibles | RSS de 10 ítems | n/a | No; opcional vigilar el RSS |

## Hallazgos que condicionan el diseño

1. **El texto HTML es mejor que el PDF.** CREG, SUIN y el CNO reciente publican el texto completo en
   HTML con estructura de artículos. El PDF se guarda como original oficial, pero el texto para
   buscar y citar sale del HTML cuando existe.
2. **Hay que hacer OCR.** CREG 2015–2016, buena parte de MinEnergía, UPME antes de 2020 y CNO 2015
   son escaneos sin capa de texto o con texto ruidoso.
3. **Electricidad vs gas.** En CREG, desde 2022 el prefijo del número lo indica (101, 501, 701 son
   electricidad). Antes de 2022 solo lo dicen los listados por servicio, que se actualizan con
   retraso. En MinEnergía, `id_sector=2` más una pasada por palabras clave.
4. **Plazos de consultas.** MinEnergía los expone como campos (`start_date`, `end_date`). En CREG
   son texto libre dentro del proyecto ("por el término de un (1) mes…") y las ampliaciones llegan
   como documentos aparte: hay que extraerlos y marcarlos como calculados.
5. **Vigencia.** SUIN tiene la relación más completa (Ley 1715: 33 relaciones; Decreto 1073: 1.054
   de 83 normas), pero solo en la norma afectada y con unas 6–10 semanas de retraso. CREG trae
   notas de modificación dentro del texto. Ninguna fuente certifica la vigencia: el producto debe
   mostrar "según SUIN/CREG al <fecha>".
6. **Fragilidad.** Las APIs de SUIN y MinEnergía son internas y sin documentar, y varios sitios ya
   cambiaron de plataforma una vez. Cada fuente necesita verificaciones de salud que alerten si un
   listado deja de devolver lo esperado.
7. **Cortesía.** Los servidores de MinEnergía cortaron conexiones ante una ráfaga de peticiones
   (~190 a ~1,2 s durante el spike, causadas por un error de paginación ya identificado). Límite
   por host de al menos 2 s, una conexión, peticiones condicionales y backoff ante cortes.

## Diseño propuesto para la etapa 1

**Worker sobre Postgres** con `pgqueuer` (usa asyncpg, el mismo driver de la API; tareas
periódicas tipo cron; sin Redis). Un proceso `worker` separado de la API en `docker-compose`.

**Modelo de datos**

| Tabla | Contenido |
| --- | --- |
| `sources` | Fuente, URL base, límite de peticiones, estado de salud |
| `norms` | Identidad canónica (`creg:resolucion:101-127:2026`), entidad, tipo, número, año, fecha, título, estado de vigencia, sector |
| `norm_versions` | Cada versión obtenida: hash, fecha, texto, origen (HTML/PDF/OCR), archivos |
| `norm_files` | Originales (PDF, DOCX, anexos) con hash y ruta en el almacenamiento |
| `norm_articles` | Texto por artículo con embedding y `tsvector` en español (búsqueda híbrida) |
| `norm_relations` | modifica, deroga, adiciona, reglamenta, sustituye; con fuente y fecha |
| `consultations` | Proyectos en consulta: apertura, cierre (dado o calculado), documentos |
| `crawl_runs` | Cada corrida: fuente, inicio, fin, items vistos, nuevos, cambiados, errores |
| `norm_events` | Norma nueva, versión nueva, consulta abierta o cerrada: alimenta el monitor y las alertas |

**Pipeline por norma:** `discover` (listados) → `fetch` (condicional, guarda originales) →
`extract` (HTML, PDF o OCR) → `structure` (metadatos, artículos, relaciones) → `enrich` (resumen,
temas, embeddings) → `publish` (evento).

**Orden de construcción:** worker y modelo de datos → CREG (la fuente más rica y fácil) →
MinEnergía (normas y consultas) → CNO → UPME → SUIN para leyes, decretos y vigencia.

## Pendientes para fases siguientes

- Diario Oficial para detectar decretos antes de que SUIN los consolide.
- XM como lista de páginas vigiladas y RSS de Superservicios.
- Histórico de UPME 2015–2019 (sin etiqueta de sector y aparentemente incompleto).
