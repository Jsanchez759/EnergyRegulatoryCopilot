# CREG crawl feasibility spike (electricity, 2015+)

Date of spike: 2026-10-20. ~60 polite HTTP requests (UA `EnergyRegCopilot-spike/0.1`, ~1 s sleep).
Raw samples: `spike/creg/dl/` (HTML), `spike/creg/pdfs/` (3 PDFs + pdftotext output), parser: `spike/creg/scripts/parse_listing.py`.

Legend: **[V]** verified with real requests, **[I]** inferred.

## TL;DR

- The Gestor Normativo ("Alejandría 2.0", built by Avance Jurídico) is a **fully static HTML site on IIS** — no API, no JS data calls except plain `XMLHttpRequest` of static per-year HTML fragments. Crawling = fetching a handful of index pages + per-year fragments + one `.htm` per document. Very crawl-friendly.
- Every document has **consolidated full text as HTML with vigencia notes** ("<Artículo modificado por el artículo 50 de la Resolución 101 99 de 2026...>"), plus links to original PDF/DOCX/annexes.
- IIS serves **Last-Modified + ETag and honours conditional GET (304)** on listings and docs. This is the core change-detection mechanism.
- Since 2022 CREG numbering encodes sector: **101 xxx = electricity general, 501 xxx = electricity particular, 701 xxx = electricity draft** (102/502/702 gas natural, 103/503/703 GLP, 104/504/704 combustibles líquidos, 105/505/705 cross-sector e.g. contribución). Pre-2022 needs the per-service listings.
- Drafts: listed in gestor (`proyectos_resolucion_*`) and on creg.gov.co (last 10 only). **No structured comment-deadline field**: the deadline is stated in the draft's preamble as a relative period ("por el término de un (1) mes contado a partir del día siguiente a su publicación").

## 1. robots.txt, blocking, WAF

- [V] `https://gestornormativo.creg.gov.co/robots.txt` → **404** (no robots; IIS `X-Powered-By: ASP.NET`, `Access-Control-Allow-Origin: *`).
- [V] `https://www.creg.gov.co/robots.txt` → 301 to `https://creg.gov.co/robots.txt` → 200:
  ```
  User-agent: *
  Sitemap: https://creg.gov.co/sitemap.xml
  Disallow: /ErrorPages
  Disallow: /backup
  Disallow: /info
  Allow: */media
  Allow: */img
  ```
- [V] `https://creg.gov.co/sitemap.xml` → 200 but **empty `<sitemapindex>`** (useless).
- [V] creg.gov.co: nginx + `server: Nexura/7.3` CMS behind Google Cloud LB (`via: 1.1 google`, `GCLB` cookie). No Cloudflare, no captcha, no 403/429 seen in ~60 requests at 1 req/s. TLS fine with curl defaults.
- [V] gestornormativo: plain IIS, no WAF headers, no rate limiting observed. gzip supported (`--compressed`: 237 KB listing → 17 KB).

## 2. How documents are listed

Base: `https://gestornormativo.creg.gov.co/gestor/` (root `/` is a JS redirect to `/gestor/`). All under `/gestor/entorno/`.

- [V] Section pages (static): `resoluciones_originales.html`, `proyectos_resolucion.html`, `circulares.html`, `conceptos_juridicos.html`, `actos_administrativos_especiales.html`, `notificaciones.html`, `novedades_regulatorias.html`, `compilaciones_unicas.html`.
- [V] Resolution listings:
  - `resoluciones_por_orden_cronologico.html` (all services; current year inline)
  - `resoluciones_caracter_general_servicio_energia_electrica.html`
  - `resoluciones_caracter_particular_servicio_energia_electrica.html`
  - also `..._por_actividad`, `..._por_empresa`, `..._por_regimen_tarifario`, `..._derogadas` / `..._no_derogadas` variants.
- [V] **Pagination = per-year fragment files.** The year `<select>` (1994–2026) triggers `js/gotoDocumentosYear_customSelect_v2.js` → `loadDocumentosYear(fileName, ext, year)` → `XMLHttpRequest GET fileName + "_" + year + ".html"`. E.g.
  - `https://gestornormativo.creg.gov.co/gestor/entorno/resoluciones_por_orden_cronologico_2015.html` (200, 194 items)
  - `.../resoluciones_caracter_general_servicio_energia_electrica_2021.html` (200, 32 items, includes 174/2021)
  - `.../resoluciones_caracter_particular_servicio_energia_electrica_2015.html` (200, 58 items)
- [V] Drafts: `proyectos_resolucion_por_orden_cronologico.html`, `proyectos_resolucion_servicio_energia_electrica.html` (+ `_YYYY` fragments, inferred same pattern).
- [V] Novedades (what's new): `novedades_regulacion_servicio_energia_electrica.html` — single "integrado" page (1.3 MB, 1,193 items, all types: Resolución, Circular, Auto, Proyecto de Resolución) since 30-Nov-2019. Also `novedades_resoluciones.html`, `novedades_circulares.html`, `novedades_autos.html`, `novedades_proyectos_resolucion.html`, `novedades_regulacion_servicio_gas_natural.html`, etc.
- [V] Listing item HTML (stable, easy to parse):
  ```html
  <div id="PU20261016-112240-0d3691-98439685" class="opcion-nueva">   <!-- id only on novedades pages -->
    <a href="docs/resolucion_creg_101-127_2026.htm" target="_blank">
      ... <div class="id-documento">Resoluci&oacute;n 101_127 de 2026 CREG</div>
          <div class="descripcion-documento"><p>Por la cual se adoptan medidas ...</p></div>
  ```
  Derogated items carry a "Derogado" prefix in the description.
- [V] No JSON/API: only static `.html` loaded by XHR. The "Búsqueda avanzada" page (`herramientas_busqueda.html`) loads an Angular-style bundle (`main_creg_V3.js`, `polyfills_creg_V3.js`, `inline_creg_V3.js`) — not inspected further [I: likely a client-side search over a static index; not needed].
- [V] No RSS/Atom found on either site; creg.gov.co sitemap empty.
- [V] creg.gov.co listings (Nexura CMS): `https://creg.gov.co/loader.php?lServicio=Documentos&lFuncion=infoCategoriaConsumo&tipo=PR` ("últimos 10 proyectos"), `tipo=RE` (resoluciones), `tipo=CI` (circulares), `tipo=AU` (autos). Each item: file name, size, "Fecha Publicación", "Fecha Expedición", description, annex list; files via `https://creg.analitica.com.co/AZDigital/ControlAdmin/BajarArchivo.php?ArId=<n>`. Page is 2.7 MB — heavy; only the last 10.

## 3. Document detail pages

- [V] URL pattern: `https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_<NNNN>_<YYYY>.htm` (pre-2022, zero-padded; suffix letters for duplicates e.g. `0004a_2021`) and `resolucion_creg_<PPP>-<N>_<YYYY>.htm` (2022+, e.g. `101-127_2026`, `701-116_2026`).
- [V] Contents (full text as HTML, consolidated): header block "RESOLUCIÓN 174 DE 2021 (octubre 7) / Diario Oficial No. 51.867 de 23 de noviembre de 2021 / COMISIÓN… / epígrafe", "Resumen de Notas de Vigencia", "Concordancias", considerandos, articulado with inline notes such as `<Artículo modificado por el artículo 50 de la Resolución 101 99 de 2026. El nuevo texto es el siguiente:>` and collapsible "Notas de Vigencia / Legislación Anterior / Doctrina Concordante" boxes (`class="caja_vja_*"`, `display:none`). Panel buttons: DESCARGAS, ÍNDICE, MODIFICACIONES, CONCORDANCIAS, NOTIFICACIONES, ACTOS DE TRÁMITE. Cross-references are links to other `docs/*.htm` (e.g. `decreto_1078_2015.htm#INICIO`) → a citation graph for free.
- [V] `<title>Alejandr&iacute;a - Resoluci&oacute;n 174 de 2021 CREG</title>`. No machine-readable metadata (no meta tags for date/sector/vigencia).
- [V] Original files (DESCARGAS) — two storage generations:
  - Legacy Lotus Notes: `https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/<docUNID>/$FILE/Creg174-2021.pdf` (+ `.docx`, annexes `.xlsx`, "D-142-2021 ANÁLISIS DE COMENTARIOS…pdf"). The `<docUNID>.html` page gives "Publicación Diario Oficial No.: 51.867, el día: 23/November/2021 / Publicada en la WEB CREG el: 22/November/2021" and the file list.
  - New (2022+): `https://gestornormativo.creg.gov.co/gestor/entorno/docs/originales/Resoluci%c3%b3n_CREG_101_127_2026/Resoluci%C3%B3n_CREG_101_127_2026.pdf` (+ `.docx`, `Documento_CREG_901_431_2026.pdf` = soporte). The folder URL itself (200) renders an "originales" page with "Publicación Diario Oficial No.: 53611, el día: lunes, 31 de agosto de 2026 / Publicada en la WEB CREG el: sábado, 29 de agosto de 2026" + text.
  - Autos/circulares on novedades link to `docs/originales/Auto_Inicio_20260000959_EXP_20260198`, `docs/originales/Circular_CREG_351_2026`.
- Metadata available per doc: number, year, type (from title), epígrafe, expedition date (header), Diario Oficial number/date, web publication date (originales/NSF page), vigencia notes/derogado flag, modifications. **Sector is not on the detail page** [V for samples] — comes from listing membership or number prefix.

## 4. Separating electricity from gas

- [V] 2022+: number prefix. Counts in 2026 chronological listing: 501:57, 101:33, 502:21, 104:6, 102:5, 103:2, 105:1. Drafts 2026: 701:36, 702:6, 704:6, 703:2, 705:2. Sequences are **continuous across years** (101-69…96 in 2025, 101-97…131 in 2026; 501-115…188 in 2025, 501-189…248 in 2026) → gaps are detectable.
- [V] All years: per-service listings `resoluciones_caracter_{general,particular}_servicio_energia_electrica[_YYYY].html`, `proyectos_resolucion_servicio_energia_electrica.html`, `novedades_regulacion_servicio_energia_electrica.html`.
- [V] Caveat: per-service classification **lags**: particular-electricity 2026 lists 501-189…221 while chronological already has up to 501-248; general lists up to 101-128/131 but misses 129/130. Also 101-96 (2025) appears in the 2026 general page. So: use prefix for 2022+, per-service lists for 2015–2021, and treat 105/705 (cross-sector) and circulars/autos via the novedades electricity page or keyword fallback ("energía eléctrica", "SIN", "OR", "comercializador"…) [I].

## 5. Draft resolutions (consultas)

- [V] Gestor: `proyectos_resolucion_por_orden_cronologico.html` (52 items 2026), `proyectos_resolucion_servicio_energia_electrica.html`, `novedades_proyectos_resolucion.html`. Detail e.g. `docs/resolucion_creg_701-116_2026.htm`:
  "PROYECTO DE RESOLUCIÓN 701 116 DE 2026 (enero 8) <Publicado en la página web de la CREG: 8 de enero de 2026> … aprobó someter a consulta pública el presente proyecto de resolución por el término de un (1) mes contado a partir del día siguiente a su publicación … a la cuenta creg@creg.gov.co con asunto: “Comentarios sobre la Resolución CREG 701 116 de 2026”. En el Documento CREG 901 336 de 2026 …"
  Files: `docs/originales/Proyecto_Resoluci%C3%B3n_CREG_701_116_2026/{Proyecto_…pdf, Documento_CREG_901_336_2026.pdf/.docx, Anexo_Formato comentarios (1).xlsm}`.
- [V] creg.gov.co: `loader.php?lServicio=Documentos&lFuncion=infoCategoriaConsumo&tipo=PR` (last 10) and per-draft `…&tipo=PR&idDirectorio=501186` with "Fecha Publicación 09-10-2026", "Fecha Expedición 08-10-2026", description, annexes. `idDirectorio` increases [I: monotonic CMS id].
- [V] **No structured deadline field anywhere.** Deadline = publication date + period parsed from preamble ("un (1) mes", "diez (10) días hábiles" [I]). Extensions are published as separate circulars/resolutions ("se amplía el plazo…") [I] → need an LLM/regex extraction step and re-check.

## 6. Change-detection signals

- [V] IIS `Last-Modified` + `ETag` on every static file; conditional requests work:
  - `If-Modified-Since: Fri, 16 Oct 2026 16:25:26 GMT` on `novedades_regulacion_servicio_energia_electrica.html` → **304**, 0 bytes.
  - `If-None-Match: "ad831828dc5cdd1:0"` on `proyectos_resolucion.html` → **304**.
- [V] Listing Last-Modified values: chronological 15-Oct-2026, novedades electricity 16-Oct-2026, year fragment 2015 15-Oct-2026 (regenerated together → don't trust a fragment's date alone; diff item sets).
- [V] Doc pages change when consolidated: 174/2021 `.htm` Last-Modified 13-Aug-2026 (amended by 101 99/2026); 024/2015 12-Jun-2026. → re-fetching doc pages with conditional GET catches vigencia updates.
- [V] Original PDFs are immutable-ish (Creg024-2015.pdf Last-Modified 2016-12-27).
- [V] Novedades items have ids `PU<YYYYMMDD>-<HHMMSS>-…` (887 of 1,193 items; 2023+) = publication timestamp in the gestor; listing is newest-first.
- [V] Cache-Control on gestor is `no-cache, no-store` — irrelevant for 304 handling, which works anyway.
- creg.gov.co CMS pages: `cache-control: no-store`, no Last-Modified [V for home]; rely on content diff and increasing `idDirectorio`.

## 7. Volume (electricity)

- [V] Resolutions (general + particular electricity per-service lists): 2015: 38+58=96; 2018: 23+15=38; 2021: 32+45=77; 2023: 32+23=55; 2025: 23+52=75; 2026 (to Oct, lagging): 32+32=64 (by prefix: 33 general + 57 particular = 90).
- [V] All-sector resolutions in chronological list: 2015: 194, 2021: 181, 2025: 169, 2026: 125 (to Oct).
- [V] Novedades electricity (resoluciones + circulares + autos + proyectos): 2019*: 31, 2020: 71, 2021: 86, 2022: 113, 2023: 152, 2024: 252, 2025: 246, 2026: 241 (to mid-Oct). 2025 split: 96 resoluciones, 46 circulares, 68 autos, 35 proyectos.
- [I] Total since 2015: ~800–1,000 electricity resolutions + ~400 drafts + circulars/autos ≈ **2,000–3,000 documents**, i.e. ~2–3 k detail pages + similar number of PDFs. Backfill at 1 req/s ≈ 1–2 hours.

## 8. Samples

| | Detail (HTML full text) | File | Text extraction |
|---|---|---|---|
| 2015 | `https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0024_2015.htm` (autogeneración a gran escala; 86 KB) | `https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/67513914c35d6b8c05257e2d007cf0b0/$FILE/Creg024-2015.pdf` (+ `.docx`) | **PDF is scanned** (Canon DR-3080CII; pdftotext → 9 chars). HTML full text and DOCX available instead. |
| 2021 | `.../docs/resolucion_creg_0174_2021.htm` (344 KB, consolidated with 2026 amendments) | `.../Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/c99b2f316a59fffb052587950077d9d1/$FILE/Creg174-2021.pdf` (3.4 MB, 67 pp) | Selectable text (219 k chars) but letter-spaced header artefacts ("Re p úb l ic a"). |
| 2026 | `.../docs/resolucion_creg_101-127_2026.htm` (El Niño 2026-2027 measures) | `.../docs/originales/Resoluci%c3%b3n_CREG_101_127_2026/Resoluci%C3%B3n_CREG_101_127_2026.pdf` (25 pp, e-signed AZSign) | Clean text (106 k chars). |

## 9. Recommended strategy

**Primary source = gestornormativo (static).** creg.gov.co only as a secondary early-warning feed (drafts/circulars may appear there first [I]) and for "Fecha Publicación".

Backfill:
1. For year in 2015..current: GET `resoluciones_caracter_general_servicio_energia_electrica_<Y>.html`, `resoluciones_caracter_particular_servicio_energia_electrica_<Y>.html`, `resoluciones_por_orden_cronologico_<Y>.html` (keep 101/501/105 prefixes for 2022+), drafts `proyectos_resolucion_servicio_energia_electrica_<Y>.html` (+ chronological variant), plus `novedades_regulacion_servicio_energia_electrica.html` for circulars/autos 2019+. Also `circulares.html` / `conceptos_juridicos.html` trees for older circulars [I: same pattern, not verified].
2. Parse items (`a[href^=docs/] .id-documento .descripcion-documento`) → doc key = href.
3. Fetch each detail `.htm`; store raw HTML + ETag/Last-Modified; extract header metadata, epígrafe, vigencia notes, modification links, DESCARGAS links.
4. Use **HTML text as the canonical text** (works even for scanned 2015 PDFs); download PDF/DOCX originals lazily/as citation artefacts. OCR only as fallback.
5. Fetch the `originales/<folder>` or NSF `<UNID>.html` page for publication/Diario Oficial dates.

Incremental (every 2–4 h):
1. Conditional GET (If-None-Match/If-Modified-Since) on ~6 pages: `novedades_regulacion_servicio_energia_electrica.html`, `resoluciones_por_orden_cronologico.html`, `proyectos_resolucion_por_orden_cronologico.html`, `proyectos_resolucion_servicio_energia_electrica.html`, `novedades_circulares.html`, `novedades_autos.html`; plus creg.gov.co `tipo=PR` and `tipo=RE` (content hash).
2. On 200: diff item hrefs/PU ids vs DB → fetch new docs. Check number-sequence gaps (101/501/701) to catch items missing from lists.
3. Weekly/nightly: conditional GET of all known electricity doc `.htm` pages (~2–3 k × 304 ≈ cheap) to catch new vigencia notes/derogations; diff text → "modified" events.
4. Drafts: compute deadline from preamble; mark open consultations; watch for extension circulars and for the final resolution (101 xxx citing the 701 number).

Risks:
- Site is maintained by a contractor (Avance Jurídico); file naming/layout may change (two storage generations already: NSF and `originales`). Build tolerant parsers + alerts on parse-yield drops.
- Classification lag in per-service lists; inconsistent year assignment (101-96 in two years). Use multiple listings + prefix rules.
- Gestor publication lags the official act by days [I: 101-127 expedited 27-Aug, web 29-Aug]; creg.gov.co may be faster.
- Deadlines are free text; extensions come as separate documents.
- Older PDFs are scanned → rely on HTML consolidated text (which is an editorial product, not the official text).
- Large pages (1.3 MB novedades, 2.7 MB creg.gov.co listing) — use gzip and conditional GET.
- No robots restrictions on gestor, but be polite (1 req/s, identifying UA); robots on creg.gov.co disallows `/info`, `/backup`, `/ErrorPages` only.
