# Spike: laws/decrees + vigencia sources (Colombian electricity, 2015+)

Date of probing: 2026-10-20. ~55 polite HTTP requests total (UA `EnergyRegCopilot-Spike/0.1`, ~1 s spacing).
Scripts: `spike/vigencia/scripts/` (es.py, parse_suin.py, diario.py). Samples: `spike/vigencia/samples/`.

Legend: **[V]** = verified by a request in this spike. **[I]** = inference / not verified.

---

## TL;DR

| | SUIN-Juriscol (MinJusticia) | Gestor Normativo (Función Pública) | Diario Oficial (Imprenta) |
|---|---|---|---|
| Access | Front-end is now an Angular SPA; **data comes from open JSON endpoints** on `lexis.minjusticia.gov.co` (Elasticsearch `_search` proxy + document API) [V] | Server-rendered PHP HTML, plus an AJAX search endpoint returning HTML fragments [V] | JSF/PrimeFaces (GlassFish 4.1.1), stateful (jsessionid + ViewState); issues are PDFs served via session-bound `dynamiccontent` URLs [V] |
| robots.txt | none (SPA shell returned for `/robots.txt`) on both hosts [V] | `User-Agent: * / Disallow:` (allow all) [V] | `svrpubindc` → 404 (no robots) [V]; `www.imprenta.gov.co` allows all except `?lightbox=` [V] |
| Full text w/ article structure | Yes – HTML with per-article anchors (`<a name="ver_NNN">`), consolidated text [V] | Yes – HTML with article/numeral anchors (`<a id="5.10">`), consolidated text + "Norma Anterior" hidden blocks with prior text [V] | PDF only (whole issue, text layer, 2-column InDesign layout) [V] |
| Vigencia / relations | **Best**: per-norm "RESUMEN DE MODIFICACIONES" list, per article: action (Modificado, Adicionado, Derogado, Reglamentado, Prorrogado, Sustituido, Subrogado, Corregido yerro…), scope ("numerales 23 y 24"), link to the modifying article (`viewDocument.asp?id=X#ver_Y`) [V] | Per-norm "Vigencias(N)" list (norm-level only) + inline free-text notes per article/numeral "(Numeral 10, Modificado por el Art. 235 de la Ley 2294 de 2023)" with links [V] | None (source of publication only) |
| Change detection | ES query sorted by `fecha` desc, filter by `tipo`/`entidad`/`epigrafe` [V]; per-doc `updateTimestamp` in doc API [V] | AJAX search by type+year+entity [V]; monotonic `norma.php?i=` ids [I] | Listing of last 100 issues, daily (issue 53.663 of 19/10/2026 listed on 20/10) [V]; search by tipo de norma + date range returns issues [V] |
| Lag observed | Newest MinMinas decree indexed: Decreto 983 of 2026-08-04; newest energy-related: Decreto 1380 of 2026-09-09 → ~6–10 weeks [V] | Only 4 MinMinas 2026 decrees found vs ≥10 in SUIN → incomplete for energy [V] | ~same day (PDF created 19/10 21:08 -05) [V] |

**Recommendation:** SUIN (via the lexis JSON endpoints) as system of record for laws/decrees text **and** vigencia; Diario Oficial as the fast "new norm" signal / authoritative publication evidence; Gestor Normativo as a secondary cross-check (and for "Norma Anterior" historical text). Details and risks in §6.

---

## 1. SUIN-Juriscol

### 1.1 Access, robots, WAF, rendering [V]
- `https://www.suin-juriscol.gov.co/` → 301 → `/suin/` (nginx 1.30.2). Body is an Angular micro-frontend shell ("GovcoFrontendBase"), i.e. **JS-only**. `/robots.txt` and `/sitemap.xml` both return the same SPA HTML (200, text/html) → effectively **no robots.txt, no sitemap**.
- Legacy URLs `viewDocument.asp?id=…` / `?ruta=Leyes/…` on www.suin-juriscol.gov.co now also return the SPA shell (the old ASP viewer is gone from that host). The legacy numeric IDs survive as the document IDs.
- Bundle config (`/299.*.js`): `API_URL=https://utu.minjusticia.gov.co` (CMS), CMS menu links "Buscador" → `https://lexis.minjusticia.gov.co/buscador/suin-buscador` (a second Angular SPA, "Gov Co Suin Buscador").
- Lexis bundle env (`chunk-W2RLAMFV.js`):
  ```
  ELASTIC_URL:"https://lexis.minjusticia.gov.co/elasticsearch",
  SUIN_SERCH:"https://lexis.minjusticia.gov.co/suin-search/api/",
  visualdocs:"minjusticia/viewDocument/"
  ```
- No WAF/captcha/TLS problems with plain curl/urllib; only a `cookiesession1` cookie (Fortinet-style) is set. No auth needed.

### 1.2 Finding norms: open Elasticsearch proxy [V]
`POST https://lexis.minjusticia.gov.co/elasticsearch/documents_pdn/_search` (JSON body, full ES Query DSL incl. `sort`, `aggs`, `track_total_hits`) → 200 `application/json`.

Fields in `_source` (Ley 1715 hit): `id, visualizacion, titulo ("LEY 1715 2014"), tipo, tipodocumento, subtipo ("LEY ORDINARIA" / "DECRETO ÚNICO"), numero, anio, fecha, fechatimestamp, estado ("Vigente"), entidad, materia, sector, epigrafe, noart, documentofuente ("DIARIO OFICIAL. AÑO CL. N. 49150. 13, MAYO, 2014. PÁG. 1."), numerodiariooficial, paginadiariooficial, @timestamp`.

Example queries used (see `scripts/es.py`):
```json
{"query":{"bool":{"filter":[{"term":{"tipo.keyword":"LEY"}},{"term":{"numero.keyword":"1715"}}]}}}
{"sort":[{"fecha":{"order":"desc"}}],"query":{"bool":{"filter":[{"term":{"tipo.keyword":"DECRETO"}},
  {"term":{"entidad.keyword":"MINISTERIO DE MINAS Y ENERGIA"}},{"range":{"fecha":{"gte":"2015-01-01"}}}]}}}
```
- MinMinas decrees since 2015 by `entidad.keyword`: **127** (joint decrees signed by other ministries are missed → also search `epigrafe` for "1073 de 2015", "energía eléctrica", "143 de 1994", "1715 de 2014", etc.).
- `match_phrase epigrafe:"1715 de 2014"` → 4 decrees (2143/2015, 829/2020, 895/2022, 1318/2022).
- Aggregations work (`terms` on `entidad.keyword`, `materia.keyword`, `estado.keyword`).

**Important ID gotcha [V]:** the document API must be called with `visualizacion`, not `id`. For Decreto 1073/2015 the ES hit has `id=30020046, visualizacion=30020048`; `documentos/30020046` returns an unrelated norm (Decreto 1960 de 2015, sports). For Ley 1715 both are 1687143.

**Data-quality notes [V]:** `materia`, `sector`, `estado` are frequently empty for recent docs; duplicates exist (e.g. "DECRETO 44 2026" twice, visualizacion 30056019/30056020); `@timestamp` is a bulk reindex time (all ≈ 2026-10-02) — not a change signal.

### 1.3 Full text + vigencia: document API [V]
`GET https://lexis.minjusticia.gov.co/suin-search/api/documentos/{visualizacion}` → 200 JSON:
```
{"success":true,"data":{"id":1687143,"textoHtml":"<html…>","url":"https://www.suin-juriscol.gov.co/viewDocument.asp?id=1687143",
 "updateTimestamp":1777911968296,"collectionId":2,"subtipo":"LEY ORDINARIA","vigencia":"Vigente",
 "entidadEmisora":"CONGRESO DE COLOMBIA","epigrafe":"…","materia":"", ...}}
```
- Sizes/latency: Ley 1715 146 KB; Ley 2099 171 KB (0.29 s); Decreto 1073 **3.5 MB** (0.94 s).
- `textoHtml` structure:
  - Header table "ESTADO DE VIGENCIA" (fechas de expedición/publicación/entrada en vigencia).
  - Índice/TOC (Libro/Parte/Título/Capítulo/Artículo).
  - **`<div id="{id}ResumenNotasVigencia">`** – the machine-parseable relationship summary:
    ```html
    <li class="resumenvigencias"><b>Artículo 5:</b><ul class="resumenvigencias">
      <li class="referencia"><span>Adicionado </span>(numerales 23 y 24 )
        <a href="/viewDocument.asp?id=30041997#ver_30248106">Artículo 5 LEY 2099 de 2021</a></li>
    ```
    → `(articulo_afectado, accion, alcance, doc_id_modificador, anchor_articulo_modificador, etiqueta)`.
  - Article bodies: `<a name="ver_{articleId}"></a><div …><p><strong>Artículo 2°.</strong>…`. Text shown is the consolidated (current) version. Per-article inline "Notas de vigencia" are **not** in the body (only the summary list) in the two samples.
- Direction: the summary lives on the *affected* norm. The modifying norm (Ley 2099) does not link back to Ley 1715 in its HTML (0 links to id=1687143) → to build the graph you crawl affected norms (or invert).
- Parser: `scripts/parse_suin.py` (regex, ~25 lines).

### 1.4 Change detection [V/I]
- [V] ES sorted by `fecha` desc with type/entity/epigrafe filters returns the newest indexed norms; poll daily and diff on `visualizacion`.
- [V] `updateTimestamp` per document (Ley 1715 → 2026-05-04; Decreto 1073 → 2026-07-21) can trigger re-parse of vigencia when a consolidated norm is updated. It is not in the ES `_source`, so it requires fetching the doc (cheap except for big DURs) [I: maybe sortable elsewhere, not found].
- [V] Lag: newest indexed MinMinas decree Decreto 983 (2026-08-04); newest energy decree Decreto 1380 (2026-09-09); Decreto 1073's summary already cites 2026 decrees (375, 393, 708, 742 de 2026) → consolidation lag of ~1–3 months.

### 1.5 Open data [V]
datos.gov.co `fiev-nid6` "Lista de normas cargadas en SUIN-Juriscol" (MinJusticia, rows updated 2026-07-08, 89,366 rows). Columns: `tipo, n_mero, a_o, sector, subtipo, vigencia, entidad, materia, art_culos`. **No SUIN id, no URL, no relationships** → only useful as an inventory/coverage check. (Older `pj9b-rf8w` also exists, updated 2024.)

---

## 2. Función Pública – Gestor Normativo

### 2.1 Access [V]
- `https://www.funcionpublica.gov.co/robots.txt` → `User-Agent: * / Disallow: / Sitemap: …/sitemap.xml` (empty Disallow = allow all). Sitemap is a Liferay index (811 locs) with **0 gestornormativo URLs**.
- Pages are server-rendered PHP (no JS needed for content). HSTS, CSP frame-ancestors; `cookiesession1` cookie. No blocking observed.
- `/rss` → Liferay page linking to `/web/guest/noticias-rss/...` (site news) – no norms feed [I: name suggests news only, feed not fetched].

### 2.2 Finding norms [V]
- Stable document URL: `https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i={id}` (Ley 1715 = `i=57353`, Ley 2099 = `i=166326`, Ley 2294 = `i=209510`, Decreto 1073 = `i=77887`).
- Search: `consulta_avanzada.php` UI calls
  `GET /eva/gestornormativo/gestion/funphp/funajax.php?t=ejecuta_busqueda_avanzada2&tipdoc=11&nrodoc=1073&ano=2015&entidad1=243&temarl=&palabras=&pagina=1`
  → HTML fragment with `norma.php?i=` links. Codes: `tipdoc` 11=Decreto, 18=Ley, 986=Decreto Ley; `entidad1` 243=Ministerio de Minas y Energía; `temarl` 2524=ENERGIA ELECTRICA, 1035=SERVICIO PUBLICO DOMICILIARIO DE ENERGIA, 3014=CREG, 36609/35661=sector Minas y Energía.
- 2026 MinMinas decrees: **4 results** (213, 242, 268, 376) vs ≥10 in SUIN → coverage of the energy sector is partial (GN focuses on public-administration law).

### 2.3 Full text + vigencia [V]
- Left accordion "Vigencias(N)": norm-level list, e.g. Ley 1715 → **4 entries** (`Adicionado por Ley 2294 de 2023`, `Modificado por Ley 2294 de 2023`, `Modificado parcialmente por Ley 2099 de 2021`, `Modificado por Decreto 2106 de 2019`) each linking `norma.php?i=…`. Missing vs SUIN: Ley 1955/2019, Decreto 829/2020 & 895/2022 (reglamenta), Decreto 142/2015 (yerro).
- Decreto 1073: "Vigencias(59)" vs SUIN's 83 distinct modifying norms.
- In the body: free-text notes per numeral/article, e.g. `(Numeral 10, Modificado por el Art. <a href="norma.php?i=209510#235">235</a> de la Ley 2294 de 2023)` → regex-parseable but not structured; 220 such notes in Decreto 1073.
- Unique value: `<button>Norma Anterior</button>` + hidden `<div class="card-body jurisprudencia">` with the **previous wording** of modified articles/numerals (useful for diffs/history).
- Anchors: `<a id="5.10">` per numeral. Decreto 1073 page = 2.3 MB HTML (2.4 s).

### 2.4 Change detection [V/I]
- [V] AJAX search filtered by `tipdoc`+`ano`+`entidad1`/`temarl`; poll and diff `i` values.
- [I] `i` ids look monotonic (2026 decrees ≈ 273k–275k) → probing new ids is possible but not recommended.
- No RSS/sitemap for norms [V for sitemap].

---

## 3. Diario Oficial (Imprenta Nacional)

### 3.1 Access [V]
- `https://svrpubindc.imprenta.gov.co/diario/` still the current public viewer: GlassFish 4.1.1, JSF/PrimeFaces, `jsessionid` in URLs, `javax.faces.ViewState`. `/robots.txt` → 404 (none). `www.imprenta.gov.co/robots.txt` (Wix site) allows all except `?lightbox=`.
- No WAF/TLS problems with urllib (cookie jar needed). Not JS-only for listing (server-rendered table), but every action is a JSF postback.

### 3.2 Finding publications [V]
- Landing lists last **100 issues** (paged 10): `53.663 Ordinaria 19/10/2026`, `53.662 … 18/10/2026`, …
- Search form fields: `numeroDiarioOf, numeroRecibo, tipoNorma_input` (LEY, DECRETO, RESOLUCIÓN, …), `numeroNorma, entidad_input, fechaInicial_input, fechaFinal_input`. PrimeFaces partial-ajax POST (`javax.faces.partial.ajax=true`, `javax.faces.source=btnBuscar`, `render=@all`) works (see `scripts/diario.py`). Result granularity is the **issue**, not the norm (DECRETO 01–19/10/2026 → 19 issues).
- Opening an issue = postback of `dtbDiariosOficiales:N:j_idt34` → detail page with a `javax.faces.resource/dynamiccontent.properties.xhtml?…pfdrid=…` link → **whole-issue PDF** (53.663: 65 pages, 7.0 MB, InDesign, text layer present, two columns, Range header ignored). URLs are session-bound → not stable/shareable.
- No API/JSON/RSS/sitemap found. Not in datos.gov.co (catalog search "diario oficial" → unrelated datasets).

### 3.3 Text / vigencia
- Only raw publication text; decrees are delimited by headings ("Ministerio de X / Decretos / DECRETO NÚMERO 1540 DE 2026 (octubre 19) por el cual …"). No vigencia data. Segmenting a 2-column PDF into norms is feasible but error-prone [I].

### 3.4 Change detection [V]
- Daily: diff the issue list; issue 53.663 (19/10) PDF created 19/10 21:08 -05 and listed by 20/10 → ~same-day. Then extract text and grep for "Ministerio de Minas y Energía", "DECRETO NÚMERO", "1073 de 2015", "energía eléctrica".

---

## 4. Samples

### Ley 1715 de 2014
- SUIN: ES `visualizacion=1687143`; doc API `https://lexis.minjusticia.gov.co/suin-search/api/documentos/1687143`; legacy canonical `https://www.suin-juriscol.gov.co/viewDocument.asp?id=1687143`; viewer route `https://lexis.minjusticia.gov.co/minjusticia/viewDocument/1687143` (SPA).
  Metadata: estado Vigente, D.O. 49150 13/05/2014 pág. 1, 51 artículos, updateTimestamp 2026-05-04.
  **33 relations** parsed (`samples/rels_1715.json`): 15 Modificado(+parcialmente), 9 Adicionado(+parcialmente), 8 Reglamentado, 1 Corregido yerro. By source: Ley 2099/2021 (arts 2–16, 33, 55 → arts 1, 4, 4-1, 5, 8, 10–14, 14-1, 19, 21, 21-1, 21-2, 34), Ley 2294/2023 (arts 235, 237 → arts 5, 30), Ley 1955/2019 (arts 174, 292 → 11, 30), Decreto 2106/2019 (arts 129, 130 → 6, 12), Decreto 829/2020 & 895/2022 (reglamentan 11–14), Decreto 142/2015 (yerro art. 40).
  Sample rows: `Artículo 5 | Adicionado | (numerales 23 y 24) | Art. 5 LEY 2099 de 2021 | /viewDocument.asp?id=30041997#ver_30248106`.
- Gestor Normativo: `https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=57353` – Vigencias(4) (norm-level) + 17 inline "Modificado por" / 7 "Adicionado por" notes + "Norma Anterior" text.
- Diario Oficial: D.O. 49150 (2014) – PDF only.

### Decreto 1073 de 2015 (DUR Minas y Energía)
- SUIN: ES `id=30020046` but **`visualizacion=30020048`**; doc API `…/suin-search/api/documentos/30020048` (3.5 MB, ~2,655 article headings, 3,462 anchors), D.O. 49523 26/05/2015 pág. 619, updateTimestamp 2026-07-21.
  **1,054 relations** from **83 distinct modifying norms** (`samples/rels_1073.json`): Adicionado 707, Prorrogado 128, Derogado 72, Modificado 60, Subrogado 22, Sustituido 17, Adicionado/Modificado parcialmente 29, Corregido 6, Eliminado 3, Reglamentado 3, Suprimido 2 … By year of modifying norm: 2015:71, 2016:44, 2017:82, 2018:180, 2019:14, 2020:68, 2021:26, 2022:106, 2023:94, 2024:154, 2025:63, 2026:152. Top modifiers: Decreto 2496/2018, 375/2026, 2236/2023, 328/2020, 1353/2018 …
  Sample row: `Artículo 2.2.1.1.2.2.1.4. | Eliminado | (la definición de "Almacenamiento comercial") | Art. 1 DECRETO 1281 de 2020 | /viewDocument.asp?id=30039923#ver_30228029`.
- Gestor Normativo: `norma.php?i=77887` – Vigencias(59), 220 inline notes, 2.3 MB.

---

## 5. Pipeline sketch (inference)
1. **Discover** (daily): (a) Diario Oficial issue list → PDF → text → candidate energy decrees/laws (fast, noisy); (b) SUIN ES query `tipo in (LEY, DECRETO)`, `fecha >= last_seen`, plus epigrafe/entidad filters for energy → canonical record (`visualizacion`).
2. **Fetch** SUIN doc API by `visualizacion`; store raw `textoHtml` + `updateTimestamp`; split articles by `<a name="ver_…">`.
3. **Vigencia graph**: parse `ResumenNotasVigencia` of every in-scope norm (Ley 142/143 de 1994, 1715/2014, 2099/2021, DUR 1073/2015, 1625/2015 tributario parts, etc.) → edges `(affected_norm, affected_article, action, scope) ← (modifier_doc_id, modifier_article_anchor)`. Re-parse when `updateTimestamp` changes (weekly sweep of in-scope norms is ~dozens of requests).
4. **Gap bridging**: a decree seen in Diario Oficial but not yet in SUIN → mark "pending consolidation"; optionally parse its epígrafe ("por el cual se modifica el artículo 2.2.3.2.4.11 del Decreto 1073 de 2015") to create provisional edges.
5. **Cross-check**: GN `Vigencias(N)` and inline notes as a secondary signal / previous-text source.

## 6. System of record & risks

**Laws/decrees (text + metadata): SUIN** via `lexis.minjusticia.gov.co` JSON. It is the official MinJusticia consolidated repository, covers all decrees (not only public-administration), returns structured metadata (tipo, número, año, fecha, entidad, D.O. number/page), consolidated article HTML with stable anchors, and is queryable with full ES DSL.

**Vigencia: SUIN** – the only source with article-level, typed, linked relationships (incl. Reglamentado, Prorrogado, Derogado, scope text). GN is norm-level and incomplete (4 vs 33 relations for Ley 1715; 59 vs 83 modifiers for DUR 1073). Both carry disclaimers: SUIN "no implica una función de certificación … de la vigencia"; GN "no se hace responsable de la vigencia".

**Freshness / legal publication: Diario Oficial** – authoritative and same-day, but PDF-only, session-bound, issue-level.

Risks:
- **Undocumented endpoints**: the ES proxy and `suin-search/api` are internal APIs of a front-end that was redeployed recently (lexis index.html Last-Modified 2026-10-03; legacy viewDocument.asp already gone). They can change or be locked down (CORS/auth) without notice. Keep request rates low, cache aggressively, and monitor for schema drift. Consider asking MinJusticia for formal access.
- **ID mismatch** `id` vs `visualizacion` (must use `visualizacion`); duplicates; sparse `materia/sector/estado`.
- **Consolidation lag** of SUIN (weeks–months) → need Diario Oficial (or ministry sites) for early detection.
- **Relationship direction**: only the affected norm carries the summary; new modifiers appear only after SUIN re-consolidates the affected norm.
- Vigencia labels are free-ish text (≥17 action variants, "Adicionado en lo pertinente", "Modificado y adicionado") → normalise to an enum; scope is free text.
- Large DURs (3.5 MB HTML) – fine, but avoid frequent refetch; use `updateTimestamp`.
- Diario Oficial: stateful JSF on old GlassFish 4.1.1, 7 MB PDFs, 2-column extraction errors, no per-norm index.
- GN: partial coverage of energy; HTML notes require regex; still useful for previous wording.
- Legal: none of the sources certify vigencia; the product should present it as "according to SUIN as of <date>".
