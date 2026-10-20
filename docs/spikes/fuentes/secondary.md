# Secondary sources feasibility spike: UPME, Superservicios, XM, CNO

Date: 2026-10-20. About 50 HTTP requests in total, 1 s apart, with User-Agent `EnergyRegCopilot-feasibility-spike/0.1 (contact: ...)`.
Raw samples are in `spike/secondary/samples/` (PDFs in `samples/pdf/`). The fetch helper is `spike/secondary/scripts/f.sh`.
Notation: **[V]** = verified by request in this spike. **[I]** = inference or recommendation.

---

## 1. UPME (www.upme.gov.co)

### Access
- [V] robots.txt is permissive (`Disallow:` empty, plus one JSON file). It points to the sitemap `https://www.upme.gov.co/sitemap_index.xml` (Yoast).
- [V] WordPress + Elementor. No WAF challenge, all requests returned 200. Pages are heavy (0.7–1.1 MB of HTML each).
- [V] No headless browser is needed for the legal library: it is server-rendered.

### Where normative documents live (three surfaces, and only one is authoritative)
1. **[V] "Biblioteca Jurídica" (custom plugin `biblioteca-juridica-main`). This is the live, complete listing to use.**
   - `https://www.upme.gov.co/nosotros/biblioteca-juridica/resoluciones-upme/` showed "Mostrando 50 de 940 resultados", "Página 1 de 19".
   - Filtering is done server-side with GET params, as read from `assets/js/frontend.js`: `categoria` (resoluciones_upme, circulares_upme, proyectos_normativos, concepto_juridico, agenda_regulatoria, otra_normativa, citaciones_notificaciones, ppda), `subseccion` (energia_electrica, hidrocarburos, mineria, transversal, asociadas_tramites, anteriores_2020), `anio`, `q`, `orderby` (recientes|antiguos|titulo), `pagina`.
   - Example: `.../circulares-upme/?categoria=circulares_upme&subseccion=energia_electrica&pagina=2` returned "50 de 161", "Página 2 de 4".
   - There is also an AJAX endpoint, `admin-ajax.php` with action `bj_front_search` (per_page 40). It needs a page nonce. We did not need it.
   - Each card carries: title (`h3.bj-tarjeta-titulo`), summary ("Por la cual…"), `Fecha Publicación: dd/mm/yyyy`, category, subcategory (e.g. "Energía Eléctrica"), and a file link to `https://docs.upme.gov.co/Normatividad/<n>_<yyyy>.pdf`. Annex links (.xlsx) appear as separate cards.
   - **Caveat [V]:** documents before 2020 are only under `subseccion=anteriores_2020`, with no sector tag. Resoluciones for 2017 returned only 16 results, all "Anteriores al 2020", and these mix mining royalties, fuel quotas, and an electricity GHG emission factor (Res 804/2017). [I] The pre-2020 archive looks partial and needs keyword or LLM classification to pick out the electricity documents.
2. [V] WP REST custom post type `circular_resolucion`: `https://www.upme.gov.co/wp-json/wp/v2/circular_resolucion?per_page=100`.
   - Returns X-WP-Total 523 and clean JSON: date, modified, title, excerpt, `link` → PDF on docs.upme.gov.co, and taxonomy `categoria_circular_o_resolucion`, where 513 = circulares energía eléctrica and 514 = resoluciones energía eléctrica.
   - **But it is stale:** the oldest entry is 2024-01 and the newest is Circular 076/2026 (2026-08-03). The homepage and the Biblioteca already show Circular 100/2026 and Res 749/2026 (13/10/2026). The RSS feed `/feed/?post_type=circular_resolucion` has lastBuildDate 2026-08-03. [I] Do not rely on it.
   - The Yoast sitemap `circular_resolucion-sitemap.xml` is empty (a single `<loc></loc>`).
3. [V] SIMEC "Normatividad – Energía eléctrica" page. It is a hand-maintained JS array (`const coberturaEnergiaElectrica = [{URL, TITULO, DESCRIPCION, ANO_PUBLICACION, CATEGORIA}]`) of mostly third-party norms (CREG, MME, decrees). It is useful as a curated seed list, not as a feed.

### Files and text
- [V] docs.upme.gov.co is served with `Last-Modified` and `ETag` headers (e.g. Circular_067_2025.pdf: Last-Modified 15 Jul 2025, ETag `"687689a9-25049"`).
- [V] Recent PDFs are born-digital (LibreOffice). Text extracts cleanly and includes the Radicado ORFEO, PARA/DE/ASUNTO/FECHA fields.
- [V] 2017 Res 804 is a scanner image (`PFU PDF Library` / "SS Manager for fi Series"). pdftotext extracts 4 characters, so it needs OCR.

### Change detection
- [I] Poll `?categoria=resoluciones_upme&orderby=recientes` and `?categoria=circulares_upme` (page 1) every few hours. Diff on the PDF URL and title. Optionally also poll `proyectos_normativos`, which carries drafts open for comment.
- [I] Optionally do conditional HEAD/GET on PDFs using ETag.
- [V] The homepage also lists the latest circulars.

### Samples
- https://docs.upme.gov.co/Normatividad/Circular_067_2025.pdf. Information on connection requests for autogeneración/GD to the SIN/ZNI. Connection-related.
- https://docs.upme.gov.co/Normatividad/691_2025.pdf. Resolución 691/2025 (interventor for the Urrá–Urabá 220 kV call).

### Rating
- Difficulty: **easy** for 2020 onward. **Medium** for 2015–2019, which has no sector tag, scanned PDFs needing OCR, and possible gaps.
- Value: **medium-high**. It covers connection and transmission calls, generation/transmission plans, and FNCER incentives.
- **Recommendation: include in the first release** (Biblioteca Jurídica, energía eléctrica + transversal, 2020 onward). Backfill 2015–2019 later with classification and OCR.

---

## 2. Superservicios (www.superservicios.gov.co)

### Access
- [V] Drupal 9 behind **Imperva/Incapsula** (`x-cdn: Imperva`, `visid_incap_*` cookies).
- [V] HTML pages return a 200 with an ~840-byte JS challenge page (`/_Incapsula_Resource?...`, `<META NAME="robots" CONTENT="noindex,nofollow">`). This happened for `/`, `/Normativa`, `/Normativa/Resoluciones-y-circulares-Superservicios`, `/Normativa/Compilacion-juridica-del-sector`, `/node/10014`, and `/jsonapi`.
- [V] Keeping cookies or using a browser-like UA did not help. WebFetch also got an empty page.
- [V] robots.txt is a standard Drupal file and does not restrict /Normativa.
- [V] **Not challenged:** `https://www.superservicios.gov.co/rss.xml`, and files under `/sites/default/files/...`. For example, `/sites/default/files/2026-03/Concepto-028-de-2026.pdf` returned 200 application/pdf.
- [V] One 404 page (`/sitemap.xml`) came back as full HTML and exposed the menu links `/Normativa/Resoluciones-y-circulares-Superservicios` and `/Normativa/Compilacion-juridica-del-sector`.
- [I] Backfill from the listing UI would need a headless browser (Playwright) that solves the Incapsula JS challenge. That is fragile and may conflict with the site's intent. A permission or contact request is advisable.

### Listing, metadata, and change detection
- [V] `rss.xml` is a site-wide feed with 10 items (news, normativa, nombramientos, banners) and no pager (`?page=50` returns the same items). It covered about one week (13–20 Oct 2026).
- [V] The `<description>` carries the full Drupal fields for normativa nodes: `field-sspd-normativa-tipo-norma` (Resolución/Circular…), `-numero-norm`, `-descripcion`, `-sp-categori` ("Servicio público / Categoría", e.g. Energía, Institucionales), `-tema`, `-estado`, `-fecha-publi`, `-anio`, `-adjunto` (PDF link).
- [V] Node IDs are sequential (node/10014 was the latest).
- [I] For change detection, poll rss.xml every 1–3 h and filter on tipo-norma, plus categoría = energía. This works without a browser, but if the gap between polls exceeds about 10 items, items will be missed.
- [V] Conceptos are PDFs at `/sites/default/files/YYYY-MM/Concepto-NNN-de-YYYY.pdf`, born-digital (MS Word). Concepto SSPD-OJ-2026-028 has 13 pages, with recipient redacted to "XXXXXXXXXX".
- [I] Alternative backfill: CREG's gestor normativo and CRA's normas compile SSPD acts of general application as HTML. Examples: `https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_superservicios_35615_2019.htm` and `.../circular_superservicios_0074_2021.htm`, both found by web search and not fetched here. These are probably the cheapest route for historic SSPD circulars and resolutions affecting energy.

### Samples
- https://www.superservicios.gov.co/sites/default/files/2026-03/Concepto-028-de-2026.pdf. Concepto on contributions/subsidies in electricity. Selectable text.
- https://www.superservicios.gov.co/system/files/2026-10/Circular-externa-20261000000794.pdf. Circular externa, linked from the RSS item. Not downloaded.

### Rating
- Difficulty: **hard** for backfill (WAF). **Easy-medium** for forward monitoring through RSS.
- Value: **medium**. Most SSPD output is individual enforcement or administrative acts (sanctions, nombramientos). General circulars and conceptos on energy billing, subsidies, and user rights are relevant.
- **Recommendation: not in the first release as a crawled source.** Optionally add a lightweight RSS watcher for new energy resoluciones and circulares. Source historic SSPD general acts through the CREG gestor normativo instead.

---

## 3. XM (www.xm.com.co)

### Access
- [V] Drupal and no WAF challenge (200 responses).
- [V] robots.txt disallows `/node/`, `/search/`, `/resultados-de-busqueda-general`, and some others. The Allow and Disallow rules leave content paths open.
- [V] Content is server-rendered and needs no JS.

### Where documents are
- [V] No central normative library. "Procedimientos" and "formatos-instructivos-y-procedimientos" are hand-edited landing pages scattered by area (transacciones, operación, generación, administración financiera).
- [V] Example: `https://www.xm.com.co/generación/informes-despacho/procedimientos-aplicables-al-despacho` holds a CKEditor `<ul>` with 5 links to `/sites/default/files/documents/*.pdf`, such as "Procedimiento Res CREG 101 083 de 2025" and "Modelo SAEB CREG 101 113 2026".
- [V] The links carry no dates or numbers beyond the link text.
- [V] `sitemap.xml` is an index of 2 pages. Page 1 alone has 5,000 URLs with `<lastmod>`: 3,803 /noticias, 709 /servicios-de-informacion, 146 /transacciones, and about 40 "formatos-instructivos-y-procedimientos…" or "procedimientos…" pages.
- [V] `https://www.xm.com.co/rss.xml` (10 items) carries bulletin articles. For example, "Conoce el Procedimiento de aplicación Resolución CREG 101 114 de 2026" (2026-09-29) and "Nuevos requisitos para la conexión y entrada en operación de proyectos solares".
- [V] PDFs have `Last-Modified` and `ETag` headers. "Procedimiento Res CREG 101 083 de 2025.pdf" is born-digital (Word) and its text extracts cleanly.

### Change detection
- [I] Poll the sitemap `lastmod` for the curated set of procedure pages, hash their link lists, and HEAD new PDFs.
- [I] Watch rss.xml and noticias for "Resolución CREG", "Acuerdo CNO", and "procedimiento".

### Samples
- https://www.xm.com.co/sites/default/files/documents/Procedimiento%20Res%20CREG%20101%20083%20de%202025.pdf
- https://www.xm.com.co/sites/default/files/documents/Modelo%20SAEB%20101%20113%202026.pdf (not downloaded)

### Rating
- Difficulty: **medium**. It is easy to fetch, but there is no structured index and the metadata is poor.
- Value: **medium**. These are implementation procedures for CREG rules and are useful context, but they are not themselves regulation.
- **Recommendation: defer.** Possibly add a curated watch-list of about 10–20 procedure pages and the RSS feed in v1.1.

---

## 4. CNO (www.cno.org.co)

### Access
- [V] Drupal and no WAF. robots.txt is a standard Drupal file that allows /acuerdos and /content.
- [V] Server-rendered. "Infinite scroll" is plain `?page=N` links (`<a href="?page=1" rel="next">`).

### Listings
- [V] `https://www.cno.org.co/acuerdos` has 9 items per page, newest first.
  - Page 0 starts at Acuerdo 2275 (16 Oct 2026). Page 80 is at Acuerdo 1553 (Apr 2022). Page 170 covers Acuerdos 742…734 (Apr–Mar 2015).
  - [I] About 172 pages back to 2015, so a backfill of about 175 listing requests plus about 1,500 detail pages.
- [V] Each card has: number + title ("Acuerdo 2274 Por el cual se aprueba…"), `<time datetime="2026-10-14T12:00:00Z">`, summary, detail link `/content/acuerdo-NNNN-…`, and **relationship fields** ("Sustituye: Acuerdo 2001…").
- [V] `https://www.cno.org.co/circular` is also server-rendered, but its first page starts at 2015 entries ("Circular 1 - Acuerdo CNO 722 de 2015"), so it appears to be sorted ascending.
- [V] The homepage has `/calendar-acuerdos`, `/calendar-circular`, `/consulta-documentos`, `/cno-documentos`, and cybersecurity variants.
- [V] `rss.xml` exists but is site-wide noise (10 items, meeting presentations, annexes).

### Detail pages and files
- [V] Recent acuerdos carry their **full text in HTML**, plus annexes on S3 (`https://cnostatic.s3.amazonaws.com/cno-public/archivosAdjuntos/Anexo_1_Acuerdo_2274.pdf`, `.xlsx`).
- [V] Older acuerdos link to `https://cnostatic.s3.amazonaws.com/cno-public/documentos/acuerdos/Acuerdo742.pdf` and `.doc`.
- [V] S3 returns ETag and Last-Modified.
- [V] The 2015 PDF is a Canon scan with an embedded OCR layer of noisy quality ("2Ot5", "Ley I43 de |gg4"). [I] Re-OCR or use the .doc where one is provided.

### Change detection
- [I] Poll `/acuerdos` page 0 (and `/circular` with descending sort or its last page) every few hours, and diff on the acuerdo number.

### Samples
- https://www.cno.org.co/content/acuerdo-2274-por-el-cual-se-aprueba-el-protocolo-para-las-pruebas-y-aplicacion-del. Full text in HTML plus 3 annexes.
- https://cnostatic.s3.amazonaws.com/cno-public/documentos/acuerdos/Acuerdo742.pdf. EDAC 2015, scanned with OCR.

### Rating
- Difficulty: **easy**.
- Value: **high** for technical and operational rules: connection requirements for solar and wind, protections, tests, and EDAC. The "Sustituye" field gives supersession data for free.
- **Recommendation: include in the first release** (acuerdos; circulares second).

---

## Summary table

| Source | Backfill since 2015 | New-doc detection | WAF/JS | Difficulty | Value | v1? |
|---|---|---|---|---|---|---|
| CNO | `?page=N` HTML listing, ~172 pages | listing page 0 | none | easy | high | **yes** |
| UPME | Biblioteca Jurídica GET filters (2020+ by sector; pre-2020 untagged, scanned) | Biblioteca page 1 | none | easy/medium | medium-high | **yes** (2020+) |
| XM | no index; sitemap + curated pages | sitemap lastmod + RSS | none | medium | medium | defer |
| Superservicios | blocked (Incapsula); use CREG gestor normativo | rss.xml (10 items, full fields) | Imperva JS challenge | hard | medium | no (RSS watcher optional) |
