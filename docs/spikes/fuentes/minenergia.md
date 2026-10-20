# MinEnergía crawl feasibility spike (2026-10-20)

Raw samples: `spike/minenergia/raw/`, PDFs: `spike/minenergia/pdfs/`, scripts: `spike/minenergia/scripts/`,
Wagtail page dump: `spike/minenergia/all_pages.json`, all draft forums: `spike/minenergia/foros_all.json`.

## TL;DR

MinEnergía runs two separate systems, and the crawler needs both:

| What | Host | Tech | Access |
|---|---|---|---|
| Issued norms (resoluciones, decretos, circulares...) | `normativame.minenergia.gov.co` | Nexura PHP normograma (server-rendered HTML, `X-Powered-By: PHP/7.0.33`, `Server: Nexura/7.3`) | GET form with filters, 10 rows/page, `genPag=N` |
| Draft regulations for comments ("Foros") | `www.minenergia.gov.co` | Wagtail/Django CMS | **Open Wagtail API v2** (`/api/v2/pages/`) returns clean JSON |

No JS rendering, captcha or Cloudflare was seen. Plain HTTP and HTML/JSON parsing is enough, so no headless browser is needed.

---

## 1. robots.txt, WAF and rate limiting

**Verified**
- `https://www.minenergia.gov.co/robots.txt` returns **404** (no robots.txt). There is no sitemap either: `/sitemap.xml` returns 404.
- `https://normativame.minenergia.gov.co/robots.txt` returns 200:
  ```
  User-agent: *
  Sitemap: https://normativame.minenergia.gov.co/sitemap.xml
  Disallow: /ErrorPages
  Disallow: /backup
  Disallow: /info
  Allow: */media
  Allow: */img
  ```
  The sitemap it points to is an **empty** `<sitemapindex/>`.
- Both hosts set a `cookiesession1=...` cookie. The format looks like a Fortinet FortiWeb/FortiADC load balancer or WAF (that is an inference). Neither host showed a captcha or JS challenge. Every HTML page is fully server-rendered.
- **Transient blocking seen.** During a burst of sequential requests (about 1 per 1.2 s; see the note below), both hosts briefly failed with `curl: (35) Send failure: Broken pipe` and Python `URLError [Errno 51] Network is unreachable`. Requests worked again after about 20 s. This points to connection-level throttling at the WAF or load balancer, with no HTTP 403 or 429 sent back.
- `www.minenergia.gov.co/api/elasticsearch/search/?q=...` returns **302**. The site search is not open to the public.

**Politeness note:** My first volume-count script had a bug. Nexura sends back **page 1** for any `genPag` past the last page, and the script treated that as "more pages" and kept doubling the page number. It sent about 190 extra requests to normativame, about 1.2 s apart, before I stopped it. That burst is probably what caused the throttling above. Lesson for the crawler: detect wrap-around by checking whether the first row equals page 1's first row.

## 2. How documents are listed

### 2a. Issued norms: Nexura normograma
- Advanced search page (the `www` home links to it from "Normatividad"):
  `https://normativame.minenergia.gov.co/loader.php?lServicio=Normatividad&lTipo=User&lFuncion=avanzado`
- GET parameters (from `<form id="form-busqueda-avanzada">`):
  - `id_tema` (1 Ambiental, **2 Energía**, 3 Gas, 4 General, 5 Hidrocarburos, 6 Minas, 15 Talento Humano, 19 Regulatorio, 21 Presupuesto, 24 Regalías)
  - `id_subtema` (loaded through `lFuncion=getSubTemas` with a token)
  - `id_sector` (1 Ambiental, **2 Energía**, 3 Gas, 4 General, 5 Hidrocarburos, 6 Minas, 7 Regalías)
  - `id_tiponorma` (1 Resolución, 2 Decreto, 3 Circular, 4 Acuerdo, 5 Ley, 6 Concepto, 7 Sentencia, 8 Proyecto de Ley, 9 Acto Legislativo, 10 Convenio, 11 Certificación, 12 Política)
  - `nor_estado` (1 Vigente, 2 Derogada), `id_entidad` (1 = MinEnergía), `nor_palabras` (keywords)
  - `vigencia_inicio` / `vigencia_fin` in `dd/mm/yyyy`
  - `genPag=N` for the page number
- Filters work as **plain URL params, with no token or session needed** (verified). Example:
  `...lFuncion=avanzado&id_sector=2&vigencia_inicio=01/01/2025&vigencia_fin=31/12/2025&genPag=2` returns 200 with 10 rows.
- Results table `#date_table` has the columns *Nombre estandar* (number, linked to detail), *Tipo de Norma*, *Vigencia* (date) and *Resumen* (epígrafe):
  ```html
  <td><a target="_blank" href="https://normativame.minenergia.gov.co/normatividad/7865/norma/">40562</a></td>
  <td> Resoluci&oacute;n </td><td> 15/10/2026</td><td> Por la cual se fija el precio base ... </td>
  ```
- Rows are sorted by date, newest first. There are 10 rows per page and no total count is shown. The pagination widget shows at most 10 page links.
- **Gotcha:** `genPag` past the last page silently returns page 1 again. Stop when a page has fewer than 10 rows, or when its first row repeats page 1's.
- There is no JSON, RSS or sitemap.

### 2b. Drafts for comment: Wagtail "Foros"
- Public JSON API: `https://www.minenergia.gov.co/api/v2/pages/?type=servicio_ciudadano.ForosPage&fields=start_date,end_date,sectors_list,tags,document_file,first_published_at&order=-first_published_at&limit=400&offset=0`
  - `limit` is capped at 400 (`{"message": "limit cannot be higher than 400"}`).
  - `total_count` is 1025 forums. Three requests fetched all of them.
  - `/api/v2/pages/` overall has 5,291 pages. The page types include `NoticiaPage` (2,950), `ForosPage` (1,025) and `ConceptoPage` (306).
  - You cannot filter by `sectors_list` in the query: it returns 400 "not a recognised field". Filter on the client instead. `last_published_at` is also not exposed.
- HTML index: `https://www.minenergia.gov.co/es/servicio-al-ciudadano/foros/` is a single 3.2 MB page with every forum rendered. The sector filter there runs client-side.
- `https://www.minenergia.gov.co/api/v2/documents/?order=-id` lists all 16,441 Wagtail documents with `download_url`. IDs increase, so this works as a global "new file" signal.

## 3. Detail structure

### Norm detail (Nexura): `https://normativame.minenergia.gov.co/normatividad/{id}/norma/`
- Labeled fields: `Tema`, `Subtema` (often "SIN DEFINIR"/"Por definir", sometimes useful such as "FSSRI"), `Entidad`, `Vigencia` (date), `Tipo`, `Nombre estandar` (number) and `Estado` (Vigente/Derogada). The parse pattern is `<strong>Tema:</strong></span> <span>Energía</span>`.
- The *title/epígrafe is only in the listing's "Resumen" column*, not on the detail page.
- There is a "Normatividad relacionada" table of links to other norm IDs. Its displayed dates are wrong: they repeat the current norm's date, which looks like a template bug.
- There is sometimes an "Anexos" section, with links like `loader.php?lServicio=Tools2&lTipo=descargas&lFuncion=descargar&idFile=8109`.
- The main PDF appears in two ways, and both work:
  - `<iframe src="public_html/info/minergia/media/tmp/<file>_<n>.pdf">`. This resolves to `https://normativame.minenergia.gov.co/info/minergia/media/tmp/...pdf`, which returns 200 with `application/pdf`, an ETag and a Last-Modified set to the request time. That makes it a temp copy, and the `/info` path is Disallowed in robots.
  - A preferred form, `https://normativame.minenergia.gov.co/loader.php?lServicio=Tools2&lTipo=viewpdf&id={fileId}`. It returns 200 with `application/pdf` and `Content-Disposition: inline; filename="..."`. The file ID is in the iframe `title`, sometimes inside an HTML comment. This is a separate ID from the norm ID.
- Detail responses send `Cache-Control: no-store` and no ETag or Last-Modified.
- A nonexistent ID (`/normatividad/9999/norma/`) returns **302**.

### PDF text quality (verified with pdftotext)
| Sample | Creator | Text? |
|---|---|---|
| 2016 Res 41269 (file 668) | Hewlett Packard MFP | **Scanned, no text layer**, so OCR is needed |
| 2021 Res 01480 (file 5946) | LibreOffice 7.2 | Native text, clean |
| 2025 Res 40632 (file 7798) | PDFsharp | Scanned **with** an OCR layer, noisy in the letterhead but readable |

Plan for an OCR fallback.

### Draft forum detail (Wagtail)
- API `GET /api/v2/pages/{id}/` fields: `title`, `start_date`, `end_date` (comment window), `tags`, `sectors_list`, `introduction` (legal boilerplate plus the draft's "Por la cual..." title), `city`, `author_list`, `document_file` (almost always null: 1 of 1,025), `meta.first_published_at` and `parent`.
- **Attachments are not in the API.** They are only in the HTML body, as Wagtail document links such as:
  `/documents/15648/1-Resoluci%C3%B3n_Plan_de_Expansi%C3%B3n_Transmisi%C3%B3n_2025-2039.pdf` (Proyecto de Resolución), `/documents/15649/...Memoria_Justificativa...pdf` and `/documents/15650/...xlsx` (the comment form). The comment deadline also appears in the prose ("hasta el día viernes 15 de mayo de 2026").
- Wagtail document PDFs return `Last-Modified`, a strong `ETag` and `Cache-Control: max-age=3600`.

## 4. Filtering to electricity only

- **Issued norms:** use `id_sector=2` (Energía) server-side. Hidrocarburos, Gas, Minas and Regalías are separate values. I verified that `id_sector=2` gives electricity content: FOES and ZNI subsidies, FSSRI, Plan de Expansión de Transmisión, PNER and similar. `Tema` on the detail page matched "Energía". I did not test `id_tema=2` against `id_sector=2` to see whether they return the same set.
- **Drafts:** filter client-side on `sectors_list` containing `"Energía"`. Values seen: Hidrocarburos 343, **Energía 308**, Minas 136, General 132, Regulatorio 51, Gas 34, Ambiental y Social 11, Regalías 9, and Hidrocarburos+Energía 1.
- **Inference / risk:** some electricity items are probably tagged General or Regulatorio, for example DUR 1073 changes or cross-sector decrees. "Energía" may also include non-grid energy topics. Add a keyword pass over title and summary for those buckets: eléctric, energía eléctrica, RETIE, RETILAP, transmisión, distribución, ZNI, FOES, FSSRI, FAZNI, FNCE, subsidio, comunidades energéticas, autogeneración, CREG, UPME, SIN, generación.

## 5. Drafts for comments

- Listing: `https://www.minenergia.gov.co/es/servicio-al-ciudadano/foros/` (HTML), or the API query in 2b.
- Fields: `start_date` (publication or comment-period start), `end_date` (comment deadline), `sectors_list`, `tags`, `title` and `introduction`. Attachments come from the HTML body (`/documents/{docId}/{name}`).
- History: there are records from 2008. Electricity drafts by `start_date` year are 2015:11, 2016:22, 2017:21, 2018:12, 2019:28, 2020:37, 2021:42, 2022:25, 2023:30, 2024:36, 2025:26, 2026 (to Oct):12.
- Before 2022, `first_published_at` is the CMS-migration date (2022-06/08). Use `start_date` as the real date.
- Detecting new drafts: page IDs grow over time (5312 May 2026, then 5587 Oct 2026). Poll `?type=servicio_ciudadano.ForosPage&order=-id&limit=20` and diff the IDs. The API sends no ETag or Last-Modified (only `Vary: Accept, Accept-Language, Cookie`).

## 6. Change-detection signals

| Signal | Normograma | Foros (Wagtail) |
|---|---|---|
| Increasing IDs | `/normatividad/{id}/norma/` IDs grow (7154–7600 for 2025, 7855–7867 for Oct 2026), but **are not in date order**: Res 40632 of 29/12/2025 has id 7445 while 7502 is 30/12/2025, so items are uploaded late or backfilled | page `id` grows; `first_published_at` |
| Dates | `Vigencia` (issue date) in listing | `start_date`, `end_date`, `first_published_at` |
| Ordering | listing sorted by date desc | `order=-id` / `-first_published_at` |
| HTTP validators | none (no-store, no ETag) | API none; `/documents/*` files have ETag + Last-Modified |
| Status changes | `Estado` Vigente→Derogada (needs re-crawl of detail) | not tracked |

## 7. Volume (electricity)

Verified counts from `id_sector=2` with a calendar-year `vigencia` range:
- **2016: 84**, **2021: 106**, **2025: 105** norms. These are mostly Resoluciones, plus some Decretos.
- **Inference:** the full catalog is about 90–110 per year, or roughly 1,000–1,200 for 2015–2026. Many are routine subsidy-transfer resolutions (FOES, ZNI, menores tarifas), which a product could down-rank.
- **Possible gaps:** the 2021 results end at 02/03/2021 and the 2025 results end at 20/01/2025, which is thin for January–February. Either the catalog is incomplete or some norms carry odd dates (for example, Decreto 1538 is listed under 20/01/2025).

Drafts: about 25–40 electricity forums per year (section 5).

## 8. Samples

| Period | Doc | Detail URL | File URL |
|---|---|---|---|
| 2016 | Res. 41269 of 29/12/2016 (distribución recursos menores tarifas sector eléctrico) | https://normativame.minenergia.gov.co/normatividad/1743/norma/ | https://normativame.minenergia.gov.co/loader.php?lServicio=Tools2&lTipo=viewpdf&id=668 (scanned) |
| 2021 | Res. 01480 of 31/12/2021 (corrige Res. 01285, subsidios ZNI solar) | https://normativame.minenergia.gov.co/normatividad/6232/norma/ | https://normativame.minenergia.gov.co/loader.php?lServicio=Tools2&lTipo=viewpdf&id=5946 (native text) |
| 2025 | Res. 40632 of 29/12/2025 (modifica Res. 40279/2021, Plan de Expansión) | https://normativame.minenergia.gov.co/normatividad/7445/norma/ | https://normativame.minenergia.gov.co/loader.php?lServicio=Tools2&lTipo=viewpdf&id=7798 (+ annex `...lFuncion=descargar&idFile=8109`) |
| 2026 draft | Proyecto: Adoptar el Plan de Expansión de Transmisión 2025-2039 (5–15 May 2026) | https://www.minenergia.gov.co/es/servicio-al-ciudadano/foros/adoptar-el-plan-de-expansion-de-transmision-2025-2039-05052026/ (API id 5312) | https://www.minenergia.gov.co/documents/15648/1-Resoluci%C3%B3n_Plan_de_Expansi%C3%B3n_Transmisi%C3%B3n_2025-2039.pdf |
| 2016 draft | RETIE (2016-02-03 → 2016-02-12) | https://www.minenergia.gov.co/es/servicio-al-ciudadano/foros/reglamento-t%C3%A9cnico-de-instalaciones-el%C3%A9ctricas-retie/ (id 2898) | (in HTML body) |

## 9. Recommended strategy

**Backfill**
1. Norms: for each year from 2015 to the current year, page through the normograma with `id_sector=2&vigencia_inicio=01/01/Y&vigencia_fin=31/12/Y&genPag=N`. Stop when a page has fewer than 10 rows or when page 1's first row repeats. That is about 11 pages per year, or about 130 listing requests in total. Take the number, type, date and title from the listing.
   - Fetch each detail page once (about 1,200) for Tema, Subtema, Estado, related norms, the viewpdf ID and annex `idFile`s. Then fetch the PDF through `viewpdf&id=`, not the `/info/.../tmp/` path, which robots.txt disallows.
   - Run the same per-year pass with `id_sector` set to General (4) and Regulatorio (19), and keyword-filter those results for electricity.
2. Drafts: three API calls get all forums. Filter `sectors_list ∋ Energía`, plus keyword matches in General/Regulatorio, with `start_date >= 2015-01-01`. Fetch each forum's HTML to collect its `/documents/` attachments (about 300 pages).
3. Throttle to at least 2 s per request with a single connection, run off-hours, and retry with backoff on connection resets. The WAF drops connections rather than sending 429. The total backfill is about 3,000 requests, roughly 2 hours at 2 s per request.

**Incremental (every 3–6 h)**
- Norms: fetch pages 1–3 of the unfiltered listing, or of `id_sector` 2, 4 and 19, with `vigencia_inicio` set to today minus 90 days. Diff the norm IDs. The window has to be wide because items show up late with older dates. About 3–9 requests.
- Weekly: re-check `Estado` for norms from the last N years, or re-crawl the last 12 months of listings to catch derogations, late uploads and changed files.
- Drafts: `GET /api/v2/pages/?type=servicio_ciudadano.ForosPage&order=-id&limit=20` and diff the IDs. While a forum's comment window (`start_date`..`end_date`) is still open, re-fetch its HTML daily to catch added attachments or new deadlines.
- Optional: poll `/api/v2/documents/?order=-id&limit=50` as a cheap trigger for any new file on www.

**Risks**
- Connection-level throttling by the WAF, with no explicit 429.
- `genPag` wrap-around: page 1 is returned forever.
- Nexura markup is fragile: commented-out iframes and a temp PDF path.
- Old PDFs are scanned, so OCR is needed.
- Sector tagging is incomplete or inconsistent, and the catalog may be missing documents.
- Upload delay or backdating breaks "new = newest date".
- Draft attachments are only in HTML, and forums dated before 2022 have migration-era `first_published_at`.
- The two systems are independent, and the normograma's listing drops the full title on the detail page.
- Normograma detail pages give no change validators, so a hash of the extracted fields is needed.
