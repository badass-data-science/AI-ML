# job-radar

Pulls job postings directly from ATS platforms' public job-board JSON APIs
(Greenhouse, Lever, Ashby), scores each one for ghost-job risk with
deterministic heuristics, and writes plain posting files ready to hand to
[`job-hunt-agent`](../job-hunt-agent/) unmodified.

> **This never filters anything for you.** Every posting is written out,
> always. Ghost-risk is a score + reasons attached to each posting, not a
> silent exclusion — see "Design decisions" below.

---

## How it works

```
Phase 1 — Load company config   [deterministic]
  config/companies.json ──► load_companies() ──► list[CompanyConfig]

Phase 2 — Fetch postings        [async HTTP, no LLM]
  CompanyConfig ──► ats_clients.fetch_{greenhouse,lever,ashby}() ──► list[RawPosting]
  (concurrent via asyncio.gather, per-company error isolation)

Phase 3 — Ghost-risk scoring    [deterministic, no LLM]
  RawPosting + SeenPostingStore history ──► score_posting() ──► GhostSignal
  (age, repost/staleness across runs, evergreen-language regex)

Phase 4 — Persist + hand off    [local, file-based]
  ScoredPosting ──► SeenPostingStore (output/seen_store.json)
                ──► output/postings/{date}/{company}--{title}--{id}.json (full record)
                ──► output/postings/{date}/{company}--{title}--{id}.txt  (raw text —
                     directly consumable by job-hunt-agent's `match --posting`)
```

---

## Design decisions

This project is a sibling to
[`../job-hunt-agent/`](../job-hunt-agent/) and
[`../strategic-reports/`](../strategic-reports/) — same repo-layout
conventions (`core/`, flat-JSON stores, Typer CLI, `pytest` + `asyncio_mode =
auto`) — but diverges from both in a few deliberate ways.

### ATS APIs, not scraping or aggregators

Input comes from each ATS's own public job-board JSON API — the same data
their hosted careers pages render from, no authentication or HTML scraping
involved. This was chosen over polling LinkedIn/Indeed-style aggregators for
two reasons: it's far more ToS-friendly (a company's own public API vs.
scraping a third-party site's HTML), and these are companies' real internal
req feeds, not degraded copies — which is also what makes ghost-job scoring
possible at all, since each API returns real posting/update timestamps.
Coverage is limited to companies you explicitly configure in
`config/companies.json`, not the open web — a deliberate tradeoff for
reliability over breadth.

### Ghost-job risk is flagged, never silently filtered

Every scored posting keeps its `GhostSignal{score, reasons}` — nothing is
dropped from `pull`'s output. `list` has an opt-in `--max-ghost-score` flag
for when you explicitly want a shorter list, but the default behavior shows
everything, sorted safest-first. This mirrors `job-hunt-agent`'s
`guardrails.py` philosophy: a heuristic can be wrong, so it surfaces
evidence for a human to weigh rather than making the call itself.

### Ghost-risk scoring is deterministic, not an LLM call

Unlike both sibling projects, job-radar makes **zero** LLM calls — no
`litellm`, `instructor`, or `tenacity` dependency at all. Every signal here
is structured (a date, a repeat count, a regex match against a short list of
"evergreen hiring" phrases), so an LLM call would add cost and latency
without improving on plain rules. Weights are coarse tiers, not a fitted
model — there's no labeled ghost-job dataset to fit against, and coarse,
explainable thresholds (see `core/ghost_scoring.py`) are easier to trust and
hand-tune than an opaque score.

### The seen-store is the strongest signal, not the ATS's own dates

`core/seen_store.py` is a flat JSON record of every posting job-radar has
ever pulled, keyed by `{ats}:{slug}:{external_id}`, tracking `first_seen_at`
and `seen_count` across runs — directly parallel to `job-hunt-agent`'s
`tracker.py::ApplicationStore`. An ATS's own posted-date field can be wrong,
backdated, or (Greenhouse's job-list endpoint specifically) simply absent.
But if job-radar has independently observed the same posting still listed
across multiple `pull` runs spanning weeks, that's first-party evidence a
single scraped date can't fake. This is also why the staleness signal only
gets more accurate the more often you run `pull` — a single run has much
weaker signal than a history of them.

### No scheduling / no Prefect

`pull` is on-demand only, run manually when you're ready to search — same
reasoning `job-hunt-agent` already documented for itself (no recurring
orchestration needed for a manually-triggered command). Revisit if this ever
needs to run unattended on a schedule; that's a real architectural change
(needs Prefect or cron), not something designed in speculatively now.

### Independent from job-hunt-agent — connected only by plain files

job-radar never imports `job_hunt_agent`, and vice versa. `pull` writes a
`.txt` file per posting that's already in the exact shape `job-hunt-agent`'s
`match --posting <file>` expects — no glue code, no shared schema coupling.
Keeping the boundary at the filesystem, not a Python import, means either
project's tests stay fully independent and either could be swapped out
without touching the other.

---

## Quick start

```bash
cd job-radar
pip install -r requirements.txt

# edit config/companies.json with your real target companies first —
# the shipped file only has placeholder examples

python -m job_radar.cli pull

python -m job_radar.cli list
python -m job_radar.cli list --max-ghost-score 0.3   # hide riskier postings
python -m job_radar.cli list --company Acme
python -m job_radar.cli list --title-contains "Data Scientist"
python -m job_radar.cli list --location-contains Remote
python -m job_radar.cli list --title-contains "ML Engineer" --location-contains Austin

python -m job_radar.cli show output/postings/<date>/<slug>.json

# find a company's ATS slug automatically instead of guessing by hand
python -m job_radar.cli discover-slug "Acadia Pharmaceuticals"
python -m job_radar.cli discover-slug "Acadia Pharmaceuticals" --add

# hand a pulled posting straight to job-hunt-agent
python -m job_hunt_agent.cli match \
    --posting job-radar/output/postings/<date>/<slug>.txt \
    --company Acme --role "Data Scientist"
```

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `JOB_RADAR_HOME` | Project root (for `output/`) | current directory |
| `JOB_RADAR_COMPANIES_PATH` | Path to `companies.json` | `$JOB_RADAR_HOME/config/companies.json` |
| `JOB_RADAR_SEEN_STORE_PATH` | Path to the seen-posting store | `$JOB_RADAR_HOME/output/seen_store.json` |

## Running the full pipeline from the command line (no Claude Code)

Everything in this project runs as a plain CLI — nothing here depends on
Claude Code to orchestrate. This section is the complete, standalone
sequence: one-time setup, then either the manual step-by-step commands or
the bundled convenience script.

### One-time setup

Both projects need their own virtualenv — they're independent (see "Design
decisions" above), so there are two separate installs, not one:

```bash
cd AI-workflows/job-radar
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd ../job-hunt-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`job-hunt-agent` also needs an LLM configured — it's the only one of the two
that makes LLM calls. Same env vars either project's README documents
(`LLM_MODEL` plus whatever your provider needs, e.g. `OLLAMA_API_BASE` /
`OLLAMA_API_KEY` for a hosted Ollama endpoint):

```bash
export LLM_MODEL="ollama_chat/gpt-oss:120b"
export OLLAMA_API_BASE="https://your-ollama-host"
export OLLAMA_API_KEY="..."
```

### Manual step-by-step chaining

This is what the convenience script below automates — worth knowing by hand
since it's the same three commands regardless of platform, and makes it
obvious where to customize (different filters, skip the draft step, run
`match` instead of `match-and-draft`, etc.):

```bash
# 1. Pull fresh postings (job-radar)
cd AI-workflows/job-radar
.venv/bin/python -m job_radar.cli pull

# 2. Pick postings to act on — --paths prints .txt file paths instead of
#    the table, one per line, so a shell loop can consume them directly
.venv/bin/python -m job_radar.cli list --location-contains "San Diego" --paths
#   /path/to/job-radar/output/postings/2026-07-02/acadia-pharmaceuticals--...txt
#   /path/to/job-radar/output/postings/2026-07-02/fate-therapeutics--...txt

# 3. Hand each one to job-hunt-agent (job-hunt-agent) — company/role/url come
#    from the sibling .json file's posting.company / posting.title / posting.url
#    fields, and the ghost score/reasons from its top-level ghost field.
#    --url threads the original apply link into match.json and both drafts'
#    metadata comment; --ghost-score/--ghost-reasons do the same for the
#    ghost-risk signal, so you see it before investing editing time, not after.
cd ../job-hunt-agent
.venv/bin/python -m job_hunt_agent.cli match-and-draft \
    --posting /path/to/job-radar/output/postings/2026-07-02/acadia-pharmaceuticals--....txt \
    --company "Acadia Pharmaceuticals" \
    --role "Associate Director, AI/ML Engineering" \
    --url "https://acadia.com/en-us/careers/job-board/8565787002?gh_jid=8565787002" \
    --ghost-score 0.0 --ghost-reasons ""
```

Looping over several matched postings without the convenience script:

```bash
cd AI-workflows/job-radar
for txt in $(.venv/bin/python -m job_radar.cli list --location-contains Remote --paths); do
    json="${txt%.txt}.json"
    company=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['posting']['company'])" "$json")
    role=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['posting']['title'])" "$json")
    url=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['posting']['url'])" "$json")
    ghost_score=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['ghost']['score'])" "$json")
    ghost_reasons=$(.venv/bin/python -c "import json,sys; print('; '.join(json.load(open(sys.argv[1]))['ghost']['reasons']))" "$json")
    ( cd ../job-hunt-agent && .venv/bin/python -m job_hunt_agent.cli match-and-draft \
        --posting "$txt" --company "$company" --role "$role" --url "$url" \
        --ghost-score "$ghost_score" --ghost-reasons "$ghost_reasons" )
done
```

### `scripts/run_pipeline.sh` — the same thing, bundled

```bash
scripts/run_pipeline.sh --location-contains "San Diego"
scripts/run_pipeline.sh --title-contains "Data Scientist" --location-contains Remote
scripts/run_pipeline.sh --company "Acadia Pharmaceuticals" --yes
```

Runs `pull`, shows what matched your filters, asks for confirmation (each
posting is a real LLM call — costs time and, depending on your provider,
money), then runs `match-and-draft` against every matched posting. Every
flag except `--yes` passes straight through to `job_radar.cli list`, so any
combination of `--company`/`--title-contains`/`--location-contains`/
`--max-ghost-score` works. `--yes` skips the confirmation prompt, for when
you've already reviewed the list and want to automate the rest (e.g. from
cron or another script — though see "No scheduling / no Prefect" above for
why job-radar itself doesn't do that scheduling for you).

`match-and-draft` itself does more than just write `resume.md`/
`cover_letter.md` — by default it also creates the editable `-filled.md`
copies to do your human-review pass into, writes a diff between each draft
and its filled copy (trivial on a first run, genuinely useful once you've
edited), prints the posting's best-effort "About &lt;Company&gt;" blurb
for writing the company-specific paragraph, and copies the posting text
itself into the same drafts directory as `posting.txt` (no flag for this
last one — it's a plain copy, not a generated artifact, so there's nothing
to opt out of). See job-hunt-agent's README ("Convenience commands that
inform the human pass, never write it") for the `--no-*` flags on the rest.

The script is pure shell calling each project's CLI as a subprocess — it
doesn't import either project's Python code, keeping the same independence
the two projects have from each other.

## Finding companies to add — including localized to you or remote

There is deliberately no company-discovery/search feature here (see "ATS
APIs, not scraping or aggregators" above) — none of Greenhouse/Lever/Ashby
expose a public "search postings across all companies by location" endpoint,
only per-company job lists, so there's no API-level way to ask "which
companies near me use Greenhouse." Finding candidate companies stays a manual
step, same as picking which RSS feeds go in `strategic-reports`:

1. Use a location- or remote-filtered search on a general job aggregator
   (LinkedIn Jobs, Indeed, Google for Jobs, etc.) to find companies actively
   hiring in your area or fully remote — you're using the aggregator for
   company *discovery* only, not as job-radar's data source.
2. For each company you're interested in, check whether their careers page
   is hosted on one of the three supported ATS platforms and grab the slug
   from the URL:
   - Greenhouse: `job-boards.greenhouse.io/<slug>` or `boards.greenhouse.io/<slug>`
   - Lever: `jobs.lever.co/<slug>`
   - Ashby: `jobs.ashbyhq.com/<slug>`
3. Add `{"name": ..., "ats": ..., "slug": ...}` to `config/companies.json` —
   or use `discover-slug` below to automate step 2 once you already know
   which company you want.

## Automatically finding a company's slug

Once you know *which* company you want (from the manual discovery process
above, or just because you already know its name), `discover-slug` automates
the guess-and-verify step — the same process used by hand to seed
`config/companies.json`'s starter list:

```bash
python -m job_radar.cli discover-slug "Acadia Pharmaceuticals"
#   greenhouse acadiapharmaceuticals          53 jobs

python -m job_radar.cli discover-slug "Acadia Pharmaceuticals" --add
#   Added Acadia Pharmaceuticals (greenhouse:acadiapharmaceuticals) to config/companies.json
```

It generates a handful of plausible slug spellings from the name (the full
name concatenated, hyphenated, and just the first word — e.g. "Acadia
Pharmaceuticals" → `acadiapharmaceuticals`, `acadia-pharmaceuticals`,
`acadia`) and checks each one against the real Greenhouse/Lever/Ashby APIs
concurrently. This is **not** company search — it can only confirm or reject
a slug for a company name you already have in mind, not discover companies
you haven't heard of; that part stays the manual aggregator-search process
above, by the same "no scraping" design decision.

Only slugs that come back with actual live postings are reported, but a
short or generic company name can still coincidentally collide with an
unrelated real board (a two-person startup that happens to also be called
"Acme"), so review `job_count` and the company name before trusting a match,
especially when `--add` is involved:

- `--add` only writes to `config/companies.json` when there's **exactly one**
  confirmed match — it never guesses which of several is the real company.
- If there are multiple matches, re-run with `--pick '<ats>:<slug>'` (shown
  in the plain listing) to explicitly choose one.
- Re-running `--add` for an already-added `(ats, slug)` pair is a safe no-op,
  not a duplicate entry.
- Existing `_comment`/`_note` fields and other entries in `companies.json`
  are preserved untouched — only a new entry is appended.

## Narrowing to a location or profession

`pull` always fetches and writes out every posting a configured company has
listed — narrowing happens at `list` time, not by asking the ATS for a
subset, since query-param support for location/department filters is
inconsistent across the three platforms and it's more reliable to always
pull everything once and filter client-side as many times as you want:

```bash
python -m job_radar.cli list --title-contains "Data Scientist"
python -m job_radar.cli list --location-contains Remote
python -m job_radar.cli list --location-contains Austin
```

Both are case-insensitive substring matches, combinable with each other and
with `--company`/`--max-ghost-score`. `--location-contains` matches whatever
string the ATS itself reports for that posting — wording varies by company
(`"Remote"`, `"Remote - US"`, `"Remote (US)"`, or just a bare city name with
no remote tag at all), so it's worth trying a few terms rather than assuming
one canonical spelling.

When your `--location-contains` term itself contains "remote" (matched
case-insensitively, so `Remote`/`remote`/`REMOTE` all count), results are
further limited to US-remote postings: any posting whose location names a
non-US country/region and no US one (`"Remote - Australia"`,
`"France (Remote)"`, `"Canada - Remote"`) is excluded. A posting that lists
several regions, including a US-remote option, still passes (e.g.
`"London, UK; Remote-Friendly, United States; San Francisco, CA"`) — only
listings that are exclusively non-US get dropped.

## Running the tests

```bash
pytest tests/
```

59 tests, no network access — every ATS response is mocked at the same
call-site granularity `strategic-reports/tests/test_ingestion.py` documents
for `feedparser.parse` (patch where the name is looked up, not where it's
defined).

## Project structure

```
job-radar/
├── job_radar/
│   ├── cli.py                  <- Typer CLI: pull, list, show, discover-slug
│   └── core/
│       ├── models.py           <- CompanyConfig, RawPosting, GhostSignal, ScoredPosting, DiscoveredSlug
│       ├── ats_clients.py      <- Greenhouse/Lever/Ashby async fetchers, error-isolated
│       ├── ghost_scoring.py    <- deterministic heuristic scorer
│       ├── seen_store.py       <- SeenPostingStore, flat-JSON backed
│       ├── slug_discovery.py   <- candidate slug generation + live-probe against ATS APIs
│       └── source.py           <- load_companies() + pull_postings() orchestration
├── config/
│   └── companies.json          <- your target companies (placeholders shipped)
├── scripts/
│   └── run_pipeline.sh         <- chains job-radar + job-hunt-agent from the shell
├── tests/                      <- 59 tests, all HTTP mocked, no real network
└── output/                     <- gitignored; postings/, seen_store.json generated at runtime
```

## Output

- `output/postings/{date}/{company}--{title}--{id}.json` — the full `ScoredPosting`.
- `output/postings/{date}/{company}--{title}--{id}.txt` — plain posting text, ready for `job-hunt-agent match --posting`.
- `output/seen_store.json` — cross-run history that powers the staleness signal.

None of this is committed (`output/` is gitignored) — it's real,
company-specific posting data pulled from live APIs, not portfolio content.
