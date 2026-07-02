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

python -m job_radar.cli show output/postings/<date>/<slug>.json

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

## Finding a company's ATS slug

Check their careers page URL:
- Greenhouse: `job-boards.greenhouse.io/<slug>` or `boards.greenhouse.io/<slug>`
- Lever: `jobs.lever.co/<slug>`
- Ashby: `jobs.ashbyhq.com/<slug>`

## Running the tests

```bash
pytest tests/
```

39 tests, no network access — every ATS response is mocked at the same
call-site granularity `strategic-reports/tests/test_ingestion.py` documents
for `feedparser.parse` (patch where the name is looked up, not where it's
defined).

## Project structure

```
job-radar/
├── job_radar/
│   ├── cli.py                  <- Typer CLI: pull, list, show
│   └── core/
│       ├── models.py           <- CompanyConfig, RawPosting, GhostSignal, ScoredPosting
│       ├── ats_clients.py      <- Greenhouse/Lever/Ashby async fetchers, error-isolated
│       ├── ghost_scoring.py    <- deterministic heuristic scorer
│       ├── seen_store.py       <- SeenPostingStore, flat-JSON backed
│       └── source.py           <- load_companies() + pull_postings() orchestration
├── config/
│   └── companies.json          <- your target companies (placeholders shipped)
├── tests/                      <- 39 tests, all HTTP mocked, no real network
└── output/                     <- gitignored; postings/, seen_store.json generated at runtime
```

## Output

- `output/postings/{date}/{company}--{title}--{id}.json` — the full `ScoredPosting`.
- `output/postings/{date}/{company}--{title}--{id}.txt` — plain posting text, ready for `job-hunt-agent match --posting`.
- `output/seen_store.json` — cross-run history that powers the staleness signal.

None of this is committed (`output/` is gitignored) — it's real,
company-specific posting data pulled from live APIs, not portfolio content.
