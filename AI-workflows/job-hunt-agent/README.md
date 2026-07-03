# job-hunt-agent

An agentic pipeline that reads Emily Marie Williams' `vault-Resume` Obsidian
vault (a decomposed resume/skills/cover-letter knowledge base), scores a job
posting against it, and assembles a draft resume + cover letter for human
review — plus a local tracker for real application history.

> **Human review required.** Every draft this tool produces is exactly that —
> a draft. Nothing here auto-submits an application, and nothing here writes
> back into the vault. See "Design decisions" below for why.

---

## How it works

```
Phase 1 — Vault parsing  [deterministic, no LLM]

  vault-Resume/  ──►  load_vault()  ──►  VaultSnapshot
  (read-only input)                      (variants, experience bullets, skills,
                                           cover-letter building blocks, ...)

Phase 2 — Keyword prefilter  [deterministic, no LLM, no network]

  job posting text  ──►  prefilter_candidates()  ──►  top-N candidate skills/bullets
                          (plain token-overlap ranking, tracks "available but
                           not yet used" content separately from "used" content)

Phase 3 — Matching  [single LLM call]

  VaultSnapshot + candidates + posting  ──►  one instructor/litellm call  ──►  JobMatchLLMOutput
                                              (scores all 3 variants comparatively,
                                               recommends variant + cover-letter
                                               register, surfaces unused content)

Phase 4 — Draft assembly  [deterministic, no LLM]

  JobMatchResult + VaultSnapshot  ──►  assemble_draft_resume()          ──►  output/drafts/{slug}/resume.md
                                   ──►  assemble_draft_cover_letter()   ──►  output/drafts/{slug}/cover_letter.md
                                        (guardrails.py scans every draft for the
                                         vault's standing content rules before writing)

Phase 5 — Tracking  [local, file-based]

  ApplicationRecord  ──►  ApplicationStore  ──►  output/tracker/applications.json
```

---

## Design decisions

This project deliberately mirrors the architecture of its sibling project,
[`../strategic-reports/`](../strategic-reports/) — same `litellm` +
`instructor` + `tenacity` LLM client pattern, same Pydantic structured-output
models, same async error-isolation philosophy, same test-mocking technique.
A few things are different here, on purpose:

### The vault is read-only input, full stop

`vault-Resume` was built and reviewed across several earlier sessions and is
considered feature-complete. This project never writes into it — not the
parsed content, not a "sent" application, nothing. The vault's own
`used_for_applications` frontmatter field exists for exactly that kind of
status tracking, but it stays manually maintained by Emily if she chooses to
touch it. `vault_reader.py` only ever reads.

### One LLM call per posting, not three

Scoring all three resume variants (`data-science`, `bioinformatics`,
`ai-engineering`) happens in a **single comparative call**, not three
independent ones — this produces better-calibrated relative scores and costs
a third as much. A deterministic keyword prefilter (`matcher.py`) shrinks the
vault down to a digest before that call, since the full vault easily exceeds
a comfortable prompt size.

### Surfacing unused content is the point, not a side effect

Every `Skills/*.md` file in the vault already tracks which keywords are
"used in current resumes" vs. "available, not yet used" — real, evidenced
content that's never made it into any resume. The matching prompt is built
specifically to surface that unused content when it's relevant to a posting,
not just to re-confirm what's already in the recommended variant. Surfaced
content is always marked `<!-- NEW: surfaced by matcher... -->` in a draft
resume — never silently indistinguishable from vetted content. The dedup
check also catches an acronym/expansion pair when only one form is already
present ("LLM" already listed means "Large Language Models" is the same
real duplicate, not new) — see `_already_present()` in `assembler.py`.

### Guardrails run on every assembled draft

The vault has standing content rules (never reintroduce "FDA"; never claim
CrewAI/RAG/Finetuning/Hugging Face as real skills until they actually are).
`guardrails.py` scans every assembled draft for these, case-insensitively,
with word-boundary matching (a naive substring check on a short acronym like
"RAG" false-positives inside ordinary words like "paragraph" — this was a
real bug caught during manual verification, see `tests/test_guardrails.py`).
A hit never silently strips text — it's surfaced as a warning so a human sees
exactly what tripped the check.

### Experience renders reverse-chronologically, not alphabetically

`vault_reader.py` loads `Resumes/experience/*.md` via `sorted(glob(...))` —
alphabetical by filename, purely for deterministic iteration. `assembler.py`
originally rendered the Experience section in that same order with no
date-based sort on top, which happened to look almost right in the real
vault only because the most recent employer's filename also starts with
"A" — a real bug, caught by noticing Best Buy Health (ends 2021) rendering
before Epic Sciences (ends 2023), which is backwards. `employer_bullets` is
now explicitly sorted by a best-effort `(year, month)` key parsed from each
employer's free-text `dates` field before rendering.

### The vault's own naming conventions are read as structured signals, not just prose

`CoverLetters/building-blocks/soft-skills-and-work-ethic.md` names its
blocks `-prose` (complete, safe to render standalone) or `-fragment` (a
phrase meant to be worked into other prose, per that file's own usage
guidance) — a real, already-existing convention the code didn't act on
until a real bug surfaced it: the LLM recommended
`stakeholder-collaboration-fragment` as the soft skill for a real draft, and
the template rendered its raw text as its own paragraph, producing a
dangling "..." sentence.
`BuildingBlock.is_fragment` (derived from the `-fragment` suffix) now
routes those blocks to a `<!-- NOTE -->` comment plus a CLI warning instead
of body prose, so the fragment text is preserved but never presented as
finished.

### `*-filled.md` siblings, not in-place edits, for the human-review pass

`resume.md`/`cover_letter.md` are the pristine, reproducible output of
`draft`/`match-and-draft` — meant to be safe to regenerate any time the
match or the code changes (which happened three times in one real session
building this project). But the actual per-application work — integrating
surfaced skills/bullets, writing the company-specific paragraph, filling in
closing-line specifics — has to be written fresh, per the vault's own
rules, and there was nowhere safe to put it without risking it being
clobbered by a later re-draft. `init-filled` creates `resume-filled.md`/
`cover_letter-filled.md` as copies, never overwriting an existing one
without `--force`; `match-and-draft` calls it automatically. `diff-filled`
writes `resume_filled_diff.md`/`cover_letter_filled_diff.md` — a real
unified diff, not just a suggestion to run one by hand — showing exactly
what a given human-review pass changed.

### Convenience commands that inform the human pass, never write it

`--ghost-score`/`--ghost-reasons` (job-radar's ghost-risk score, if you have
it) carries the same way `--url` does — into `match.json` and both drafts'
metadata comment, so you see the risk signal before investing editing time,
not after. `company-brief` does a best-effort extraction of a posting's
"About &lt;Company&gt;" blurb to save re-reading the whole raw posting when
writing the company-specific paragraph. `diff-filled` writes a real unified
diff between each pristine draft and its `-filled.md` sibling. None of these
write anything into a draft itself — the company-specific paragraph and
closing line still have to be written fresh, per the vault's own rules;
these just put the raw material, and a record of what changed, one command
away instead of a full re-read or a by-hand `diff`.

`match-and-draft` runs `init-filled`, `diff-filled`, and `company-brief`
automatically after every draft (`--no-init-filled`/`--no-diff-filled`/
`--no-company-brief` to skip any of them) — a `diff-filled` run immediately
after generation will just say "no differences," since nothing's been
edited yet, but it stays useful: re-running `match-and-draft` later (say,
after a code fix, which happened more than once building this project)
refreshes the diff to reflect whatever you've actually changed by then.

### Markdown until the very last step

Every draft artifact in this pipeline — `resume.md`, `*-filled.md`, diffs,
`posting.txt` — stays plain markdown, since that's what's easy to diff,
template, and hand-edit. `to-docx` is the one command that leaves that
world: a thin wrapper over the system `pandoc` binary (not a Python
dependency — nothing else here needs a document-conversion library), run
by hand once editing is actually finished, against whatever file the human
review pass produced (a `-filled.md`, a differently-named hand-edited copy,
or the pristine draft). It doesn't guess which file is "done" — you name it
explicitly with `--file`, repeatable for both documents in one call. Uses
pandoc's default `.docx` styling; reformatting to match a specific
company's template is a by-hand step in Word after that, not something this
tool tries to solve.

### No job-board scraping in v1

Input is a job posting's text — pasted, or a local file. This is
deliberately not a job-board polling/scraping tool. That's a plausible future
phase, not something designed in as if already decided.

### No Prefect in v1

`strategic-reports` uses Prefect because it runs on a daily cron schedule.
Job postings here arrive one at a time, on demand — there's no recurring job
to orchestrate, and `tenacity` already handles retry at the LLM-call level.
Adopting Prefect now would mean running a Prefect server just to invoke a CLI
command. Revisit if a future phase adds real scheduled/batch work.

### Flat JSON tracker, not a database

`tracker.py`'s `ApplicationStore` is backed by a single pretty-printed JSON
file — consistent with `strategic-reports`' own precedent for growing,
date-queryable history (`bullet_history.json`, `urgency_history.json` are
flat JSON there too). The store starts empty; Emily explicitly chose not to
backfill the real, unstructured application history at
`~/Desktop/Employment/Job-Hunt/`.

---

## Quick start

```bash
cd job-hunt-agent
pip install -r requirements.txt   # or: uv pip install -r requirements.txt

# to-docx also needs pandoc on PATH (not a pip package):
#   apt install pandoc   /   brew install pandoc

# sanity-check the vault parser
python -m job_hunt_agent.cli load-vault

# find the "About <Company>" blurb, to speed up writing the
# company-specific paragraph (never auto-writes it — see Design decisions)
python -m job_hunt_agent.cli company-brief --posting posting.txt

# score a posting against the vault
python -m job_hunt_agent.cli match --posting posting.txt --company Acme --role "Data Scientist"

# ...with job-radar's ghost-risk score carried along too, if you have it
python -m job_hunt_agent.cli match --posting posting.txt --company Acme --role "Data Scientist" \
    --ghost-score 0.35 --ghost-reasons "posted 113 days ago (>90d)"

# assemble a draft resume + cover letter from that match
python -m job_hunt_agent.cli draft --match output/matches/acme-data-scientist-<date>/match.json

# or do both in one step — the common real-world path. Also creates
# resume-filled.md/cover_letter-filled.md automatically, the editable
# copies to do your actual human-review pass into (--no-init-filled to skip),
# and copies the posting itself in as posting.txt for easy side-by-side review
python -m job_hunt_agent.cli match-and-draft --posting posting.txt --company Acme --role "Data Scientist"

# re-create the *-filled.md copies standalone (e.g. after manually deleting
# one, or with --force to intentionally reset one back to the pristine draft)
python -m job_hunt_agent.cli init-filled --draft-dir output/drafts/acme-data-scientist-<date>

# see exactly what your human-review pass changed
python -m job_hunt_agent.cli diff-filled --draft-dir output/drafts/acme-data-scientist-<date>

# last step before sending: convert whatever you actually finished editing to Word (requires pandoc on PATH)
python -m job_hunt_agent.cli to-docx \
    --file output/drafts/acme-data-scientist-<date>/resume-filled.md \
    --file output/drafts/acme-data-scientist-<date>/cover_letter-filled.md

# track what you actually send
python -m job_hunt_agent.cli track add --company Acme --role "Data Scientist" \
    --variant data-science --register formal-professional
python -m job_hunt_agent.cli track update <id> --status applied
python -m job_hunt_agent.cli track list
```

`--posting -` reads the posting text from stdin instead of a file.

Postings don't have to be pasted by hand — see
[`../job-radar/`](../job-radar/) for pulling and ghost-risk-scoring postings
from ATS APIs, and its README's "Running the full pipeline from the command
line" section for chaining the two projects together with a script.

## Configuration

Same pattern as `strategic-reports`: every option is a CLI flag *and* an
environment variable.

| Env var | Purpose | Default |
|---|---|---|
| `LLM_MODEL` | litellm model string | `ollama_chat/llama3.1:70b` |
| `JOB_HUNT_AGENT_VAULT_PATH` | Path to `vault-Resume/` | `~/Desktop/vaults/vault-Resume` |
| `JOB_HUNT_AGENT_HOME` | Project root (for `output/`) | current directory |
| `JOB_HUNT_AGENT_TRACKER_PATH` | Path to the tracker JSON file | `$JOB_HUNT_AGENT_HOME/output/tracker/applications.json` |
| `OLLAMA_API_BASE` / `OLLAMA_API_KEY` | Hosted Ollama server | unset |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Enable Langfuse tracing | unset |
| `PHOENIX_TRACING` | Enable local Phoenix tracing UI | unset |

## Tracing

Same opt-in mechanism as `strategic-reports` (`core/tracing.py`, ported
unchanged) — set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` or
`PHOENIX_TRACING=true` and every LLM call is automatically traced. Neither
backend raises on misconfiguration; tracing is observability, not business
logic.

## Running the tests

```bash
pytest tests/
```

150 tests, all against a synthetic fixture vault built fresh under `tmp_path`
in `tests/conftest.py` — no test ever touches the real `vault-Resume`.
`asyncio_mode = auto` (pytest.ini), same as `strategic-reports`.

## Project structure

```
job-hunt-agent/
├── job_hunt_agent/
│   ├── cli.py                  <- Typer CLI: load-vault, company-brief, match, draft, init-filled, diff-filled, to-docx, match-and-draft, track *
│   └── core/
│       ├── llm_client.py       <- litellm + instructor client (ported from strategic-reports)
│       ├── tracing.py          <- Langfuse/Phoenix setup (ported from strategic-reports)
│       ├── models.py           <- pipeline models: JobPosting, JobMatchResult, ApplicationRecord, ...
│       ├── vault_models.py     <- typed representations of vault-Resume content
│       ├── vault_reader.py     <- markdown -> typed objects; load_vault() entry point
│       ├── prompts.py          <- system/user prompt builders for the single matching call
│       ├── matcher.py          <- deterministic prefilter + score_job()
│       ├── assembler.py        <- deterministic draft resume/cover-letter assembly
│       ├── guardrails.py       <- forbidden-term / excluded-skill scanning
│       ├── posting_utils.py    <- best-effort "About <Company>" extraction from raw posting text
│       └── tracker.py          <- ApplicationStore, flat-JSON backed
│   └── templates/               <- Jinja2 templates for the two draft documents
├── tests/                       <- 150 tests, fixture-vault-only, no real-vault or real-LLM calls
└── output/                      <- gitignored; matches/, drafts/, tracker/ generated at runtime
```

## Output

- `output/matches/{slug}/match.json` — the full `JobMatchResult` from a `match` run, re-loadable by `draft`.
- `output/drafts/{slug}/resume.md`, `cover_letter.md` — assembled drafts, always marked as needing human review. Pristine, reproducible output of `draft` — safe to regenerate any time the match or the code changes, since nothing gets hand-edited into these directly. If job-radar's ghost-risk score was passed in (`--ghost-score`/`--ghost-reasons`), it's in the metadata comment alongside the source URL.
- `output/drafts/{slug}/posting.txt` — a copy of the exact posting text the drafts were built from, written automatically by both `draft` and `match-and-draft`. No flag to skip it — it's a plain copy, not a generated artifact, so there's no meaningful "no" to opt into. Keeps the posting one file away during the human-review pass instead of a re-lookup of wherever the original `--posting` path (or the match.json it came from) happened to point.
- `output/drafts/{slug}/resume-filled.md`, `cover_letter-filled.md` — editable copies of the above, created automatically by `match-and-draft` (or explicitly via `init-filled`). This is where the actual human-review pass happens (surfaced-content integration, the company-specific paragraph, closing-line specifics), so re-running `draft`/`match-and-draft` never clobbers that work.
- `output/drafts/{slug}/resume_filled_diff.md`, `cover_letter_filled_diff.md` — a unified diff between each pristine draft and its `-filled.md` sibling, from `diff-filled`. Shows exactly what a given human-review pass actually changed; always safe to regenerate, since a diff is disposable.
- `output/drafts/{slug}/*.docx` — a Word version of whichever markdown file you point `to-docx` at, same stem, written next to it. Not created automatically by anything — a deliberate last step run by hand once editing is actually done.
- `output/tracker/applications.json` — the local application tracker.

None of this is committed (`output/` is gitignored) — it's real, potentially
company-specific application data, not portfolio content.

## FAQ

### What does "Surfaced skills not yet in this resume" mean, and how should I work with it?

It's real, evidenced content from your vault's `Skills/` files that's never
made it into any resume before — every skill file tracks a "used" list and
an "available, not yet used" list, and the matching step's whole job is to
check that unused list against the specific posting and pull out anything
genuinely relevant. It's kept in its own labeled section, separate from the
main Skills list, specifically so nothing surfaced ever looks pre-vetted
before you've actually looked at it (see "Surfacing unused content is the
point, not a side effect" above).

Working through it, item by item:

1. **Read the `why_relevant` note critically, not as given.** Sometimes
   it's a sharp, correct read on why a skill fits this posting; sometimes
   it's stretching a real skill to a tenuous connection. Keep it only if the
   connection actually holds up.
2. **Ask if you could defend it in an interview.** The vault's whole design
   principle is that provenance is mandatory — everything traces to
   something real you actually did. If a surfaced skill would make you
   hesitate under "tell me about a time you used that," it isn't ready yet.
3. **Check for soft duplicates the tooling won't catch.** The dedup logic
   catches exact-phrase duplicates already rendered elsewhere in the draft
   (see the `word_boundary_pattern` check in `assembler.py`), but not
   near-duplicates worded differently — e.g. "Large Language Models"
   surfacing as new when the summary already says "LLM." Worth a quick
   manual scan.
4. **If you keep one, move it — don't leave it in the surfaced section.**
   Fold it into the main Skills list wherever it fits, then delete it from
   "Surfaced skills." That section is a staging area, not somewhere content
   should live in a resume you actually send.
5. **Ignore the `(keyword) (file_title)` double-labeling if it looks
   redundant.** Known cosmetic quirk — the model sometimes fills
   `file_title` with the keyword itself instead of the real vault filename.
   Not a correctness problem, safe to drop when editing it in.

### What does "Surfaced bullets not yet in any resume variant" mean, and how should I work with it?

Same idea as surfaced skills, one level down — real experience bullets from
your vault's `Resumes/experience/` files, individually ID'd and tagged with
which resume variant(s) currently use them. The matcher surfaces bullets
relevant to the posting that aren't tagged as used in the *recommended*
variant specifically (the heading says "any," but the check is really "this
one" — a bullet already used in a different variant could still show up
here).

The dedup check also knows about retired bullets: some employer files keep
pre-trim "original" versions of a bullet around for reference, explicitly
marked `**Used in:** none (retired — superseded by \`newer-bullet-id\`)`.
Those get cross-checked against the bullet they were superseded by — if
that replacement is already in the recommended variant, the retired
original is dropped too, not surfaced as if it were new (see
`superseded_by` in `vault_models.py` and the check in `assembler.py`).

What that check *doesn't* catch: two bullets that happen to describe
similar work without an explicit "superseded by" relationship recorded in
the vault. If a surfaced bullet reads like something you're pretty sure is
already in the Experience section above it, trust that instinct and check
by hand — the same "ground truth from the vault, not from the LLM's claim"
principle applies, but it only works where the vault actually records the
relationship.
