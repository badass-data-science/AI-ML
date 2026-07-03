# Job Radar

## An ATS Pipeline That Scores Every Posting Before Our Heroine Wastes Her Time On It

Our heroine the data scientist's Ultimate Cunning Master Plan™ has a weak link, and it isn't the resume. The resume is solved — decomposed into a knowledge graph of bullets and skills and cover-letter blocks, cross-referenced, tagged with what's used where and why, ready for an agent to reason over. The weak link is upstream of all that: *which postings does she even bother running through the pipeline?*

LinkedIn and Indeed are, by most independent estimates, somewhere between "mostly noise" and "actively hostile" to a real job search. A meaningful share of what shows up there is evergreen — reposted, stale, "always hiring" listings that exist to farm resumes for a role that isn't really open, or isn't open the way the posting implies. Scrolling that firehose and manually judging which listings are real is its own full-time job, and it's exactly the kind of task that should be automatable: fetch, score, filter, done.

So she built it.

Job Radar pulls postings directly from the job-board APIs that Greenhouse, Lever, and Ashby expose — the same JSON endpoints their own hosted careers pages render from — scores each one for ghost-job risk using nothing but deterministic heuristics, and writes out plain files ready to feed straight into the resume-matching pipeline from her last build. No scraping. No LLM call. Just three public APIs, a handful of tunable rules, and a flat JSON file that remembers what it's seen before.

## Why Not LinkedIn, and Why Not Scrape

The obvious move is a scraper aimed at LinkedIn or Indeed. It was also the first idea discarded. Scraping an aggregator means fighting ToS, fighting layout changes, and — worse for the actual goal here — inheriting the aggregator's own ghost-job problem, since a scrape of Indeed is a scrape of exactly the noisy signal she's trying to filter out.

Greenhouse, Lever, and Ashby solve both problems for free. Their public job-board APIs are the same data a company's own hosted careers page renders from — not a third-party's degraded copy of it, and not something anyone's ToS forbids reading, since it's the identical request a browser makes. More importantly, they're a company's *real* internal requisition feed, with real timestamps: when a role was actually posted, when it was last updated. An aggregator flattens all of that into "posted 3 days ago," and there's no way to know if that's true.

The tradeoff, and it's a real one: coverage is limited to whatever companies you explicitly tell it about. There's no way to ask any of these APIs "show me every AI/ML role within 50 miles of San Diego" — each one only exposes a single company's board at a time. Breadth for reliability. Given the alternative was reliability *of noise*, that trade was easy to make.

## Architecture Overview

```
Phase 1 — Load company config      [deterministic]
  config/companies.json  ──►  load_companies()  ──►  list[CompanyConfig]

Phase 2 — Fetch postings           [async HTTP, no LLM]
  CompanyConfig  ──►  ats_clients.fetch_{greenhouse,lever,ashby}()  ──►  list[RawPosting]
  (concurrent via asyncio.gather, one company's failure never blocks another's)

Phase 3 — Ghost-risk scoring       [deterministic, no LLM]
  RawPosting + cross-run history  ──►  score_posting()  ──►  GhostSignal{score, reasons}

Phase 4 — Persist and hand off     [local, file-based]
  ScoredPosting  ──►  seen_store.json          (cross-run memory)
                ──►  {slug}.json               (full record)
                ──►  {slug}.txt                (plain text, ready for the next pipeline)
```

Sixteen companies get fetched concurrently in Phase 2 — total pull time is the slowest single board, not the sum of all sixteen, the same concurrency logic already proven out in the strategic-reports pipeline. Each company's fetch owns its own error handling; one dead slug or one 404 doesn't take down the other fifteen.

## The Decision Not to Reach for an LLM

Both sibling projects in this pipeline — strategic-reports and the resume-matching agent — make a real LLM call. Job Radar makes zero. That wasn't the obvious call at the start; ghost-job detection sounds like exactly the kind of fuzzy judgment call an LLM is good at. It turned out not to be.

Every signal that actually distinguishes a ghost listing from a real one is structured, not linguistic: how long ago was it posted, has it been sitting there across multiple checks, does the copy use a specific handful of evergreen-hiring phrases. None of that needs a model that reads English — it needs a date subtraction, a repeat count, and a regex:

```python
_EVERGREEN_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"always[\s-]hiring",
        r"talent (community|pool|network)",
        r"general application",
        r"future opportunit(y|ies)",
        r"evergreen",
        r"we'?re always looking",
    ]
]
```

An LLM call here would add real latency and a real dollar cost to score something a five-line regex already catches, and — the part that actually mattered in the decision — it would make the score *less* explainable, not more. Every signal that contributes to a ghost-risk score appends a plain-English reason to the result. Nothing is a silent number:

```
score=0.35
reasons=["posted 275 days ago (>90d)"]
```

That's an actual score from an actual pull, on a real "Software Engineer, Growth Platform" posting that has, in fact, been open at Ramp for the better part of a year. The pipeline never hides the postings it thinks are risky, either — every one gets written out; the score just lets her sort safest-first or filter with `--max-ghost-score` when she wants a shorter list. A heuristic can be wrong, so it surfaces the evidence instead of making the call unilaterally.

## Making Staleness a Verifiable Claim, Not a Guess

The single strongest ghost-risk signal in this pipeline isn't anything an ATS reports about itself. Greenhouse's job-list endpoint, specifically, doesn't reliably return a creation date at all — only `updated_at`, which a company can bump without the role having changed in any meaningful way. Trusting any single ATS's self-reported dates means trusting a number that has every incentive to look fresher than it is.

So the pipeline keeps its own memory instead. Every posting it has ever seen gets a record in a flat JSON store, keyed by `{ats}:{slug}:{external_id}`:

```json
{
  "key": "lever:fatetherapeutics:a0b82256-00da-4cd9-81b3-7c8e0df48b2a",
  "first_seen_at": "2026-07-02T19:38:29.097930",
  "last_seen_at": "2026-07-02T19:38:29.097930",
  "seen_count": 1,
  "last_title": "Director, Regulatory Affairs"
}
```

Each `pull` updates every record it touches. If a posting is still showing up across a dozen pulls spanning six weeks, that's evidence no ATS's own timestamp can fake, because it isn't the ATS's claim at all — it's a claim job-radar can independently stand behind, having actually watched. The staleness signal only gets *more* trustworthy with time, which is a strange property for a piece of software to have, and also exactly the point: this store is one-pull-in, more-confident-out, by design.

## What Broke, and What That Taught Her

The mocked test suite — 59 tests, no real network call in any of them — passed cleanly through every bug described here. None of them were unit-test-detectable, because the fixtures used to write those tests weren't adversarial enough to trip the real edge cases. Two bugs only surfaced once real postings from real companies got pulled through the real pipeline, which is the same lesson the resume-matching agent's build already taught her once and apparently needed to teach twice:

**A timezone crash on the very first real Greenhouse posting.** Greenhouse and Ashby's ISO timestamps parse into timezone-*aware* datetimes; the seen-store's own `datetime.now()` calls, and Lever's epoch-millisecond conversion, are timezone-*naive*. The instant the scorer tried to subtract one from the other: `TypeError: can't subtract offset-naive and offset-aware datetimes`. Fixed by stripping tzinfo at parse time — a few hours of UTC-versus-local skew doesn't matter at the day-granularity thresholds this scorer uses, and consistency mattered more than precision here.

**Double-HTML-escaped job descriptions leaking straight into drafts.** This one was subtler and worse. Greenhouse's `content` field doesn't contain real HTML markup — it contains the *literal text* `&lt;div class=&quot;content-intro&quot;&gt;`, an HTML-escaped string sitting inside a JSON string. The markdown converter had no real tags to strip, so raw escaped-looking markup was leaking straight through into what should have been clean job-description text — and would have gone straight into an assembled resume draft unnoticed, if a real Figma posting hadn't been read end-to-end by hand first. One `html.unescape()` call before conversion fixed it. The regression test for this now ships with genuinely escaped fixture content, not the clean, well-behaved HTML the original test used — the exact kind of fixture-realism gap that let it through the first time.

Neither bug is exotic. Both are the kind of thing that only shows up when a pipeline meets data it didn't imagine having to handle, which is most real data, most of the time.

## Closing the Loop That Got Left Open Last Time

The last post about this whole system ended on an open question — an agent that watches job postings come in, scores each one, and closes the loop between "the infrastructure exists" and "applications actually get sent." That loop is closed now, and Job Radar is the half of it that watches.

Every `pull` writes two files per posting: the full scored record as JSON, and the plain description text as `.txt` — already in the exact shape the resume-matching agent's `match` command expects, no reformatting required:

```bash
python -m job_radar.cli pull
python -m job_radar.cli list --location-contains "San Diego" --title-contains "AI"
#   0.00  Acadia Pharmaceuticals   San Diego, CA   Associate Director, AI/ML Engineering

python -m job_hunt_agent.cli match-and-draft \
    --posting output/postings/2026-07-02/acadia-pharmaceuticals--....txt \
    --company "Acadia Pharmaceuticals" --role "Associate Director, AI/ML Engineering" \
    --url "https://acadia.com/en-us/careers/job-board/8565787002?gh_jid=8565787002"
```

That `--url` flag is a small thing that took a real conversation to notice was missing: a matched, drafted resume is useless without the link back to where it came from, and the original design didn't carry one anywhere. It does now — threaded from the ATS API's own listing URL, through the scored posting, through the match, into a metadata comment on the assembled draft, so the apply link is never more than one file away from the document built for it.

The two projects still don't know about each other in any way that matters. Job Radar never imports the resume-matching agent's code, and the reverse is also true — the only thing that crosses the boundary is a `.txt` file and, now, a `--url` string. That's deliberate: either project's tests stay fully independent, and either one could be swapped out or rebuilt without touching the other. `scripts/run_pipeline.sh` chains them together for real, end-to-end command-line use — pull, filter, confirm (each downstream match is a real LLM call, and confirmation matters when that costs real time and, depending on the provider, real money), then match-and-draft against everything that made the cut — but it's a shell script calling two CLIs, not a shared codebase.

There's also `discover-slug` now, which exists because seeding that company list by hand got tedious fast. Give it a company name and it generates a small set of plausible ATS slugs — the full name, hyphenated, and just the first word — and checks every one concurrently against all three real APIs:

```
$ python -m job_radar.cli discover-slug "Acadia Pharmaceuticals"
  greenhouse acadiapharmaceuticals          53 jobs
```

It never guesses when more than one candidate turns out to be a real, live board — `--add` refuses to write anything to the config file unless exactly one match is confirmed, or a specific one is picked by hand. A generic enough company name really can coincidentally collide with someone else's board, and silently trusting a job count over a genuine name match is exactly the kind of shortcut this whole project exists to avoid taking.

## Next Steps

A few things on the list:

* **More ATS platforms.** Greenhouse, Lever, and Ashby cover a lot of the venture-backed and biotech world, but not all of it — Workday-hosted boards in particular are common at larger, more established employers and currently invisible to this pipeline entirely.

* **A real scheduling story, if the workflow ever asks for one.** `pull` is on-demand today, run by hand when she's actually searching — deliberately not wired into Prefect the way the strategic-reports pipeline is, since there's no recurring job to orchestrate yet. That's a real architectural change to make later, not something to design in speculatively now on the chance it's needed.

* **A "what's new since last time" view.** The seen-store already knows exactly which postings first appeared in this pull versus three pulls ago — `list` doesn't expose that distinction yet, and it should, since "what changed" is a more useful question than "show me everything" once there's real pull history to compare against.

## Code

Job Radar lives alongside the resume-matching agent and the strategic-reports pipeline from the last post, in the same AI-ML repo.

## AI Use Statement

Our heroine asked Claude Code to build Job Radar from a standing start — architecture, the ghost-scoring heuristics, the ATS clients, the test suite, all of it — across a planned, reviewed session, not a one-shot prompt. She made the real calls herself, confirmed one at a time rather than assumed: flag ghost-risk instead of silently filtering it, ship the config with placeholder companies rather than guessed real ones, keep it on-demand instead of scheduled. Every bug described above was caught by actually running the pipeline against live company data and reading the output by hand, not by trusting a green test suite — the same discipline the resume-matching agent's build already required of itself once. Claude drafted this article in her voice, from two of her own posts supplied as reference, which she then edited before publication.
