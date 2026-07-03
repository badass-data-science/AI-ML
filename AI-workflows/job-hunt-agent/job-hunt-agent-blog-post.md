# Job Hunt Agent

## The Half of the Loop That Judges What Job Radar Finds

Our heroine the data scientist now has a resume that's actually a knowledge graph, and a pipeline that watches sixteen companies' ATS boards and hands her back real, scored, non-ghost postings. Neither of those, on its own, gets an application out the door. A graph of decomposed bullets doesn't read a job posting. A scored posting doesn't know which of three resume variants fits it, or which real-but-unused skill from her own vault happens to be exactly what this particular hiring manager wants to see.

Most advice about AI and job hunting stops at "paste your resume into ChatGPT and ask it to tailor it." That's the thing she explicitly decided not to build. Pasting a resume as prose treats a career as a block of text to be reworded — it throws away the structure the vault exists to preserve, the tagged relevance judgments, the "used here, cut there, and here's why" reasoning sitting right next to every bullet. An agent that just rewords prose can't tell the difference between "this skill isn't relevant to this posting" and "this skill is real but nobody's ever put it on a resume yet."

So she built something that actually reads the graph.

Job Hunt Agent takes a job posting and the vault, scores the posting against all three resume variants in a single comparative LLM call, and assembles a draft resume and cover letter from the result — grounded entirely in real vault content, never invented, and never written back into the vault it reads from.

## Architecture Overview

```
Phase 1 — Vault parsing            [deterministic, no LLM]
  vault-Resume/  ──►  load_vault()  ──►  VaultSnapshot
  (read-only input; nothing here ever writes back)

Phase 2 — Keyword prefilter        [deterministic, no LLM, no network]
  posting text  ──►  prefilter_candidates()  ──►  top-N candidate skills/bullets
  (plain token-overlap ranking — shrinks the vault to a digest that fits one prompt)

Phase 3 — Matching                 [single LLM call]
  VaultSnapshot + candidates + posting  ──►  JobMatchLLMOutput
  (scores all 3 variants comparatively, recommends variant + register,
   surfaces real-but-unused content)

Phase 4 — Draft assembly           [deterministic, no LLM]
  JobMatchResult + VaultSnapshot  ──►  resume.md + cover_letter.md
  (guardrails.py scans every draft before it's written)

Phase 5 — Tracking                 [local, file-based]
  ApplicationRecord  ──►  applications.json
```

Five phases, and only one of them touches an LLM at all. That's not an accident — it's the same design instinct behind Job Radar's ghost-scoring: reach for a model only where the judgment genuinely requires one, and do everything else with plain, auditable code.

## One LLM Call, Not Three

The obvious approach — score each of the three resume variants (generalist data-science, bioinformatics, AI-engineering) with its own independent call — was also the first one discarded. Three separate calls triple the cost for no real benefit, and worse, they produce scores that don't actually mean anything relative to each other. A 0.85 from one call and a 0.72 from a different call, made with no knowledge of the first, are not directly comparable numbers — they just look like they are.

One comparative call, scoring all three variants against each other in the same context, fixes that. The prompt gets the model to reason about *which one wins and why* rather than *rate this one in isolation* — and it costs a third as much doing it. The deterministic keyword prefilter exists specifically to make that single call affordable: the full vault is easily too large for a comfortable prompt, so `prefilter_candidates()` ranks every skill file and experience bullet by plain token overlap with the posting text — no LLM, no network, just set intersection — before anything gets sent to a model at all.

## Surfacing What's Real But Unused Is the Point

Every `Skills/` file in the vault already tracks two lists: keywords used in a current resume, and keywords that are real, evidenced, and simply never made it into one. That second list is not a byproduct of how the vault happens to be organized — it's the reason this whole matching step exists. The prompt is built specifically to surface genuinely relevant unused content when a posting calls for it, not just re-confirm whatever's already in the recommended variant.

When that happens, the draft never lets the surfaced content blend in silently:

```
### Surfaced skills not yet in this resume — human review required
- Model Context Protocol (Model Context Protocol) — Enables standardized
  context passing between agents, a core component of multi-agent
  orchestration frameworks such as LangGraph or AutoGen.
```

That's real output, from a real posting — Acadia Pharmaceuticals' Associate Director, AI/ML Engineering role, matched against the actual vault. The skill is real; it's just never been on a resume before this match surfaced it. Every surfaced item stays visibly marked as unreviewed, forever, until a human decides otherwise. Nothing here gets to look like vetted content by accident.

## Guardrails, Not Filters

The vault carries a short list of standing content rules — forbidden terms that shouldn't reappear on a resume, aspirational skills that aren't real yet and shouldn't be claimed as if they were. `guardrails.py` scans every assembled draft against that list before it's written to disk, and it does not get to quietly fix what it finds. A hit becomes a warning attached to the draft; the text itself is never silently stripped or altered. If a violation is real, a human should see it in context and decide, not have it vanish before they ever knew it was there.

That design choice caught its own bug early. The first version scanned for forbidden acronyms with plain substring matching, and "RAG" — an excluded, not-yet-real skill — is a literal substring of ordinary words like "paragraph" and "leveraging." Every draft with either word in it tripped a false violation. The fix was a one-line word-boundary regex, but the lesson underneath it was the real point: a safety check needs adversarial test cases, not just the positive and negative examples that happen to occur to you first. The unit tests that existed at the time didn't happen to use either word, so they passed cleanly straight through a real bug.

## What Broke, and What That Taught Her

The suite sits at 122 tests now, entirely against a synthetic fixture vault, never the real one. The most serious bug wasn't caught by any of them — it needed a second real posting, in a different domain from the first, to surface.

The first real end-to-end run, against a synthetic-but-realistic healthcare-data-scientist posting, worked. It correctly recommended the data-science variant, surfaced genuinely relevant real content, and nothing looked wrong. The second real run — an actual pasted contract posting, in a different domain, deliberately mentioning "FDA regulations" as an adversarial test of the forbidden-term guardrail, which held clean — looked wrong immediately: every single "surfaced" bullet in the assembled resume was an exact-text duplicate of a bullet already sitting in the Experience section above it. The model had tagged bullets it had *already used* in the recommended variant as "surfaced," implying they were unused, directly against what the prompt explicitly told it not to do.

Prompt instructions aren't guarantees. The fix doesn't trust the model's claim at all — `assemble_draft_resume` cross-checks every LLM-claimed surfaced bullet and skill against the vault's own ground-truth `used_in` tags, and silently drops anything already present in the recommended variant, enforced deterministically in code rather than requested politely in a prompt:

```python
if variant_slug in full.used_in:
    # Ground-truth check, not LLM-trust: the vault's own `used_in` tag says
    # this bullet is already in the recommended variant's Experience
    # section, despite the LLM having called it "surfaced" (implying
    # unused). Silently drop rather than render a bullet twice.
    continue
```

The test fixture that was supposed to catch exactly this had the identical bug baked into it — its "surfaced" example bullet was already tagged as used in the recommended variant, so the regression test built on it was never actually testing a real surfaced-content scenario. A green test suite is not the same claim as "verified against reality," and this pipeline needed to relearn that lesson from job-radar's own build afterward, the hard way, a second time.

## The Vault Kept Telling the Truth in More Specific Ways Than the Code Was Listening For

"Done" didn't stay done. Four more real bugs turned up in the weeks after this post's first draft, once real postings from Job Radar started flowing through instead of hand-pasted ones — and every single one has the same shape: the vault already recorded the exact fact that would have prevented the bug, and the code was checking something coarser instead.

**A synonym the ground-truth check didn't know was a synonym.** The vault tags `MCP` (the abbreviation) as used in the ai-engineering resume — but the résumé's actual Skills line renders `Model Context Protocol (MCP)`, both forms together. When the matcher surfaced `Model Context Protocol` as a "not yet used" skill, the dedup check looked up the exact keyword `MCP` in the vault's tracked-used list, didn't find `Model Context Protocol` as a separate entry, and let a skill that was already sitting in the résumé's own Skills section get flagged as new. The fix stopped trusting the tracked keyword list alone and started checking the literal rendered text too — reusing `guardrails.py`'s own word-boundary matcher, since "is this term already present" turned out to be exactly the same question guardrails was already answering for forbidden terms.

**A relationship recorded in prose, not a field.** Best Buy Health's employer file keeps two retired, pre-trim bullets around for reference, each explicitly marked `Used in: none (retired — superseded by bbh-topic-modeling)`. The ground-truth check for surfaced bullets only looked at a bullet's own `used_in` tag — which is empty for a retired bullet, by design — so it had no way to know the retired bullet's content already existed in the résumé under a different, newer bullet ID. Three of four "surfaced" bullets in one real draft turned out to be near-duplicate pre-trim originals of content already on the page. The fix parses that `superseded by` relationship into a real field and follows it one hop before deciding a bullet is genuinely new.

**Alphabetical order, mistaken for chronological.** `Resumes/experience/*.md` loads via `sorted(glob(...))` — alphabetical by filename, chosen purely so file loading is deterministic. Nothing ever sorted the Experience section by date on top of that. It looked almost right in the real vault only because the most recent employer's filename happens to start with "A" — until a second employer, ending later but filed earlier alphabetically, rendered in the wrong position. The fix parses a best-effort `(year, month)` key out of each employer's free-text `dates` field and sorts by that before rendering, not by however the filesystem happened to hand the files over.

**A naming convention that was never read as a signal.** The vault's soft-skill blocks are named `-prose` or `-fragment` on purpose — one is a complete sentence safe to drop into a letter, the other is a phrase meant to be worked into the company-specific paragraph, and the file says so directly in its own usage guidance. Nothing in the code ever looked at the block ID's suffix or that guidance text. When the matcher picked a `-fragment` block, the template rendered its raw text — which starts with "..." — as its own paragraph, producing a dangling half-sentence in a real letter. The fix reads the `-fragment` suffix as the structured signal it already was, and routes those blocks to a review-comment instead of the letter body.

None of these are the kind of bug a fixture-vault test suite was ever going to catch by accident — each one needed either the real vault's real content shape, or a second real posting different enough from the first to expose a gap the first one didn't touch. The pattern across all nine bugs this project has now shipped, from the first build through this one, is the same pattern each time: trust the vault's actual, specific, already-recorded truth over whatever coarser signal the code happened to be checking instead.

## Where It Sits in the Pipeline

Job Hunt Agent is deliberately the middle of three pieces, not the whole thing. Upstream, the vault is strictly read-only — not even the `used_for_applications` frontmatter field, built for exactly this kind of status tracking, ever gets written to automatically. That stays a manual choice, made by a human, on purpose. Downstream, a drafted resume and cover letter are exactly that: drafts, always marked as needing a human review pass before anything gets sent.

The other upstream connection is newer. Job Radar hands off a scored posting as a plain `.txt` file, already in the exact shape `match` expects — no reformatting, no shared code, just a file:

```bash
python -m job_radar.cli list --location-contains "San Diego" --title-contains "AI"
#   0.00  Acadia Pharmaceuticals   San Diego, CA   Associate Director, AI/ML Engineering

python -m job_hunt_agent.cli match-and-draft \
    --posting output/postings/2026-07-02/acadia-pharmaceuticals--....txt \
    --company "Acadia Pharmaceuticals" --role "Associate Director, AI/ML Engineering" \
    --url "https://acadia.com/en-us/careers/job-board/8565787002?gh_jid=8565787002"
```

That `--url` flag didn't exist in the original design — a drafted resume with no link back to where the posting came from is a real gap, not a small one, and it only became obvious once she went looking for the apply link after a real draft was already sitting there. It's threaded now: the ATS's own listing URL rides along from Job Radar's scored posting, through the match, into a metadata comment on both assembled documents, so the link is never more than one file away from the draft built for it.

Neither project imports the other's code. Job Hunt Agent has no idea Job Radar exists, structurally — it just reads whatever `.txt` file it's pointed at, whether that came from a real ATS pull, a hand-pasted posting, or stdin. `scripts/run_pipeline.sh`, over in Job Radar, is what actually chains the two together for real command-line use — pull, filter, confirm, then `match-and-draft` against everything that made the cut — but it's a shell script calling two independent CLIs, not a merger of the two codebases. Either one can be rebuilt, replaced, or tested completely on its own.

Local tracking closes the last piece of the loop. `tracker.py`'s `ApplicationStore` is a flat, pretty-printed JSON file recording what actually got drafted, sent, and what happened after — matching the same flat-JSON precedent the strategic-reports pipeline already set for its own growing history, deliberately not a database. It starts empty by design; the real, messy, unstructured folder of past applications sitting elsewhere on her machine stays untouched rather than force-fit into a new schema it was never built for.

## Two Files, Not One, for Every Draft

The company-specific paragraph and the closing line can never come from the vault — that's not a limitation, it's a rule: `CoverLetters/INDEX.md` says those must always be written fresh, per application, and the code has never tried to get around that. But writing them meant editing `resume.md`/`cover_letter.md` directly, and those same files are also the pristine, safe-to-regenerate output of `draft` — which got re-run three separate times across the fixes described above. Editing in place meant every code fix was also a small bet against overwriting real review work.

`init-filled` splits the two concerns into two files. `resume.md`/`cover_letter.md` stay exactly what they've always been — regenerate them any time, no hesitation. `resume-filled.md`/`cover_letter-filled.md` are plain copies, created automatically now by `match-and-draft`, that never get touched again once they exist unless `--force` says otherwise:

```python
def _init_filled_files(draft_dir: Path, force: bool) -> tuple[list[Path], list[Path]]:
    created, skipped = [], []
    for base_name in ("resume.md", "cover_letter.md"):
        source = draft_dir / base_name
        target = draft_dir / f"{source.stem}-filled{source.suffix}"
        if target.exists() and not force:
            skipped.append(target)
            continue
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        created.append(target)
    return created, skipped
```

It held up under real use immediately: after the chronological-ordering fix, `draft` regenerated a real `resume.md` from scratch while `resume-filled.md` — already carrying real hand-integrated content for a real Acadia Pharmaceuticals application — was correctly left untouched. Diffing the two now shows, for any application, exactly what the pipeline produced automatically versus what a human actually decided to say.

## Next Steps

* **Batch mode.** Right now `match-and-draft` handles one posting at a time. `scripts/run_pipeline.sh` already loops over several from the shell, but a native batch command inside job-hunt-agent itself — with its own summary table across postings — would make a real multi-posting review session faster.

* **Tracker analytics.** The flat JSON file already has everything needed to answer "what's my actual response rate by variant" or "by register" — nothing currently reads it that way. That's a pure reporting layer over data that already exists, no new capture needed.

* **The cosmetic `file_title` mismatch.** The model sometimes fills a surfaced skill's `file_title` field with the keyword itself rather than the real Skills filename it came from — not a correctness bug, since the draft still marks it clearly as unreviewed, but worth a prompt-engineering pass eventually.

## Code

Job Hunt Agent lives alongside Job Radar and the strategic-reports pipeline in the same AI-ML repo.

## AI Use Statement

Our heroine asked Claude Code to build Job Hunt Agent end to end — the vault parser, the two-phase matcher, the guardrails, the assembler, all of it — deliberately mirroring the strategic-reports pipeline's own architecture, since that project is the flagship example on her AI-engineering resume and this gives her a second one. Every real scope boundary was her explicit call, confirmed one at a time rather than assumed: the vault stays fully read-only with no sync option at all, no Prefect since postings arrive on demand rather than on a schedule, no job-board scraping in this project specifically (that became Job Radar, later, as its own project), no backfill of old application history. Every bug described above was caught by actually running the pipeline against the real vault and real postings, not by trusting a passing test suite — nine real bugs now, across two build phases and several real end-to-end runs, all covered by regression tests written *after* the fact, from the failure, not before it — several confirmed via `git stash`-ing just the fix to prove the new test genuinely failed without it, not just that it passed with it. Claude drafted this article in her voice, and its later revisions, from two of her own posts supplied as reference, which she then edited before publication.
