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

The suite sits at 108 tests now, entirely against a synthetic fixture vault, never the real one. The most serious bug wasn't caught by any of them — it needed a second real posting, in a different domain from the first, to surface.

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

## Next Steps

* **Batch mode.** Right now `match-and-draft` handles one posting at a time. `scripts/run_pipeline.sh` already loops over several from the shell, but a native batch command inside job-hunt-agent itself — with its own summary table across postings — would make a real multi-posting review session faster.

* **Tracker analytics.** The flat JSON file already has everything needed to answer "what's my actual response rate by variant" or "by register" — nothing currently reads it that way. That's a pure reporting layer over data that already exists, no new capture needed.

* **The cosmetic `file_title` mismatch.** The model sometimes fills a surfaced skill's `file_title` field with the keyword itself rather than the real Skills filename it came from — not a correctness bug, since the draft still marks it clearly as unreviewed, but worth a prompt-engineering pass eventually.

## Code

Job Hunt Agent lives alongside Job Radar and the strategic-reports pipeline in the same AI-ML repo.

## AI Use Statement

Our heroine asked Claude Code to build Job Hunt Agent end to end — the vault parser, the two-phase matcher, the guardrails, the assembler, all of it — deliberately mirroring the strategic-reports pipeline's own architecture, since that project is the flagship example on her AI-engineering resume and this gives her a second one. Every real scope boundary was her explicit call, confirmed one at a time rather than assumed: the vault stays fully read-only with no sync option at all, no Prefect since postings arrive on demand rather than on a schedule, no job-board scraping in this project specifically (that became Job Radar, later, as its own project), no backfill of old application history. Every bug described above was caught by actually running the pipeline against the real vault and real postings, not by trusting a passing test suite — five real bugs across two real end-to-end runs, all now covered by regression tests written *after* the fact, from the failure, not before it. Claude drafted this article in her voice, from two of her own posts supplied as reference, which she then edited before publication.
