"""job-hunt-agent CLI entrypoint.

Usage:
    python -m job_hunt_agent.cli load-vault
    python -m job_hunt_agent.cli match --posting posting.txt --company Acme --role "Data Scientist"
    python -m job_hunt_agent.cli match --posting posting.txt --company Acme --role "Data Scientist" --url "https://..." --ghost-score 0.35 --ghost-reasons "posted 113 days ago (>90d)"
    python -m job_hunt_agent.cli draft --match output/matches/acme-data-scientist-2026-01-01/match.json
    python -m job_hunt_agent.cli init-filled --draft-dir output/drafts/acme-data-scientist-2026-01-01

    # match-and-draft also creates resume-filled.md/cover_letter-filled.md automatically (pass --no-init-filled to skip)
    # and copies the source posting into the draft directory as posting.txt (both draft and match-and-draft do this; no opt-out)
    python -m job_hunt_agent.cli match-and-draft --posting posting.txt --company Acme --role "Data Scientist"
    python -m job_hunt_agent.cli track add --company Acme --role "Data Scientist" --variant data-science --register formal-professional
    python -m job_hunt_agent.cli track list

vault-Resume is treated as strictly read-only input everywhere in this CLI —
nothing here ever writes into it, including on a "sent" application. See
Resumes/INDEX.md's `used_for_applications` field in the vault itself if you
want to record that by hand.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import sys
from datetime import datetime
from pathlib import Path

import typer

from job_hunt_agent.core.assembler import assemble_draft_cover_letter, assemble_draft_resume, slugify
from job_hunt_agent.core.llm_client import LLMClient
from job_hunt_agent.core.matcher import score_job
from job_hunt_agent.core.models import ApplicationRecord, JobMatchResult, JobPosting
from job_hunt_agent.core.posting_utils import extract_company_brief
from job_hunt_agent.core.tracker import ApplicationStore
from job_hunt_agent.core.tracing import generate_run_id, setup_tracing
from job_hunt_agent.core.vault_reader import load_vault

app = typer.Typer(add_completion=False, help="Match job postings against vault-Resume and track applications.")
track_app = typer.Typer(add_completion=False, help="Manage the local application tracker.")
app.add_typer(track_app, name="track")

_DEFAULT_HOME = Path(os.environ.get("JOB_HUNT_AGENT_HOME", Path.cwd()))
_DEFAULT_VAULT_PATH = Path(
    os.environ.get(
        "JOB_HUNT_AGENT_VAULT_PATH", str(Path.home() / "Desktop" / "vaults" / "vault-Resume")
    )
)
_DEFAULT_MODEL = os.environ.get("LLM_MODEL", "ollama_chat/llama3.1:70b")
_DEFAULT_TRACKER_PATH = Path(
    os.environ.get(
        "JOB_HUNT_AGENT_TRACKER_PATH", str(_DEFAULT_HOME / "output" / "tracker" / "applications.json")
    )
)


def _read_posting_text(posting: str) -> str:
    if posting == "-":
        return sys.stdin.read()
    path = Path(posting)
    if not path.is_file():
        raise typer.BadParameter(f"posting file not found: {posting}")
    return path.read_text(encoding="utf-8")


def _write_posting_copy(draft_dir: Path, posting_text: str) -> Path:
    """Copy the raw posting text into the draft directory as posting.txt.

    Keeps the source posting one file away from resume.md/cover_letter.md
    for the human-review pass, without requiring a re-lookup of wherever
    the original --posting path or match.json came from.
    """
    draft_dir.mkdir(parents=True, exist_ok=True)
    out_path = draft_dir / "posting.txt"
    out_path.write_text(posting_text, encoding="utf-8")
    return out_path


async def _run_match(job: JobPosting, model: str, vault_path: Path) -> JobMatchResult:
    setup_tracing()
    run_id = generate_run_id()
    vault = load_vault(vault_path)
    client = LLMClient(
        model=model, run_metadata={"trace_id": run_id, "trace_name": "job-hunt-agent-match"}
    )
    return await score_job(job, vault, client)


def _print_match_summary(result: JobMatchResult) -> None:
    if result.error or result.llm_output is None:
        typer.echo(f"Match failed: {result.error}", err=True)
        raise typer.Exit(code=1)

    out = result.llm_output
    typer.echo(f"Recommended variant: {out.recommended_variant}")
    for vs in out.variant_scores:
        typer.echo(f"  {vs.variant}: {vs.fit_score:.2f}")
    typer.echo(f"Cover letter register: {out.cover_letter_register} — {out.register_reasoning}")
    if out.surfaced_skills:
        typer.echo("Surfaced skills not yet in any resume:")
        for s in out.surfaced_skills:
            typer.echo(f"  - {s.keyword} ({s.file_title}) — {s.why_relevant}")
    if out.surfaced_bullets:
        typer.echo("Surfaced bullets not yet in any resume:")
        for b in out.surfaced_bullets:
            typer.echo(f"  - [{b.employer}] {b.bullet_id} — {b.why_relevant}")


@app.command("load-vault")
def load_vault_cmd(
    vault_path: Path = typer.Option(
        _DEFAULT_VAULT_PATH, "--vault-path", envvar="JOB_HUNT_AGENT_VAULT_PATH"
    ),
) -> None:
    """Parse the vault and print a summary — smoke test for the vault parser."""
    snap = load_vault(vault_path)
    typer.echo(f"Vault: {vault_path}")
    typer.echo(f"Variants: {sorted(snap.variants.keys())}")
    typer.echo(f"Employers: {list(snap.experience.keys())}")
    typer.echo(f"Experience bullets: {len(snap.all_experience_bullets())}")
    typer.echo(f"Skill files: {len(snap.skills)}")
    typer.echo(f"Projects: {len(snap.projects)}")
    typer.echo(f"Patents/publications: {len(snap.patents)}")
    typer.echo(f"Education entries: {len(snap.education)}")
    typer.echo(f"Voice examples: {len(snap.voice_examples)}")
    typer.echo(f"Achievement paragraphs: {len(snap.achievement_paragraphs)}")
    typer.echo(f"Soft skills: {len(snap.soft_skills)}")
    typer.echo(f"Greetings: {len(snap.greetings)}")
    typer.echo(f"Closings: {len(snap.closings)}")
    if snap.warnings:
        typer.echo(f"\nWarnings ({len(snap.warnings)}):")
        for w in snap.warnings:
            typer.echo(f"  - {w}")
    else:
        typer.echo("\nNo warnings.")


@app.command("company-brief")
def company_brief_cmd(
    posting: str = typer.Option(
        ..., "--posting", help="Path to a job posting text file, or '-' to read from stdin"
    ),
) -> None:
    """Best-effort extraction of a posting's "About <Company>" blurb.

    A convenience for writing the company-specific paragraph — that
    paragraph must always be written fresh (per the vault's own rules,
    see CoverLetters/INDEX.md), this command doesn't write anything, it
    just saves re-reading the whole raw posting to find the mission
    language to write from. Not authoritative: always sanity-check
    against the real posting.
    """
    posting_text = _read_posting_text(posting)
    brief = extract_company_brief(posting_text)
    if brief is None:
        typer.echo(
            "Couldn't find an 'About <Company>' section or any substantial paragraph.", err=True
        )
        raise typer.Exit(code=1)
    typer.echo(brief)


@app.command("match")
def match_cmd(
    posting: str = typer.Option(
        ..., "--posting", help="Path to a job posting text file, or '-' to read from stdin"
    ),
    company: str | None = typer.Option(None, "--company"),
    role: str | None = typer.Option(None, "--role"),
    url: str | None = typer.Option(None, "--url", help="The posting's apply/listing URL, if known — carried into the saved match.json and both drafts' metadata comment."),
    ghost_score: float | None = typer.Option(
        None, "--ghost-score", help="job-radar's ghost-risk score for this posting (0-1), if known — carried into the saved match.json and both drafts' metadata comment."
    ),
    ghost_reasons: str | None = typer.Option(
        None, "--ghost-reasons", help="job-radar's ghost-risk reasons, semicolon-separated, if known."
    ),
    model: str = typer.Option(_DEFAULT_MODEL, "--model", envvar="LLM_MODEL"),
    vault_path: Path = typer.Option(
        _DEFAULT_VAULT_PATH, "--vault-path", envvar="JOB_HUNT_AGENT_VAULT_PATH"
    ),
    home: Path = typer.Option(_DEFAULT_HOME, "--home", envvar="JOB_HUNT_AGENT_HOME"),
) -> None:
    """Score a job posting against the vault; writes output/matches/{slug}/match.json."""
    posting_text = _read_posting_text(posting)
    job = JobPosting(
        raw_text=posting_text,
        company=company,
        role_title=role,
        url=url,
        ghost_score=ghost_score,
        ghost_reasons=[r.strip() for r in ghost_reasons.split(";") if r.strip()] if ghost_reasons else [],
    )
    result = asyncio.run(_run_match(job, model, vault_path))

    slug = slugify(company or "unknown", role or "role", datetime.now().strftime("%Y-%m-%d"))
    out_path = home / "output" / "matches" / slug / "match.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    _print_match_summary(result)
    typer.echo(f"Match written to: {out_path}")


@app.command("draft")
def draft_cmd(
    match_path: Path = typer.Option(..., "--match", help="Path to a saved match.json"),
    variant: str | None = typer.Option(None, "--variant", help="Override the recommended variant"),
    register: str | None = typer.Option(None, "--register", help="Override the recommended register"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    vault_path: Path = typer.Option(
        _DEFAULT_VAULT_PATH, "--vault-path", envvar="JOB_HUNT_AGENT_VAULT_PATH"
    ),
    home: Path = typer.Option(_DEFAULT_HOME, "--home", envvar="JOB_HUNT_AGENT_HOME"),
) -> None:
    """Deterministically assemble a draft resume + cover letter from a saved match.json."""
    result = JobMatchResult.model_validate_json(match_path.read_text(encoding="utf-8"))
    if result.llm_output is None:
        typer.echo("This match.json has no llm_output (the match run failed) — nothing to draft.", err=True)
        raise typer.Exit(code=1)

    vault = load_vault(vault_path)

    if output_dir is None:
        slug = slugify(
            result.job.company or "unknown",
            result.job.role_title or "role",
            datetime.now().strftime("%Y-%m-%d"),
        )
        output_dir = home / "output" / "drafts" / slug

    resume = assemble_draft_resume(result, vault, output_dir, variant_override=variant)
    letter = assemble_draft_cover_letter(result, vault, output_dir, register_override=register)
    posting_copy = _write_posting_copy(output_dir, result.job.raw_text)

    typer.echo(f"Resume draft ({resume.variant}): {resume.output_path}")
    if resume.warnings:
        typer.echo("  WARNINGS:")
        for w in resume.warnings:
            typer.echo(f"    - {w}")
    typer.echo(f"Cover letter draft ({letter.letter_register}): {letter.output_path}")
    if letter.warnings:
        typer.echo("  WARNINGS:")
        for w in letter.warnings:
            typer.echo(f"    - {w}")
    typer.echo(f"Posting copy: {posting_copy}")


def _init_filled_files(draft_dir: Path, force: bool) -> tuple[list[Path], list[Path]]:
    """Shared by init_filled_cmd and match_and_draft_cmd's automatic call.

    Returns (created, skipped). Never overwrites an existing *-filled.md
    unless force=True — this is what keeps a hand-review pass safe across a
    later re-draft, whether that re-draft is invoked explicitly (`draft`) or
    implicitly (`match-and-draft` run again for the same slug).
    """
    created: list[Path] = []
    skipped: list[Path] = []
    for base_name in ("resume.md", "cover_letter.md"):
        source = draft_dir / base_name
        if not source.is_file():
            continue
        target = draft_dir / f"{source.stem}-filled{source.suffix}"
        if target.exists() and not force:
            skipped.append(target)
            continue
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        created.append(target)
    return created, skipped


@app.command("init-filled")
def init_filled_cmd(
    draft_dir: Path = typer.Option(
        ..., "--draft-dir", help="A draft directory containing resume.md and/or cover_letter.md"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing *-filled.md file instead of leaving it untouched."
    ),
) -> None:
    """Create resume-filled.md / cover_letter-filled.md as editable copies of the assembled drafts.

    resume.md and cover_letter.md stay the pristine, reproducible output of
    `draft` — always safe to re-run when the match or the code changes,
    since nothing gets hand-edited into them directly. The *-filled.md
    siblings are where the actual human-review pass happens: integrating
    worthwhile surfaced skills/bullets into the main sections, writing the
    company-specific paragraph, filling in closing-line specifics — content
    that, per the vault's own rules, must always be written fresh rather
    than templated. Re-running `draft` never clobbers that work, and you can
    diff draft vs. filled to see exactly what changed for a given
    application, which is the point when doing this across many postings.

    `match-and-draft` calls this automatically (pass --no-init-filled to
    skip); this command is for running it standalone against an existing
    draft directory, or re-running it with --force.
    """
    created, skipped = _init_filled_files(draft_dir, force)

    if not created and not skipped:
        typer.echo(f"No resume.md or cover_letter.md found under {draft_dir}", err=True)
        raise typer.Exit(code=1)

    for t in created:
        typer.echo(f"Created: {t}")
    for t in skipped:
        typer.echo(f"Already exists, left untouched (use --force to overwrite): {t}")


# Deliberately not "<stem>-filled-diff.md" to match resume-filled.md's own
# naming — these are a distinct artifact (a diff, not a draft), so a visually
# distinct underscore convention keeps the two from being confused at a glance.
_DIFF_FILENAMES = {
    "resume.md": "resume_filled_diff.md",
    "cover_letter.md": "cover_letter_filled_diff.md",
}


def _write_diff(source: Path, filled: Path, out_path: Path) -> bool:
    """Write a unified diff from source -> filled at out_path. Returns False
    (writes nothing) if either file is missing — there's nothing to diff."""
    if not source.is_file() or not filled.is_file():
        return False
    source_lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    filled_lines = filled.read_text(encoding="utf-8").splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(source_lines, filled_lines, fromfile=source.name, tofile=filled.name)
    )
    if diff_lines:
        body = f"# Diff: {source.name} → {filled.name}\n\n```diff\n{''.join(diff_lines)}```\n"
    else:
        body = f"# Diff: {source.name} → {filled.name}\n\nNo differences — {filled.name} is identical to {source.name}.\n"
    out_path.write_text(body, encoding="utf-8")
    return True


@app.command("diff-filled")
def diff_filled_cmd(
    draft_dir: Path = typer.Option(
        ..., "--draft-dir", help="A draft directory containing resume.md/resume-filled.md and/or cover_letter.md/cover_letter-filled.md"
    ),
) -> None:
    """Write resume_filled_diff.md / cover_letter_filled_diff.md showing exactly
    what the human-review pass changed, for each pair that exists.

    This is what "diff resume.md against resume-filled.md" from the README
    actually means made concrete — a thin wrapper over difflib, not new
    judgment. Skips (with a note) any pair where either half doesn't exist
    yet; always overwrites its own output, since a diff file is disposable
    and regenerating it is cheap.
    """
    if not any((draft_dir / base_name).is_file() for base_name in _DIFF_FILENAMES):
        typer.echo(f"No resume.md or cover_letter.md found under {draft_dir}", err=True)
        raise typer.Exit(code=1)

    written: list[Path] = []
    skipped: list[str] = []
    for base_name, diff_name in _DIFF_FILENAMES.items():
        source = draft_dir / base_name
        filled = draft_dir / f"{source.stem}-filled{source.suffix}"
        out_path = draft_dir / diff_name
        if _write_diff(source, filled, out_path):
            written.append(out_path)
        else:
            skipped.append(base_name)

    for w in written:
        typer.echo(f"Written: {w}")
    for s in skipped:
        stem = Path(s).stem
        typer.echo(f"Skipped {s}: needs both {s} and {stem}-filled.md to exist first", err=True)


@app.command("match-and-draft")
def match_and_draft_cmd(
    posting: str = typer.Option(
        ..., "--posting", help="Path to a job posting text file, or '-' to read from stdin"
    ),
    company: str | None = typer.Option(None, "--company"),
    role: str | None = typer.Option(None, "--role"),
    url: str | None = typer.Option(None, "--url", help="The posting's apply/listing URL, if known — carried into the saved match.json and both drafts' metadata comment."),
    ghost_score: float | None = typer.Option(
        None, "--ghost-score", help="job-radar's ghost-risk score for this posting (0-1), if known — carried into the saved match.json and both drafts' metadata comment."
    ),
    ghost_reasons: str | None = typer.Option(
        None, "--ghost-reasons", help="job-radar's ghost-risk reasons, semicolon-separated, if known."
    ),
    model: str = typer.Option(_DEFAULT_MODEL, "--model", envvar="LLM_MODEL"),
    register: str | None = typer.Option(None, "--register", help="Override the recommended register"),
    vault_path: Path = typer.Option(
        _DEFAULT_VAULT_PATH, "--vault-path", envvar="JOB_HUNT_AGENT_VAULT_PATH"
    ),
    home: Path = typer.Option(_DEFAULT_HOME, "--home", envvar="JOB_HUNT_AGENT_HOME"),
    init_filled: bool = typer.Option(
        True,
        "--init-filled/--no-init-filled",
        help="Also create resume-filled.md/cover_letter-filled.md (skipped if they already exist — never overwritten here).",
    ),
    diff_filled: bool = typer.Option(
        True,
        "--diff-filled/--no-diff-filled",
        help="Also (re)write resume_filled_diff.md/cover_letter_filled_diff.md. On a first run this just says "
        "'no differences' — the *-filled.md files were only just created — but it stays useful: re-running "
        "match-and-draft later (e.g. after a code fix) refreshes the diff to reflect whatever you've since edited.",
    ),
    company_brief: bool = typer.Option(
        True,
        "--company-brief/--no-company-brief",
        help="Also print the posting's best-effort 'About <Company>' blurb, for writing the company-specific paragraph.",
    ),
) -> None:
    """Convenience: chain match + draft in one invocation — the common real-world path."""
    posting_text = _read_posting_text(posting)
    job = JobPosting(
        raw_text=posting_text,
        company=company,
        role_title=role,
        url=url,
        ghost_score=ghost_score,
        ghost_reasons=[r.strip() for r in ghost_reasons.split(";") if r.strip()] if ghost_reasons else [],
    )
    result = asyncio.run(_run_match(job, model, vault_path))
    _print_match_summary(result)

    vault = load_vault(vault_path)
    slug = slugify(company or "unknown", role or "role", datetime.now().strftime("%Y-%m-%d"))

    match_path = home / "output" / "matches" / slug / "match.json"
    match_path.parent.mkdir(parents=True, exist_ok=True)
    match_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Match written to: {match_path}")

    output_dir = home / "output" / "drafts" / slug
    resume = assemble_draft_resume(result, vault, output_dir)
    letter = assemble_draft_cover_letter(result, vault, output_dir, register_override=register)
    posting_copy = _write_posting_copy(output_dir, posting_text)

    typer.echo(f"Resume draft ({resume.variant}): {resume.output_path}")
    if resume.warnings:
        for w in resume.warnings:
            typer.echo(f"  WARNING: {w}")
    typer.echo(f"Cover letter draft ({letter.letter_register}): {letter.output_path}")
    if letter.warnings:
        for w in letter.warnings:
            typer.echo(f"  WARNING: {w}")
    typer.echo(f"Posting copy: {posting_copy}")

    if init_filled:
        created, skipped = _init_filled_files(output_dir, force=False)
        for t in created:
            typer.echo(f"Filled copy created: {t}")
        for t in skipped:
            typer.echo(f"Filled copy already exists, left untouched: {t}")

    if diff_filled:
        for base_name, diff_name in _DIFF_FILENAMES.items():
            source = output_dir / base_name
            filled = output_dir / f"{source.stem}-filled{source.suffix}"
            if _write_diff(source, filled, output_dir / diff_name):
                typer.echo(f"Diff written: {output_dir / diff_name}")

    if company_brief:
        brief = extract_company_brief(posting_text)
        if brief:
            typer.echo(f"\nCompany brief (for the company-specific paragraph):\n{brief}")


def _tracker(tracker_path: Path) -> ApplicationStore:
    return ApplicationStore(tracker_path)


@track_app.command("add")
def track_add_cmd(
    company: str = typer.Option(..., "--company"),
    role: str = typer.Option(..., "--role"),
    variant: str = typer.Option(..., "--variant"),
    register: str = typer.Option(..., "--register"),
    status: str = typer.Option("drafted", "--status"),
    notes: str | None = typer.Option(None, "--notes"),
    job_posting_path: str | None = typer.Option(None, "--job-posting-path"),
    tracker_path: Path = typer.Option(
        _DEFAULT_TRACKER_PATH, "--tracker-path", envvar="JOB_HUNT_AGENT_TRACKER_PATH"
    ),
) -> None:
    """Record a new application in the local tracker."""
    record = ApplicationRecord(
        company=company,
        role=role,
        resume_variant=variant,
        cover_letter_register=register,
        status=status,
        notes=notes,
        job_posting_path=job_posting_path,
    )
    _tracker(tracker_path).add(record)
    typer.echo(f"Added application {record.id} — {company} / {role} [{status}]")


@track_app.command("update")
def track_update_cmd(
    record_id: str = typer.Argument(...),
    status: str | None = typer.Option(None, "--status"),
    outcome: str | None = typer.Option(None, "--outcome"),
    notes: str | None = typer.Option(None, "--notes"),
    tracker_path: Path = typer.Option(
        _DEFAULT_TRACKER_PATH, "--tracker-path", envvar="JOB_HUNT_AGENT_TRACKER_PATH"
    ),
) -> None:
    """Update status/outcome/notes on an existing application record."""
    fields = {k: v for k, v in {"status": status, "outcome": outcome, "notes": notes}.items() if v is not None}
    if not fields:
        typer.echo("Nothing to update — pass at least one of --status/--outcome/--notes.", err=True)
        raise typer.Exit(code=1)
    try:
        updated = _tracker(tracker_path).update(record_id, **fields)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Updated {updated.id} — {updated.company} / {updated.role} [{updated.status}]")


@track_app.command("list")
def track_list_cmd(
    status: str | None = typer.Option(None, "--status"),
    company: str | None = typer.Option(None, "--company"),
    tracker_path: Path = typer.Option(
        _DEFAULT_TRACKER_PATH, "--tracker-path", envvar="JOB_HUNT_AGENT_TRACKER_PATH"
    ),
) -> None:
    """List application records, optionally filtered by status/company."""
    records = _tracker(tracker_path).filter(status=status, company=company)
    if not records:
        typer.echo("No matching applications.")
        return
    for r in records:
        typer.echo(f"{r.id}  {r.company:<25} {r.role:<30} {r.status:<12} {r.resume_variant}")


@track_app.command("show")
def track_show_cmd(
    record_id: str = typer.Argument(...),
    tracker_path: Path = typer.Option(
        _DEFAULT_TRACKER_PATH, "--tracker-path", envvar="JOB_HUNT_AGENT_TRACKER_PATH"
    ),
) -> None:
    """Show the full detail of one application record."""
    record = _tracker(tracker_path).get(record_id)
    if record is None:
        typer.echo(f"No application record found for id {record_id!r}", err=True)
        raise typer.Exit(code=1)
    typer.echo(record.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
