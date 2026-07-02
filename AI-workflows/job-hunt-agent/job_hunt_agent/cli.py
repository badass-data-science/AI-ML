"""job-hunt-agent CLI entrypoint.

Usage:
    python -m job_hunt_agent.cli load-vault
    python -m job_hunt_agent.cli match --posting posting.txt --company Acme --role "Data Scientist"
    python -m job_hunt_agent.cli draft --match output/matches/acme-data-scientist-2026-01-01/match.json
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
import os
import sys
from datetime import datetime
from pathlib import Path

import typer

from job_hunt_agent.core.assembler import assemble_draft_cover_letter, assemble_draft_resume, slugify
from job_hunt_agent.core.llm_client import LLMClient
from job_hunt_agent.core.matcher import score_job
from job_hunt_agent.core.models import ApplicationRecord, JobMatchResult, JobPosting
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


@app.command("match")
def match_cmd(
    posting: str = typer.Option(
        ..., "--posting", help="Path to a job posting text file, or '-' to read from stdin"
    ),
    company: str | None = typer.Option(None, "--company"),
    role: str | None = typer.Option(None, "--role"),
    model: str = typer.Option(_DEFAULT_MODEL, "--model", envvar="LLM_MODEL"),
    vault_path: Path = typer.Option(
        _DEFAULT_VAULT_PATH, "--vault-path", envvar="JOB_HUNT_AGENT_VAULT_PATH"
    ),
    home: Path = typer.Option(_DEFAULT_HOME, "--home", envvar="JOB_HUNT_AGENT_HOME"),
) -> None:
    """Score a job posting against the vault; writes output/matches/{slug}/match.json."""
    posting_text = _read_posting_text(posting)
    job = JobPosting(raw_text=posting_text, company=company, role_title=role)
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


@app.command("match-and-draft")
def match_and_draft_cmd(
    posting: str = typer.Option(
        ..., "--posting", help="Path to a job posting text file, or '-' to read from stdin"
    ),
    company: str | None = typer.Option(None, "--company"),
    role: str | None = typer.Option(None, "--role"),
    model: str = typer.Option(_DEFAULT_MODEL, "--model", envvar="LLM_MODEL"),
    register: str | None = typer.Option(None, "--register", help="Override the recommended register"),
    vault_path: Path = typer.Option(
        _DEFAULT_VAULT_PATH, "--vault-path", envvar="JOB_HUNT_AGENT_VAULT_PATH"
    ),
    home: Path = typer.Option(_DEFAULT_HOME, "--home", envvar="JOB_HUNT_AGENT_HOME"),
) -> None:
    """Convenience: chain match + draft in one invocation — the common real-world path."""
    posting_text = _read_posting_text(posting)
    job = JobPosting(raw_text=posting_text, company=company, role_title=role)
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

    typer.echo(f"Resume draft ({resume.variant}): {resume.output_path}")
    if resume.warnings:
        for w in resume.warnings:
            typer.echo(f"  WARNING: {w}")
    typer.echo(f"Cover letter draft ({letter.letter_register}): {letter.output_path}")
    if letter.warnings:
        for w in letter.warnings:
            typer.echo(f"  WARNING: {w}")


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
