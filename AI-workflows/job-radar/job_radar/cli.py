"""job-radar CLI entrypoint.

Usage:
    python -m job_radar.cli pull
    python -m job_radar.cli pull --companies-path config/companies.json
    python -m job_radar.cli list
    python -m job_radar.cli list --max-ghost-score 0.4
    python -m job_radar.cli show output/postings/2026-07-02/acme--data-scientist.json

Produces posting files ready to hand to job-hunt-agent unmodified:
    python -m job_radar.cli pull
    python -m job_hunt_agent.cli match --posting job-radar/output/postings/<date>/<slug>.txt \\
        --company Acme --role "Data Scientist"

job-radar never imports or calls job-hunt-agent — the two projects stay
independent, connected only by these plain files.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import date
from pathlib import Path

import typer

from job_radar.core.models import ScoredPosting
from job_radar.core.seen_store import SeenPostingStore
from job_radar.core.source import load_companies, pull_postings

app = typer.Typer(add_completion=False, help="Pull job postings from ATS APIs and score them for ghost-job risk.")

_DEFAULT_HOME = Path(os.environ.get("JOB_RADAR_HOME", Path.cwd()))
_DEFAULT_COMPANIES_PATH = Path(
    os.environ.get("JOB_RADAR_COMPANIES_PATH", str(_DEFAULT_HOME / "config" / "companies.json"))
)
_DEFAULT_SEEN_STORE_PATH = Path(
    os.environ.get("JOB_RADAR_SEEN_STORE_PATH", str(_DEFAULT_HOME / "output" / "seen_store.json"))
)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def _posting_slug(sp: ScoredPosting) -> str:
    return f"{_slugify(sp.posting.company)}--{_slugify(sp.posting.title)}--{sp.posting.external_id}"


@app.command("pull")
def pull_cmd(
    companies_path: Path = typer.Option(
        _DEFAULT_COMPANIES_PATH, "--companies-path", envvar="JOB_RADAR_COMPANIES_PATH"
    ),
    seen_store_path: Path = typer.Option(
        _DEFAULT_SEEN_STORE_PATH, "--seen-store-path", envvar="JOB_RADAR_SEEN_STORE_PATH"
    ),
    home: Path = typer.Option(_DEFAULT_HOME, "--home", envvar="JOB_RADAR_HOME"),
) -> None:
    """Fetch all configured companies' postings, score ghost-risk, write output files."""
    companies = load_companies(companies_path)
    seen_store = SeenPostingStore(seen_store_path)
    scored = asyncio.run(pull_postings(companies, seen_store))

    out_dir = home / "output" / "postings" / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    for sp in scored:
        slug = _posting_slug(sp)
        (out_dir / f"{slug}.json").write_text(sp.model_dump_json(indent=2), encoding="utf-8")
        (out_dir / f"{slug}.txt").write_text(sp.posting.raw_text, encoding="utf-8")

    typer.echo(f"Pulled {len(scored)} postings from {len(companies)} companies -> {out_dir}")
    typer.echo(f"{'GHOST':>6}  {'COMPANY':<20} {'TITLE'}")
    for sp in scored:
        typer.echo(f"{sp.ghost.score:>6.2f}  {sp.posting.company:<20} {sp.posting.title}")


@app.command("list")
def list_cmd(
    postings_dir: Path = typer.Option(None, "--postings-dir", help="Defaults to <home>/output/postings"),
    max_ghost_score: float | None = typer.Option(
        None, "--max-ghost-score", help="Hard-exclude postings scored above this (default: show everything)."
    ),
    company: str | None = typer.Option(None, "--company"),
    title_contains: str | None = typer.Option(
        None, "--title-contains", help="Case-insensitive substring match against the posting title, e.g. 'Data Scientist'."
    ),
    location_contains: str | None = typer.Option(
        None,
        "--location-contains",
        help="Case-insensitive substring match against the posting location, e.g. 'Remote' or a city name. "
        "Matches whatever string the ATS itself reports — wording varies by company (some tag a posting "
        "'Remote', others 'Remote - US', others just a city with no remote tag at all), so try a few terms.",
    ),
    home: Path = typer.Option(_DEFAULT_HOME, "--home", envvar="JOB_RADAR_HOME"),
) -> None:
    """Re-list previously pulled postings, sorted by ghost score ascending (safest first)."""
    postings_dir = postings_dir or (home / "output" / "postings")
    if not postings_dir.exists():
        typer.echo(f"No postings found under {postings_dir} — run `pull` first.")
        return

    records = [
        ScoredPosting.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(postings_dir.glob("**/*.json"))
    ]
    if company is not None:
        records = [r for r in records if r.posting.company.lower() == company.lower()]
    if title_contains is not None:
        records = [r for r in records if title_contains.lower() in r.posting.title.lower()]
    if location_contains is not None:
        records = [
            r
            for r in records
            if r.posting.location is not None and location_contains.lower() in r.posting.location.lower()
        ]
    if max_ghost_score is not None:
        records = [r for r in records if r.ghost.score <= max_ghost_score]

    records.sort(key=lambda r: r.ghost.score)
    if not records:
        typer.echo("No matching postings.")
        return

    typer.echo(f"{'GHOST':>6}  {'COMPANY':<20} {'LOCATION':<20} {'TITLE'}")
    for r in records:
        location = r.posting.location or "—"
        typer.echo(f"{r.ghost.score:>6.2f}  {r.posting.company:<20} {location:<20} {r.posting.title}")


@app.command("show")
def show_cmd(json_path: Path = typer.Argument(..., help="Path to a scored posting .json file")) -> None:
    """Show full detail of one scored posting, including every ghost-signal reason."""
    if not json_path.is_file():
        typer.echo(f"No such file: {json_path}", err=True)
        raise typer.Exit(code=1)
    record = ScoredPosting.model_validate_json(json_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    app()
