"""job-radar CLI entrypoint.

Usage:
    python -m job_radar.cli pull
    python -m job_radar.cli pull --companies-path config/companies.json
    python -m job_radar.cli list
    python -m job_radar.cli list --max-ghost-score 0.4
    python -m job_radar.cli show output/postings/2026-07-02/acme--data-scientist.json
    python -m job_radar.cli discover-slug "Acadia Pharmaceuticals"
    python -m job_radar.cli discover-slug "Acadia Pharmaceuticals" --add

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
import logging
import os
import re
from datetime import date
from pathlib import Path

import structlog
import typer

from job_radar.core.models import DiscoveredSlug, ScoredPosting
from job_radar.core.seen_store import SeenPostingStore
from job_radar.core.slug_discovery import discover_company
from job_radar.core.source import load_companies, pull_postings

app = typer.Typer(add_completion=False, help="Pull job postings from ATS APIs and score them for ghost-job risk.")

_US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}
_US_MARKERS = {"united states", "usa", "us"} | _US_STATES
# Not exhaustive — covers the countries that actually show up on Greenhouse/
# Lever/Ashby boards as an explicit non-US remote option. Ambiguous names that
# collide with US states/cities (e.g. "Georgia") are deliberately left out;
# false negatives here just mean an international posting slips through, which
# a human still filters out at review time, vs. false-positives hiding a real
# US posting.
_NON_US_REMOTE_MARKERS = {
    "canada", "mexico", "brazil", "argentina", "chile", "colombia", "peru", "uruguay",
    "costa rica", "panama", "uk", "united kingdom", "ireland", "france", "germany",
    "spain", "italy", "portugal", "netherlands", "poland", "sweden", "switzerland",
    "austria", "belgium", "denmark", "norway", "finland", "romania", "ukraine",
    "greece", "czech republic", "hungary", "australia", "new zealand", "japan",
    "china", "hong kong", "singapore", "india", "philippines", "indonesia",
    "malaysia", "thailand", "vietnam", "korea", "taiwan", "israel",
    "united arab emirates", "south africa", "nigeria", "kenya", "egypt", "turkey",
    "europe", "emea", "apac", "latam",
}


def _is_us_remote(location: str) -> bool:
    """Whether a 'remote' location string reads as US-based remote.

    ATS location fields are free text with no separate country field, and
    often list several offices/regions in one string (e.g. "London, UK;
    Remote-Friendly, United States; San Francisco, CA") — so this only
    excludes a posting when it names a non-US region *and* names no US one;
    a mixed listing that includes a US remote option still passes.
    """
    loc = location.lower()
    has_us_marker = any(re.search(rf"\b{re.escape(m)}\b", loc) for m in _US_MARKERS)
    has_non_us_marker = any(re.search(rf"\b{re.escape(m)}\b", loc) for m in _NON_US_REMOTE_MARKERS)
    return has_us_marker or not has_non_us_marker

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
        "'Remote', others 'Remote - US', others just a city with no remote tag at all), so try a few terms. "
        "When the term itself contains 'remote', results are further limited to US-remote postings — "
        "international-only remote listings (e.g. 'Remote - Australia', 'France (Remote)') are excluded.",
    ),
    paths: bool = typer.Option(
        False,
        "--paths",
        help="Print each matched posting's .txt file path (one per line) instead of the table — "
        "for piping into a shell loop, e.g. `job_hunt_agent.cli match --posting <path>`.",
    ),
    home: Path = typer.Option(_DEFAULT_HOME, "--home", envvar="JOB_RADAR_HOME"),
) -> None:
    """Re-list previously pulled postings, sorted by ghost score ascending (safest first)."""
    postings_dir = postings_dir or (home / "output" / "postings")
    if not postings_dir.exists():
        if not paths:
            typer.echo(f"No postings found under {postings_dir} — run `pull` first.")
        return

    entries = [
        (p, ScoredPosting.model_validate_json(p.read_text(encoding="utf-8")))
        for p in sorted(postings_dir.glob("**/*.json"))
    ]
    if company is not None:
        entries = [(p, r) for p, r in entries if r.posting.company.lower() == company.lower()]
    if title_contains is not None:
        entries = [(p, r) for p, r in entries if title_contains.lower() in r.posting.title.lower()]
    if location_contains is not None:
        entries = [
            (p, r)
            for p, r in entries
            if r.posting.location is not None and location_contains.lower() in r.posting.location.lower()
        ]
        if "remote" in location_contains.lower():
            entries = [(p, r) for p, r in entries if _is_us_remote(r.posting.location)]
    if max_ghost_score is not None:
        entries = [(p, r) for p, r in entries if r.ghost.score <= max_ghost_score]

    entries.sort(key=lambda pr: pr[1].ghost.score)

    if paths:
        for p, _r in entries:
            typer.echo(str(p.with_suffix(".txt")))
        return

    if not entries:
        typer.echo("No matching postings.")
        return

    typer.echo(f"{'GHOST':>6}  {'COMPANY':<20} {'LOCATION':<20} {'TITLE'}")
    for _p, r in entries:
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


def _add_company_to_config(companies_path: Path, name: str, ats: str, slug: str) -> bool:
    """Append {name, ats, slug} to companies.json, preserving any _comment/
    _note fields already there. Returns False without writing if (ats, slug)
    is already present, so re-running --add is always safe to repeat."""
    raw = json.loads(companies_path.read_text(encoding="utf-8"))
    existing = raw.setdefault("companies", [])
    for entry in existing:
        if entry.get("ats") == ats and entry.get("slug") == slug:
            return False
    existing.append({"name": name, "ats": ats, "slug": slug})
    companies_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return True


@app.command("discover-slug")
def discover_slug_cmd(
    company_name: str = typer.Argument(..., help="Company name to guess ATS slugs for, e.g. 'Acadia Pharmaceuticals'"),
    add: bool = typer.Option(
        False,
        "--add",
        help="Append the confirmed match to companies.json. Only writes when exactly one match is found, "
        "or when --pick disambiguates among several — never guesses which one is the real company.",
    ),
    pick: str | None = typer.Option(
        None,
        "--pick",
        help="Disambiguate among multiple matches: '<ats>:<slug>', e.g. 'greenhouse:acadiapharmaceuticals'.",
    ),
    companies_path: Path = typer.Option(
        _DEFAULT_COMPANIES_PATH, "--companies-path", envvar="JOB_RADAR_COMPANIES_PATH"
    ),
) -> None:
    """Guess plausible ATS slugs for a company name and verify each against the live API.

    Every candidate slug is checked against the real Greenhouse/Lever/Ashby
    APIs (no scraping, no guessing without verification) — only slugs that
    come back with actual live postings are reported. A short/common company
    name can coincidentally match an unrelated real board, so review the
    matches before trusting one, especially when job_count is small.
    """
    # Most candidate slugs are expected to 404 — that's the whole point of trying
    # several. ats_clients.py logs each miss at warning level (appropriate for a
    # real `pull`, where a 404 means something's actually wrong), which would
    # otherwise flood this command with noise for every wrong guess. Quieted to
    # errors-only for the probe, then restored immediately after.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
    try:
        matches = asyncio.run(discover_company(company_name))
    finally:
        structlog.reset_defaults()

    if not matches:
        typer.echo(f"No live ATS board found for '{company_name}' among the slug spellings tried.")
        typer.echo("Try a shorter or differently-spelled name, or find the slug manually from the careers page URL.")
        return

    typer.echo(f"Confirmed live boards for '{company_name}':")
    for m in matches:
        typer.echo(f"  {m.ats:<10} {m.slug:<30} {m.job_count} jobs")

    if not add:
        return

    chosen: DiscoveredSlug
    if pick is not None:
        ats_pick, _, slug_pick = pick.partition(":")
        found = [m for m in matches if m.ats == ats_pick and m.slug == slug_pick]
        if not found:
            typer.echo(f"\n'{pick}' isn't among the confirmed matches above — nothing added.", err=True)
            raise typer.Exit(code=1)
        chosen = found[0]
    elif len(matches) == 1:
        chosen = matches[0]
    else:
        typer.echo(
            "\nMultiple matches found — refusing to guess which one is the real company. "
            "Re-run with --pick '<ats>:<slug>' to choose.",
            err=True,
        )
        raise typer.Exit(code=1)

    added = _add_company_to_config(companies_path, company_name, chosen.ats, chosen.slug)
    if added:
        typer.echo(f"\nAdded {company_name} ({chosen.ats}:{chosen.slug}) to {companies_path}")
    else:
        typer.echo(f"\n{chosen.ats}:{chosen.slug} is already in {companies_path} — nothing added.")


if __name__ == "__main__":
    app()
