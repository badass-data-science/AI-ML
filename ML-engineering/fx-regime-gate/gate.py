"""
FX regime confirmation gate -- a two-signal situational-awareness dashboard.

Combines VIX level with fx-pcn direction-flip burst status to answer one
question: is a current volatility spike actually showing up as structural
change in FX pair correlations, or is it (so far) an equity-only event?

Background / where the thresholds come from:
    forex-partial-correlation-network/EDA/FINDINGS.md, 2026-08-31 entry.
    - Burst (>=7 directed-edge flips in a week) is a real but tail-only
      signal: high-burst weeks have significantly higher VIX than the rest
      (median 22 vs 16, Mann-Whitney p=0.0007, robust to excluding COVID).
    - VIX leads bursts by 1-10 weeks; bursts do NOT lead VIX (Granger
      causality, placebo-tested). So this gate is a CONFIRMATION/FILTER
      tool, not an early-warning one -- it never fires before VIX moves.

Data sources:
    - fx-pcn `density`/`direction` parquet output, config:
      window-days-60, step-days-7, min-observations-30, max-lag-3,
      fdr-alpha-0.05, granularity-D. Read from
      ~/output/forex-partial-correlation-network/ (populated by fx-pcn's
      own Prefect pipeline -- this script only reads it).
    - VIX daily close via yfinance (^VIX).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FX_PCN_OUTPUT_DIR = Path.home() / "output" / "forex-partial-correlation-network"
FX_PCN_CONFIG = (
    "window-days-60---step-days-7---min-observations-30---max-lag-3"
    "---fdr-alpha-0.05---granularity-D"
)
NETWORK_NAME = "network-name-forex-network-seven-majors"

# Thresholds, sourced directly from FINDINGS.md's empirical results.
BURST_THRESHOLD = 7  # flips/week; ~90th percentile of the historical distribution
VIX_ELEVATED_THRESHOLD = 20.0  # standard industry "elevated" band; close to the
                                # 22-median VIX observed among historical burst weeks
BURST_LOOKBACK_STEPS = 2  # a burst up to 2 weekly steps ago still counts as "recent",
                           # since VIX leading bursts by 1-2 weeks was the strongest lag


@dataclass
class GateStatus:
    as_of_date: pd.Timestamp
    vix_level: float
    vix_elevated: bool
    current_flip_count: int
    weeks_since_last_burst: int | None
    recent_burst: bool
    recent_flip_counts: list[int]
    color: str


def load_fx_pcn_series() -> tuple[pd.DatetimeIndex, pd.Series]:
    density_path = (
        FX_PCN_OUTPUT_DIR / f"density---{NETWORK_NAME}---{FX_PCN_CONFIG}.parquet"
    )
    direction_path = (
        FX_PCN_OUTPUT_DIR / f"direction---{NETWORK_NAME}---{FX_PCN_CONFIG}.parquet"
    )
    if not density_path.exists() or not direction_path.exists():
        raise FileNotFoundError(
            f"Expected fx-pcn output not found under {FX_PCN_OUTPUT_DIR}. "
            "Run fx-pcn's pipeline first."
        )

    density = pd.read_parquet(density_path)
    direction = pd.read_parquet(direction_path)
    direction["date"] = pd.to_datetime(direction["date"])

    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(density["date"]).unique()))
    flip_counts = direction.groupby("date").size().reindex(all_dates, fill_value=0)
    return all_dates, flip_counts


def fetch_vix(as_of: pd.Timestamp) -> float:
    vix = yf.download(
        "^VIX",
        start=(as_of - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        end=(as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
    )
    vix.columns = vix.columns.get_level_values(0)
    close = vix["Close"].dropna()
    if close.empty:
        raise RuntimeError("Could not fetch a recent VIX close from yfinance.")
    return float(close.iloc[-1])


def compute_gate_status() -> GateStatus:
    all_dates, flip_counts = load_fx_pcn_series()
    as_of_date = all_dates[-1]
    vix_level = fetch_vix(as_of_date)
    vix_elevated = vix_level >= VIX_ELEVATED_THRESHOLD

    current_flip_count = int(flip_counts.iloc[-1])
    recent_window = flip_counts.iloc[-BURST_LOOKBACK_STEPS:]
    recent_burst = bool((recent_window >= BURST_THRESHOLD).any())

    burst_dates = flip_counts[flip_counts >= BURST_THRESHOLD].index
    if len(burst_dates) > 0:
        weeks_since_last_burst = int(
            (as_of_date - burst_dates[-1]).days // 7
        )
    else:
        weeks_since_last_burst = None

    if vix_elevated and recent_burst:
        color = "RED"
    elif vix_elevated or recent_burst:
        color = "YELLOW"
    else:
        color = "GREEN"

    return GateStatus(
        as_of_date=as_of_date,
        vix_level=vix_level,
        vix_elevated=vix_elevated,
        current_flip_count=current_flip_count,
        weeks_since_last_burst=weeks_since_last_burst,
        recent_burst=recent_burst,
        recent_flip_counts=[int(x) for x in flip_counts.iloc[-8:]],
        color=color,
    )


INTERPRETATION = {
    "RED": (
        "VIX is elevated AND the fx-pcn network has burst recently. This is "
        "the one state where fx-pcn's own analysis found a real, "
        "placebo-surviving link to volatility. It CONFIRMS that whatever is "
        "driving VIX up is also showing up as structural change in FX "
        "correlations -- not just an equity-side event. Treat as: the "
        "regime shift is FX-relevant, worth re-validating pair models "
        "against (see idea #2, ideas.txt) and worth distrusting any "
        "currently 'validated' forex-ML model until re-checked."
    ),
    "YELLOW": (
        "Only one of the two signals is firing. Historically this is common "
        "and NOT on its own evidence of an FX-relevant regime shift -- most "
        "elevated-VIX weeks never produce a burst, and it's possible to see "
        "a burst without VIX being elevated. Treat as: worth watching, not "
        "worth acting on by itself."
    ),
    "GREEN": (
        "Neither signal is firing. No evidence of elevated volatility or "
        "structural FX change this week. Normal/quiet regime by both "
        "measures."
    ),
}

HOW_TO_READ_PARAGRAPHS = [
    "This is a CONFIRMATION tool, not an early-warning one. VIX has been "
    "shown to lead fx-pcn bursts by 1-10 weeks (Granger causality, "
    "placebo-tested) -- bursts do NOT lead VIX. So a color change here "
    "never happens before VIX has already moved; the fx-pcn side only "
    "tells you whether that VIX move is showing up structurally in FX "
    "pair relationships, which VIX alone can't tell you (VIX can spike "
    "for reasons that never touch FX).",
    "The burst signal itself is real but narrow: it's a TAIL effect. "
    "Across the full historical range, VIX level barely correlates with "
    "weekly flip count (Spearman r=0.055, not significant) -- the "
    "relationship only shows up when comparing the most extreme burst "
    "weeks (top ~6%, >=7 flips) against everything else. Most "
    "elevated-VIX weeks do NOT produce a burst. A burst is close to "
    'necessary-but-not-sufficient for "VIX is genuinely FX-relevant '
    'right now," not a general-purpose volatility predictor.',
]

COLOR_EMOJI = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}
COLOR_CSS = {"RED": "#c0392b", "YELLOW": "#b7950b", "GREEN": "#1e8449"}

COLOR_LEGEND = [
    (
        "RED",
        f"VIX elevated (>= {VIX_ELEVATED_THRESHOLD}) AND a burst "
        f"(>= {BURST_THRESHOLD} flips) in the last {BURST_LOOKBACK_STEPS} "
        "week(s)",
        "Real, placebo-surviving co-occurrence signal from FINDINGS.md. "
        "Treat as FX-relevant.",
    ),
    (
        "YELLOW",
        "Only one of the two signals firing",
        "Common; not on its own evidence of anything FX-specific.",
    ),
    (
        "GREEN",
        "Neither signal firing",
        "Quiet regime by both measures.",
    ),
]


def render_report(status: GateStatus) -> str:
    lines = []
    lines.append("# FX Volatility Regime Confirmation Gate (vis-a-vis VIX)")
    lines.append("")
    lines.append(f"*As of fx-pcn evaluation step: {status.as_of_date.date()}*")
    lines.append("")
    lines.append(f"## Gate status: {COLOR_EMOJI[status.color]} {status.color}")
    lines.append("")
    lines.append("### Signal 1 — VIX")
    lines.append("")
    lines.append(f"- **Level:** {status.vix_level:.2f}")
    lines.append(
        f"- **Status:** "
        f"{'ELEVATED (>= ' + str(VIX_ELEVATED_THRESHOLD) + ')' if status.vix_elevated else 'calm'}"
    )
    lines.append("")
    lines.append("### Signal 2 — fx-pcn burst (60d window / 7d step / daily)")
    lines.append("")
    lines.append(f"- **Current week flip count:** {status.current_flip_count}")
    lines.append(f"- **Last {BURST_LOOKBACK_STEPS} weeks:** {status.recent_flip_counts[-BURST_LOOKBACK_STEPS:]}")
    lines.append(
        f"- **Recent burst** (>= {BURST_THRESHOLD} flips within {BURST_LOOKBACK_STEPS} wks): "
        f"{'**YES**' if status.recent_burst else 'no'}"
    )
    if status.weeks_since_last_burst is not None:
        lines.append(f"- **Weeks since last burst:** {status.weeks_since_last_burst}")
    else:
        lines.append("- No burst in available history")
    lines.append(f"- **Last 8 weeks flip counts:** {status.recent_flip_counts}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(INTERPRETATION[status.color])
    lines.append("")
    lines.append("## How to read this gate")
    lines.append("")
    for paragraph in HOW_TO_READ_PARAGRAPHS:
        lines.append(paragraph)
        lines.append("")
    lines.append("| Color | Condition | Meaning |")
    lines.append("| --- | --- | --- |")
    for color, condition, meaning in COLOR_LEGEND:
        lines.append(f"| {COLOR_EMOJI[color]} {color} | {condition} | {meaning} |")
    lines.append("")
    lines.append(
        "Full methodology and all validation tests (shuffle placebo, "
        "reverse-direction placebo, circular-shift placebo): "
        "`forex-partial-correlation-network/EDA/FINDINGS.md`"
    )
    return "\n".join(lines)


def render_report_html(status: GateStatus) -> str:
    color_hex = COLOR_CSS[status.color]
    signal2_rows = "".join(
        f"<li><strong>{label}</strong> {value}</li>"
        for label, value in [
            ("Current week flip count:", status.current_flip_count),
            (
                f"Last {BURST_LOOKBACK_STEPS} weeks:",
                status.recent_flip_counts[-BURST_LOOKBACK_STEPS:],
            ),
            (
                f"Recent burst (&gt;= {BURST_THRESHOLD} flips within "
                f"{BURST_LOOKBACK_STEPS} wks):",
                "<strong>YES</strong>" if status.recent_burst else "no",
            ),
        ]
    )
    weeks_since_row = (
        f"<li><strong>Weeks since last burst:</strong> {status.weeks_since_last_burst}</li>"
        if status.weeks_since_last_burst is not None
        else "<li>No burst in available history</li>"
    )
    legend_rows = "".join(
        f"<tr><td>{COLOR_EMOJI[color]} {color}</td><td>{condition}</td>"
        f"<td>{meaning}</td></tr>"
        for color, condition, meaning in COLOR_LEGEND
    )
    paragraphs = "".join(f"<p>{p}</p>" for p in HOW_TO_READ_PARAGRAPHS)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FX Volatility Regime Confirmation Gate (vis-a-vis VIX)</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 780px; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }}
  .status {{ display: inline-block; padding: 0.25em 0.75em; border-radius: 0.4em;
            color: white; background: {color_hex}; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.5em; text-align: left; vertical-align: top; }}
  th {{ background: #f2f2f2; }}
  code {{ background: #f2f2f2; padding: 0.1em 0.3em; border-radius: 0.3em; }}
</style>
</head>
<body>
<h1>FX Volatility Regime Confirmation Gate (vis-a-vis VIX)</h1>
<p><em>As of fx-pcn evaluation step: {status.as_of_date.date()}</em></p>

<h2>Gate status: <span class="status">{COLOR_EMOJI[status.color]} {status.color}</span></h2>

<h3>Signal 1 &mdash; VIX</h3>
<ul>
  <li><strong>Level:</strong> {status.vix_level:.2f}</li>
  <li><strong>Status:</strong> {'ELEVATED (&gt;= ' + str(VIX_ELEVATED_THRESHOLD) + ')' if status.vix_elevated else 'calm'}</li>
</ul>

<h3>Signal 2 &mdash; fx-pcn burst (60d window / 7d step / daily)</h3>
<ul>
  {signal2_rows}
  {weeks_since_row}
  <li><strong>Last 8 weeks flip counts:</strong> {status.recent_flip_counts}</li>
</ul>

<h2>Interpretation</h2>
<p>{INTERPRETATION[status.color]}</p>

<h2>How to read this gate</h2>
{paragraphs}
<table>
  <tr><th>Color</th><th>Condition</th><th>Meaning</th></tr>
  {legend_rows}
</table>

<p>Full methodology and all validation tests (shuffle placebo,
reverse-direction placebo, circular-shift placebo):
<code>forex-partial-correlation-network/EDA/FINDINGS.md</code></p>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html", action="store_true", help="Render the report as HTML instead of Markdown."
    )
    args = parser.parse_args()
    status = compute_gate_status()
    if args.html:
        print(render_report_html(status))
    else:
        print(render_report(status))


if __name__ == "__main__":
    main()
