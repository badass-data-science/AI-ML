#!/usr/bin/env bash
# Chains job-radar and job-hunt-agent from the command line, no Claude Code
# involved: pull postings, list the ones matching your filters, then (after
# confirmation) run job-hunt-agent's match-and-draft against each one.
# match-and-draft also creates resume-filled.md/cover_letter-filled.md
# automatically for each posting (job-hunt-agent's init-filled, on by
# default) -- those are the editable copies to do your actual human-review
# pass into; re-running this script later never clobbers that work.
#
# This script is pure shell orchestration over each project's own CLI --
# job-radar and job-hunt-agent still never import each other. See each
# project's README.md for one-time setup (venv + pip install + env vars)
# before running this.
#
# Usage:
#   scripts/run_pipeline.sh [--title-contains TEXT] [--location-contains TEXT] \
#                            [--company NAME] [--max-ghost-score FLOAT] [--yes]
#
# Every flag except --yes is passed straight through to `job_radar.cli list`
# to select which pulled postings to match. --yes skips the confirmation
# prompt before running match-and-draft (each call is a real LLM call --
# costs time and, depending on your provider, money).
#
# Examples:
#   scripts/run_pipeline.sh --location-contains "San Diego"
#   scripts/run_pipeline.sh --title-contains "Data Scientist" --location-contains Remote
#   scripts/run_pipeline.sh --company "Acadia Pharmaceuticals" --yes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_RADAR_DIR="$(dirname "$SCRIPT_DIR")"
AI_WORKFLOWS_DIR="$(dirname "$JOB_RADAR_DIR")"
JOB_HUNT_AGENT_DIR="$AI_WORKFLOWS_DIR/job-hunt-agent"

JOB_RADAR_PY="$JOB_RADAR_DIR/.venv/bin/python"
JOB_HUNT_AGENT_PY="$JOB_HUNT_AGENT_DIR/.venv/bin/python"

for py in "$JOB_RADAR_PY" "$JOB_HUNT_AGENT_PY"; do
    if [ ! -x "$py" ]; then
        echo "Missing venv: $py" >&2
        echo "Run the 'Quick start' setup in both job-radar/README.md and job-hunt-agent/README.md first." >&2
        exit 1
    fi
done

YES=false
LIST_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--yes" ]; then
        YES=true
    else
        LIST_ARGS+=("$arg")
    fi
done

echo "=== job-radar: pull ==="
( cd "$JOB_RADAR_DIR" && "$JOB_RADAR_PY" -m job_radar.cli pull )

echo
echo "=== job-radar: postings matching your filters ==="
( cd "$JOB_RADAR_DIR" && "$JOB_RADAR_PY" -m job_radar.cli list "${LIST_ARGS[@]}" )

TXT_PATHS=$(cd "$JOB_RADAR_DIR" && "$JOB_RADAR_PY" -m job_radar.cli list "${LIST_ARGS[@]}" --paths)
if [ -z "$TXT_PATHS" ]; then
    echo
    echo "No postings matched — nothing to hand to job-hunt-agent."
    exit 0
fi
COUNT=$(printf '%s\n' "$TXT_PATHS" | grep -c .)

if [ "$YES" != true ]; then
    echo
    read -rp "Run job-hunt-agent match-and-draft against these $COUNT posting(s)? Each is a real LLM call. [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Stopping here. Re-run with different filters, or 'job_hunt_agent.cli match' by hand against any .txt path above."
        exit 0
    fi
fi

echo
echo "=== job-hunt-agent: match-and-draft ==="
printf '%s\n' "$TXT_PATHS" | while IFS= read -r txt_path; do
    json_path="${txt_path%.txt}.json"
    company=$("$JOB_RADAR_PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['posting']['company'])" "$json_path")
    role=$("$JOB_RADAR_PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['posting']['title'])" "$json_path")
    url=$("$JOB_RADAR_PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['posting']['url'])" "$json_path")
    ghost_score=$("$JOB_RADAR_PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['ghost']['score'])" "$json_path")
    ghost_reasons=$("$JOB_RADAR_PY" -c "import json,sys; print('; '.join(json.load(open(sys.argv[1]))['ghost']['reasons']))" "$json_path")
    echo
    echo "--- $company: $role ---"
    ( cd "$JOB_HUNT_AGENT_DIR" && "$JOB_HUNT_AGENT_PY" -m job_hunt_agent.cli match-and-draft \
        --posting "$txt_path" --company "$company" --role "$role" --url "$url" \
        --ghost-score "$ghost_score" --ghost-reasons "$ghost_reasons" )
done
