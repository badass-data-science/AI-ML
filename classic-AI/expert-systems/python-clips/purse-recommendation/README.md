# Python-CLIPS Purse Recommendation Expert System

An expert system that recommends purses to customers based on customer traits, implemented using the [`clips`](https://pypi.org/project/clips/) Python package — the official Python bindings for the CLIPS rule engine.

This is one of three side-by-side implementations of the same expert system:

| Implementation | Language | Engine |
|---|---|---|
| [`CLIPS/purse-recommendation`](../../CLIPS/purse-recommendation/) | CLIPS | CLIPS 6.30 shell |
| [`clojure-and-clara-rules/purse-recommendation`](../../clojure-and-clara-rules/purse-recommendation/) | Clojure | Clara Rules |
| `python-clips/purse-recommendation` (this project) | Python | CLIPS via `clips` package |

## Overview

Python acts as the orchestration layer: it creates the CLIPS environment, builds the same three rules and templates from the CLIPS version, and hands off to the CLIPS inference engine. The rules themselves are identical to `purse.clp`.

The system demonstrates:

- **Structured facts** via `deftemplate`
- **Initial facts** via `deffacts`
- **Forward chaining**: the size recommendation rules fire first, asserting `purse-recommendation` facts
- **Cascade effect**: the asserted `purse-recommendation` facts trigger a second rule that adds the person's favorite color

## Requirements

```bash
pip install clips
```

## Running

```bash
python purse.py
```

## Expected output

```text
Sally should carry a small purse.
CASCADE EFFECT: Sally should sport a small, blue purse!
Emily should carry a large purse.
CASCADE EFFECT: Emily should sport a large, pink purse!
```

Rule firing order may vary; both people always receive both recommendations.

## How it compares to the other implementations

**vs. CLIPS shell** — The rules and templates are identical. Python replaces the interactive CLIPS shell (`load`, `reset`, `run`) with `clips.Environment` method calls.

**vs. Clara Rules (Clojure)** — Both use the host language to assert initial facts and invoke the engine. Clara uses Clojure records and `defrule` macros; this version uses the CLIPS rule syntax unchanged inside Python strings.
