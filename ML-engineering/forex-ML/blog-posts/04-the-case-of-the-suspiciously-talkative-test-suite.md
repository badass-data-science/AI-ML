# The Case of the Suspiciously Talkative Test Suite

Every origin story needs a mystery, and this one arrived disguised as the most boring
possible complaint: a test was "flaky." It passed sometimes. It failed sometimes. Our
heroine had seen a thousand flaky tests before, usually guilty of the usual crimes —
a race condition, an unseeded random number generator, a timezone having a bad day.
This one was going to be different, and by the end of the afternoon, our heroine was
going to wish, a little, that it hadn't been.

**The Setup**

To actually trust the InfluxDB integration boundary — the part of the pipeline that
pulls real candlestick data out of a real database — our heroine had spun up a real,
disposable InfluxDB instance in Docker, seeded it with fifty rows of clearly synthetic
data (suspiciously smooth prices, a suspiciously round starting timestamp, tags
nobody would mistake for a real trading pair), and pointed the code at it. The test
would query the container, get back exactly fifty rows of obviously-fake data, and
declare victory.

Except sometimes it didn't get back fifty rows. Sometimes it got back a number that
looked uncomfortably like *real trading volume*, spanning a date range that looked
uncomfortably like *actual market history*. The test wasn't flaky. The test was
telling on someone.

**Following the Money**

The credentials for the real, production InfluxDB instance are lazily loaded — fetched
from AWS Secrets Manager only when actually accessed, via a module-level `__getattr__`
trick, specifically so nothing resolves a real secret before it's genuinely needed.
Reasonable design. Foiled entirely by one small, easy-to-miss habit: several modules
imported those credentials the ordinary way —

```python
from database_config import INFLUXDB_URL
```

— instead of importing the module and reaching into it at the point of use. And
`from module import NAME` does something our heroine had, until this exact afternoon,
never had cause to think hard about: it triggers that lazy `__getattr__` loader
*immediately*, at import time, and freezes the resolved value into the importing
module's own namespace, permanently, for the entire life of the process. Not when the
function that needs it finally runs. The instant the module is imported. Which, thanks
to how pytest collects test files before running any of them, meant simply having a
sibling test file in the same directory — one that never even executed, just sat
there, minding its own business, waiting to be collected — was enough to quietly
resolve and permanently bake in the *real* production credentials before a single
`monkeypatch` had any chance to intervene.

Our heroine's test container was completely innocent. It had been sitting there the
whole time, patiently serving its fifty fake rows to nobody, while the code under test
had already, invisibly, made other plans.

**Was Anything Actually Wrong?**

This is the point in a detective story where the story could go one of two very
different directions, and our heroine did not skip the step of finding out which one
she was in. A frozen-at-import-time credential affecting a *read* is an embarrassing
test-isolation bug. A frozen-at-import-time credential affecting a *write* is a
production data incident, and those get treated with an entirely different level of
seriousness. So: every write path in every affected test was checked, by hand, and
every single one of them constructed its own database client directly, with
explicit test-container credentials, bypassing the frozen module-level constants
entirely. Real production data was queried at the exact synthetic timestamps in
question, across the full real historical range, to independently confirm nothing had
been touched. Only then was this filed under "bug," rather than "incident."

**The Fix, and Why It Wasn't Glamorous**

The fix was almost insultingly small: reference the config module and reach into it
at the point of use —

```python
from forex.etl.config import database_config
# ...
database_config.INFLUXDB_URL
```

— rather than importing the frozen value by name. Five files across two repositories
carried the same habit, including one sneaky variant wearing a different hat
entirely: a function default parameter set to the frozen constant directly, which is
just the mutable-default-argument bug's quieter cousin — evaluated once, at
definition time, and never again. Regression tests now assert, in so many words, that
merely *importing* these modules must never freeze a secret — only actually calling
something that needs a live connection is allowed to.

**A Season of Bugs Nobody Introduced**

This one wasn't a solo case, as it turns out. Modernizing a pipeline — actually
running it end to end, against real Spark, real Docker containers, real data shaped
like real data — turned out to be an extremely effective way of discovering bugs that
had been sitting in the *original* notebooks the entire time, undisturbed, because
nobody had ever built the scaffolding required to notice them:

- An LSTM input tensor with its time and feature axes silently swapped — a faithful,
  inherited bug from the original stacking logic, invisible because Keras checks
  shape arithmetic, not shape *meaning*, and two axes of similar size will happily
  multiply together into a number that looks correct right up until you check what
  it actually represents.
- A forward-fill step in the sibling ETL pipeline that ran its cleanup step in the
  wrong order, calling a blanket `dropna()` *before* the forward-fill instead of
  after — meaning every gap the fill was supposed to bridge got deleted first,
  making the entire forward-fill step a very expensive no-op.
- A forward-filled dataset that was faithfully computed in memory and then never
  actually written back to the database at all — a whole pipeline stage that did
  real work and then quietly threw it away.
- A shared Spark session getting torn down out from under other flows sharing the
  same process, because one flow's cleanup `finally` block assumed it was the only
  tenant in the building.

None of these were introduced by the modernization work. All of them were *found* by
it — because tests, real integration environments, and actually running the thing
end to end are the only way "it worked in the notebook" and "it actually works" ever
get compared against each other honestly.

**A Closing Note on Trust**

Somewhere around the fifth bug, our heroine's paranoia stopped being funny and started
being load-bearing. Which is, perhaps, the real ending to this particular case: when
it came time to actually regenerate the real production forward-fill data — a genuine
write to a real database, using a script that had, by that point, been built,
documented, and thoroughly tested against synthetic data — our heroine chose to clear
and regenerate it herself, independently, outside the very session that had built the
tool. Not because the tool wasn't trusted. Because *independent* verification is worth
more than convenient verification, and by this point in the story, our heroine had
earned the right to be exactly that stubborn about it.

**AI Use Statement:** Claude Code wrote both the code described in this post — the
fix for the eager secret-resolution bug, and the fixes for the LSTM axis-order bug,
the forward-fill ordering bug, the missing write-back, and the shared-Spark-session
teardown bug — and this post's prose itself, across an extended collaborative session
with the author, spanning both repositories involved.
