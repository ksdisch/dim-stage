#!/usr/bin/env python3
"""CI guard: prove the suite actually ran, and that it still runs what it claims.

Two failure modes, both of which shipped in this portfolio and neither of which
an ordinary green build catches:

1. CI reports success without executing a single assertion. `uv run <file>`
   did exactly this for weeks — every suite imported, defined its test
   functions, and exited 0. A `--collect-only` count cannot detect it, because
   collection is a separate process from the run: revert the run step to the
   broken form and the count is still right. So this reads the JUnit report the
   run step itself produced. No report means pytest never ran.

2. The advertised test count is true only on the author's machine. Suites
   gated on gitignored artifacts skip wholesale in a clean checkout while
   collection still counts them, so "1002 tests" quietly becomes 835 for
   everyone else. That is why there are two floors: `collected` guards against
   tests disappearing, `executed` guards against them silently going dark.

Usage: guard.py <junit.xml> <collected-floor> <executed-floor>
"""
import sys
import xml.etree.ElementTree as ET

report = sys.argv[1]
collected_floor = int(sys.argv[2])
executed_floor = int(sys.argv[3])

try:
    root = ET.parse(report).getroot()
except (OSError, ET.ParseError) as exc:
    sys.exit(f"::error::no JUnit report at {report} ({exc}) — pytest did not run")

suites = root.findall("testsuite") or [root]
collected = sum(int(s.get("tests", 0)) for s in suites)
skipped = sum(int(s.get("skipped", 0)) for s in suites)
bad = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
executed = collected - skipped

print(f"run reported: {collected} collected, {executed} executed, {skipped} skipped, {bad} failed/errored")
print(f"floors:       {collected_floor} collected, {executed_floor} executed")

if bad:
    sys.exit(f"::error::{bad} test(s) failed or errored")
if executed == 0:
    sys.exit("::error::0 tests executed — the vacuous-CI mode this guard exists to catch")
if collected < collected_floor:
    sys.exit(f"::error::{collected} collected, expected >= {collected_floor} — tests have disappeared")
if executed < executed_floor:
    sys.exit(
        f"::error::{executed} executed, expected >= {executed_floor} — "
        f"{skipped} skipped; a suite has gone dark rather than failing"
    )
print("ok")
