#!/usr/bin/env python3
"""CI guard: prove the suite actually ran, and that it still has its tests.

The bug this exists to prevent is not "tests failed" — CI catches that. It is
"CI reported success without executing a single assertion", which is what
`uv run <file>` did here for weeks under a green badge: every suite imported,
defined its test functions, and exited 0.

A `--collect-only` count cannot catch that, because collection is a separate
process from the run: revert the run step to the broken form and the count is
still right. So this reads the JUnit report the run step itself produced. No
report means pytest never ran. A report with too few executed tests means the
suite shrank. Either way the build goes red.
"""
import sys
import xml.etree.ElementTree as ET

report, floor = sys.argv[1], int(sys.argv[2])

try:
    root = ET.parse(report).getroot()
except (OSError, ET.ParseError) as exc:
    sys.exit(f"::error::no JUnit report at {report} ({exc}) — pytest did not run")

suites = root.findall("testsuite") or [root]
total = sum(int(s.get("tests", 0)) for s in suites)
skipped = sum(int(s.get("skipped", 0)) for s in suites)
bad = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
executed = total - skipped

print(f"reported by the run: {total} collected, {executed} executed, {skipped} skipped, {bad} failed/errored")

if bad:
    sys.exit(f"::error::{bad} test(s) failed or errored")
if executed == 0:
    sys.exit("::error::0 tests executed — this is the vacuous-CI mode this guard exists to catch")
if total < floor:
    sys.exit(f"::error::{total} tests collected, expected >= {floor} — the advertised count no longer holds")
print(f"ok: {total} >= floor {floor}, {executed} actually executed")
