# A pytest collection error scores as a legitimate model failure

## Context

First FP4 (NVFP4) agent-OPD run on `Qwen3.8-27B-NVFP4`, round-0 held-out eval
over 12 tasks. Five of the twelve came back `reward=0.000 edited=true [exit 4]`,
all from the same repo (`matthewwithanm__python-markdownify`). Round-0 read
`pass_rate=0.2500 mean_dense=0.4647`.

## Root Cause

pytest exit 4 is a usage/collection error, not a test failure. The repo's tests
import the package under test, which imports `bs4`, which was not installed on
the box:

    markdownify/__init__.py:1: from bs4 import BeautifulSoup, ...
    E   ModuleNotFoundError: No module named 'bs4'

`score_workdir` returns `Ok((0.0, log))` for this — the run is scored, not
errored, so `score_err` stays false and the reward enters training as a real
0.0. In the metrics it is indistinguishable from a model that edited the files
and failed the tests.

A corpus-wide sweep found the same class in two more repos: `exceptiongroup`
and `prettytable` both import a `_version.py` that setuptools-scm generates at
install time and that a source checkout does not carry. Together the three
repos are 48 of the 113 held-out tasks — 42% of the eval split was scoring 0.0
for reasons that have nothing to do with the model.

Two false leads while sweeping, both from the pre-flight not reproducing the
harness:

- collecting in the staged tree reports "no tests collected" for all 11 repos —
  the f2p test files ship in `test_patch`, so the patch has to land first;
- without `PYTHONPATH` set the way `workdir_pythonpath()` sets it (task tree
  ahead of site-packages), self-referential packages fail to import and look
  like missing dependencies.

## Fix

- Installed `beautifulsoup4`; wrote a `_version.py` stub into the 23 staged
  `exceptiongroup` / `prettytable` instances. All 11 repos now collect.
- `scripts/opd_corpus_preflight.py` — applies each repo's test_patch into a temp
  copy, collects its f2p tests under the harness's PYTHONPATH, exits 1 on any
  repo that cannot collect. Run it against a corpus root before spending GPU
  hours on that box.
- Restarted the run (`fp4rl-a` → `fp4rl-b`); a baseline measured half in the
  broken environment and half in the fixed one is not a baseline.

## Rule

A reward of 0.0 is only evidence about the model if the tests actually ran.
Scoring must separate "the tests failed" from "the tests could not run" —
until it does, treat an all-zero cluster sharing one repo as an environment
fault, not a capability reading.
