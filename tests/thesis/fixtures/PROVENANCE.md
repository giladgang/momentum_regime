# Smoke Fixture Provenance — TEMPLATE

This file is overwritten by `tests/thesis/make_smoke_fixture.py` after
each fixture-generation run. The template below documents the schema.

## Schema

- **Generated**: ISO-8601 UTC timestamp of the run
- **git SHA**: HEAD commit at the time of the run (must be a clean tree
  for the fixture to be valid)
- **env hash**: sha256[0:12] of `pip freeze`, identifying the Python
  environment used (must match `requirements.lock`)
- **Wall time**: total runtime in seconds and minutes (target < 30 min)
- **Output root**: temp dir used for the run (deleted after copy unless
  `--keep-output` was passed)
- **Files copied**: the fixture artefacts now committed under
  `tests/thesis/fixtures/`

## Generation

```bash
# Local only — the cloud agent does NOT run this:
python tests/thesis/make_smoke_fixture.py
```

## Required environment

Use the env defined by `requirements.lock` (per `CLAUDE.md`). After
fixture generation, verify that
`pytest tests/thesis/test_e2e_smoke.py` passes against the freshly
written fixtures. Only then commit them.
