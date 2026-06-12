"""AutoFlatten benchmark / autoresearch harness.

Dev tooling (not part of the installed ``autoflatten`` package) for a reproducible,
provenance-tracked optimization benchmark of the pyflatten flattening stage.

See ``benchmark/PLAN.md`` for the full design and ``.claude/skills/autoflatten-autoresearch/``
for session-recovery instructions.

Code lives in the repo; **all generated artifacts** (ledger, manifest, configs, flat
patches, caches) are written under :data:`benchmark.paths.DATA_ROOT`.
"""
