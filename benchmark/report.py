"""Render the experiment ledger into a human-readable lab notebook.

Reads ``experiments.jsonl`` and writes ``NOTEBOOK.md`` (under
:data:`benchmark.paths.DATA_ROOT`) — the inspectable, shareable narrative of every
experiment, each pinned to its commit SHA and repro command.

Usage
-----
    python -m benchmark.report
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import paths
from .ledger import Ledger

_METRIC_KEYS = [
    "n_patches",
    "mean_distortion",
    "worst_distortion",
    "total_flipped",
    "frac_patches_with_flips",
    "mean_runtime_s",
]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render(records: list[dict[str, Any]]) -> str:
    lines = ["# AutoFlatten autoresearch — lab notebook", ""]
    lines.append(
        "Auto-generated from `ledger/experiments.jsonl`. Each row is one experiment; "
        "see the ledger for full provenance (env, seeds, per-subject metrics, diffs)."
    )
    lines.append("")

    if not records:
        lines.append("_No experiments logged yet._")
        return "\n".join(lines) + "\n"

    # Summary table
    header = ["id", "kind", "label", "commit"] + _METRIC_KEYS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in records:
        commit = (r.get("git", {}).get("commit") or "")[:8]
        dirty = "*" if r.get("git", {}).get("dirty") else ""
        m = r.get("metrics", {})
        row = [
            r.get("experiment_id", ""),
            r.get("kind", ""),
            r.get("label", ""),
            commit + dirty,
        ] + [_fmt(m.get(k, "")) for k in _METRIC_KEYS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-experiment detail
    for r in records:
        lines.append(
            f"## {r.get('label', r.get('experiment_id'))}  (`{r.get('experiment_id')}`)"
        )
        lines.append("")
        lines.append(
            f"- **kind**: {r.get('kind')}  •  **timestamp**: {r.get('timestamp')}  •  **status**: {r.get('status')}"
        )
        git = r.get("git", {})
        lines.append(
            f"- **commit**: `{git.get('commit')}` ({git.get('branch')})"
            + ("  ⚠️ dirty tree" if git.get("dirty") else "")
        )
        env = r.get("environment", {})
        lines.append(
            f"- **env**: jax {env.get('jax_version')} [{env.get('jax_backend')}], "
            f"python {env.get('python')}, host {env.get('hostname')}"
        )
        if r.get("method"):
            lines.append(f"- **method**: {r['method'].get('name')}")
        if r.get("repro_command"):
            lines.append(f"- **repro**: `{r['repro_command']}`")
        decision = r.get("decision", {})
        if decision.get("hypothesis"):
            lines.append(f"- **hypothesis**: {decision['hypothesis']}")
        if decision.get("conclusion"):
            lines.append(f"- **conclusion**: {decision['conclusion']}")
        if decision.get("next_step"):
            lines.append(f"- **next step**: {decision['next_step']}")
        if "determinism_check" in decision:
            lines.append(
                f"- **determinism**: {decision['determinism_check'].get('deterministic')}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=paths.LEDGER_PATH)
    ap.add_argument("--out", type=Path, default=paths.NOTEBOOK_PATH)
    args = ap.parse_args()

    records = Ledger(args.ledger).read()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(records))
    print(f"Wrote {args.out}  ({len(records)} experiments)")


if __name__ == "__main__":
    main()
