"""Fast, JAX-free unit tests for the benchmark harness plumbing.

These cover the pure-Python logic (ledger I/O, aggregation, manifest selection, metric
helpers) without running the flattening optimizer, so they're quick and deterministic.
The end-to-end flatten path is verified separately via ``run_baseline``.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import metrics
from benchmark.harness import select_entries
from benchmark.ledger import ExperimentRecord, Ledger


# --- ledger round-trip ------------------------------------------------------------
def _make_record(eid: str, kind: str = "baseline") -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=eid,
        timestamp="2026-01-01T00:00:00+00:00",
        kind=kind,
        label=f"test:{eid}",
        git={"commit": "abc123", "dirty": False},
        environment={"jax_backend": "cpu"},
        metrics={"mean_distortion": 12.5, "n_patches": 3},
    )


def test_ledger_append_and_read_roundtrip(tmp_path):
    ledger = Ledger(tmp_path / "experiments.jsonl")
    assert ledger.read() == []  # absent ledger reads empty

    ledger.append(_make_record("aaa"))
    ledger.append(_make_record("bbb", kind="probe"))

    records = ledger.read()
    assert [r["experiment_id"] for r in records] == ["aaa", "bbb"]
    assert records[0]["metrics"]["mean_distortion"] == 12.5
    assert ledger.latest()["experiment_id"] == "bbb"
    assert ledger.latest(kind="baseline")["experiment_id"] == "aaa"


def test_ledger_is_append_only(tmp_path):
    ledger = Ledger(tmp_path / "experiments.jsonl")
    ledger.append(_make_record("first"))
    ledger.append(_make_record("second"))
    # Second append must not overwrite the first.
    assert len(ledger.read()) == 2


# --- aggregation ------------------------------------------------------------------
def test_aggregate_basic_stats():
    per_subject = [
        {
            "status": "ok",
            "mean_distortion": 10.0,
            "p90_distortion": 20.0,
            "n_flipped": 0,
            "runtime_s": 5.0,
        },
        {
            "status": "ok",
            "mean_distortion": 20.0,
            "p90_distortion": 40.0,
            "n_flipped": 3,
            "runtime_s": 7.0,
        },
    ]
    agg = metrics.aggregate(per_subject)
    assert agg["n_patches"] == 2
    assert agg["mean_distortion"] == pytest.approx(15.0)
    assert agg["worst_distortion"] == pytest.approx(20.0)
    assert agg["total_flipped"] == 3
    assert agg["frac_patches_with_flips"] == pytest.approx(0.5)
    assert agg["mean_runtime_s"] == pytest.approx(6.0)


def test_aggregate_excludes_errors():
    per_subject = [
        {
            "status": "ok",
            "mean_distortion": 10.0,
            "p90_distortion": 10.0,
            "n_flipped": 0,
            "runtime_s": 1.0,
        },
        {"status": "error", "error": "boom"},
    ]
    agg = metrics.aggregate(per_subject)
    assert agg["n_patches"] == 1
    assert agg["n_failed"] == 1


def test_aggregate_all_failed():
    agg = metrics.aggregate([{"status": "error"}, {"status": "error"}])
    assert agg["n_patches"] == 0
    assert agg["n_failed"] == 2


# --- manifest selection -----------------------------------------------------------
def _manifest():
    return {
        "entries": [
            {"subject": "sub-001", "hemi": "lh", "split": "train"},
            {"subject": "sub-001", "hemi": "rh", "split": "train"},
            {"subject": "sub-002", "hemi": "lh", "split": "holdout"},
        ]
    }


def test_select_entries_split_and_subset():
    m = _manifest()
    assert len(select_entries(m)) == 3
    assert len(select_entries(m, split="train")) == 2
    assert len(select_entries(m, split="holdout")) == 1
    assert len(select_entries(m, subset=1)) == 1
    assert select_entries(m, split="train", subset=1)[0]["hemi"] == "lh"


# --- metric helpers (numpy-only) --------------------------------------------------
def test_area_distortion_unit_square():
    # Two triangles tiling a unit square: total 2D area = 1.0.
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    # Matching 3D area -> zero distortion; doubled 3D area -> 0.5.
    assert metrics._area_distortion(uv, faces, orig_area=1.0) == pytest.approx(0.0)
    assert metrics._area_distortion(uv, faces, orig_area=2.0) == pytest.approx(0.5)


def test_per_vertex_p90_zero_when_isometric():
    # A perfectly isometric embedding: 2D distances equal targets -> 0 error.
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    neighbors = np.array([[1, 2], [0, 2], [0, 1]])
    targets = np.array([[1.0, 1.0], [1.0, np.sqrt(2)], [1.0, np.sqrt(2)]])
    mask = np.ones_like(targets, dtype=bool)
    assert metrics._per_vertex_p90(uv, neighbors, targets, mask) == pytest.approx(0.0)
