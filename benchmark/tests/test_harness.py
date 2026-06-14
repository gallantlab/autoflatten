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


# --- flip-free init probe ---------------------------------------------------------
def _disk_mesh(n: int = 12):
    """A flat triangle-fan disk: center vertex + ``n`` boundary vertices on a circle."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rim = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(n)])
    vertices = np.vstack([[0.0, 0.0, 0.0], rim])
    faces = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)], dtype=np.int64)
    return vertices, faces


def _signed_areas(uv, faces):
    v0, v1, v2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    return 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )


def test_tutte_init_is_flip_free():
    from benchmark.probe_tutte_init import flipfree_init

    vertices, faces = _disk_mesh()
    uv = flipfree_init(vertices, faces, method="tutte")
    areas = _signed_areas(uv, faces)
    # All triangles share one orientation -> zero flips (Tutte guarantee).
    assert np.all(areas > 0) or np.all(areas < 0)


def test_scale_to_area_matches_target():
    from benchmark.probe_tutte_init import scale_to_area

    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)  # area 1
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    scaled = scale_to_area(uv, faces, target_area=9.0)
    assert np.abs(_signed_areas(scaled, faces)).sum() == pytest.approx(9.0)


# --- plot helpers -----------------------------------------------------------------
def test_parse_subject_hemi():
    from benchmark.plot import _parse_subject_hemi

    assert _parse_subject_hemi("/runs/abc/sub-022.lh.flat.patch.3d") == (
        "sub-022",
        "lh",
    )
    assert _parse_subject_hemi("sub-005.rh.flat.patch.3d") == ("sub-005", "rh")


def test_subtitle_formats_known_fields():
    from benchmark.plot import _subtitle

    assert _subtitle({}) == ""
    s = _subtitle({"mean_distortion": 15.25, "n_flipped": 24, "runtime_s": 454.0})
    assert "15.25% dist" in s and "24 flipped" in s and "454s" in s
