"""Vulture whitelist: names that look unused but are intentionally kept.

Referenced from ``[tool.vulture] paths`` in pyproject.toml so these
references count as "used" and don't show up in dead-code reports.
"""

# pytest fixtures: looked up by name via dependency injection, never imported.
from autoflatten.tests import conftest

conftest.mock_freesurfer_env
conftest.temp_subject_dir
conftest.no_freesurfer_env
