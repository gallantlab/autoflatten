"""Tests for autoflatten.config module."""

from pathlib import Path

from autoflatten import config


class TestConfig:
    """Tests for configuration module."""

    def test_here_is_path(self):
        """HERE should be a Path object."""
        assert isinstance(config.HERE, Path)

    def test_here_exists(self):
        """HERE should point to an existing directory."""
        assert config.HERE.exists()
        assert config.HERE.is_dir()

    def test_default_template_dir_is_path(self):
        """DEFAULT_TEMPLATE_DIR should be a Path object."""
        assert isinstance(config.DEFAULT_TEMPLATE_DIR, Path)

    def test_default_template_dir_exists(self):
        """DEFAULT_TEMPLATE_DIR should point to an existing directory."""
        assert config.DEFAULT_TEMPLATE_DIR.exists()
        assert config.DEFAULT_TEMPLATE_DIR.is_dir()

    def test_fsaverage_cut_template_is_path(self):
        """fsaverage_cut_template should be a Path object."""
        assert isinstance(config.fsaverage_cut_template, Path)

    def test_fsaverage_cut_template_exists(self):
        """fsaverage_cut_template should point to an existing file."""
        assert config.fsaverage_cut_template.exists()
        assert config.fsaverage_cut_template.is_file()

    def test_fsaverage_cut_template_is_json(self):
        """fsaverage_cut_template should be a JSON file."""
        assert config.fsaverage_cut_template.suffix == ".json"

    def test_template_dir_is_inside_package(self):
        """DEFAULT_TEMPLATE_DIR should be inside the package directory."""
        assert config.DEFAULT_TEMPLATE_DIR.parent == config.HERE
