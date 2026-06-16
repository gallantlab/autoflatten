"""Tests for autoflatten.cli module."""

import argparse
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

import pytest

from autoflatten.cli import (
    add_backend_args,
    add_common_args,
    add_freesurfer_args,
    add_projection_args,
    add_pyflatten_args,
    check_freesurfer_environment,
    cmd_plot_flatmap,
    cmd_plot_projection,
    main,
)


class TestArgumentParsers:
    """Tests for argument parser helper functions."""

    def test_add_common_args(self):
        """add_common_args should add output-dir, overwrite, hemispheres, parallel."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        # Parse with defaults
        args = parser.parse_args([])
        assert args.output_dir is None
        assert args.overwrite is False
        assert args.hemispheres == "both"
        assert args.parallel is False

    def test_add_common_args_with_values(self):
        """add_common_args arguments should accept values."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(
            [
                "--output-dir",
                "/tmp/out",
                "--overwrite",
                "--hemispheres",
                "lh",
                "--parallel",
            ]
        )
        assert args.output_dir == "/tmp/out"
        assert args.overwrite is True
        assert args.hemispheres == "lh"
        assert args.parallel is True

    def test_add_common_args_hemispheres_choices(self):
        """hemispheres should only accept lh, rh, or both."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        # Valid choices
        for hemi in ["lh", "rh", "both"]:
            args = parser.parse_args(["--hemispheres", hemi])
            assert args.hemispheres == hemi

        # Invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--hemispheres", "invalid"])

    def test_add_projection_args(self):
        """add_projection_args should add template-file and no-refine-geodesic."""
        parser = argparse.ArgumentParser()
        add_projection_args(parser)

        args = parser.parse_args([])
        assert args.template_file is None
        assert args.no_refine_geodesic is False

    def test_add_projection_args_with_values(self):
        """add_projection_args arguments should accept values."""
        parser = argparse.ArgumentParser()
        add_projection_args(parser)

        args = parser.parse_args(
            ["--template-file", "/path/to/template.json", "--no-refine-geodesic"]
        )
        assert args.template_file == "/path/to/template.json"
        assert args.no_refine_geodesic is True

    def test_add_backend_args(self):
        """add_backend_args should add backend choice with pyflatten default."""
        parser = argparse.ArgumentParser()
        add_backend_args(parser)

        args = parser.parse_args([])
        assert args.backend == "pyflatten"

    def test_add_backend_args_choices(self):
        """backend should only accept pyflatten or freesurfer."""
        parser = argparse.ArgumentParser()
        add_backend_args(parser)

        # Valid choices
        for backend in ["pyflatten", "freesurfer"]:
            args = parser.parse_args(["--backend", backend])
            assert args.backend == backend

        # Invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "invalid"])

    def test_add_pyflatten_args(self):
        """add_pyflatten_args should add pyflatten-specific options."""
        parser = argparse.ArgumentParser()
        add_pyflatten_args(parser)

        args = parser.parse_args([])
        assert args.k_ring == 7
        assert args.n_neighbors == 12
        assert args.print_every == 1
        assert args.skip_phase is None
        assert args.skip_spring_smoothing is False
        assert args.skip_neg_area is False
        assert args.pyflatten_config is None
        assert args.n_cores == -1
        assert args.debug_save_distances is False

    def test_add_pyflatten_args_with_values(self):
        """add_pyflatten_args arguments should accept values."""
        parser = argparse.ArgumentParser()
        add_pyflatten_args(parser)

        args = parser.parse_args(
            [
                "--k-ring",
                "5",
                "--n-neighbors",
                "8",
                "--print-every",
                "10",
                "--skip-phase",
                "epoch_1",
                "epoch_2",
                "--skip-spring-smoothing",
                "--skip-neg-area",
                "--pyflatten-config",
                "/path/to/config.json",
                "--n-cores",
                "4",
                "--debug-save-distances",
            ]
        )
        assert args.k_ring == 5
        assert args.n_neighbors == 8
        assert args.print_every == 10
        assert args.skip_phase == ["epoch_1", "epoch_2"]
        assert args.skip_spring_smoothing is True
        assert args.skip_neg_area is True
        assert args.pyflatten_config == "/path/to/config.json"
        assert args.n_cores == 4
        assert args.debug_save_distances is True

    def test_add_pyflatten_args_skip_phase_choices(self):
        """skip-phase should only accept valid epoch names."""
        parser = argparse.ArgumentParser()
        add_pyflatten_args(parser)

        # Valid choices
        for phase in ["epoch_1", "epoch_2", "epoch_3"]:
            args = parser.parse_args(["--skip-phase", phase])
            assert args.skip_phase == [phase]

        # Invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--skip-phase", "invalid_phase"])

    def test_add_freesurfer_args(self):
        """add_freesurfer_args should add FreeSurfer-specific options."""
        parser = argparse.ArgumentParser()
        add_freesurfer_args(parser)

        args = parser.parse_args([])
        assert args.seed is None
        assert args.nthreads == 1
        assert args.distances == [15, 80]
        assert args.n_iterations == 200
        assert args.dilate == 1
        assert args.passes == 1
        assert args.tol == 0.005
        assert args.debug is False

    def test_add_freesurfer_args_with_values(self):
        """add_freesurfer_args arguments should accept values."""
        parser = argparse.ArgumentParser()
        add_freesurfer_args(parser)

        args = parser.parse_args(
            [
                "--seed",
                "42",
                "--nthreads",
                "4",
                "--distances",
                "10",
                "100",
                "--n-iterations",
                "500",
                "--dilate",
                "2",
                "--passes",
                "3",
                "--tol",
                "0.001",
                "--debug",
            ]
        )
        assert args.seed == 42
        assert args.nthreads == 4
        assert args.distances == [10, 100]
        assert args.n_iterations == 500
        assert args.dilate == 2
        assert args.passes == 3
        assert args.tol == 0.001
        assert args.debug is True


class TestCheckFreesurferEnvironment:
    """Tests for check_freesurfer_environment function."""

    def test_missing_freesurfer_home(self):
        """Should return False when FREESURFER_HOME is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure FREESURFER_HOME and SUBJECTS_DIR are not set
            os.environ.pop("FREESURFER_HOME", None)
            os.environ.pop("SUBJECTS_DIR", None)

            success, env_vars = check_freesurfer_environment()

            assert success is False
            assert env_vars["FREESURFER_HOME"] is None

    def test_missing_subjects_dir(self):
        """Should return False when SUBJECTS_DIR is not set."""
        with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}, clear=True):
            success, env_vars = check_freesurfer_environment()

            assert success is False
            assert env_vars["SUBJECTS_DIR"] is None

    def test_missing_mri_label2label(self, mock_freesurfer_env):
        """Should return False when mri_label2label is not in PATH."""
        with patch.object(shutil, "which", return_value=None):
            success, env_vars = check_freesurfer_environment()

            assert success is False

    def test_missing_mri_info(self, mock_freesurfer_env):
        """Should return False when mri_info is not in PATH."""

        def mock_which(cmd):
            if cmd == "mri_label2label":
                return "/opt/freesurfer/bin/mri_label2label"
            return None

        with patch.object(shutil, "which", side_effect=mock_which):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                success, env_vars = check_freesurfer_environment()

                assert success is False

    def test_success_with_valid_environment(self, mock_freesurfer_env):
        """Should return True when environment is properly configured."""

        def mock_which(cmd):
            return f"/opt/freesurfer/bin/{cmd}"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "freesurfer-linux-ubuntu22_x86_64-7.4.1"

        with patch.object(shutil, "which", side_effect=mock_which):
            with patch("subprocess.run", return_value=mock_result):
                success, env_vars = check_freesurfer_environment()

                assert success is True
                assert env_vars["FREESURFER_HOME"] is not None
                assert env_vars["SUBJECTS_DIR"] is not None

    def test_version_below_7_rejected(self, mock_freesurfer_env):
        """Should return False when FreeSurfer version is below 7.0."""

        def mock_which(cmd):
            return f"/opt/freesurfer/bin/{cmd}"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "freesurfer-linux-ubuntu22_x86_64-6.0.0"

        with patch.object(shutil, "which", side_effect=mock_which):
            with patch("subprocess.run", return_value=mock_result):
                success, env_vars = check_freesurfer_environment()

                assert success is False

    def test_version_7_or_higher_accepted(self, mock_freesurfer_env):
        """Should return True when FreeSurfer version is 7.0 or higher."""

        def mock_which(cmd):
            return f"/opt/freesurfer/bin/{cmd}"

        for version in ["7.0.0", "7.4.1", "8.0.0"]:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = f"freesurfer-linux-ubuntu22_x86_64-{version}"

            with patch.object(shutil, "which", side_effect=mock_which):
                with patch("subprocess.run", return_value=mock_result):
                    success, env_vars = check_freesurfer_environment()

                    assert success is True, f"Version {version} should be accepted"

    def test_warning_when_version_unknown(self, mock_freesurfer_env):
        """Should still return True when version cannot be determined but commands exist."""

        def mock_which(cmd):
            return f"/opt/freesurfer/bin/{cmd}"

        mock_result = MagicMock()
        mock_result.returncode = 1  # Non-zero return code
        mock_result.stdout = ""

        with patch.object(shutil, "which", side_effect=mock_which):
            with patch("subprocess.run", return_value=mock_result):
                success, env_vars = check_freesurfer_environment()

                # Should still succeed since commands are available
                assert success is True


class TestMainFunction:
    """Tests for the main() CLI entry point."""

    def test_no_args_prints_help(self, capsys):
        """main() with no arguments should print help and return 1."""
        with patch.object(sys, "argv", ["autoflatten"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "autoflatten" in captured.out or "usage" in captured.out.lower()

    def test_run_subcommand_requires_subject_dir(self):
        """run subcommand should require subject_dir argument."""
        with patch.object(sys, "argv", ["autoflatten", "run"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 for missing required arguments
            assert exc_info.value.code == 2

    def test_project_subcommand_requires_subject_dir(self):
        """project subcommand should require subject_dir argument."""
        with patch.object(sys, "argv", ["autoflatten", "project"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_flatten_subcommand_requires_patch_file(self):
        """flatten subcommand should require patch_file argument."""
        with patch.object(sys, "argv", ["autoflatten", "flatten"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_plot_flatmap_subcommand_requires_flat_patch(self):
        """plot-flatmap subcommand should require flat_patch argument."""
        with patch.object(sys, "argv", ["autoflatten", "plot-flatmap"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_path_without_subcommand_inserts_run(self):
        """A path as first argument should trigger implicit 'run' subcommand."""
        # Mock check_freesurfer_environment to return False (avoid running pipeline)
        with patch.object(sys, "argv", ["autoflatten", "/path/to/subject"]):
            with patch(
                "autoflatten.cli.check_freesurfer_environment",
                return_value=(False, {}),
            ):
                result = main()

        # Should fail because FreeSurfer check fails, not because of parsing
        assert result == 1

    def test_flatten_subcommand_parsing(self, tmp_path):
        """flatten subcommand should parse all its arguments."""
        patch_file = tmp_path / "lh.patch.3d"
        patch_file.touch()

        with patch.object(
            sys,
            "argv",
            [
                "autoflatten",
                "flatten",
                str(patch_file),
                "--backend",
                "pyflatten",
                "--k-ring",
                "5",
            ],
        ):
            # Mock the flatten command to avoid running it
            with patch("autoflatten.cli.cmd_flatten", return_value=0) as mock_flatten:
                result = main()

                assert result == 0
                mock_flatten.assert_called_once()
                args = mock_flatten.call_args[0][0]
                assert args.patch_file == str(patch_file)
                assert args.backend == "pyflatten"
                assert args.k_ring == 5

    def test_plot_flatmap_subcommand_parsing(self, tmp_path):
        """plot-flatmap subcommand should parse all its arguments."""
        flat_patch = tmp_path / "lh.flat.patch.3d"
        flat_patch.touch()

        with patch.object(
            sys,
            "argv",
            [
                "autoflatten",
                "plot-flatmap",
                str(flat_patch),
                "--subject-dir",
                "/path/to/subject/surf",
                "--output",
                "/tmp/output.png",
            ],
        ):
            # Mock the plot command to avoid running it
            with patch(
                "autoflatten.cli.cmd_plot_flatmap", return_value=0
            ) as mock_plot_flatmap:
                result = main()

                assert result == 0
                mock_plot_flatmap.assert_called_once()
                args = mock_plot_flatmap.call_args[0][0]
                assert args.flat_patch == str(flat_patch)
                assert args.subject_dir == "/path/to/subject/surf"
                assert args.output == "/tmp/output.png"

    def test_plot_projection_subcommand_requires_patch(self):
        """plot-projection subcommand should require patch argument."""
        with patch.object(sys, "argv", ["autoflatten", "plot-projection"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_plot_projection_subcommand_parsing(self, tmp_path):
        """plot-projection subcommand should parse all its arguments."""
        patch_file = tmp_path / "lh.autoflatten.patch.3d"
        patch_file.touch()

        with patch.object(
            sys,
            "argv",
            [
                "autoflatten",
                "plot-projection",
                str(patch_file),
                "--subject-dir",
                "/path/to/subject",
                "--output",
                "/tmp/output.png",
            ],
        ):
            with patch(
                "autoflatten.cli.cmd_plot_projection", return_value=0
            ) as mock_cmd:
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()
                args = mock_cmd.call_args[0][0]
                assert args.patch == str(patch_file)
                assert args.subject_dir == "/path/to/subject"
                assert args.output == "/tmp/output.png"

    def test_cmd_plot_flatmap_auto_detects_subject_dir_from_surf(self, tmp_path):
        """cmd_plot_flatmap should auto-detect subject_dir when patch is in surf/."""
        # Create subject structure: tmp_path/sub-01/surf/
        subject_dir = tmp_path / "sub-01"
        surf_dir = subject_dir / "surf"
        surf_dir.mkdir(parents=True)

        # Create patch file in surf/
        patch_file = surf_dir / "lh.flat.patch.3d"
        patch_file.touch()

        # Create required surface file
        (surf_dir / "lh.inflated").touch()

        # Create mock args
        args = argparse.Namespace(
            flat_patch=str(patch_file),
            subject_dir=None,  # Not provided - should auto-detect
            output=None,
        )

        with patch(
            "autoflatten.cli.plot_patch", return_value=str(tmp_path / "out.png")
        ):
            result = cmd_plot_flatmap(args)
            assert result == 0

    def test_cmd_plot_flatmap_fails_without_subject_dir(self, tmp_path):
        """cmd_plot_flatmap should fail when patch not in surf/ and no --subject-dir."""
        # Create patch file NOT in surf/ directory
        patch_file = tmp_path / "lh.flat.patch.3d"
        patch_file.touch()

        args = argparse.Namespace(
            flat_patch=str(patch_file),
            subject_dir=None,
            output=None,
        )

        result = cmd_plot_flatmap(args)
        assert result == 1  # Should fail

    def test_cmd_plot_projection_with_valid_args(self, tmp_path):
        """cmd_plot_projection should call plot_projection with correct args."""
        patch_file = tmp_path / "lh.autoflatten.patch.3d"
        patch_file.touch()

        args = argparse.Namespace(
            patch=str(patch_file),
            subject_dir="/path/to/subject",
            output=str(tmp_path / "output.png"),
        )

        with patch(
            "autoflatten.cli.plot_projection",
            return_value=str(tmp_path / "output.png"),
        ):
            result = cmd_plot_projection(args)
            assert result == 0

    def test_project_subcommand_parsing(self, tmp_path):
        """project subcommand should parse all its arguments."""
        subject_dir = tmp_path / "sub-01"
        subject_dir.mkdir()

        with patch.object(
            sys,
            "argv",
            [
                "autoflatten",
                "project",
                str(subject_dir),
                "--hemispheres",
                "lh",
                "--no-refine-geodesic",
            ],
        ):
            # Mock the project command to avoid running it
            with patch("autoflatten.cli.cmd_project", return_value=0) as mock_project:
                result = main()

                assert result == 0
                mock_project.assert_called_once()
                args = mock_project.call_args[0][0]
                assert args.subject_dir == str(subject_dir)
                assert args.hemispheres == "lh"
                assert args.no_refine_geodesic is True
