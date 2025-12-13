"""Tests for threading configuration module."""

import os


def get_max_threads():
    """Get maximum available threads for testing."""
    try:
        import numba

        return numba.get_num_threads()
    except ImportError:
        return os.cpu_count() or 1


class TestConfigureThreading:
    """Tests for configure_threading function."""

    def test_sets_xla_flags(self, monkeypatch):
        """Test that XLA_FLAGS is set correctly."""
        # Clear any existing XLA_FLAGS
        monkeypatch.delenv("XLA_FLAGS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        # Use a value that's always valid (env vars don't check core count)
        configure_threading(4)

        assert "XLA_FLAGS" in os.environ
        assert "--xla_force_host_platform_device_count=4" in os.environ["XLA_FLAGS"]
        assert "--xla_cpu_multi_thread_eigen_thread_count=4" in os.environ["XLA_FLAGS"]
        assert "--xla_cpu_multi_thread_eigen=true" in os.environ["XLA_FLAGS"]

    def test_xla_flags_are_not_duplicated(self, monkeypatch):
        """Ensure repeated calls do not duplicate XLA flags."""
        monkeypatch.setenv("XLA_FLAGS", "--xla_some_other_flag=true")

        from autoflatten.flatten.threading import configure_threading

        configure_threading(2)
        first = os.environ["XLA_FLAGS"]
        configure_threading(2)
        second = os.environ["XLA_FLAGS"]

        assert first == second

    def test_preserves_existing_eigen_flags(self, monkeypatch):
        """Existing eigen flags should not be duplicated or overridden."""
        monkeypatch.setenv(
            "XLA_FLAGS",
            "--xla_cpu_multi_thread_eigen=false "
            "--xla_cpu_multi_thread_eigen_thread_count=8",
        )

        from autoflatten.flatten.threading import configure_threading

        configure_threading(2)

        xla_flags = os.environ["XLA_FLAGS"]
        tokens = xla_flags.split()
        assert sum(t.startswith("--xla_cpu_multi_thread_eigen=") for t in tokens) == 1
        assert (
            sum(
                t.startswith("--xla_cpu_multi_thread_eigen_thread_count")
                for t in tokens
            )
            == 1
        )

    def test_sets_omp_num_threads(self, monkeypatch):
        """Test that OMP_NUM_THREADS is set correctly."""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        # Env vars accept any value - they don't validate against core count
        configure_threading(8)

        assert os.environ.get("OMP_NUM_THREADS") == "8"

    def test_sets_mkl_num_threads(self, monkeypatch):
        """Test that MKL_NUM_THREADS is set correctly."""
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        configure_threading(2)

        assert os.environ.get("MKL_NUM_THREADS") == "2"

    def test_sets_openblas_num_threads(self, monkeypatch):
        """Test that OPENBLAS_NUM_THREADS is set correctly."""
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        configure_threading(6)

        assert os.environ.get("OPENBLAS_NUM_THREADS") == "6"

    def test_sets_veclib_maximum_threads(self, monkeypatch):
        """Test that VECLIB_MAXIMUM_THREADS is set correctly (macOS)."""
        monkeypatch.delenv("VECLIB_MAXIMUM_THREADS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        configure_threading(4)

        assert os.environ.get("VECLIB_MAXIMUM_THREADS") == "4"

    def test_sets_numexpr_num_threads(self, monkeypatch):
        """Test that NUMEXPR_NUM_THREADS is set correctly."""
        monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        configure_threading(4)

        assert os.environ.get("NUMEXPR_NUM_THREADS") == "4"

    def test_sets_numba_threads(self):
        """Test that numba threads are configured (capped at available cores)."""
        from autoflatten.flatten.threading import configure_threading

        import numba

        max_threads = numba.get_num_threads()
        requested = 4

        configure_threading(requested)

        # Should be min(requested, available)
        expected = min(requested, max_threads)
        assert numba.get_num_threads() == expected

    def test_numba_threads_capped_at_available(self):
        """Test that numba threads are capped at available cores."""
        from autoflatten.flatten.threading import configure_threading

        import numba

        max_threads = numba.get_num_threads()

        # Request more than available
        configure_threading(1000)

        # Should not exceed available
        assert numba.get_num_threads() <= max_threads

    def test_none_does_not_set_limits(self, monkeypatch):
        """Test that None means no limit (don't set env vars)."""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        configure_threading(None)

        # Should not set these vars when None
        assert "OMP_NUM_THREADS" not in os.environ
        assert "MKL_NUM_THREADS" not in os.environ

    def test_negative_one_does_not_set_limits(self, monkeypatch):
        """Test that -1 means use all CPUs (don't set env vars)."""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        configure_threading(-1)

        assert "OMP_NUM_THREADS" not in os.environ

    def test_zero_does_not_set_limits(self, monkeypatch):
        """Test that 0 means use all CPUs (don't set env vars)."""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

        from autoflatten.flatten.threading import configure_threading

        configure_threading(0)

        assert "OMP_NUM_THREADS" not in os.environ

    def test_respects_existing_env_vars(self, monkeypatch):
        """Test that existing env vars are not overwritten."""
        monkeypatch.setenv("OMP_NUM_THREADS", "16")
        monkeypatch.setenv("MKL_NUM_THREADS", "32")

        from autoflatten.flatten.threading import configure_threading

        configure_threading(4)

        # Should keep original values
        assert os.environ.get("OMP_NUM_THREADS") == "16"
        assert os.environ.get("MKL_NUM_THREADS") == "32"

    def test_appends_to_existing_xla_flags(self, monkeypatch):
        """Test that XLA_FLAGS appends to existing flags."""
        monkeypatch.setenv("XLA_FLAGS", "--xla_some_other_flag=true")

        from autoflatten.flatten.threading import configure_threading

        configure_threading(4)

        xla_flags = os.environ.get("XLA_FLAGS", "")
        assert "--xla_some_other_flag=true" in xla_flags
        assert "--xla_force_host_platform_device_count=4" in xla_flags

    def test_does_not_duplicate_xla_device_count(self, monkeypatch):
        """Test that device count flag is not added if already present."""
        monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

        from autoflatten.flatten.threading import configure_threading

        configure_threading(4)

        xla_flags = os.environ.get("XLA_FLAGS", "")
        # Should keep original value, not add another
        assert xla_flags.count("--xla_force_host_platform_device_count") == 1
        assert "--xla_force_host_platform_device_count=8" in xla_flags


class TestIsConfigured:
    """Tests for is_configured function."""

    def test_returns_true_after_configure(self):
        """Test that is_configured returns True after calling configure_threading."""
        from autoflatten.flatten.threading import configure_threading, is_configured

        configure_threading(1)

        assert is_configured() is True


class TestGetEffectiveThreads:
    """Tests for get_effective_threads function."""

    def test_returns_dict(self):
        """Test that get_effective_threads returns a dictionary."""
        from autoflatten.flatten.threading import get_effective_threads

        result = get_effective_threads()

        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        """Test that result contains all expected keys."""
        from autoflatten.flatten.threading import get_effective_threads

        result = get_effective_threads()

        expected_keys = [
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "numba_threads",
        ]
        for key in expected_keys:
            assert key in result

    def test_returns_configured_values(self, monkeypatch):
        """Test that get_effective_threads returns configured values."""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

        from autoflatten.flatten.threading import (
            configure_threading,
            get_effective_threads,
        )

        import numba

        max_threads = numba.get_num_threads()

        configure_threading(4)
        result = get_effective_threads()

        assert result["OMP_NUM_THREADS"] == "4"
        assert result["MKL_NUM_THREADS"] == "4"
        # Numba threads capped at available
        assert result["numba_threads"] == min(4, max_threads)
