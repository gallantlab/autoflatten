"""Tests for autoflatten.logging module."""

import sys
from io import StringIO

import pytest

from autoflatten.logging import TeeStream, restore_logging, setup_logging


class TestTeeStream:
    """Tests for TeeStream class."""

    def test_tee_stream_writes_to_single_stream(self):
        """TeeStream should write to a single stream."""
        output = StringIO()
        tee = TeeStream(output)

        tee.write("hello")
        assert output.getvalue() == "hello"

    def test_tee_stream_writes_to_multiple_streams(self):
        """TeeStream should write to multiple streams simultaneously."""
        output1 = StringIO()
        output2 = StringIO()
        tee = TeeStream(output1, output2)

        tee.write("hello world")

        assert output1.getvalue() == "hello world"
        assert output2.getvalue() == "hello world"

    def test_tee_stream_returns_length(self):
        """TeeStream.write() should return the number of characters written."""
        output = StringIO()
        tee = TeeStream(output)

        result = tee.write("hello")
        assert result == 5

    def test_tee_stream_flush(self):
        """TeeStream.flush() should flush all streams."""
        output1 = StringIO()
        output2 = StringIO()
        tee = TeeStream(output1, output2)

        # Write some data
        tee.write("test")

        # Flush should not raise
        tee.flush()

        # Data should still be there
        assert output1.getvalue() == "test"
        assert output2.getvalue() == "test"

    def test_tee_stream_multiple_writes(self):
        """TeeStream should accumulate multiple writes."""
        output = StringIO()
        tee = TeeStream(output)

        tee.write("hello ")
        tee.write("world")

        assert output.getvalue() == "hello world"

    def test_tee_stream_empty_write(self):
        """TeeStream should handle empty writes."""
        output = StringIO()
        tee = TeeStream(output)

        result = tee.write("")
        assert result == 0
        assert output.getvalue() == ""


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_log_file(self, tmp_path):
        """setup_logging should create a log file."""
        output_path = tmp_path / "test_output"
        log_path = tmp_path / "test_output.log"

        original_stdout, log_file = setup_logging(str(output_path))

        try:
            assert log_path.exists()
        finally:
            restore_logging(original_stdout, log_file)

    def test_setup_logging_creates_directory(self, tmp_path):
        """setup_logging should create parent directories if needed."""
        output_path = tmp_path / "subdir" / "test_output"
        log_path = tmp_path / "subdir" / "test_output.log"

        original_stdout, log_file = setup_logging(str(output_path))

        try:
            assert log_path.parent.exists()
            assert log_path.exists()
        finally:
            restore_logging(original_stdout, log_file)

    def test_setup_logging_returns_original_stdout(self, tmp_path):
        """setup_logging should return the original stdout."""
        output_path = tmp_path / "test_output"
        current_stdout = sys.stdout

        original_stdout, log_file = setup_logging(str(output_path))

        try:
            assert original_stdout is current_stdout
        finally:
            restore_logging(original_stdout, log_file)

    def test_setup_logging_returns_file_handle(self, tmp_path):
        """setup_logging should return a file handle."""
        output_path = tmp_path / "test_output"

        original_stdout, log_file = setup_logging(str(output_path))

        try:
            assert log_file is not None
            assert hasattr(log_file, "write")
            assert hasattr(log_file, "close")
        finally:
            restore_logging(original_stdout, log_file)

    def test_setup_logging_redirects_stdout(self, tmp_path):
        """setup_logging should redirect sys.stdout to TeeStream."""
        output_path = tmp_path / "test_output"
        original_stdout = sys.stdout

        orig, log_file = setup_logging(str(output_path))

        try:
            assert sys.stdout is not original_stdout
            assert isinstance(sys.stdout, TeeStream)
        finally:
            restore_logging(orig, log_file)

    def test_setup_logging_verbose_true(self, tmp_path):
        """With verbose=True, TeeStream should have 2 streams (stdout + file)."""
        output_path = tmp_path / "test_output"

        original_stdout, log_file = setup_logging(str(output_path), verbose=True)

        try:
            assert isinstance(sys.stdout, TeeStream)
            assert len(sys.stdout.streams) == 2
        finally:
            restore_logging(original_stdout, log_file)

    def test_setup_logging_verbose_false(self, tmp_path):
        """With verbose=False, TeeStream should only write to file."""
        output_path = tmp_path / "test_output"

        original_stdout, log_file = setup_logging(str(output_path), verbose=False)

        try:
            assert isinstance(sys.stdout, TeeStream)
            assert len(sys.stdout.streams) == 1
        finally:
            restore_logging(original_stdout, log_file)

    def test_print_writes_to_log_file(self, tmp_path):
        """print() should write to the log file when logging is set up."""
        output_path = tmp_path / "test_output"
        log_path = tmp_path / "test_output.log"

        original_stdout, log_file = setup_logging(str(output_path))

        try:
            print("test message")
        finally:
            restore_logging(original_stdout, log_file)

        # Check log file contents
        content = log_path.read_text()
        assert "test message" in content


class TestRestoreLogging:
    """Tests for restore_logging function."""

    def test_restore_logging_restores_stdout(self, tmp_path):
        """restore_logging should restore the original stdout."""
        output_path = tmp_path / "test_output"
        original_stdout = sys.stdout

        orig, log_file = setup_logging(str(output_path))
        restore_logging(orig, log_file)

        assert sys.stdout is original_stdout

    def test_restore_logging_closes_file(self, tmp_path):
        """restore_logging should close the log file handle."""
        output_path = tmp_path / "test_output"

        original_stdout, log_file = setup_logging(str(output_path))
        restore_logging(original_stdout, log_file)

        assert log_file.closed

    def test_restore_logging_handles_none_file(self, tmp_path):
        """restore_logging should handle None log file gracefully."""
        output_path = tmp_path / "test_output"

        original_stdout, log_file = setup_logging(str(output_path))
        log_file.close()  # Close it manually first

        # Should not raise even with None
        restore_logging(original_stdout, None)

        assert sys.stdout is original_stdout
