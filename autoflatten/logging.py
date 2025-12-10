"""Logging utilities for autoflatten.

Provides TeeStream-based logging that captures all print() output to both
console and log file.
"""

import os
import sys
from typing import Optional, TextIO


class TeeStream:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_logging(
    output_path: str, verbose: bool = True
) -> tuple[TextIO, Optional[TextIO]]:
    """Setup dual logging to stdout and file.

    Parameters
    ----------
    output_path : str
        Path to the output file. The log file will be created at output_path + ".log"
    verbose : bool
        If True, output to both console and file; if False, only file

    Returns
    -------
    tuple
        (original_stdout, log_file_handle)
        The log_file_handle must be closed by the caller via restore_logging().
    """
    log_path = f"{output_path}.log"

    # Create directory if needed
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_file = open(log_path, "w")
    original_stdout = sys.stdout

    if verbose:
        sys.stdout = TeeStream(sys.__stdout__, log_file)
    else:
        sys.stdout = TeeStream(log_file)

    return original_stdout, log_file


def restore_logging(original_stdout: TextIO, log_file: Optional[TextIO]) -> None:
    """Restore original stdout and close log file.

    Parameters
    ----------
    original_stdout : TextIO
        The original stdout stream to restore
    log_file : TextIO or None
        The log file handle to close
    """
    sys.stdout = original_stdout
    if log_file:
        log_file.close()
