"""
Tests for the utils module.
"""

import json
import os
import tempfile

import numpy as np

from autoflatten.utils import load_json, save_json


def test_load_json():
    """
    Test loading a JSON file.

    This test creates a temporary JSON file with known content,
    then verifies that load_json correctly loads it.
    """
    test_data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_file:
        json.dump(test_data, temp_file)
        temp_filename = temp_file.name

    try:
        loaded_data = load_json(temp_filename)
        assert loaded_data == test_data
        assert loaded_data["key1"] == "value1"
        assert loaded_data["key2"] == 42
        assert loaded_data["key3"] == [1, 2, 3]
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_save_json_basic():
    """
    Test saving a basic dictionary to a JSON file.

    Verifies saving and loading roundtrip for basic Python types.
    """
    test_data = {
        "string": "text",
        "int": 123,
        "float": 3.14,
        "list": [1, 2, 3],
        "bool": True,
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        save_json(temp_filename, test_data)
        loaded = load_json(temp_filename)

        assert loaded == test_data
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_save_json_numpy_array():
    """
    Test saving a dictionary with NumPy arrays to a JSON file.

    Verifies the NumPy array conversion functionality.
    """
    test_data = {
        "array_1d": np.array([1, 2, 3]),
        "array_2d": np.array([[1, 2], [3, 4]]),
        "mixed": {"array": np.array([5, 6, 7]), "value": 42},
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        save_json(temp_filename, test_data)
        loaded = load_json(temp_filename)

        # Arrays should be converted to lists
        assert loaded["array_1d"] == [1, 2, 3]
        assert loaded["array_2d"] == [[1, 2], [3, 4]]
        assert loaded["mixed"]["array"] == [5, 6, 7]
        assert loaded["mixed"]["value"] == 42
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_save_json_numpy_types():
    """
    Test saving a dictionary with NumPy scalar types to a JSON file.

    Verifies the NumPy scalar type conversion functionality.
    """
    test_data = {
        "np_int": np.int64(42),
        "np_float": np.float64(3.14159),
        "mixed": [np.int32(10), np.float32(2.5)],
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        save_json(temp_filename, test_data)
        loaded = load_json(temp_filename)

        # NumPy types should be converted to Python types
        assert loaded["np_int"] == 42
        assert isinstance(loaded["np_int"], int)

        assert loaded["np_float"] == 3.14159
        assert isinstance(loaded["np_float"], float)

        assert loaded["mixed"] == [10, 2.5]
        assert isinstance(loaded["mixed"][0], int)
        assert isinstance(loaded["mixed"][1], float)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_save_json_indent():
    """
    Test the indent parameter of save_json.

    Verifies that the indent parameter correctly formats the JSON file.
    """
    test_data = {"key1": "value1", "key2": [1, 2, 3]}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Save with indent=2
        save_json(temp_filename, test_data, indent=2)

        # Read the raw file content to check formatting
        with open(temp_filename, "r") as f:
            content = f.read()

        # Check that the content contains newlines (which wouldn't be there without indent)
        assert "\n" in content

        # Verify the content is still valid JSON and matches our data
        loaded = load_json(temp_filename)
        assert loaded == test_data
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
