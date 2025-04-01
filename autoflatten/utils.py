"""Various utils"""
import json


def load_json(fn):
    """
    Load a json file and return the content as a dictionary.
    """
    with open(fn, "r") as f:
        data = json.load(f)
    return data


def save_json(fn, data, indent=None):
    """
    Save a dictionary to a json file.
    """
    with open(fn, "w") as f:
        json.dump(data, f, indent=indent)
