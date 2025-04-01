"""Various utils"""

import json

import numpy as np


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
    Handles numpy arrays by converting them to lists.
    """

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    with open(fn, "w") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
