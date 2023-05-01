import numpy as np
from typing import Dict, Any
from numbers import Number


def assert_dict_allclose(actual: Dict[Any, Number], expected: [Any, Number]):
    """check the equivalence of two dicts, assuming the value field of both dict are numeric"""
    assert set(actual.keys()) == (expected.keys())

    for k in actual.keys():
        np.testing.assert_allclose(actual[k], expected[k])
