import itertools
import json
import math
import os
import pickle as pkl
import tempfile
import gmpy2 as gmp
import numpy as np
import pandas as pd

from collections import deque
from functools import reduce
from itertools import chain, combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
from scipy.sparse import csc_matrix, csr_matrix


from gmpy2 import mpz
from logzero import logger, setup_logger

from .common import loglevel

ii32 = np.iinfo(np.int32)


def randints(num, vmin=0, vmax=ii32.max - 1) -> np.ndarray:
    return np.random.randint(vmin, vmax, num, dtype=np.int32)


def int_floor(num: float):
    return int(math.floor(num))


def int_ceil(num: float):
    return int(math.ceil(num))


def flatten(stuff):
    """flatten an array"""
    return np.asarray(stuff).flatten()


def get_tempdir(prefix=None, suffix=None, dir=None):
    if dir is not None:
        makedir(dir, usedir=False)
    return tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=dir)


def makedir(d, usedir=True):
    if usedir:
        d = os.path.dirname(d)

    if not os.path.exists(d):
        os.makedirs(d)


def save_pickle(obj, path):
    return pkl.dump(obj, open(path, "wb"))


def save_file(string, path):
    with open(path, "w") as f:
        f.write(string)


def load_pickle(path):
    return pkl.load(open(path, "rb"))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def convert_numerical_columns_to_bool(df: pd.DataFrame):
    """
    given a dataframe, convert its columns of numerical types to have boolean

    this operation is inplace
    """
    numerical_column_names = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numerical_column_names] = df[numerical_column_names].astype(bool)


def powerset(iterable, min_size=0):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(min_size, len(s) + 1))


def bin_array(arr):
    """create a binary array"""
    return np.array(arr, dtype=bool)


def bin_zeros(shape):
    return np.zeros(shape, dtype=bool)


def bin_zeros_mpz(n: int):
    """create a binary vector, where n is the number of bits"""
    raise NotImplementedError()


def bin_ones(shape):
    return np.ones(shape, dtype=bool)


def bin_random(shape):
    return np.random.randint(0, 2, size=shape, dtype=bool)


def assert_binary_array(arr):
    assert isinstance(arr, np.ndarray), f"{type(arr)}"
    assert arr.dtype == bool


def solutions_to_dict(sols: List[Tuple[Set[int], float]]) -> Dict[Tuple[int], float]:
    """transform a list of solutions (each is a tuple of rule set indices + objective value)
    to a dictionary of rule set (as a tuple of ints) to the objective value"""
    return dict(map(lambda tpl: (tuple(sorted(tpl[0])), tpl[1]), sols))


def fill_array_until(arr, end_idx, val):
    """fill the values in arr from index 0 to idx with value `val`"""
    for i in range(0, end_idx + 1):
        arr[i] = val


def fill_array_from(arr, start_idx, val):
    """fill the values in arr from index idx until the end with value `val`"""
    for i in range(start_idx, len(arr)):  # typo in paper |S| -> |S| - 1
        arr[i] = val


def mpz_set_bits(n: mpz, bits: np.ndarray) -> mpz:
    """return a copy of n and set `bits` to 1 in `n`"""
    for i in bits:
        n = gmp.bit_set(n, int(i))
    return n


def mpz_clear_bits(n: mpz, bits: np.ndarray) -> mpz:
    """return a copy of n and set `bits` to 0 in `n`"""
    for i in bits:
        n = gmp.bit_clear(n, int(i))
    return n


def mpz_all_ones(n: int) -> mpz:
    """make a number of value 0b111..111, where the number of 1 equals n"""
    assert n >= 0
    if n > 0:
        return mpz("0b" + "1" * n)
    else:
        return mpz()


def mpz2bag(n: mpz):
    """given a mpz() this function returns the indices of non-zero entries"""
    i = 0
    bag = set()
    thisBit = gmp.bit_scan1(n, i)
    while thisBit is not None:
        bag.add(thisBit)
        i += 1
        thisBit = gmp.bit_scan1(n, i)

    return bag


def debug1(msg):
    logger.log(loglevel.DEBUG1, msg)


def debug2(msg):
    logger.log(loglevel.DEBUG2, msg)


def debug3(msg):
    logger.log(loglevel.DEBUG2, msg)


def count_iter(it: Iterable) -> int:
    """counter the number of elements in an iterable

    reference: https://stackoverflow.com/a/15112059/557067
    """
    counter = itertools.count()
    deque(zip(it, counter), maxlen=0)
    return next(counter)


def reconstruct_array(n: gmp.mpz) -> np.ndarray:
    # Count the number of bits in `n`
    bit_count = gmp.bit_length(n)

    # Calculate the number of bytes needed to store the bits
    byte_count = (bit_count + 7) // 8

    # Extract the bits of `n` into a bit string
    bit_str = ""
    for i in range(bit_count - 1, -1, -1):
        bit_str += "1" if gmp.bit_test(n, i) else "0"

    # Reverse the bit string and pad it with zeros to a multiple of 8
    bit_str = bit_str[::-1].ljust(byte_count * 8, "0")

    # Convert the bit string to a byte string and then to a numpy array
    byte_str = int(bit_str, 2).to_bytes(byte_count, byteorder="little")
    return np.frombuffer(byte_str, dtype=np.uint8)[: len(bit_str) // 8]


def get_indices_and_indptr(A: np.ndarray, axis: int = 1):
    """get the indptr and indices attributes of A's sparse matrix

    axis determines if A is a csc_matrix (axis=0) or csr_matrix (axis=1)
    """
    assert A.ndim == 2
    assert axis in (0, 1)
    if axis == 0:  # treat A as a csr_matrix
        A_sp = csr_matrix(A)
    else:  # treat A as a csc_matrix
        A_sp = csc_matrix(A)
    return A_sp.indices, A_sp.indptr


def lor_of_truthtable(rules: List["Rule"]) -> mpz:
    """take the logical OR of rules' truth tables"""
    bit_vec_list = [r.truthtable for r in rules]
    return reduce(lambda x, y: x | y, bit_vec_list, mpz())


def calculate_obj(
    rules: List["Rule"], y_np: np.ndarray, y_mpz: mpz, sol: Tuple[int], lmbd: float
) -> float:
    """calcuclate the objective for a given decision rule set (indicated by `sol`)
    by convention, `sol` is sorted and `0` is included
    """
    assert tuple(sorted(sol)) == tuple(sol), f"{sol} is not sorted lexicographically"
    # print("sol: {}".format(sol))
    ds_rules = [rules[i] for i in sol]
    # print("ds_rules: {}".format(ds_rules))
    pred = lor_of_truthtable(ds_rules)
    # print("bin(pred): {}".format(bin(pred)))
    # print("bin(y_mpz): {}".format(bin(y_mpz)))
    num_mistakes = gmp.popcount(y_mpz ^ pred)
    # print("num_mistakes: {}".format(num_mistakes))
    obj = len(sol) * lmbd + num_mistakes / y_np.shape[0]
    return float(obj)


def calculate_lower_bound(
    rules: List["Rule"], y_np: np.ndarray, y_mpz: mpz, sol: Tuple[int], lmbd: float
) -> float:
    """calcuclate the lower bound for a given decision rule set (indicated by `sol`)
        by convention, `sol` is sorted and `0` is included

        the lower bound is basically number of false positives / total number of points + lambda \times (|sol| - 1)
    1"""
    assert tuple(sorted(sol)) == tuple(sol), f"{sol} is not sorted lexicographically"
    # print("sol: {}".format(sol))
    ds_rules = [rules[i] for i in sol]
    # print("ds_rules: {}".format(ds_rules))
    pred = lor_of_truthtable(ds_rules)
    # print("bin(pred): {}".format(bin(pred)))
    # print("bin(y_mpz): {}".format(bin(y_mpz)))
    num_fp = gmp.popcount((y_mpz ^ pred) & pred)
    # print("(y_mpz ^ pred) & y_mpz: {}".format(bin((y_mpz ^ pred) & y_mpz)))
    # print("num_mistakes: {}".format(num_mistakes))
    lb = len(sol) * lmbd + num_fp / y_np.shape[0]
    return float(lb)
