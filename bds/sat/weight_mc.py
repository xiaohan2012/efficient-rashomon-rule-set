from typing import Callable

import numpy as np
import ray
from logzero import logger
from tqdm import tqdm

from ..common import CPVarList, Program, Solver
from ..ray_pbar import RayProgressBar
from .bounded_weight_sat import weight_mc_core
from .solver import construct_solver


@ray.remote
def weight_mc_core_job(*args, **kwargs):
    c, _ = weight_mc_core(*args, **kwargs)
    return c


def weight_mc(
    program: Program,
    S: CPVarList,
    make_callback: Callable,
    epsilon: float,
    delta: float,
    r: float,
    weight_func: Callable,
    show_progress: bool = False,
    solver: Solver = None,
    parallel: bool = False,
    verbose: bool = False,
):
    """return an approximate estimate of the total weight of satisfying solutions to a given constrained program"""
    if solver is None:
        solver = construct_solver()

    # initially w_max is 1.0
    w_max = 1.0

    pivot = 2 * np.ceil(np.exp(3.0 / 2) * np.power(1 + 1.0 / epsilon, 2))
    t = int(np.ceil(35 * np.log2(3 / delta)))

    if verbose:
        logger.info("weight_mc starts")
        logger.info(f"r={r}, w_max={w_max}")
        logger.info(f"pivot={pivot}, t={t}")

    iter_obj = range(t)

    if not parallel:
        # sequential run of weight_mc_core, w_max may get updated
        if verbose:
            logger.info("sequentially call weight_mc_core multiple times")
        C = []  # list of estimates
        if show_progress:
            iter_obj = tqdm(iter_obj)

        for i in iter_obj:
            c, w_max_new = weight_mc_core(
                program,
                S,
                make_callback,
                weight_func,
                pivot,
                r,
                w_max,
                solver,
                return_details=False,
                rand_seed=None,
                verbose=verbose,
            )
            if w_max_new != w_max:
                if verbose:
                    logger.debug(f"w_max is updated to {w_max_new}")
                w_max = w_max_new
            if c is not None:
                C.append(c * w_max)
    else:
        # paralell run of weight_mc_core, w_max does not get updated
        # not sure if the probabilistic guarantee still holds
        if verbose:
            logger.info("parallelly call weight_mc_core multiple times")

        promise = [
            weight_mc_core_job.remote(
                program,
                S,
                make_callback,
                weight_func,
                pivot,
                r,
                w_max,
                solver=None,  # do not pass in a solver, otherwise ray complains it cannot be pickled
                return_details=False,
                rand_seed=None,
            )
            for i in iter_obj
        ]

        RayProgressBar.show(promise)

        C = ray.get(promise)
        w_max_new = w_max  # we simply copy the initial w_max

    C = list(filter(None, C))
    if len(C) == 0:
        msg = "the number of valid estimates is zero!"
        logger.error(msg)
        raise RuntimeError(msg)

    final_estimate = np.median(C)
    return final_estimate, w_max_new
