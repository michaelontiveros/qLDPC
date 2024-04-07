#!/usr/bin/env python3
"""Script to perform a brute-force search for quasi-cyclic codes

   Copyright 2023 The qLDPC Authors and Infleqtion Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import concurrent.futures
import itertools
import os

import diskcache
import numpy as np
from sympy.abc import x, y

import qldpc
import qldpc.cache

NUM_TRIALS = 1000
COMMUNICATION_DISTANCE_CUTOFF = 12
CACHE_NAME = "qldpc_" + os.path.basename(os.path.dirname(__file__))


def get_quasi_cyclic_code_params(
    dim_x: int, dim_y: int, exponents: tuple[int, int, int, int], num_trials: int
) -> tuple[int, int, int, float] | None:
    """Compute communication distance and code distance for a quasi-cyclic code.

    If the code is trivial or the communication distance is beyond the cutoff, return None.
    """
    # construct the code itself
    dims = (dim_x, dim_y)
    exp_ax, exp_ay, exp_bx, exp_by = exponents
    poly_a = 1 + x + x**exp_ax * y**exp_ay
    poly_b = 1 + y + x**exp_bx * y**exp_by
    code = qldpc.codes.QCCode(dims, poly_a, poly_b)

    if code.dimension == 0:
        return None

    # identify maximum Euclidean distance between check/data qubits required for each toric layout
    max_distances = []
    toric_mappings = code.get_toric_mappings()
    for plaquette_map, torus_shape in toric_mappings:
        shifts_x, shifts_z = code.get_check_shifts(plaquette_map, torus_shape, open_boundaries=True)
        distances = set(np.sqrt(xx**2 + yy**2) for xx, yy in shifts_x | shifts_z)
        max_distances.append(max(distances))

    # minimize distance requirement over possible toric layouts
    comm_distance = min(max_distances)

    if comm_distance > COMMUNICATION_DISTANCE_CUTOFF:
        return None

    distance = code.get_distance_bound(num_trials=num_trials)
    assert isinstance(distance, int)

    return code.num_qubits, code.dimension, distance, comm_distance


def run_and_save(
    dim_x: int,
    dim_y: int,
    exponents: tuple[int, int, int, int],
    num_trials: int,
    cache: diskcache.Cache,
    *,
    silent: bool = False,
) -> None:
    """Compute and save quasi-cyclic code parameters."""
    params = get_quasi_cyclic_code_params(dim_x, dim_y, exponents, num_trials)
    if params is not None:
        nn, kk, dd, comm_dist = params
        cache[dim_x, dim_y, exponents, num_trials] = (nn, kk, dd, comm_dist)
        if not silent:
            print(dim_x, dim_y, exponents, params)


if __name__ == "__main__":
    min_dim_y = 3
    min_dim_x = 3
    max_dim = 10

    silent = False
    cache = qldpc.cache.get_disk_cache(CACHE_NAME)
    max_concurrent_jobs = num_cpus // 2 if (num_cpus := os.cpu_count()) else 1

    # run multiple jobs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_jobs) as executor:

        for dim_x in range(min_dim_x, max_dim + 1):
            for dim_y in range(min_dim_y, dim_x + 1):
                for exponents in itertools.product(
                    range(dim_x), range(dim_y), range(dim_x), range(dim_y)
                ):
                    # submit this job to the job queue
                    executor.submit(run_and_save, dim_x, dim_y, exponents, NUM_TRIALS, cache)
