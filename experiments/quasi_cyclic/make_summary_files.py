#!/usr/bin/env python3
"""Script to save quasi-cyclic search results to data files

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
import os

import diskcache
import numpy as np
import platformdirs
from run_search import CACHE_NAME, NUM_TRIALS, get_code_params


comm_cutoffs = [5, 8, 10]  # communication distance "cutoffs" by which to organize results

dirname = os.path.dirname(__file__)
save_dir = os.path.join(dirname, "codes")


headers = [
    "AUTHOR: Michael A. Perlin, 2024",
    "quasi-cyclic codes of arXiv:2308.07915, with generating polynomials",
    "    A = 1 + x + x**ax * y**ay",
    "    B = 1 + y + x**bx * y**by",
    "here x and y are generators of cyclic groups with orders L and M",
    "code parameters [[n, k, d]] indicate",
    "    n = number of physical qubits",
    "    k = number of logical qubits",
    "    d = code distance (minimal weight of a nontrivial logical operator)",
    "code distance is estimated by the method of arXiv:2308.07915,"
    + f" minimizing over {NUM_TRIALS} trials",
    "also included:",
    "    D = (Euclidean) communication distance required for a 'folded toric layout' of the code",
    "    r = k d^2 / n",
    "topological 2D codes such as the toric code strictly satisfy r <= 1",
    "we only keep track of codes with r > 1",
    "",
    "L, M, ax, ay, bx, by, n, k, d, D, r",
]
fmt = "%d, %d, %d, %d, %d, %d, %d, %d, %d, %.3f, %.3f"

##################################################

comm_cutoffs = sorted(comm_cutoffs)
data_groups: list[list[tuple[int | float, ...]]] = [[] for _ in range(len(comm_cutoffs))]

# iterate over all entries in the cache
cache = diskcache.Cache(platformdirs.user_cache_dir(CACHE_NAME))
for key in cache.iterkeys():
    if len(key) != 3:
        continue

    # cyclic group orders and polynomial exponents
    ll, mm, (gg, hh, ii, jj) = key

    # communication distance
    comm = cache[key]

    if comm > comm_cutoffs[-1]:
        # we don't care about this code, the communication distance is too large
        continue

    # code parameters
    nn, kk, dd = get_code_params(*key)

    # figure of merit, relative to the surface code
    merit = kk * dd**2 / nn
    if merit <= 1:
        # this code doesn't even beat the surface code, so we don't care about it
        continue

    # add a summary of this code to the appropriate group of data
    code = (ll, mm, gg, hh, ii, jj, nn, kk, dd, comm, merit)
    for cutoff, data in zip(comm_cutoffs, data_groups):
        if comm <= cutoff:
            data.append(code)
            break

##################################################

os.makedirs(save_dir, exist_ok=True)
header = "\n".join(headers)

# save data groups to files
for last_comm, comm, data in zip([0] + comm_cutoffs, comm_cutoffs, data_groups):
    file = f"codes_D{last_comm}-{comm}.csv"
    path = os.path.join(save_dir, file)
    np.savetxt(path, data, header=header, fmt=fmt)
    last_comm = comm

# save all data
path = os.path.join(save_dir, "codes_all.csv")
data_all = [code for data_group in data_groups for code in data_group]
np.savetxt(path, data_all, header=header, fmt=fmt)
