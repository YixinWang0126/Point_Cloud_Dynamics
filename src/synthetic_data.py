from __future__ import annotations

import numpy as np


def get_potential_rvp(x1, x2, h):
    r2 = x1**2 + x2**2
    return 0.5 * (r2**2) + h * r2


def sample_rvp_reproducible(h, n_samples=2000, T=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    samples = []
    curr_x = np.random.normal(0, 0.5, 2)
    curr_E = get_potential_rvp(curr_x[0], curr_x[1], h)

    burn_in = 5000
    thinning = 50
    total_steps = burn_in + n_samples * thinning
    step_sigma = 0.3

    for i in range(total_steps):
        proposal = curr_x + np.random.normal(0, step_sigma, 2)
        prop_E = get_potential_rvp(proposal[0], proposal[1], h)
        delta_E = prop_E - curr_E

        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            curr_x = proposal
            curr_E = prop_E

        if i >= burn_in and (i - burn_in) % thinning == 0:
            samples.append(curr_x)

    return np.array(samples)


def generate_full_dataset(seed=42):
    full_data = []
    np.random.seed(seed)

    for idx, h in enumerate(np.linspace(-1, 1, 51)):
        current_seed = seed + idx
        pts = sample_rvp_reproducible(
            h,
            n_samples=200,
            T=0.001,
            seed=current_seed,
        )
        h_col = np.full((len(pts), 1), h)
        full_data.append(np.hstack((h_col, pts)))

    return np.vstack(full_data)


def get_potential_1to2(x1, x2, h):
    return (x1**2 - h) ** 2 + x2**2


def sample_1to2(h, n_samples=2000, T=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)

    samples = []
    curr_x = np.random.normal(0, 0.5, 2)
    curr_E = get_potential_1to2(curr_x[0], curr_x[1], h)

    burn_in = 5000
    thinning = 200
    total_steps = burn_in + n_samples * thinning
    step_sigma = 1

    for i in range(total_steps):
        proposal = curr_x + np.random.normal(0, step_sigma, 2)
        prop_E = get_potential_1to2(proposal[0], proposal[1], h)
        delta_E = prop_E - curr_E

        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            curr_x = proposal
            curr_E = prop_E

        if i >= burn_in and (i - burn_in) % thinning == 0:
            samples.append(curr_x)

    return np.array(samples)


def generate_bistable_dataset(seed=42):
    full_data = []
    np.random.seed(seed)

    for idx, h in enumerate(np.linspace(-1, 1, 51)):
        current_seed = seed + idx
        pts = sample_1to2(
            h,
            n_samples=300,
            T=0.04,
            seed=current_seed,
        )
        h_col = np.full((len(pts), 1), h)
        full_data.append(np.hstack((h_col, pts)))

    return np.vstack(full_data)
