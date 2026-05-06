from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R

from coot import dot, eta


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "tools").is_dir():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


def get_affinities(X, gaussian=True):
    C = sp.spatial.distance.cdist(X, X, metric="sqeuclidean")
    C /= np.mean(C)
    if gaussian:
        C = np.exp(-C)
    return C


def geodesic_distances(a, b, t, indices):
    n_points = len(indices)
    distances = np.zeros((n_points, n_points))
    for ii in range(n_points):
        p_i = np.asarray(a[indices[ii][0]])
        q_i = np.asarray(b[indices[ii][1]])
        for jj in range(ii + 1, n_points):
            p_j = np.asarray(a[indices[jj][0]])
            q_j = np.asarray(b[indices[jj][1]])
            da = np.linalg.norm(p_i - p_j)
            db = np.linalg.norm(q_i - q_j)
            distances[ii, jj] = (1 - t) * da + t * db
    return distances + distances.T


def convex_point(a, b, t, indices):
    points = []
    for ii in range(len(indices)):
        p_i = np.asarray(a[indices[ii][0]])
        q_i = np.asarray(b[indices[ii][1]])
        points.append(p_i * (1 - t) + q_i * t)
    return points


def geodesic_distances_gw(a, b, t, indices):
    n_points = len(indices)
    distances = np.zeros((n_points, n_points))
    for ii in range(n_points):
        p_i = np.asarray(a[indices[ii][0]])
        q_i = np.asarray(b[indices[ii][1]])
        for jj in range(ii + 1, n_points):
            p_j = np.asarray(a[indices[jj][0]])
            q_j = np.asarray(b[indices[jj][1]])
            da = np.linalg.norm(p_i - p_j)
            db = np.linalg.norm(q_i - q_j)
            distances[ii, jj] = t * da + (1 - t) * db
    return distances + distances.T


def geodesic_distances_coot(a, b, t, point_indices, cycle_indices):
    distances = np.zeros(a.shape)
    for ii in range(len(point_indices)):
        p_i = np.asarray(a[point_indices[ii][0]])
        q_i = np.asarray(b[point_indices[ii][1]])
        for jj in range(len(cycle_indices) - 1):
            if cycle_indices[jj][1] <= len(cycle_indices) - 2:
                distances[ii, jj] = t * p_i[cycle_indices[jj][0]] + (1 - t) * q_i[cycle_indices[jj][1]]
            else:
                distances[ii, jj] = t * p_i[cycle_indices[jj][0]]
    return distances


def geodesic_distances_iota(a, b, t, indices):
    n_points = len(indices) - 1
    interpolated = np.zeros((n_points, 2))
    for ii in range(n_points):
        p_i = a[indices[ii][0]]
        if indices[ii][1] == n_points:
            q_i = [(p_i[0] + p_i[1]) / 2, (p_i[0] + p_i[1]) / 2]
        else:
            q_i = b[indices[ii][1]]
        interpolated[ii, 0] = t * p_i[0] + (1 - t) * q_i[0]
        interpolated[ii, 1] = t * p_i[1] + (1 - t) * q_i[1]
    return interpolated


def align_2d(reference, target):
    ref_3d = [[point[0], point[1], 0] for point in reference]
    tgt_3d = [[point[0], point[1], 0] for point in target]
    rotation, _, _ = R.align_vectors(tgt_3d, ref_3d, return_sensitivity=True)
    return rotation.apply(ref_3d)[:, :2]


def reflect_points(points):
    return np.array([[point[0], -point[1]] for point in points])


def compute_coot_ot_gw_distances(X1, X2, C1, C2, C_pd, pi_s, pi_f, alpha, beta):
    _X1 = np.hstack([X1, np.zeros((X1.shape[0], 1))])
    _X2 = np.hstack([X2, np.zeros((X2.shape[0], 1))])

    coot_term = dot(
        eta(_X1, _X2, pi_f.sum(-1), pi_f.sum(0)) - _X1 @ pi_f @ _X2.T,
        pi_s,
    )
    ot_term = dot(C_pd, pi_f)
    gw_term = dot(
        eta(C1, C2, pi_s.sum(-1), pi_s.sum(0)) - C1 @ pi_s @ C2.T,
        pi_s,
    )
    return beta * coot_term, (1 - alpha) * ot_term, alpha * gw_term


def zscore(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    return (values - mean) / std if std > 0 else values * 0.0


def align_cycles(reference_matrix, target_matrix, cycle_matching):
    pairs = np.array(cycle_matching, dtype=int)
    if len(pairs) == 0:
        return np.zeros((target_matrix.shape[0], reference_matrix.shape[1]), dtype=float)

    target_idx, reference_idx = pairs[:, 0], pairs[:, 1]
    keep = reference_idx < reference_matrix.shape[1]
    target_idx = target_idx[keep]
    reference_idx = reference_idx[keep]

    aligned = np.zeros((target_matrix.shape[0], reference_matrix.shape[1]), dtype=float)
    if len(target_idx) > 0:
        aligned[:, reference_idx] = target_matrix[:, target_idx]
    return aligned


def col_normalize(matrix, eps=1e-12):
    matrix = np.clip(matrix, 0.0, None).astype(float)
    colsum = matrix.sum(axis=0, keepdims=True)
    colsum = np.maximum(colsum, eps)
    return matrix / colsum


def cycle_entropy(matrix, eps=1e-12, normalize_to_01=True):
    clipped = np.clip(matrix, eps, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=0)
    if normalize_to_01:
        entropy /= np.log(matrix.shape[0] + 1e-9)
    return entropy
