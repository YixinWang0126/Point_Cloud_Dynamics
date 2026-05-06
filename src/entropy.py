from __future__ import annotations

import numpy as np


def persist_entropy(pd_point):
    durations = pd_point[:, 1] - pd_point[:, 0]
    total = durations.sum()
    entropy = 0.0
    for duration in durations:
        ratio = duration / total
        entropy -= ratio * np.log(ratio)
    return entropy


def hypergraph_shannon_entropy(incidence_matrix: np.ndarray) -> float:
    laplacian = incidence_matrix @ incidence_matrix.T
    eigvals = np.linalg.eigvalsh(laplacian)
    probs = eigvals / eigvals.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def hyper_graph_entropy(incidence_matrix):
    vertex_mass = incidence_matrix.sum(axis=1)
    active = vertex_mass[vertex_mass > 0]
    if len(active) == 0:
        return 0.0
    probs = active / active.sum()
    return -np.sum(probs * np.log(probs)) / np.log(len(active))


def hyper_edge_entropy(incidence_matrix):
    edge_mass = incidence_matrix.sum(axis=0)
    active = edge_mass[edge_mass > 0]
    if len(active) == 0:
        return 0.0
    probs = active / active.sum()
    return -np.sum(probs * np.log(probs)) / np.log(len(active))
