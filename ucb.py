import numpy as np
from typing import List, Set, Dict, Iterable, Optional, Tuple
from utilities import topo_order_from_adj

def longest_path_length(adj:np.ndarray) -> int:
    n = adj.shape[0]
    order = topo_order_from_adj(adj)
    dp = np.zeros(n, dtype = int)
    for v in order:
        for w in np.where(adj[v] != 0)[0]:
            dp[int(w)] = max(dp[int(w)], dp[v] + 1)
    return int(dp.max())


def expected_z_from_A_nu(A: np.ndarray, nu: np.ndarray, L: int) -> np.ndarray:

    n = A.shape[0]
    out = nu.copy()
    cur = nu.copy()
    for _ in range(L):
        cur = A @ cur
        out += cur
    return out


def sample_from_ellipsoid(xhat: np.ndarray, M: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    p = xhat.shape[0]
    Minv = np.linalg.pinv(M)
    w, V = np.linalg.eigh((Minv + Minv.T) / 2.0)
    w = np.maximum(w, 0.0)
    Minv_sqrt = V @ np.diag(np.sqrt(w)) @ V.T

    g = rng.normal(size = p)
    ng = np.linalg.norm(g) + 1e-12
    direction = g / ng
    d = Minv_sqrt @ direction
    d = d * np.sqrt(beta)
    return xhat + d

def build_A_tilde(
    Ahat: np.ndarray,
    Astar_hat: np.ndarray,
    pat: List[List[int]],
    action: Set[int],
    M_row_obs: Dict[int, np.ndarray],
    M_row_int: Dict[int, np.ndarray],
    beta_row: Dict[int, float],
    rng: np.random.Generator,
    hard: bool
) -> np.ndarray:
    n = Ahat.shape[0]
    Atilde = np.zeros((n, n))
    for i in range(n):
        pa = pat[i]
        if len(pa) == 0:
            continue
        in_action = (i in action)
        if hard and in_action:
            continue
        if in_action:
            xhat = Astar_hat[i, pa]
            M = M_row_int[i]
        else:
            xhat = Ahat[i, pa]
            M = M_row_obs[i]
        beta = beta_row[i]
        xt = sample_from_ellipsoid(xhat, M, beta, rng)
        Atilde[i, pa] = xt
    return Atilde

