import numpy as np
from typing import List, Set, Dict, Iterable, Optional, Tuple

# Counter for how often we fall back due to non-acyclic graph (under_explored_set + ucb_mc)
_acyclicity_fallback_count: int = 0


def get_acyclicity_fallback_count() -> int:
    """Return the number of times the estimated graph was non-acyclic and a fallback was used."""
    return _acyclicity_fallback_count


def reset_acyclicity_fallback_count() -> None:
    """Reset the acyclicity fallback counter to 0 (e.g. at the start of a new run)."""
    global _acyclicity_fallback_count
    _acyclicity_fallback_count = 0


def record_acyclicity_fallback() -> None:
    """Increment the acyclicity fallback counter (e.g. when topo_order_from_adj fails elsewhere)."""
    global _acyclicity_fallback_count
    _acyclicity_fallback_count += 1

def gamma_schedule_noise_margin(S_t, t, q_noise=0.25, q_signal=0.90, c=0.5):
    """
    gamma_t = noise_floor + margin
    noise_floor ~ lower quantile of scores
    margin ~ c * (signal_scale - noise_floor) / sqrt(t)
    """
    n = S_t.shape[0]
    vals = S_t[~np.eye(n, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0

    noise = float(np.quantile(vals, q_noise))
    signal = float(np.quantile(vals, q_signal))
    scale = max(signal - noise, 1e-12)

    margin = c * scale / np.sqrt(max(t, 1))
    return noise + margin

def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def principal_eigvec(R: np.ndarray) -> np.ndarray:
    R = sym(R)
    w, V = np.linalg.eigh(R)
    idx = int(np.argmax(np.abs(w)))
    return V[:, idx]

def cov_from_sum(sum_x: np.ndarray, sum_xxT: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        d = sum_x.shape[0]
        return np.zeros((d, d))
    mu = sum_x / n
    return (sum_xxT / n) - np.outer(mu, mu)

def pinv_precision(Sigma: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """
    compute precision matrix Theta as pseudo-inverse of Sigma:
    Theta = (Sigma + ridge * I)^+
    """
    Sigma = sym(Sigma) + ridge * np.eye(Sigma.shape[0])
    return np.linalg.pinv(Sigma)

def transitive_closure(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    reach = adj.astype(bool).copy()
    for k in range(n):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach.astype(int)

def parents_from_adj(adj: np.ndarray) -> List[List[int]]:
    """
    get parents of each node from adjacency matrix
    """
    n = adj.shape[0]
    return [list(np.where(adj[:, j] != 0)[0]) for j in range(n)]

def topo_order_from_adj(adj: np.ndarray) -> List[int]:
    n = adj.shape[0]
    indeg = adj.sum(axis = 0).astype(int).tolist()
    q = [i for i in range(n) if indeg[i] == 0]
    out = []
    while q:
        v = q.pop()
        out.append(v)
        for w in np.where(adj[v] != 0)[0]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(int(w))
    if len(out) != n:
        raise ValueError("Graph is not acyclic")
    return out


#-----------------------------------------------------------
# Doubly-weighted ridge regression weight matrices
#-----------------------------------------------------------

def doubly_weighted_diag_weights(
    Zfeat: np.ndarray,
    gates: np.ndarray,
    zeta_t: float,
    ridge: float = 1.0
) -> np.ndarray:

    """
    Implements the iterative, inverse-weighted leverage-score weights
    Returns the diagonal weights w_s
    Zfeat: (t,p) design matrix
    gates: (t,) in {0,1} gates
    """
    t, p = Zfeat.shape
    w = np.zeros(t, dtype = float)
    Vtilde = ridge * np.eye(p)
    zeta_t = float(max(zeta_t, 1e-12))

    for s in range(t):
        if gates[s] == 0.0:
            w[s] = 0.0
            continue
        z = Zfeat[s, :].reshape(p, 1)
        Vinv = np.linalg.pinv(Vtilde)
        lev = float((z.T @ Vinv @ z)[0,0])
        norm = float(np.sqrt(max(lev, 1e-12))) # ||hat{Z}_t[pa_t(i),s]||_{tilde{V}^{-1}}
        w[s] = (1.0 / zeta_t) * min(1.0, 1.0 / norm)

        Vtilde = Vtilde + (w[s] ** 2) * (z @ z.T)
    return w

def doubly_weighted_gram(
    Zfeat: np.ndarray,
    gates: np.ndarray,
    zeta_t: float,
    ridge: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same iteration as doubly_weighted_diag_weights, but also returns the final Vtilde (Gram).
    """
    t, p = Zfeat.shape
    w = np.zeros(t, dtype=float)
    Vtilde = ridge * np.eye(p)

    zeta_t = float(max(zeta_t, 1e-12))

    for s in range(t):
        if gates[s] == 0.0:
            w[s] = 0.0
            continue
        z = Zfeat[s, :].reshape(p, 1)
        Vinv = np.linalg.pinv(Vtilde)
        lev = float((z.T @ Vinv @ z)[0, 0])
        norm = float(np.sqrt(max(lev, 1e-12))) # ||hat{Z}_t[pa_t(i),s]||_{tilde{V}^{-1}}
        w[s] = (1.0 / zeta_t) * min(1.0, 1.0 / norm)
        Vtilde = Vtilde + (w[s] ** 2) * (z @ z.T)
    return Vtilde, w


def delta_schedule(delta: float, t: int) -> float:
    return float(6.0 * delta / (np.pi**2 * (t**2)))

def N_eps(d: int, eps: float, delta_t: float, C: float = 1.0) -> int:
    return int(np.ceil(C * (d + np.log(1.0 / max(delta_t, 1e-12))) / (eps**2)))


def parents_from_adj(adj: np.ndarray) -> List[List[int]]:
    n = adj.shape[0]
    return [list(np.where(adj[:, j] != 0)[0]) for j in range(n)]

def compute_u_t_from_graph(adj: np.ndarray, intervention_type: str) -> float:
    """
    Compute u_t from graph structure. If the graph is not acyclic (e.g. estimated
    graph has cycles), returns a conservative fallback sqrt(n) so thresholding
    still works.
    """
    n = adj.shape[0]
    if adj is None or n == 0:
        return float(np.sqrt(max(n, 1)))
    pa = parents_from_adj(adj)
    try:
        order = topo_order_from_adj(adj)
    except ValueError:
        # Estimated graph can be cyclic; use conservative fallback (same as no graph)
        record_acyclicity_fallback()
        return float(np.sqrt(n))
    u_i = np.zeros(n, dtype=float)
    for i in order:
        if len(pa[i]) == 0:
            u_i[i] = 0.0
        else:
            u_i[i] = float(u_i[pa[i]].sum() + np.sqrt(len(pa[i])))
    return float(u_i.sum() + np.sqrt(n))

def f_t(
    t: int,
    d: int,
    n: int,
    u_t: float,
    epsmax: float,
    delta_t: float
) -> int:
    t = max(int(t), 1)
    term1 = (d ** (1.0/3.0)) * (n ** (-2.0/3.0)) * (u_t ** (2.0/3.0)) * (t ** (2.0/3.0))
    term2 = N_eps(d, epsmax, delta_t)
    return int(np.ceil(max(term1, term2)))

def under_explored_set(
    t: int,
    counts_A0: Dict[int, int],
    Ghat_adj: np.ndarray,
    d: int,
    n: int,
    epsmax: float,
    delta: float,
    intervention_type: str
) -> Tuple[List[int], int]:
    """
    Returns (AUE, threshold) where AUE is a list of action-ids in {0,1,...,n}
    whose counts are below the current threshold.
    """
    delta_t = delta_schedule(delta, t)
    uval = compute_u_t_from_graph(Ghat_adj, intervention_type) if Ghat_adj is not None else float(np.sqrt(n))
    thresh = f_t(t, d, n, uval, epsmax, delta_t)
    AUE = [a_id for a_id in range(n + 1) if counts_A0.get(a_id, 0) < thresh]
    return AUE, thresh