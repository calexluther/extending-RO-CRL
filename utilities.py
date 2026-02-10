import numpy as np
from typing import List, Set, Dict, Iterable, Optional, Tuple


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def principal_eigvec(R: np.ndarray) -> np.ndarray:
    R = sym(R)
    w, V = np.linalg.eigh(R)
    idx = int(np.argmax(np.abs(w)))
    return V[:, idx]

def cov_from_sum(sum_x: np.ndarray, sum_xxT: np.ndarray, n: int) -> np.ndarray:
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

def doubly_weighted_diag_weights(
    Zfeat: np.ndarray,
    gates: np.ndarray,
    zeta_t: float,
    ridge: float = 1.0
) -> np.ndarray:
    
    t, p = Zfeat.shape
    w = np.zeros(t, dtype = float)
    Vtilde = ridge * np.eye(p)
    for s in range(t):
        if gates[s] == 0.0:
            w[s] = 0.0
            continue
        z = Zfeat[s, :].reshape(p, 1)
        Vinv = np.linalg.pinv(Vtilde)
        norm = float(np.sqrt(max((z.T @ Vinv @ z)[0,0], 1e-12)))
        w[s] = (1.0 / zeta_t) * min(1.0, 1.0 / norm)

        Vtilde = Vtilde + (w[s] ** 2) * (z @ z.T)
    return w

def delta_schedule(delta: float, t: int) -> float:
    return float(6.0 * delta / (np.pi**2 * (t**2)))

def N_eps(d: int, eps: float, delta_t: float, C: float = 1.0) -> int:
    return int(np.ceil(C * (d + np.log(1.0 / max(delta_t, 1e-12))) / (eps**2)))

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

def parents_from_adj(adj: np.ndarray) -> List[List[int]]:
    n = adj.shape[0]
    return [list(np.where(adj[:, j] != 0)[0]) for j in range(n)]

def compute_u_t_from_graph(adj: np.ndarray, intervention_type: str) -> float:
    
    adj_for_u = transitive_closure(adj) if intervention_type == "soft" else adj
    n = adj.shape[0]
    pa = parents_from_adj(adj)
    order = topo_order_from_adj(adj)
    u_i = np.zeros(n)
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

    term1 = (d ** (1.0/3.0)) * (n ** (-2.0/3.0)) * (u_t ** (2.0/3.0)) * (t ** (2.0/3.0))
    term2 = N_eps(d, epsmax, delta_t)
    return int(np.ceil(max(term1, term2)))

def under_explored_set(
    t: int,
    counts_A0: dict,
    Ghat_adj: np.ndarray,
    n: int,
    epsmax: float,
    delta: float,
    intervention_type: str
) -> typle[list[int], int]:
    delta_t = delta_schedule(delta, t)
    uval = compute_u_t_from_graph(Ghat_adj, intervention_type)
    thresh = f_t(t, d, n, uval, epsmax, delta_t)
    AUE = [a_id for a_id in range(n + 1) if counts_A0.get(a_id, 0) < thresh]
    return AUE, thresh