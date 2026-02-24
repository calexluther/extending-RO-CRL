import numpy as np
from typing import List, Set, Dict, Iterable, Optional, Tuple
from utilities import topo_order_from_adj, record_acyclicity_fallback

def longest_path_length(adj: np.ndarray) -> int:
    n = adj.shape[0]
    if n == 0:
        return 0
    try:
        order = topo_order_from_adj(adj)
    except ValueError:
        # Graph is not acyclic; use n as fallback (safe bound for propagation steps)
        record_acyclicity_fallback()
        return n
    dp = np.zeros(n, dtype=int)
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


def compute_ellipsoid_sqrt(M: np.ndarray) -> np.ndarray:
    """Precompute Minv_sqrt for ellipsoid sampling. O(p^3) once instead of per sample."""
    M = np.asarray(M)
    Minv = np.linalg.pinv(M)
    w, V = np.linalg.eigh((Minv + Minv.T) / 2.0)
    w = np.maximum(w, 0.0)
    return V @ np.diag(np.sqrt(w)) @ V.T


def sample_from_ellipsoid_fast(
    xhat: np.ndarray,
    Minv_sqrt: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from ellipsoid using precomputed Minv_sqrt. No pinv/eigh in the loop."""
    p = xhat.shape[0]
    g = rng.normal(size=p)
    ng = np.linalg.norm(g) + 1e-12
    direction = g / ng
    d = Minv_sqrt @ direction * np.sqrt(beta)
    return xhat + d


def sample_from_ellipsoid(xhat: np.ndarray, M: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    p = xhat.shape[0]
    Minv_sqrt = compute_ellipsoid_sqrt(M)
    return sample_from_ellipsoid_fast(xhat, Minv_sqrt, beta, rng)

def precompute_ucb_sqrt(
    M_theta: np.ndarray,
    M_row_obs: Dict[int, np.ndarray],
    M_row_int: Dict[int, np.ndarray],
) -> Dict:
    """
    Precompute ellipsoid sqrt factors once per UCB step. Returns dict with
    'M_theta_sqrt', 'M_row_obs_sqrt', 'M_row_int_sqrt' for use in ucb_mc fast path.
    """
    return {
        "M_theta_sqrt": compute_ellipsoid_sqrt(np.asarray(M_theta)),
        "M_row_obs_sqrt": {i: compute_ellipsoid_sqrt(np.asarray(M)) for i, M in M_row_obs.items()},
        "M_row_int_sqrt": {i: compute_ellipsoid_sqrt(np.asarray(M)) for i, M in M_row_int.items()},
    }


def build_A_tilde(
    Ahat: np.ndarray,
    Astar_hat: np.ndarray,
    pat: List[List[int]],
    action: Set[int],
    M_row_obs: Dict[int, np.ndarray],
    M_row_int: Dict[int, np.ndarray],
    beta_row: Dict[int, float],
    rng: np.random.Generator,
    hard: bool,
    precomputed_sqrt: Optional[Dict] = None,
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
        beta = float(beta_row.get(i, 0.0))
        if precomputed_sqrt is not None:
            if in_action:
                xhat = np.asarray(Astar_hat[i, pa])
                Minv_sqrt = precomputed_sqrt["M_row_int_sqrt"][i]
            else:
                xhat = np.asarray(Ahat[i, pa])
                Minv_sqrt = precomputed_sqrt["M_row_obs_sqrt"][i]
            xt = sample_from_ellipsoid_fast(xhat, Minv_sqrt, beta, rng)
        else:
            if in_action:
                xhat = np.asarray(Astar_hat[i, pa])
                M = M_row_int[i]
            else:
                xhat = np.asarray(Ahat[i, pa])
                M = M_row_obs[i]
            xt = sample_from_ellipsoid(xhat, np.asarray(M), beta, rng)
        Atilde[i, pa] = xt
    return Atilde

def estimate_nu_vectors(
    Z: np.ndarray,
    pat: List[List[int]],
    Ahat: np.ndarray,
    Astar_hat: np.ndarray,
    a_hist: List[Set[int]],
    hard: bool
) -> Tuple[np.ndarray, np.ndarray]:

    n, t = Z.shape
    nu_hat = np.zeros(n)
    nu_star = np.zeros(n)

    c_obs = np.zeros(n, dtype = int)
    c_int = np.zeros(n, dtype = int)
    for s in range(t):
        a_s = a_hist[s]
        z_s = Z[:, s]
        for i in range(n):
            pa = pat[i]
            pred_obs = 0.0 if len(pa) == 0 else float(Ahat[i, pa] @ z_s[pa])
            pred_int = 0.0 if len(pa) == 0 else float(Astar_hat[i, pa] @ z_s[pa])

            if i not in a_s:
                nu_hat[i] += (z_s[i] - pred_obs)
                c_obs[i] += 1
            else:
                if hard:
                    nu_star[i] += z_s[i]
                else:
                    nu_star[i] += (z_s[i] - pred_int)
                c_int[i] += 1
    for i in range(n):
        if c_obs[i] > 0:
            nu_hat[i] /= c_obs[i]
        if c_int[i] > 0:
            nu_star[i] /= c_int[i]
    return nu_hat, nu_star


def ucb_mc(
    Ahat: np.ndarray,
    Astar_hat: np.ndarray,
    theta_hat: np.ndarray,
    nu_hat: np.ndarray,
    nu_star_hat: np.ndarray,
    pat: List[List[int]],
    action: Set[int],
    M_row_obs: Dict[int, np.ndarray],
    M_row_int: Dict[int, np.ndarray],
    beta_row: Dict[int, float],
    M_theta: np.ndarray,
    adj: np.ndarray,
    beta_theta: float,
    rng: np.random.Generator,
    hard: bool,
    num_mc: int = 64,
    precomputed_sqrt: Optional[Dict] = None,
) -> float:

    L = longest_path_length(adj)
    best = -1e18

    for _ in range(int(max(1, num_mc))):
        if precomputed_sqrt is not None:
            theta_tilde = sample_from_ellipsoid_fast(
                theta_hat, precomputed_sqrt["M_theta_sqrt"], beta_theta, rng
            )
        else:
            theta_tilde = sample_from_ellipsoid(theta_hat, M_theta, beta_theta, rng)
        Atilde = build_A_tilde(
            Ahat=Ahat,
            Astar_hat=Astar_hat,
            pat=pat,
            action=action,
            M_row_obs=M_row_obs,
            M_row_int=M_row_int,
            beta_row=beta_row,
            rng=rng,
            hard=hard,
            precomputed_sqrt=precomputed_sqrt,
        )
        nu_a = nu_hat.copy()
        for i in action:
            nu_a[i] = nu_star_hat[i]
        zmean = expected_z_from_A_nu(Atilde, nu_a, L)
        val = float(theta_tilde @ zmean)
        if val > best:
            best = val

    return float(best)



    