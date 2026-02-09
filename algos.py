import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


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

@dataclass
class ActionCovStats:
    """
    update empirical covariance of interventional data
    """
    d: int
    n: int = 0
    sum_x: Optional[np.ndarray] = None
    sum_xxT: Optional[np.ndarray] = None

    def __post_init__(self):
        self.sum_x = np.zeros(self.d)
        self.sum_xxT = np.zeros((self.d, self.d))
    
    def update(self, x: np.ndarray):
        x = np.asarray(x).reshape(-1)
        assert x.shape[0] == self.d, "x must have shape (d,)"
        self.n += 1
        self.sum_x += x
        self.sum_xxT += np.outer(x,x)
    
    def cov(self) -> np.ndarray:
        return cov_from_sum(self.sum_x, self.sum_xxT, self.n)


class ROCRLLearner:

    def __init__(
        self,
        n_latent: int,
        d_obs: int,
        gamma: float,
        ridge_x: float = 1e-8,
        ridge_reg: float = 1.0
    ):
        self.n = n_latent
        self.d = d_obs
        self.gamma = gamma
        self.ridge_x = ridge_x
        self.ridge_reg = ridge_reg

        # tracks covariance for each atomic intervention, 0 is observational (no intervention)
        self.stats: Dict[int, ActionCovStats] = {k: ActionCovStats(d_obs) for k in range(n_latent + 1)}

        self.X_hist: List[np.ndarray] = []
        self.U_hist: List[np.ndarray] = []
        self.a_hist: List[Set[int]] = []

        self.H: Optional[np.ndarray] = None        # n x d
        self.Zhat: Optional[np.ndarray] = None    # n x t
        self.Ghat: Optional[np.ndarray] = None    # n x n (adjacency matrix)
        self.pat: Optional[List[List[int]]] = None # parents of each node in Ghat

        def observe(self, x_t: np.ndarray, u_t: float, a_t: Set[int]):
            """
            observe data at time t and update statistics
            """
            x_t = np.asarray(x_t).reshape(-1)
            assert x_t.shape[0] == self.d, "x_t must have shape (d,)"
            self.X_hist.append(x_t)
            self.U_hist.append(float(u_t))
            self.a_hist.append(set(a_t))

            if len(a_t) == 0: # observational data
                self.stats[0].update(x_t)
            elif len(a_t) == 1: # atomic intervention
                i = next(iter(a_t))
                self.stats[i].update(x_t)
        

        def estimate_precision_diffs(self) -> List[np.ndarray]:
            """
            compute R_{i,t} = Theta_{i,t} - Theta_{0, t} for each intervention i
            """
            Sigma0 = self.stats[0].cov()
            Theta0 = pinv_precision(Sigma0, ridge = self.ridge_x)
            R_list = []
            for i in range(1, self.n + 1):
                Sig_i = self.stats[i].cov()
                Theta_i = pinv_precision(Sig_i, ridge = self.ridge_x)
                R_list.append(sym(Theta_i - Theta0))
            return R_list
        
        def update_H(self): np.ndarray:
            """
            [H_t]_i <- principal eigenvector of R_{i,t}
            """
            R_list = self.estimate_precision_diffs()
            H = np.zeros((self.n, self.d))
            for i in range(self.n):
                H[i, :] = principal_eigvec(R_list[i])
            self.H = H
            return H

        def estimate_Zhat(self) -> np.ndarray:
            """
            Zhat_t = H_t @ X_t
            """
            X = np.stack(self.X_hist, axis = 1) # d x t
            Zhat = self.H @ X
            self.Zhat = Zhat
            return Zhat
        
        def update_graph(self) -> np.ndarray:
            """
            compute Rz_hat_{i,t} = Hdag_t @ R_{i,t} @ H_t for each intervention i
            assign edge i -> j if ||[Rz_hat_{i,t}]_j||_2 > gamma
            set transitive closure of adjacency matrix as Ghat
            """
            R_list = self.estimate_precision_diffs()
            Hdag = np.linalg.pinv(self.H)
            adj = np.zeros((self.n, self.n), dtype = int)

            for i in range(self.n):
                Rz_hat = Hdag @ R_list[i] @ H
                for j in range(self.n):
                    if j == i:
                        continue
                    if np.linalg.norm(Rz_hat[j, :], ord = 2) > self.gamma:
                        adj[i, j] = 1
            
            adj_tc = transitive_closure(adj)
            self.Ghat = adj_tc
            self.pat = parents_from_adj(adj_tc)
            return adj_tc
        
        def hard_refine_H(self) -> np.ndarray:
            """
            refine H_t under hard interventions
            """
            Sigma_a = {k: self.stats[k].cov() for k in range(self.n + 1)}
            SigmaZ = {k: sym(self.H @ Sigma_a[k] @ self.H.T) for k in range(self.n + 1)}

            Xi = np.zeros((self.n, self.d))
            for i in range(self.n):
                pa = self.pat[i]
                if len(pa) == 0:
                    continue
                env = i + 1
                S = SigmaZ[env]
                S_ip = S[i, pa]
                S_pp = S[np.ix_(pa, pa)]
                Xi[i, pa] = S_ip @ np.linalg.pinv(S_pp + 1e-8 * np.eye(len(pa)))
            H_new = (np.eye(self.n) - Xi) @ self.H
            self.H = H_new
            return H_new
