import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
from utilities import *

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

        self.counts_A0 = {k: 0 for k in range(self.n + 1)}
        self.epsmax = 0.25
        self.delta = 0.05

        def record_action_pull(self, a_t: Set[int]):
            if len(a_t) == 0:
                self.counts_A0[0] += 1
            elif len(a_t) == 1:
                i = next(iter(a_t))
                self.counts_A0[i] += 1

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
        
        def update_graph(self, use_transitive_closure: bool = True) -> np.ndarray:
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
            
            if use_transitive_closure:
                adj_tc = transitive_closure(adj)
            else:
                adj_tc = adj
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
        
        def _build_weight_diag(
            self,
            which: str,
            i: Optional[int],
            zeta_t: float, 
            Vtilde_inv_norms: Sequence[float]
        ) -> np.ndarray:
            """
            Assemble weight diagonals given V^tilde inverse norms
            """
            t = len(self.X_hist)
            w = np.zeros(t)

            for s in range(t):
                a_s = self.a_hist[s]
                if which == "obs":
                    assert i is not None
                    gate = 1.0 if (i not in a_s) else 0.0
                elif which == "int":
                    assert i is not None
                    gate = 1.0 if (i in a_s) else 0.0
                elif which == "theta":
                    gate = 1.0
            
                bonus_inv = 0.0 if Vtilde_inv_norms[s] <= 0 else 1.0 / Vtilde_inv_norms[s]
                w[s] = gate * (1.0 / zeta_t) * min(1.0, bonus_inv)
            return w
        
        def fit_sem_and_theta(self, zeta_t: float = 1.0, hard: bool = False):

            Z = self.Zhat # n x t
            t = Z.shape[1]
            A = np.zeros((selfn.n, self.n))
            for i in range(self.n):
                pa = self.pat[i]
                if len(pa) == 0:
                    continue
                X_pa = Z[pa, :].T # t x |pa|
                y_i = Z[i, :].reshape(-1, 1) # t x 1

                gates_obs = np.array([1.0 if (i not in self.a_hist[s]) else 0.0 for s in range(t)])
                w_obs = doubly_weighted_diag_weights(X_pa, gates_obs, zeta_t, ridge = self.ridge_reg)
                W_obs = np.diag(w_obs)


                parent_norms = np.linalg.norm(X_pa, axis = 1) + 1e-8
                

                XtWX = X_pa.T @ W_obs @ X_pa + self.ridge_reg * np.eye(len(pa))
                XtWY = X_pa.T @ W_obs @ y_i
                coef = np.linalg.solve(XtWX, XtWY).reshape(-1)
                A[i, pa] = coef

                if not hard:
                    gates_int = np.array([1.0 if (i in self.a_hist[s]) else 0.0 for s in range(t)], dtype = float)
                    w_int = doubly_weighted_diag_weights(X_pa, gates_int, zeta_t, ridge = self.ridge_reg)
                    W_int = np.diag(w_int)
                    XtWXs = X_pa.T @ W_int @ X_pa + self.ridge_reg * np.eye(len(pa))
                    XtWYs = X_pa.T @ W_int @ y_i
                    coef_s = np.linalg.solve(XtWXs, XtWYs).reshape(-1)
                    Astar[i, pa] = coef_s
            if hard:
                Astar[:] = 0.0
            
            U = np.asarray(self.U_hist).reshape(-1,1)
            X_theta = Z.T
            gates_theta = np.ones(t, dtype = float)
            w_theta = doubly_weighted_diag_weights(X_theta, gates_theta, zeta_t, ridge = self.ridge_reg)
            Wt = np.diag(w_theta)

            XtWX = X_theta.T @ Wt @ X_theta + self.ridge_reg * np.eye(t)
            XtWY = X_theta.T @ Wt @ U
            theta = np.linalg.solve(XtWX, XtWY).reshape(-1)

            return self.A, self.Astar, self.theta, Astar, theta

        def choose_action_with_under_sampling(self, rng: np.random.Generator, intervention_type: str) -> int:
            if self.Ghat is None:
                pick = int(rng.choice(list(range(self.n + 1))))
                return set() if pick == 0 else {pick}
            
            AUE, thresh = under_explored_set(
                t = len(self.X_hist),
                counts_A0 = self.counts_A0,
                Ghat_adj = self.Ghat,
                d = self.d,
                n = self.n,
                epsmax = self.epsmax,
                delta = self.delta,
                intervention_type = intervention_type
            )
            if len(AUE) == 0:
                return None
            pick = int(rng.choice(AUE))
            return set() if pick == 0 else {pick}
        


        def learner_update(self, intervention_type: str, zeta_t: float = 1.0):
            
            if intervention_type == "soft":
                self.update_H()
                self.estimate_Zhat()
                self.update_graph(use_transitive_closure = True)
                
            else: # intervention_type == "hard"
                self.update_H()
                self.estimate_Zhat()
                self.update_graph(use_transitive_closure = True)
                self.hard_refine_H()
                self.estimate_Zhat()
                self.update_graph(use_transitive_closure = False)
            
            t = max(1, len(self.X_hist))
            AUE, thresh = under_explored_set(
                t = t,
                counts_A0 = self.counts_A0,
                Ghat_adj = self.Ghat,
                d = self.d,
                n = self.n,
                epsmax = self.epsmax,
                delta = self.delta,
                intervention_type = intervention_type
            )

            did_fit = False
            if len(AUE) == 0:
                self.fit_sem_and_theta(zeta_t = zeta_t, hard = intervention_type == "hard")
                did_fit = True
                # UCB selection
            
            return AUE, thresh, did_fit
        