import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
from utilities import *
from ucb import ucb_mc, precompute_ucb_sqrt

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


def action_to_id(a: Set[int], n: int) -> int:
    """Map action set a to id in {0,1,...,n}. Only supports empty or singleton actions."""
    if len(a) == 0:
        return 0
    if len(a) != 1:
        raise ValueError("This minimal reproduction only supports ∅ and singleton interventions.")
    i = int(next(iter(a)))
    if i < 0 or i >= n:
        raise ValueError("action index out of range")
    return i + 1


def id_to_action(a_id: int, n: int) -> Set[int]:
    if a_id == 0:
        return set()
    i = a_id - 1
    if i < 0 or i >= n:
        raise ValueError("action id out of range")
    return {i}

def hard_refinement_update(H: np.ndarray, SigmaZ_a: np.ndarray, i: int, pa_i: List[int], ridge: float = 1e-8) -> np.ndarray:
    """
    Implements the refinement H <- (I - Xi) H where Xi[i,pa(i)] = Cov(i,pa) Cov(pa,pa)^{-1}.
    """
    n = H.shape[0]
    if len(pa_i) == 0:
        return H
    Spp = SigmaZ_a[np.ix_(pa_i, pa_i)] + ridge * np.eye(len(pa_i))
    Sip = SigmaZ_a[i, pa_i].reshape(1, -1)
    Xi_row = Sip @ np.linalg.inv(Spp)  # 1 x |pa|
    Xi = np.zeros((n, n))
    Xi[i, pa_i] = Xi_row.reshape(-1)
    return (np.eye(n) - Xi) @ H


@dataclass
class ROCRLLearner:
    """
    Minimal, end-to-end RO-CRL reproduction (Yan 2025):
      - supports actions ∅ and singleton {i}
      - uses precision-difference eigenvector latent recovery
      - graph update via thresholded latent precision differences
      - under-sampling rule with u(Ghat) depending on soft vs hard (Eq. 35 style)
      - parameter estimation via doubly-weighted ridge regressions
      - UCB by Monte-Carlo sampling from ellipsoids
    """

    n_latent: int
    d_obs: int
    gamma: float = 0.1
    gamma_update_every: int = 25
    epsmax: float = 0.25
    delta: float = 0.05
    ridge_cov: float = 1e-6
    ridge_reg: float = 1.0

    # histories
    X_hist: List[np.ndarray] = field(default_factory=list)
    U_hist: List[float] = field(default_factory=list)
    A_hist: List[Set[int]] = field(default_factory=list)

    # action counts for A0 = {∅} U {{i}}
    counts_A0: Dict[int, int] = field(default_factory=dict)

    # maintained running sums for covariances (obs and each singleton)
    sum_x: Dict[int, np.ndarray] = field(default_factory=dict)
    sum_xxT: Dict[int, np.ndarray] = field(default_factory=dict)

    # latest estimates
    H: Optional[np.ndarray] = None           # n x d
    Ghat: Optional[np.ndarray] = None        # not used explicitly; H acts as pseudo-inv
    Z_hist: Optional[np.ndarray] = None      # n x t
    G_adj: Optional[np.ndarray] = None       # n x n adjacency (0/1)
    Ghat_tc: Optional[np.ndarray] = None

    # SEM / reward param estimates
    A: Optional[np.ndarray] = None           # n x n (row i: parents -> i)
    Astar: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None

    # confidence / gram matrices for UCB sampling
    M_row_obs: Dict[int, np.ndarray] = field(default_factory=dict)
    M_row_int: Dict[int, np.ndarray] = field(default_factory=dict)
    beta_row: Dict[int, float] = field(default_factory=dict)
    M_theta: Optional[np.ndarray] = None
    beta_theta: float = 1.0
    pat: Optional[List[List[int]]] = None

    def __post_init__(self):
        # initialize action stats dicts for ids 0..n
        for a_id in range(self.n_latent + 1):
            self.counts_A0[a_id] = 0
            self.sum_x[a_id] = np.zeros(self.d_obs)
            self.sum_xxT[a_id] = np.zeros((self.d_obs, self.d_obs))

        # start with empty graph
        self.G_adj = np.zeros((self.n_latent, self.n_latent), dtype=int)
        self.Ghat_tc = np.zeros((self.n_latent, self.n_latent), dtype=int)

    # -----------------
    # observation update
    # -----------------
    def observe(self, x_t: np.ndarray, u_t: float, action: Set[int]):
        x_t = np.asarray(x_t, dtype=float).reshape(-1)
        if x_t.shape[0] != self.d_obs:
            raise ValueError("x_t has wrong dimension")
        self.X_hist.append(x_t)
        self.U_hist.append(float(u_t))
        self.A_hist.append(set(action))

        a_id = action_to_id(action, self.n_latent)
        self.counts_A0[a_id] += 1
        self.sum_x[a_id] += x_t
        self.sum_xxT[a_id] += np.outer(x_t, x_t)

    # -----------------
    # latent recovery
    # -----------------
    def _cov_for_action_id(self, a_id: int) -> np.ndarray:
        n = self.counts_A0[a_id]
        return cov_from_sum(self.sum_x[a_id], self.sum_xxT[a_id], n)
        
    def _precision_for_action_id(self, a_id: int) -> np.ndarray:
        Sigma = self._cov_for_action_id(a_id)
        return pinv_precision(Sigma, ridge=self.ridge_cov)

    def update_H_from_precision_diff(self, i: int) -> np.ndarray:
        """
        H row i <- principal eigenvector of R_{i} = Theta_{ {i} } - Theta_{ empty }.
        """
        Theta_i = self._precision_for_action_id(i + 1)
        Theta_0 = self._precision_for_action_id(0)
        R_i = Theta_i - Theta_0
        v = principal_eigvec(R_i)  # d-vector
        if self.H is None:
            self.H = np.zeros((self.n_latent, self.d_obs))
        self.H[i, :] = v.reshape(-1)
        return self.H

    def estimate_Z_history(self) -> np.ndarray:
        """
        Zhat = H X (with X stacked as d x t).
        """
        if self.H is None:
            raise ValueError("H is None; cannot estimate Z.")
        X = np.stack(self.X_hist, axis=1)  # d x t
        Zhat = self.H @ X                  # n x t
        self.Z_hist = Zhat
        return Zhat


    # -----------------
    # graph update
    # -----------------
    def update_graph(self, take_transitive_closure: bool):
        """
        Edge rule: i -> j iff i != j and || R^Z_i[j,:] ||_2 > gamma
        where R^Z_i = (H^+)^T R_i H^+
        """
        if self.H is None:
            return
        H = self.H
        Hpinv = np.linalg.pinv(H)                 # d x n
        Hpinv_T = Hpinv.T                         # n x d

        Theta_0 = self._precision_for_action_id(0)
        adj = np.zeros((self.n_latent, self.n_latent), dtype=int)

        S_t = np.zeros((self.n_latent, self.n_latent), dtype=float)
        for i in range(self.n_latent):
            # need enough samples for singleton and obs, else skip
            if self.counts_A0[i + 1] <= 0 or self.counts_A0[0] <= 0:
                continue
            Theta_i = self._precision_for_action_id(i + 1)
            R_i = Theta_i - Theta_0
            Rz = Hpinv_T @ R_i @ Hpinv           # n x n
            # edge to j based on row/col magnitude; we use row j of Rz (heuristic)
            for j in range(self.n_latent):
                if i == j:
                    continue
                score = float(np.linalg.norm(Rz[j, :], ord=2))
                S_t[i, j] = score

                if score > self.gamma:
                    adj[i, j] = 1
        
        if len(self.X_hist) % self.gamma_update_every == 0:
            adj = np.zeros((self.n_latent, self.n_latent), dtype=int)
            self.gamma = gamma_schedule_noise_margin(S_t, len(self.X_hist))
            for i in range(self.n_latent):
                for j in range(self.n_latent):
                    if S_t[i, j] > self.gamma:
                        adj[i, j] = 1

        # enforce acyclicity by removing back-edges with a simple ordering heuristic
        # (prevents topo-sort crashes; can be swapped for a proper DAG projection)
        # we order nodes by index for this minimal repro: keep only i->j with i<j
        #adj = np.triu(adj, k=1).astype(int)

        self.G_adj = adj
        self.Ghat_tc = transitive_closure(adj) if take_transitive_closure else adj

    # -----------------
    # parameter estimation
    # -----------------
    def fit_sem_and_theta(self, intervention_type: str, zeta_t: float):
        """
        Fits A (obs SEM), A* (intervened rows), theta with doubly-weighted ridge.
        Stores per-row Gram matrices for UCB sampling.
        """
        if self.Z_hist is None:
            self.estimate_Z_history()
        Z = self.Z_hist  # n x t
        n, t = Z.shape

        adj = self.G_adj if self.G_adj is not None else np.zeros((n, n), dtype=int)
        pat = parents_from_adj(adj)
        self.pat = pat
        A = np.zeros((n, n))
        Astar = np.zeros((n, n))
        hard = (intervention_type == "hard")

        # build gates for each row regression: obs vs int samples
        for i in range(n):
            pa = pat[i]
            p = len(pa)
            if p == 0:
                self.M_row_obs[i] = self.ridge_reg * np.eye(1)
                self.M_row_int[i] = self.ridge_reg * np.eye(1)
                self.beta_row[i] = 0.0
                continue

            # design matrix is Z_pa^T (t x p), response is Z_i (t,)
            Xfeat = Z[pa, :].T  # t x p
            y = Z[i, :].reshape(-1, 1)  # t x 1

            gates_obs = np.array([0.0 if (i in a) else 1.0 for a in self.A_hist], dtype=float)
            gates_int = 1.0 - gates_obs

            # obs regression weights/gram
            Vobs, w_obs = doubly_weighted_gram(Xfeat, gates_obs, zeta_t, ridge=self.ridge_reg)
            Wobs = np.diag(w_obs)
            XtWX = Xfeat.T @ Wobs @ Xfeat + self.ridge_reg * np.eye(p)
            XtWy = Xfeat.T @ Wobs @ y
            Ai = np.linalg.solve(XtWX, XtWy).reshape(-1)

            # int regression weights/gram (for row i under intervention)
            Vint, w_int = doubly_weighted_gram(Xfeat, gates_int, zeta_t, ridge=self.ridge_reg)
            Wint = np.diag(w_int)
            XtWXs = Xfeat.T @ Wint @ Xfeat + self.ridge_reg * np.eye(p)
            XtWys = Xfeat.T @ Wint @ y
            Ais = np.linalg.solve(XtWXs, XtWys).reshape(-1)

            A[i, pa] = Ai
            Astar[i, pa] = Ais

            # store gram matrices for UCB sampling
            self.M_row_obs[i] = sym(XtWX)
            self.M_row_int[i] = sym(XtWXs) if not hard else sym(Vint)
            self.beta_row[i] = float(np.sqrt(p))  # simple default; you can plug in paper's beta_t

        if hard:
            Astar[:] = 0.0

        # fit theta from (Z, U)
        U = np.asarray(self.U_hist, dtype=float).reshape(-1, 1)  # t x 1
        X_theta = Z.T                                            # t x n
        gates_theta = np.ones(t, dtype=float)
        Vth, w_th = doubly_weighted_gram(X_theta, gates_theta, zeta_t, ridge=self.ridge_reg)
        Wt = np.diag(w_th)
        XtWX_th = X_theta.T @ Wt @ X_theta + self.ridge_reg * np.eye(n)
        XtWY_th = X_theta.T @ Wt @ U
        theta = np.linalg.solve(XtWX_th, XtWY_th).reshape(-1)

        self.A = A
        self.Astar = Astar
        self.theta = theta

        self.M_theta = sym(XtWX_th)
        self.beta_theta = float(np.sqrt(n))

        return A, Astar, theta


    # -----------------
    # decision rules
    # -----------------
    def under_sampling_action(self, rng: np.random.Generator, intervention_type: str) -> Optional[Set[int]]:
        """
        Returns an action (empty or {i}) if under-explored set is non-empty, else None.
        """
        if self.Ghat is None:
            # before we have a graph estimate, just count-based under-sampling
            AUE = [a_id for a_id in range(self.n_latent + 1) if self.counts_A0.get(a_id, 0) == 0]
            if len(AUE) == 0:
                return None
            pick = int(rng.choice(AUE))
            return set() if pick == 0 else {pick - 1}

        AUE, _ = under_explored_set(
            t=len(self.X_hist),
            counts_A0=self.counts_A0,
            Ghat_adj=self.Ghat_tc,
            d=self.d_obs,
            n=self.n_latent,
            epsmax=self.epsmax,
            delta=self.delta,
            intervention_type=intervention_type,
        )
        if len(AUE) == 0:
            return None
        pick = int(rng.choice(AUE))
        return id_to_action(pick, self.n_latent)

    def choose_action_with_ucb(
        self,
        candidate_actions: Optional[List[Set[int]]],
        intervention_type: str,
        rng: np.random.Generator,
        num_mc: int = 64
    ) -> Set[int]:
        """
        If candidate_actions is None, uses [∅, {0},...,{n-1}].
        """
        if candidate_actions is None:
            candidate_actions = [set()] + [{i} for i in range(self.n_latent)]

        if self.A is None or self.Astar is None or self.theta is None or self.Z_hist is None:
            # fallback to random if not fitted
            return set(candidate_actions[int(rng.integers(0, len(candidate_actions)))])

        # need nu estimates (obs + int)
        hard = (intervention_type == "hard")
        pat = parents_from_adj(self.G_adj if self.G_adj is not None else np.zeros((self.n_latent, self.n_latent), dtype=int))
        nu_hat, nu_star_hat = ucb_mc.__globals__["estimate_nu_vectors"](
            Z=self.Z_hist,
            pat=pat,
            Ahat=self.A,
            Astar_hat=self.Astar,
            a_hist=self.A_hist,
            hard=hard
        )

        M_theta = self.M_theta if self.M_theta is not None else np.eye(self.n_latent)
        precomputed_sqrt = precompute_ucb_sqrt(
            M_theta=M_theta,
            M_row_obs=self.M_row_obs,
            M_row_int=self.M_row_int,
        )

        best_val = -1e18
        best_a = None

        for a in candidate_actions:
            val = ucb_mc(
                Ahat=self.A,
                Astar_hat=self.Astar,
                theta_hat=self.theta,
                nu_hat=nu_hat,
                nu_star_hat=nu_star_hat,
                pat=pat,
                action=set(a),
                M_row_obs=self.M_row_obs,
                M_row_int=self.M_row_int,
                beta_row=self.beta_row,
                M_theta=self.M_theta if self.M_theta is not None else np.eye(self.n_latent),
                beta_theta=self.beta_theta,
                adj=self.G_adj if self.G_adj is not None else np.zeros((self.n_latent, self.n_latent), dtype=int),
                hard=hard,
                num_mc=num_mc,
                rng=rng,
                precomputed_sqrt=precomputed_sqrt,
            )
            if val > best_val:
                best_val = val
                best_a = set(a)

        return best_a if best_a is not None else set()

    # -----------------
    # main learner update
    # -----------------
    def learner_update(self, intervention_type: str, zeta_t: float) -> Tuple[List[int], int, bool]:
        """
        Mirrors the pseudocode control flow:
          1) latent recovery + graph update each round (if enough data)
          2) compute under-explored set; if non-empty, skip parameter estimation
          3) else fit SEM/theta and return empty AUE.
        Returns: (AUE, threshold, did_fit_params)
        """
        t = len(self.X_hist)

        # Need at least one obs sample to start
        if self.counts_A0[0] >= 1:
            # update H rows if we have singleton samples
            for i in range(self.n_latent):
                if self.counts_A0[i + 1] >= 1:
                    self.update_H_from_precision_diff(i)

            if self.H is not None:
                self.estimate_Z_history()

                # update graph; following your notes: take TC in main update for soft rule
                take_tc = True
                self.update_graph(take_transitive_closure=take_tc)
                self.Ghat = self.G_adj

        # under-sampling rule
        if self.Ghat_tc is None:
            G_for_u = np.zeros((self.n_latent, self.n_latent), dtype=int)
        else:
            G_for_u = self.Ghat_tc

        AUE, thresh = under_explored_set(
            t=max(t, 1),
            counts_A0=self.counts_A0,
            Ghat_adj=G_for_u,
            d=self.d_obs,
            n=self.n_latent,
            epsmax=self.epsmax,
            delta=self.delta,
            intervention_type=intervention_type
        )

        if len(AUE) > 0:
            return AUE, thresh, False

        # fit params only if not under-sampled
        if t >= 2 and self.H is not None:
            self.fit_sem_and_theta(intervention_type=intervention_type, zeta_t=zeta_t)
            return [], thresh, True

        return [], thresh, False













        