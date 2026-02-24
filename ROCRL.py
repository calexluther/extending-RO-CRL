from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Sequence, Tuple, FrozenSet, Dict

from .lscalei import l_scale_i
from .utils_graph import parents_from_adj, causal_order, transitive_closure


Action = FrozenSet[int]          # frozenset of intervened node indices
MCA0Action = FrozenSet[int]      # either frozenset() or frozenset({i})


@dataclass
class CRLEstimate:
    enc_est: np.ndarray          # n x d (or n x d_reduced mapped back)
    dag_adj: np.ndarray          # n x n {0,1}
    pa: List[List[int]]          # parent lists
    topo: List[int]              # topological order


class ROCRLLearner:
    """
    
    """

    def __init__(
        self,
        n: int,
        d: int,
        intervention_type: str,          # "soft" or "hard"
        gamma: float,
        lam: float = 1e-3,              # ridge lambda (match script if fixed there)
        seed: int = 0,
        dim_reduction: bool = True,
        hard_unmixing: bool = False,     # whether to run l_scale_i hard branch
    ):
        assert intervention_type in ("soft", "hard")
        self.n, self.d = n, d
        self.intervention_type = intervention_type
        self.gamma = gamma
        self.lam = lam
        self.dim_reduction = dim_reduction
        self.hard_unmixing = hard_unmixing

        self.rng = np.random.default_rng(seed)

        # --- Pools exactly like the scripts ---
        self.X_all: List[np.ndarray] = []       # each is (d,)
        self.Z_all: List[np.ndarray] = []       # each is (n,)
        self.A_all: List[Action] = []           # action (intervention set)

        # CRL pool: only observational + singleton
        self.X_crl: List[np.ndarray] = []
        self.A_crl: List[MCA0Action] = []

        # Cached CRL results (recomputed many times in the scripts)
        self.crl: Optional[CRLEstimate] = None

        # For under-sampling counts on mca0
        self.mca0 = [frozenset()] + [frozenset({i}) for i in range(n)]

        # Any cached matrices for UCB (optional; script recomputes each loop)
        self._cache = {}

    # ----------------------------
    # Data ingestion (sampling results)
    # ----------------------------
    def add_sample(self, z: np.ndarray, x: np.ndarray, action: Action) -> None:
        self.Z_all.append(z.copy())
        self.X_all.append(x.copy())
        self.A_all.append(frozenset(action))

        # Only keep mca0 for CRL
        if action in self.mca0:
            self.Z_crl_append(x, action)

    def Z_crl_append(self, x: np.ndarray, action: MCA0Action) -> None:
        self.X_crl.append(x.copy())
        self.A_crl.append(frozenset(action))

    # ----------------------------
    # CRL step (exactly l_scale_i)
    # ----------------------------
    def fit_crl(self) -> CRLEstimate:
        X = np.stack(self.X_crl, axis=0)          # (t_crl, d)
        # Yan passes actions as list like [] or [i]; keep same format:
        actions_as_list = [list(a) for a in self.A_crl]

        soft_results, hard_results = l_scale_i(
            x_samples=X,
            actions_as_list=actions_as_list,
            gamma=self.gamma,
            hard_intervention=self.hard_unmixing,
            dim_reduction=self.dim_reduction,
        )

        if self.intervention_type == "soft":
            enc_est, dag_adj = soft_results
        else:
            # Yan's hard script uses "hard" semantics; many people still use soft_results
            # BUT if you want exact hard-branch behavior, use hard_results if provided.
            if hard_results is None:
                enc_est, dag_adj = soft_results
            else:
                enc_est, dag_adj = hard_results

        dag_adj = transitive_closure(dag_adj) if self.intervention_type == "soft" else dag_adj
        pa = parents_from_adj(dag_adj)
        topo = causal_order(dag_adj)

        self.crl = CRLEstimate(enc_est=enc_est, dag_adj=dag_adj, pa=pa, topo=topo)
        return self.crl

    # ----------------------------
    # Under-sampling rule (Yan’s f_t and mca0 counts)
    # ----------------------------
    def mca0_counts(self) -> Dict[MCA0Action, int]:
        counts = {a: 0 for a in self.mca0}
        for a in self.A_crl:
            counts[a] += 1
        return counts

    def compute_u(self, pa: List[List[int]], topo: List[int]) -> float:
        """
        Mirror Yan's u-computation (the recursive u_ along topo).
        You should copy-paste your existing correct implementation here.
        """
        # Placeholder structure (replace with exact script logic)
        u_vec = np.zeros(self.n)
        for v in topo:
            parents = pa[v]
            if len(parents) == 0:
                u_vec[v] = 1.0
            else:
                u_vec[v] = 1.0 + np.sum(u_vec[parents])
        return float(np.sum(u_vec))

    def f_t(self, t: int, u: float) -> float:
        # exactly as in Yan: d^(1/3) * n^(2/3) * u^(-2/3) * t^(2/3)
        return (self.d ** (1/3)) * (self.n ** (2/3)) * (u ** (-2/3)) * (t ** (2/3))

    def under_explored_set(self, t: int) -> List[MCA0Action]:
        assert self.crl is not None
        u = self.compute_u(self.crl.pa, self.crl.topo)
        threshold = self.f_t(t, u)
        counts = self.mca0_counts()
        return [a for a, c in counts.items() if c < threshold]

    def choose_action_under_sampling(self, under: Sequence[MCA0Action]) -> MCA0Action:
        # Yan: choose one under-explored action (often random or argmin count)
        counts = self.mca0_counts()
        under = list(under)
        under.sort(key=lambda a: counts[a])
        return under[0]

    # ----------------------------
    # Parameter estimation + UCB decision
    # ----------------------------
    def fit_sem_and_theta(self, zeta_t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        This should become your refactor of the giant loops in TO_CRL*.py:
        builds V, tilde_V, g__ etc. and returns A, A*, theta and any per-row Gram caches.

        For exactness: do NOT be clever initially; keep the same loop order and formulas.
        """
        assert self.crl is not None
        # TODO: implement by porting your existing version, but ensure:
        # - uses self.crl.pa and self.crl.enc_est if Yan uses it
        # - separates obs vs intervened samples
        # - handles hard intervention semantics (intervened node model)
        raise NotImplementedError

    def choose_action_ucb(
        self,
        A: np.ndarray,
        Astar: np.ndarray,
        theta: np.ndarray,
        gram_cache: dict,
        beta_t: float,
    ) -> Action:
        """
        Sequential decision along topo order: decide intervene or not per node.
        Mirrors Yan's loop over pi with UCB comparisons.
        """
        assert self.crl is not None
        pi = self.crl.topo

        chosen: set[int] = set()
        # TODO: port Yan's exact sequential logic:
        # for idx in pi:
        #    compare best predicted mean under do vs not-do, with UCB bonuses
        #    decide to add idx to chosen
        return frozenset(chosen)

    # ----------------------------
    # One step of the learner: decide next action given current history
    # ----------------------------
    def next_action(self, t: int, zeta_t: float, beta_t: float) -> Action:
        # In Yan, CRL is (re)fit frequently; to match, you likely refit every t after warm start
        if self.crl is None:
            self.fit_crl()

        under = self.under_explored_set(t)
        if len(under) > 0:
            return self.choose_action_under_sampling(under)

        A, Astar, theta, gram_cache = self.fit_sem_and_theta(zeta_t=zeta_t)
        return self.choose_action_ucb(A, Astar, theta, gram_cache, beta_t=beta_t)