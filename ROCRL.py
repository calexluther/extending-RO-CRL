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

    

    def compute_u_from_graph(self, dag_est: np.ndarray):

        return dag_est.sum(axis=1)
    
    
    def get_a0_UE(self, dag_est: np.ndarray):
        """
        Get under-explored set of actions.
        """
        u = compute_u_from_graph(dag_est)
        f_t = d ** (1/3) * n ** (2/3) * u ** (-2/3) * t ** (2/3)

    
    
    def learn(self):
        """
        Main learning loop. 
        Perform crl (LSCALE), under-sampling, and (potentially) parameter estimation and UCB selection
        """

        # CRL
        crl = l_scale_i(
            n=self.n,
            x_samples = self.X_all,
            actions_as_list = self.A_all,
            hard_intervention = self.intervention_type == "hard",
            gamma = self.gamma
        )
        enc_est_s, dag_est_s = crl[0]
        enc_est_h, dag_est_h = crl[1] if self.intervention_type == "hard" else None

        # Under-sampling

        # If all actions are well-sampled, estimate parameters and select action with UCB

