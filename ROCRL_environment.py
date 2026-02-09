#from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple, Dict, Any

import numpy as np

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class LinearSEMGenerator:
    """
    Latent: Z = BZ + eps with B strictly lower triangular
    Observed: X = GZ where G is full column rank
    """
    n: int = 5 # number of latent nodes
    d: int = 10 # dimension of observed data
    edge_prob: float = 0.4
    w_low: float = 0.25
    w_high: float = 1.0
    soft_scale: float = 0.1
    seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        assert self.d >= self.n, "Need d>=n for full column rank"
        self.rng = _rng(self.seed)

        # Strictly lower triangular mask for B with edge probability edge_prob
        mask = np.tril(self.rng.random(size = (self.n, self.n)) < self.edge_prob, k=-1)
        # Sample edge weights in B for existing edges
        B = np.zeros((self.n, self.n), dtype = float)
        B[mask] = self.rng.uniform(self.w_low, self.w_high, size = int(mask.sum()))
        self.B = B

        while True:
            G = self.rng.normal(size = (self.d, self.n))
            if np.linalg.matrix_rank(G) == self.n:
                break
        self.G = G

        self.B_star_soft = self.soft_scale * self.B
        self.B_star_hard = np.zeros_like(self.B)

    def _B_under_action(self, action: Set[int], kind: str) -> np.ndarray:
        if kind == "none" or len(action) == 0:
            return self.B
        B_a = self.B.copy()
        if kind == "soft":
            for i in action:
                B_a[i, :] = self.B_star_soft[i, :]
        else:
            for i in action:
                B_a[i, :] = self.B_star_hard[i, :]
        return B_a
    
    def sample_latents(
        self,
        num_samples: int,
        action: Optional[Iterable[int]] = None,
        kind: str = "none"
    ) -> np.ndarray:

        a: Set[int] = set(action) if action is not None else set()
        B_a = self._B_under_action(a, kind)
        eps = self.rng.normal(0.0, 1.0, size = (num_samples, self.n))

        # Solve (I - B_a)Z = eps for Z
        Z = np.zeros_like(eps)
        for t in range(num_samples):
            for i in range(self.n):
                Z[t, i] = eps[t, i] + np.dot(B_a[i, :i], Z[t, :i])
        return Z
    def latents_to_obs(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.G.T

    def sample(
        self, num_samples: int, 
        action: Optional[Iterable[int]] = None, 
        kind: str = "none",
        return_latents: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Z = self.sample_latents(num_samples, action, kind)
        X = self.latents_to_obs(Z)
        return (X, Z) if return_latents else (X, None)


    def plot_graph(self, B: Optional[np.ndarray] = None, ax=None) -> None:
        """Visualize the causal DAG: B[i,j] != 0 means edge j -> i."""
        B = B if B is not None else self.B
        show_plot = ax is None
        ax = ax or plt.gca()
        G = nx.DiGraph()
        G.add_nodes_from(range(B.shape[0]))
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if B[i, j] != 0:
                    G.add_edge(j, i, weight=B[i, j])
        pos = nx.spring_layout(G, seed=42)
        ax = ax or plt.gca()
        nx.draw(G, pos, with_labels=True, ax=ax, node_color="lightblue",
                node_size=800, font_size=12, arrows=True, arrowsize=20)
        ax.set_title("Causal DAG (B[i,j] = edge j → i)")
        plt.tight_layout()
        if show_plot:
            plt.show()


@dataclass
class LinearUtility:

    n: int
    theta: Optional[np.ndarray] = None
    noise_std: float = 0.0
    theta_dist: str = "rademacher"
    theta_scale: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = _rng(self.seed)
        if self.theta is None:
            self.theta = self.sample_theta()
        else:
            self.theta = np.asarray(self.theta, dtype = float)
            assert self.theta.shape == (self.n,), "theta must be a vector of length n"
    
    def sample_theta(self) -> np.ndarray:
        if self.theta_dist == "rademacher":
            th = self.rng.choice([-1, 1], size = self.n)
        elif self.theta_dist == "uniform":
            th = self.rng.uniform( size = self.n)
        elif self.theta_dist == "normal":
            th = self.rng.normal(size = self.n)
        else:
            raise ValueError(f"Invalid theta distribution: {self.theta_dist}")
        return self.theta_scale * th

    def set_new_task(self, theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
            self.theta = self.sample_theta()
        else:
            theta = np.asarray(theta, dtype = float)
            assert theta.shape == (self.n,), "theta must be a vector of length n"
            self.theta = theta
        return self.theta

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        #print("Z shape:", Z.shape, "theta shape:", self.theta.shape)
        U = Z @ self.theta
        if self.noise_std > 0:
            U += self.rng.normal(0, self.noise_std, size = U.shape)
        return U

@dataclass
class ROCRLEnvironment:

    sem: LinearSEMGenerator
    utility: LinearUtility
    
    def step(
        self, num_samples: int = 1,
        action: Optional[Iterable[int]] = None,
        kind: str = "none",
        return_latents: bool = False
    ) -> Dict[str, Any]:
        X, Z = self.sem.sample(
            num_samples = num_samples,
            action = action,
            kind = kind,
            return_latents = True  # Always need Z for utility
        )
        U = self.utility(Z)
        out: Dict[str, Any] = {"X": X, "U": U}
        if return_latents:
            out["Z"] = Z
        return out
    
    def new_task(self, theta: Optional[np.ndarray] = None) -> np.ndarray:
        return self.utility.set_new_task(theta = theta)



if __name__ == "__main__":
    n_latents = 5
    sem = LinearSEMGenerator(n = n_latents, d = 10, seed = 0)
    util = LinearUtility(n = n_latents, noise_std = 0.1, theta_dist = "rademacher", seed = 1)
    env = ROCRLEnvironment(sem = sem, utility = util)

    batch = env.step(num_samples = 4, kind = "none", return_latents = True)
    print("X shape:", batch["X"].shape, "U.shape:", batch["U"].shape, "Z.shape:", batch["Z"].shape)

    batch2 = env.step(num_samples = 4, action = {1,3}, kind = "soft")
    print("mean utility observed vs intervened on: ", batch["U"].mean(), batch2["U"].mean())


