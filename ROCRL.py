from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple, Dict
from LSCALE import LSCALE_i
from ucb import *
from metrics import expected_utility_under_action
import numpy as np

Action = FrozenSet[int]
MCA0Action = FrozenSet[int]   # frozenset() or frozenset({i})


@dataclass
class CRLEstimate:
    enc_est: np.ndarray
    dag_adj: np.ndarray
    pa: List[List[int]]
    topo: List[int]

@dataclass
class DecisionRecord:
    round_idx: int
    total_samples: int
    action: Action
    mode: str                    # "under_sampled" or "ucb"
    under_sampled_actions: List[Action]
    chosen_action_index: Optional[int] = None


class ROCRLLearner:
    def __init__(
        self,
        n: int,
        d: int,
        intervention_type: str,          # "soft" or "hard"
        gamma: float,
        delta: float = 0.05,
        epsilon_max: float = 0.25,
        C_const: float = 1.0,
        lam: float = 1e-3,
        seed: int = 0,
        dim_reduction: bool = True,
        hard_unmixing: bool = False,
        T0: int = 3000,
        print_debug: bool = False,
    ):
        assert intervention_type in ("soft", "hard")
        self.n, self.d = n, d
        self.intervention_type = intervention_type
        self.gamma = gamma
        self.lam = lam
        self.dim_reduction = dim_reduction
        self.hard_unmixing = hard_unmixing
        self.T0 = T0
        self.delta_t = 0.1
        self.delta = delta
        self.epsilon_max = epsilon_max
        self.C_const = C_const  
        self.rng = np.random.default_rng(seed)
        self.print_debug = print_debug
        self.Ghat: Optional[np.ndarray] = None
        self.H_t: Optional[np.ndarray] = None
        self.Zhat_all: List[np.ndarray] = []

        # MCA0 = observational + singleton actions
        self.mca0: List[MCA0Action] = [frozenset()] + [frozenset({i}) for i in range(n)]

        # forced-exploration schedule, matching the repo:
        # [] repeated T0 times, then {i} repeated T0 times for each i
        self.forced_actions: List[MCA0Action] = (
            [frozenset() for _ in range(T0)]
            + [frozenset({i}) for i in range(n) for _ in range(T0)]
        )

        self.order: List[int] = []

        # All samples/actions (repo: x_samples_all / z_samples_all / int_lists_all)
        self.X_all: List[np.ndarray] = []
        self.Z_all: List[np.ndarray] = []
        self.A_all: List[Action] = []

        # CRL pool only: observational + singleton actions
        # (repo: x_samples / int_lists)
        self.X_crl: List[np.ndarray] = []
        self.Z_crl: List[np.ndarray] = []
        self.A_crl: List[MCA0Action] = []

        self.crl: Optional[CRLEstimate] = None


        self.A_hat: Optional[np.ndarray] = None
        self.Astar_hat: Optional[np.ndarray] = None
        self.nu_hat: Optional[np.ndarray] = None
        self.nu_star_hat: Optional[np.ndarray] = None

        self.A_hat_history: List[np.ndarray] = []
        self.Astar_hat_history: List[np.ndarray] = []
        self.nu_hat_history: List[np.ndarray] = []
        self.nu_star_hat_history: List[np.ndarray] = []
        self.round_history: List[int] = []

        self.decision_history: List[DecisionRecord] = []
        self.last_decision_mode: Optional[str] = None
        self.last_under_sampled_actions: List[Action] = []

        self.instant_regret: List[float] = []
        self.cum_regret: List[float] = []
        self.expected_reward_history: List[float] = []
        self.benchmark_reward: Optional[float] = None
        self.benchmark_action: Optional[Action] = None

        self.U_all: List[float] = []

        self.theta_hat: Optional[np.ndarray] = None
        self.theta_hat_history: List[np.ndarray] = []
        self.V_theta: Optional[np.ndarray] = None
        self.tilde_V_theta: Optional[np.ndarray] = None
        self.VV_theta: Optional[np.ndarray] = None
        self.threshold_history: List[Tuple[int, float]] = []
        self.f_t_val: Optional[float] = None

        # Repo-style UCB counts
        # N_mca0 conceptually counts samples for each action in mca0.
        # Use a dict keyed by frozenset action instead of the repo's brittle list.
        self.N_mca0: Dict[MCA0Action, int] = {a: 0 for a in self.mca0}

        # Ni[i] = [count for observational regime of node i, count for intervened regime of node i]
        self.Ni: List[List[int]] = [[n * T0, T0] for _ in range(n)]

        self._cache = {}

    def get_causal_order(self, dag_est: np.ndarray) -> List[int]:
        M = np.asarray(dag_est, dtype=int)
        n = M.shape[0]
        indegree = M.sum(axis = 0).tolist()
        q = [i for i in range(n) if indegree[i] == 0]
        out = []
        while q:
            v = q.pop()
            out.append(v)
            for w in np.where(M[v] != 0)[0]:
                indegree[w] -= 1
                if indegree[w] == 0:
                    q.append(int(w))
        if len(out) != n:
            raise ValueError("Graph is not acyclic")
        return out
    
    def compute_u_from_graph(self, dag_est: np.ndarray, pa: List[List[int]]) -> np.ndarray:
        u_vec = np.zeros(self.n, dtype=float)
        self.order = self.get_causal_order(dag_est)

        for i in self.order:
            if len(pa[i]) == 0:
                u_vec[i] = 1.0
            else:
                u_vec[i] = float(np.sum(u_vec[pa[i]]) + np.sqrt(len(pa[i])))

        return float(np.sum(u_vec) + np.sqrt(self.n))

    def get_delta_t(self, t: int, delta: float) -> float:
        return 6.0 * delta / (np.pi**2 * (t**2))
    
    def N_eps(self, epsilon_max: float, delta_t: float, C_const: float = 1.0) -> float:
        return (C_const**2) * (epsilon_max**-2) * (self.d + np.log(1.0 / delta_t))

    def f_t(self, u: float, t: int, delta_t: float, epsilon_max: float, C_const: float = 1.0) -> float:
        leading = (self.d ** (1/3)) * (self.n ** (-2/3)) * (u ** (2/3)) * (t ** (2/3))
        floor = self.N_eps(epsilon_max=epsilon_max, delta_t=delta_t, C_const=C_const)
        if self.print_debug:
            if (t + 1) % 500 == 0:
                print(f"Leading: {leading}, Floor: {floor}, t: {t + 1}")
        return float(min(max(leading, floor), 1000))

    def current_mca0_counts(self) -> Dict[MCA0Action, int]:
        counts = {a: 0 for a in self.mca0}
        for a in self.A_all:
            if a in counts:
                counts[a] += 1
        return counts


    def get_a0_UE(
        self,
        dag_est: np.ndarray,
        pa: List[List[int]],
        delta: float,
        epsilon_max: float,
        C_const: float = 1.0,
    ) -> List[MCA0Action]:
        t = max(len(self.X_all), 1)
        self.delta_t = self.get_delta_t(t, delta)
        u = self.compute_u_from_graph(dag_est, pa)
        self.f_t_val = self.f_t(u=u, t=t, delta_t=self.delta_t, epsilon_max=epsilon_max, C_const=C_const)
        self.threshold_history.append((t, self.f_t_val))
        counts = self.current_mca0_counts()
        self.N_mca0 = counts
        return [a for a in self.mca0 if counts[a] < self.f_t_val]

    
    
    def initialize_weight_matrices(
        self, pa: List[List[int]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Match the repo initialization:
            V[i]       = [I_{p_i+1}, I_{p_i+1}]
            tilde_V[i] = [I_{p_i+1}, I_{p_i+1}]
            hat_b[i]   = zeros((2, p_i+1, 1))
            g__[i]     = zeros((2, p_i+1, 1))
        """
        V: List[np.ndarray] = []
        tilde_V: List[np.ndarray] = []
        hat_b: List[np.ndarray] = []
        g__: List[np.ndarray] = []

        for i in range(self.n):
            p = len(pa[i]) + 1
            V.append(np.array([np.eye(p), np.eye(p)], dtype=float))
            tilde_V.append(np.array([np.eye(p), np.eye(p)], dtype=float))
            hat_b.append(np.zeros((2, p, 1), dtype=float))
            g__.append(np.zeros((2, p, 1), dtype=float))

        return V, tilde_V, hat_b, g__



    def build_weight_matrices(
        self,
        hat_Z: np.ndarray,
        V: List[np.ndarray],
        tilde_V: List[np.ndarray],
        hat_b: List[np.ndarray],
        g__: List[np.ndarray],
        zeta_t: float,
        pa: List[List[int]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Faithful modularization of the repo's weighted-regression block.
        """
        t = hat_Z.shape[0]

        for idx in range(t):
            a_idx = self.A_all[idx]
            for i in range(self.n):
                feat = np.concatenate(([1.0], hat_Z[idx, pa[i]])).reshape(-1, 1)
                temp = float(hat_Z[idx, i]) * feat

                regime = 1 if i in a_idx else 0
                inv_tilde = np.linalg.inv(tilde_V[i][regime])
                norm_term = float(np.sqrt(feat.T @ inv_tilde @ feat))
                weight = (1.0 / zeta_t) * min(1.0, 1.0 / norm_term)

                V[i][regime] += weight * (feat @ feat.T)
                tilde_V[i][regime] += (weight ** 2) * (feat @ feat.T)
                g__[i][regime] += weight * temp

        VV: List[np.ndarray] = []
        for i in range(self.n):
            hat_b[i][0] = np.linalg.solve(V[i][0], g__[i][0])
            hat_b[i][1] = np.linalg.solve(V[i][1], g__[i][1])
            VV.append(
                np.array(
                    [
                        V[i][0] @ np.linalg.inv(tilde_V[i][0]) @ V[i][0],
                        V[i][1] @ np.linalg.inv(tilde_V[i][1]) @ V[i][1],
                    ],
                    dtype=float,
                )
            )

        return V, tilde_V, hat_b, VV

    def estimate_theta(
        self,
        hat_Z: np.ndarray,      # shape (t, n)
        U: np.ndarray,          # shape (t,)
        zeta_t: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate theta_t according to the paper's weighted ridge rule.

        Returns
        -------
        theta_hat : (n,)
        V_theta   : (n, n)
        tilde_V_theta : (n, n)
        VV_theta  : (n, n)
        """
        t, n = hat_Z.shape
        if n != self.n:
            raise ValueError(f"hat_Z should have {self.n} columns, got {n}")
        if U.shape[0] != t:
            raise ValueError(f"U should have length {t}, got {U.shape[0]}")

        V_theta = np.eye(self.n, dtype=float)
        tilde_V_theta = np.eye(self.n, dtype=float)
        g_theta = np.zeros((self.n, 1), dtype=float)

        for s in range(t):
            z_s = hat_Z[s].reshape(-1, 1)   # (n, 1)

            inv_tilde = np.linalg.inv(tilde_V_theta)
            norm_term = float(np.sqrt(z_s.T @ inv_tilde @ z_s))
            weight = (1.0 / zeta_t) * min(1.0, 1.0 / norm_term)

            V_theta += weight * (z_s @ z_s.T)
            tilde_V_theta += (weight ** 2) * (z_s @ z_s.T)
            g_theta += weight * float(U[s]) * z_s

        #print("V_theta shape:", V_theta.shape, "g_theta shape:", g_theta.shape)
        #print("V_theta:", V_theta)
        #print("g_theta:", g_theta)
        theta_hat = np.linalg.solve(V_theta, g_theta).reshape(-1)
        VV_theta = V_theta @ np.linalg.inv(tilde_V_theta) @ V_theta

        return theta_hat, V_theta, tilde_V_theta, VV_theta

    def learn(self):
        """
        Main learning loop. 
        Perform crl (LSCALE), under-sampling, and (potentially) parameter estimation and UCB selection
        """
        if len(self.X_crl) == 0:
            raise ValueError("X_crl is empty; CRL needs observational/singleton data.")
        if len(self.X_all) == 0:
            raise ValueError("X_all is empty; learner needs some collected samples.")

        X_crl = np.asarray(self.X_crl, dtype=float)
        X_all = np.asarray(self.X_all, dtype=float)

        crl = LSCALE_i(
            n=self.n,
            x_samples=X_crl,
            actions_as_list=self.A_crl,
            hard_intervention=(self.intervention_type == "hard"),
            gamma=self.gamma,
        )

        enc_est_s, dag_est_s = crl[0]
        dag_est_s = dag_est_s.astype(int)

        if self.intervention_type == "hard":
            enc_est_h, dag_est_h = crl[1]
        else:
            enc_est_h, dag_est_h = None, None

        dag_est = dag_est_s
        enc_est = enc_est_h if self.intervention_type == "hard" else enc_est_s

        pa = [list(np.nonzero(dag_est[:, i])[0]) for i in range(self.n)]
        topo = self.get_causal_order(dag_est)

        self.crl = CRLEstimate(
            enc_est=enc_est,
            dag_adj=dag_est,
            pa=pa,
            topo=topo,
        )
        self.order = topo

        a0_ue = self.get_a0_UE(dag_est, pa, delta = self.delta, epsilon_max = self.epsilon_max, C_const = self.C_const)
        self.last_under_sampled_actions = list(a0_ue)

        if len(a0_ue) > 0:
            a = a0_ue[int(self.rng.integers(len(a0_ue)))]
            self.last_decision_mode = "under_sampled"

            self.decision_history.append(
                DecisionRecord(
                    round_idx=len(self.decision_history),
                    total_samples=len(self.X_all),
                    action=a,
                    mode="under_sampled",
                    under_sampled_actions=list(a0_ue),
                    chosen_action_index = None,
                )
            )
            return a

        u = self.compute_u_from_graph(dag_est, pa)
        f_t_val = self.f_t(u, t = len(self.X_all), delta_t = self.delta_t, epsilon_max = self.epsilon_max, C_const = self.C_const)

        hat_Z = X_all @ enc_est.T

        V, tilde_V, hat_b, g__ = self.initialize_weight_matrices(pa)
        zeta_t = 0.1 * len(self.X_all) * np.sqrt((self.d + np.log(1.0 / self.delta_t)) / f_t_val)


        U_all = np.asarray(self.U_all, dtype=float)

        theta_hat, V_theta, tilde_V_theta, VV_theta = self.estimate_theta(
            hat_Z=hat_Z,
            U=U_all,
            zeta_t=zeta_t,
        )

        self.theta_hat = theta_hat
        self.V_theta = V_theta
        self.tilde_V_theta = tilde_V_theta
        self.VV_theta = VV_theta
        self.theta_hat_history.append(theta_hat.copy())

        V, tilde_V, hat_b, VV = self.build_weight_matrices(
            hat_Z=hat_Z,
            V=V,
            tilde_V=tilde_V,
            hat_b=hat_b,
            g__=g__,
            zeta_t=zeta_t,
            pa=pa,
        )

        # optional: reconstruct A_hat here if you're monitoring it
        A_hat, Astar_hat, nu_hat, nu_star_hat = self.reconstruct_A_from_hat_b(hat_b, pa)
        self.A_hat = A_hat
        self.Astar_hat = Astar_hat
        self.nu_hat = nu_hat
        self.nu_star_hat = nu_star_hat
        self.A_hat_history.append(A_hat.copy())
        self.Astar_hat_history.append(Astar_hat.copy())
        self.nu_hat_history.append(nu_hat.copy())
        self.nu_star_hat_history.append(nu_star_hat.copy())
        self.round_history.append(len(self.X_all))

        a, ucb_debug, Ni = ucb_action_level(
            n=self.n,
            topo=self.order,
            pa=pa,
            A_hat=self.A_hat,
            Astar_hat=self.Astar_hat,
            nu_hat=self.nu_hat,
            nu_star_hat=self.nu_star_hat,
            theta_hat=self.theta_hat,
            VV=VV,
            VV_theta=self.VV_theta,
            Ni=self.Ni,
            zeta_t=zeta_t,
            delta_t=self.delta_t,
        )
        self.Ni = Ni
        self._cache["last_ucb_debug"] = ucb_debug
        self.last_decision_mode = "ucb"
        self.decision_history.append(
            DecisionRecord(
                round_idx=len(self.decision_history),
                total_samples=len(self.X_all),
                action=a,
                mode="ucb",
                under_sampled_actions=[],
                chosen_action_index=None,
            )
        )
        return a


    def apply_action_and_update_pools(
        self,
        a: Action,
        z_sample: np.ndarray,
        mixing_matrix: np.ndarray,
        utility_obs: Optional[float] = None,
    ) -> None:

        z_sample = np.asarray(z_sample, dtype=float).reshape(-1)
        if z_sample.shape[0] != self.n:
            raise ValueError(f"z_sample must have shape ({self.n},), got {z_sample.shape}")

        # ------------------------------------------------------------
        # 1) Update the "all data" pool
        # ------------------------------------------------------------
        self.A_all.append(frozenset(a))
        self.Z_all.append(z_sample.copy())

        x_sample = z_sample @ mixing_matrix.T   # shape (d,)
        self.X_all.append(np.asarray(x_sample, dtype=float).reshape(-1))

        if utility_obs is not None:
            self.U_all.append(float(utility_obs))
        else:
            self.U_all.append(np.nan)

        # ------------------------------------------------------------
        # 2) Update under-sampling counts on MCA0 = {emptyset, singleton}
        # ------------------------------------------------------------
        if a in self.N_mca0:
            self.N_mca0[a] += 1

        # ------------------------------------------------------------
        # 3) Update the CRL pool (all actions are singletons)
        # ------------------------------------------------------------
        a_mca0 = frozenset(a)
        self.A_crl.append(a_mca0)
        self.Z_crl.append(z_sample.copy())
        self.X_crl.append(np.asarray(x_sample, dtype=float).reshape(-1))
                

    def seed_initial_pools(
        self,
        env,
        mixing_matrix: np.ndarray,
        kind: str = "none",
        benchmark_action: Optional[set[int]] = None,
    ) -> None:
        self.A_all.clear()
        self.Z_all.clear()
        self.X_all.clear()
        self.A_crl.clear()
        self.Z_crl.clear()
        self.X_crl.clear()

        self.N_mca0 = {a: 0 for a in self.mca0}

        self.instant_regret.clear()
        self.cum_regret.clear()
        self.expected_reward_history.clear()

        self.initialize_regret_benchmark(env, benchmark_action=benchmark_action, kind=kind)


        for a in self.forced_actions:
            kind = "none" if len(a) == 0 else self.intervention_type
            z_sample = np.asarray(env.sem.sample_latents(1, a, kind=kind), dtype=float).reshape(-1)
            
            utility_obs = float(z_sample @ env.utility.theta)
            if env.utility.noise_std > 0:
                utility_obs += float(np.random.uniform(0.0, env.utility.noise_std))
            
            self.apply_action_and_update_pools(a, z_sample, mixing_matrix, utility_obs)
            self.record_regret(env, a, kind)

    def reconstruct_A_from_hat_b(
        self,
        hat_b: List[np.ndarray],
        pa: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct observational/intervened SEM matrices and intercept vectors.

        Returns
        -------
        A_hat : (n, n)
            Observational SEM coefficient matrix.
        Astar_hat : (n, n)
            Intervened SEM coefficient matrix.
        nu_hat : (n,)
            Observational intercepts.
        nu_star_hat : (n,)
            Intervened intercepts.
        """
        A_hat = np.zeros((self.n, self.n), dtype=float)
        Astar_hat = np.zeros((self.n, self.n), dtype=float)
        nu_hat = np.zeros(self.n, dtype=float)
        nu_star_hat = np.zeros(self.n, dtype=float)

        for i in range(self.n):
            # regime 0 = observational / not intervened on i
            b0 = hat_b[i][0].reshape(-1)
            nu_hat[i] = b0[0]
            if len(pa[i]) > 0:
                A_hat[i, pa[i]] = b0[1:]

            # regime 1 = intervened on i
            b1 = hat_b[i][1].reshape(-1)
            nu_star_hat[i] = b1[0]
            if len(pa[i]) > 0:
                Astar_hat[i, pa[i]] = b1[1:]

        return A_hat, Astar_hat, nu_hat, nu_star_hat

    def initialize_regret_benchmark(
        self,
        env,
        benchmark_action: Optional[set[int]] = None,
        kind: str = "none",
    ) -> None:
        if benchmark_action is None:
            benchmark_action = set()

        self.benchmark_action = frozenset(benchmark_action)
        self.benchmark_reward = expected_utility_under_action(
            env,
            action=set(self.benchmark_action),
            kind=kind,
        )


    def record_regret(
        self,
        env,
        action: Action,
        kind: str = "none",
    ) -> float:
        if self.benchmark_reward is None:
            raise ValueError("Benchmark reward not initialized. Call initialize_regret_benchmark first.")

        reward_exp = expected_utility_under_action(env, action=set(action), kind=kind)
        regret_t = float(self.benchmark_reward - reward_exp)

        self.expected_reward_history.append(reward_exp)
        self.instant_regret.append(regret_t)

        if len(self.cum_regret) == 0:
            self.cum_regret.append(regret_t)
        else:
            self.cum_regret.append(self.cum_regret[-1] + regret_t)

        return regret_t