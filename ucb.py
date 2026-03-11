"""
UCB-based intervention selection for RO-CRL (Yan et al., NeurIPS 2025).

Implements the decision rule from Section 3.3 of the paper:
  - Eq. (31):  UCB_{a,t} = max_{{A,θ} ∈ C_{a,t}} <θ, Σ_ℓ A^ℓ ν_a>
  - Eq. (155): β_{i,t}  — confidence radius for SEM row parameters
  - Eq. (156): β_t      — confidence radius for utility parameter θ
  - Appendix B: sequential greedy maximisation for non-negative systems
  - §F.1.2:    unknown-mean reparameterisation (padded feature [1, Z[pa(i)]])
"""

from typing import Dict, FrozenSet, List, Optional, Tuple
import numpy as np

Action = FrozenSet[int]


# ---------------------------------------------------------------------------
# Candidate action generation
# ---------------------------------------------------------------------------

def all_actions(n: int) -> List[Action]:
    """Singleton interventions only: empty set + each single-node intervention."""
    return [frozenset()] + [frozenset({i}) for i in range(n)]


# ---------------------------------------------------------------------------
# Point-estimate of E[Z] under an action  (Corollary 1 / eq. 130)
# ---------------------------------------------------------------------------

def predict_latent_mean_under_action(
    topo: List[int],
    pa: List[List[int]],
    action: Action,
    A_hat: np.ndarray,
    Astar_hat: np.ndarray,
    nu_hat: np.ndarray,
    nu_star_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute E[Ẑ | a] = (I - A_a)^{-1} ν_a via forward substitution in
    causal order (Corollary 1, eq. 130).
    """
    n = A_hat.shape[0]
    mu = np.zeros(n, dtype=float)
    a = set(action)

    for i in topo:
        pa_i = pa[i]
        if i in a:
            mu[i] = nu_star_hat[i] + (
                float(Astar_hat[i, pa_i] @ mu[pa_i]) if len(pa_i) > 0 else 0.0
            )
        else:
            mu[i] = nu_hat[i] + (
                float(A_hat[i, pa_i] @ mu[pa_i]) if len(pa_i) > 0 else 0.0
            )
    return mu


# ---------------------------------------------------------------------------
# Confidence radii  (eqs. 155–156, Lemma 7)
# ---------------------------------------------------------------------------

def _beta_node(
    d_i: int, n: int, t: int,
    delta_t: float, zeta_t: float, m_tilde: float,
) -> float:
    """
    Confidence radius for SEM row-i parameters (eq. 155).

    β_{i,t}(δ_t) = 1 + √d_i + √(2 ln(n/δ_t) + d_i ln(1 + m̃²t / (d_i ζ_t²)))

    In the unknown-mean setting (§F.1.2) d_i includes the intercept dimension.
    """
    d_i = max(d_i, 1)
    zeta_sq = max(zeta_t ** 2, 1e-12)
    return (
        1.0
        + np.sqrt(d_i)
        + np.sqrt(
            2.0 * np.log(max(n / max(delta_t, 1e-30), 1.0))
            + d_i * np.log(1.0 + m_tilde ** 2 * t / (d_i * zeta_sq))
        )
    )


def _beta_theta(
    n: int, t: int,
    delta_t: float, zeta_t: float, m_tilde: float,
) -> float:
    """
    Confidence radius for utility parameter θ (eq. 156).

    β_t(δ_t) = 1 + √n + √(2 ln(1/δ_t) + n ln(1 + m̃²t / (n ζ_t²)))
    """
    zeta_sq = max(zeta_t ** 2, 1e-12)
    return (
        1.0
        + np.sqrt(n)
        + np.sqrt(
            2.0 * np.log(max(1.0 / max(delta_t, 1e-30), 1.0))
            + n * np.log(1.0 + m_tilde ** 2 * t / (n * zeta_sq))
        )
    )


# ---------------------------------------------------------------------------
# Per-action UCB score  (eq. 31 + Appendix B)
# ---------------------------------------------------------------------------

def _safe_inv(M: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def action_ucb_score(
    action: Action,
    topo: List[int],
    pa: List[List[int]],
    A_hat: np.ndarray,
    Astar_hat: np.ndarray,
    nu_hat: np.ndarray,
    nu_star_hat: np.ndarray,
    theta_hat: np.ndarray,
    VV: List[np.ndarray],
    VV_theta: np.ndarray,
    Ni: List[List[int]],
    N_theta: int,
    zeta_t: float,
    delta_t: float,
    m_tilde: float = 1.0,
) -> Tuple[float, np.ndarray, float, float]:
    """
    Compute UCB_{a,t} for a single action via the sequential greedy
    optimistic rule for non-negative systems (eq. 31, Appendix B).

    Stage 1 — Optimistic latent mean propagation in causal order:
      For each node *i* processed in topological order, the padded feature
      is feat_i = [1, μ̃[pa(i)]] (unknown-mean reparameterisation, §F.1.2).
      The optimistic latent mean at node *i* is

          μ̃[i] = b̂_a[i]·feat_i  +  β_{i,t} ‖feat_i‖_{M_{i,a}^{-1}}

      where M_{i,a} = V_{i,a} Ṽ_{i,a}^{-1} V_{i,a}  (= VV[i][r]).

    Stage 2 — Optimistic utility:
          UCB_a = θ̂·μ̃  +  β_t ‖μ̃‖_{M_θ^{-1}}

    Returns (total_score, mu_tilde, utility_mean, total_bonus).
    """
    n = len(pa)
    a = set(action)
    t = max(N_theta, 1)

    # ------------------------------------------------------------------
    # Stage 1: propagate optimistic means through the causal graph
    # ------------------------------------------------------------------
    mu_tilde = np.zeros(n, dtype=float)
    sem_bonus_total = 0.0

    for i in topo:
        r = 1 if i in a else 0
        pa_i = pa[i]

        # Padded feature (§F.1.2 unknown-mean reparameterisation)
        feat = (
            np.concatenate(([1.0], mu_tilde[pa_i]))
            if len(pa_i) > 0
            else np.array([1.0])
        )

        # Point estimate using current optimistic parent means
        if i in a:
            point_i = nu_star_hat[i] + (
                float(Astar_hat[i, pa_i] @ mu_tilde[pa_i])
                if len(pa_i) > 0 else 0.0
            )
        else:
            point_i = nu_hat[i] + (
                float(A_hat[i, pa_i] @ mu_tilde[pa_i])
                if len(pa_i) > 0 else 0.0
            )

        # Confidence radius β_{i,t} (eq. 155, +1 for intercept dimension)
        d_i = len(pa_i) + 1
        beta_i = _beta_node(d_i, n, t, delta_t, zeta_t, m_tilde)

        # Exploration bonus: β_{i,t} · ‖feat_i‖_{M_{i,a}^{-1}}  (dual norm)
        VV_inv_i = _safe_inv(VV[i][r])
        bonus_i = beta_i * float(np.sqrt(max(float(feat @ VV_inv_i @ feat), 0.0)))

        mu_tilde[i] = point_i + bonus_i
        sem_bonus_total += bonus_i

    # ------------------------------------------------------------------
    # Stage 2: optimistic utility  (eq. 31, θ part)
    # ------------------------------------------------------------------
    utility_mean = float(theta_hat @ mu_tilde)

    beta_th = _beta_theta(n, t, delta_t, zeta_t, m_tilde)
    VV_theta_inv = _safe_inv(VV_theta)
    theta_bonus = beta_th * float(
        np.sqrt(max(float(mu_tilde @ VV_theta_inv @ mu_tilde), 0.0))
    )

    total_score = utility_mean + theta_bonus
    total_bonus = sem_bonus_total + theta_bonus

    return total_score, mu_tilde, utility_mean, total_bonus


# ---------------------------------------------------------------------------
# Action-level UCB selection  (Algorithm 1, line 20–21)
# ---------------------------------------------------------------------------

def ucb_action_level(
    n: int,
    topo: List[int],
    pa: List[List[int]],
    A_hat: np.ndarray,
    Astar_hat: np.ndarray,
    nu_hat: np.ndarray,
    nu_star_hat: np.ndarray,
    theta_hat: np.ndarray,
    VV: List[np.ndarray],
    VV_theta: np.ndarray,
    Ni: List[List[int]],
    zeta_t: float,
    delta_t: float,
    candidate_actions: Optional[List[Action]] = None,
    m_tilde: float = 1.0,
) -> Tuple[Action, Dict[Action, Dict[str, object]], List[List[int]]]:
    """
    Select a_{t+1} = argmax_{a ∈ A} UCB_{a,t}  (Algorithm 1, line 21).

    Iterates over candidate actions, scores each with ``action_ucb_score``,
    and returns the maximiser together with per-action diagnostics.
    """
    if candidate_actions is None:
        candidate_actions = all_actions(n)

    # Total sample count (same for every node: Ni[i][0]+Ni[i][1])
    N_theta = Ni[0][0] + Ni[0][1]

    scores: Dict[Action, Dict[str, object]] = {}
    best_action: Optional[Action] = None
    best_score = -np.inf

    for a in candidate_actions:
        score, mu_tilde, mean_term, bonus_term = action_ucb_score(
            action=a,
            topo=topo,
            pa=pa,
            A_hat=A_hat,
            Astar_hat=Astar_hat,
            nu_hat=nu_hat,
            nu_star_hat=nu_star_hat,
            theta_hat=theta_hat,
            VV=VV,
            VV_theta=VV_theta,
            Ni=Ni,
            N_theta=N_theta,
            zeta_t=zeta_t,
            delta_t=delta_t,
            m_tilde=m_tilde,
        )

        scores[a] = {
            "score": score,
            "mu_hat": mu_tilde,
            "mean_term": mean_term,
            "bonus_term": bonus_term,
        }

        if score > best_score:
            best_score = score
            best_action = a

    # Update per-node regime counts for the chosen action
    best_action_set = set(best_action)
    for i in range(n):
        r = 1 if i in best_action_set else 0
        Ni[i][r] += 1

    return best_action, scores, Ni
