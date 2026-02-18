import numpy as np
from itertools import combinations

def powerset_actions(n: int):
    """All subsets of {0,...,n-1} as sets."""
    items = list(range(n))
    for r in range(n + 1):
        for comb in combinations(items, r):
            yield set(comb)

def expected_latent_mean_under_action(sem, action=set(), kind="none"):
    """
    Exact E[Z] under SEM + intervention, matching LinearSEMGenerator.sample_latents:
      Z_i = eps_i + B_a[i,:i] Z_:i
    eps_i ~ Uniform(0, latent_noise_std)  => E[eps_i] = latent_noise_std/2
    """
    n = sem.n
    a = set(action)
    B_a = sem._B_under_action(a, kind)

    mu_eps = (sem.latent_noise_std / 2.0) * np.ones(n)

    mu = np.zeros(n)
    for i in range(n):
        mu[i] = mu_eps[i] + float(B_a[i, :i] @ mu[:i])
    return mu

def expected_utility_under_action(env, action=set(), kind="none"):
    """
    Exact E[U] for your ROCRLEnvironment:
      U = Z @ theta + reward_noise
    reward_noise ~ Uniform(0, noise_std) => mean = noise_std/2
    """
    muZ = expected_latent_mean_under_action(env.sem, action=action, kind=kind)
    base = float(muZ @ env.utility.theta)

    noise_mean = (env.utility.noise_std / 2.0) if env.utility.noise_std > 0 else 0.0
    return base + noise_mean

def fmt_action(a: set) -> str:
    return "∅" if len(a) == 0 else "{" + ",".join(map(str, sorted(a))) + "}"