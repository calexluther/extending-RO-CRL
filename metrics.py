import numpy as np

def powerset_actions(n: int):
    """Singleton interventions only: empty set + each single-node intervention."""
    yield set()
    for i in range(n):
        yield {i}

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


import math

def delta_sched(t: int, delta: float) -> float:
    return 6.0 * delta / (math.pi**2 * t**2)

def N_eps(epsilon_max: float, delta_t: float, d: int, C_const: float = 1.0) -> float:
    return (C_const**2) * (epsilon_max**-2) * (d + math.log(1.0 / delta_t))

def choose_T0_from_paper(
    n: int,
    d: int,
    epsilon_max: float,
    delta: float = 0.1,
    C_const: float = 1.0,
    init_T0: int = 50,
    max_iter: int = 100,
) -> int:
    """
    Solve T0 ≈ N(epsilon_max, delta_{n T0}) by fixed-point iteration.
    Returns an integer T0 rounded up.
    """
    T0 = max(1, init_T0)
    for _ in range(max_iter):
        dt = delta_sched(n * T0, delta)
        T0_new = math.ceil(N_eps(epsilon_max, dt, d=d, C_const=C_const))
        if T0_new == T0:
            return T0
        T0 = T0_new
    return T0