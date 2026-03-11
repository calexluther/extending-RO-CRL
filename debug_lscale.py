"""
Debug script for the LSCALE component within ROCRLLearner.

Tests:
  1. Whether seed_initial_pools generates actual interventional data
     (BUG: kind= is never passed to sample_latents, so all data is observational)
  2. Whether LSCALE's pseudo-inverse computation is correct
  3. Whether LSCALE recovers encoder, graph, and latents from well-formed data
  4. Whether ROCRL's hat_Z = X @ enc_est.T aligns with true Z
  5. Whether SEM parameter estimates (A_hat, nu_hat, theta_hat) are reasonable
"""

import numpy as np
from ROCRL_environment import LinearSEMGenerator, LinearUtility, ROCRLEnvironment
from LSCALE import LSCALE_i
from metrics import expected_utility_under_action, expected_latent_mean_under_action

np.set_printoptions(precision=4, suppress=True, linewidth=120)

SEED = 3
N = 5
D = 10
INTERVENTION_TYPE = "hard"
T0 = 3000
GAMMA = 0.1

sem = LinearSEMGenerator(n=N, d=D, seed=SEED, latent_noise_std=1.0)
util = LinearUtility(n=N, noise_std=0.0, theta_dist="rademacher", seed=1)
env = ROCRLEnvironment(sem=sem, utility=util)

B_true = sem.B
G_true = sem.G
theta_true = util.theta

print("=" * 80)
print("GROUND TRUTH")
print("=" * 80)
print(f"B (edge weights, B[i,j]!=0 means j->i):\n{B_true}")
print(f"\nTrue DAG adjacency (B[i,j]!=0):\n{(np.abs(B_true) > 1e-10).astype(int)}")
print(f"\nG (mixing matrix) shape: {G_true.shape}")
print(f"theta: {theta_true}")
print(f"B_star_hard:\n{sem.B_star_hard}")
print(f"B_star_soft:\n{sem.B_star_soft}")


# ============================================================================
# TEST 1: seed_initial_pools generates only observational data
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Does seed_initial_pools actually intervene?")
print("=" * 80)

kind_of = lambda a: ("none" if len(a) == 0 else INTERVENTION_TYPE)

mu_obs = expected_latent_mean_under_action(sem, action=set(), kind="none")
mu_int = {i: expected_latent_mean_under_action(sem, action={i}, kind=INTERVENTION_TYPE) for i in range(N)}

print(f"\nE[Z] observational:          {mu_obs}")
for i in range(N):
    diff = mu_int[i] - mu_obs
    print(f"E[Z] under do({i}) [{INTERVENTION_TYPE}]: {mu_int[i]}  (diff from obs: {diff})")

# Simulate what seed_initial_pools does (NO kind passed)
rng_test = np.random.default_rng(42)
n_test = 5000
z_obs_bug = []
z_int_bug = {i: [] for i in range(N)}

for _ in range(n_test):
    z_obs_bug.append(sem.sample_latents(1, frozenset()).reshape(-1))  # observational
for i in range(N):
    for _ in range(n_test):
        z_int_bug[i].append(sem.sample_latents(1, frozenset({i})).reshape(-1))  # BUG: kind="none" default

z_obs_bug = np.array(z_obs_bug)
z_int_bug = {i: np.array(v) for i, v in z_int_bug.items()}

print(f"\n--- With seed_initial_pools behavior (kind='none' always) ---")
print(f"Sample mean Z (obs):          {z_obs_bug.mean(axis=0)}")
for i in range(N):
    diff = z_int_bug[i].mean(axis=0) - z_obs_bug.mean(axis=0)
    print(f"Sample mean Z (do({i})):       {z_int_bug[i].mean(axis=0)}  (diff: {diff})")

# Now do it correctly (with kind=intervention_type)
z_obs_correct = []
z_int_correct = {i: [] for i in range(N)}

sem_correct = LinearSEMGenerator(n=N, d=D, seed=SEED, latent_noise_std=1.0)
for _ in range(n_test):
    z_obs_correct.append(sem_correct.sample_latents(1, frozenset(), kind="none").reshape(-1))
for i in range(N):
    for _ in range(n_test):
        z_int_correct[i].append(sem_correct.sample_latents(1, frozenset({i}), kind=INTERVENTION_TYPE).reshape(-1))

z_obs_correct = np.array(z_obs_correct)
z_int_correct = {i: np.array(v) for i, v in z_int_correct.items()}

print(f"\n--- With CORRECT behavior (kind='{INTERVENTION_TYPE}') ---")
print(f"Sample mean Z (obs):          {z_obs_correct.mean(axis=0)}")
for i in range(N):
    diff = z_int_correct[i].mean(axis=0) - z_obs_correct.mean(axis=0)
    print(f"Sample mean Z (do({i})):       {z_int_correct[i].mean(axis=0)}  (diff: {diff})")

# Compare covariance differences (what LSCALE uses)
x_obs_bug = z_obs_bug @ G_true.T
x_int_bug_x = {i: z_int_bug[i] @ G_true.T for i in range(N)}

x_obs_correct = z_obs_correct @ G_true.T
x_int_correct_x = {i: z_int_correct[i] @ G_true.T for i in range(N)}

cov_obs_bug = np.cov(x_obs_bug, rowvar=False)
cov_obs_correct = np.cov(x_obs_correct, rowvar=False)

print(f"\n--- Precision difference norms (Frobenius) ---")
prec_obs_bug = np.linalg.pinv(cov_obs_bug)
prec_obs_correct = np.linalg.pinv(cov_obs_correct)

for i in range(N):
    prec_i_bug = np.linalg.pinv(np.cov(x_int_bug_x[i], rowvar=False))
    R_bug = prec_i_bug - prec_obs_bug

    prec_i_correct = np.linalg.pinv(np.cov(x_int_correct_x[i], rowvar=False))
    R_correct = prec_i_correct - prec_obs_correct

    print(f"  Node {i}: ||R_bug||_F = {np.linalg.norm(R_bug, 'fro'):.4f},  "
          f"||R_correct||_F = {np.linalg.norm(R_correct, 'fro'):.4f}")

print("\n>>> VERDICT: If 'bug' norms are near-zero while 'correct' norms are large,")
print("    seed_initial_pools is feeding LSCALE purely observational data.")


# ============================================================================
# TEST 2: LSCALE pseudo-inverse computation
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Is LSCALE's pseudo-inverse (enc_est_s_pt) correct?")
print("=" * 80)

# Generate proper data for LSCALE
sem2 = LinearSEMGenerator(n=N, d=D, seed=SEED, latent_noise_std=1.0)
mca0 = [frozenset()] + [frozenset({i}) for i in range(N)]
x_samples = []
actions_list = []
for a in mca0:
    kind = "none" if len(a) == 0 else INTERVENTION_TYPE
    for _ in range(T0):
        Z = sem2.sample_latents(1, a, kind=kind)
        X = Z @ sem2.G.T
        x_samples.append(X.reshape(-1))
        actions_list.append(a)

x_samples = np.array(x_samples)

# Reproduce the internal steps of LSCALE_i
n = N
d = D
dim_reduction = True

if dim_reduction:
    U_svd, S_svd, Vh_svd = np.linalg.svd(x_samples[:n + d], full_matrices=False)
    dec_colbt = Vh_svd[:n]
    x_proj = x_samples @ dec_colbt.T
else:
    x_proj = x_samples

actions_fs = list(map(frozenset, actions_list))
x_by_mca0 = {a: np.stack([x_proj[i] for i in range(len(actions_fs)) if actions_fs[i] == a]) for a in mca0}
x_covs = np.stack([np.cov(x_by_mca0[a], rowvar=False) for a in mca0])
x_precs = np.stack([np.linalg.pinv(x_covs[i]) for i in range(n + 1)])
rxs = x_precs[1:] - x_precs[0]

# _get_encoder
enc_est = np.zeros((n, x_proj.shape[1]))
for i in range(n):
    _, _, Vh_i = np.linalg.svd(rxs[i], full_matrices=False)
    enc_est[i] = Vh_i[0]

# normalize
zhat_covs = enc_est @ x_covs @ enc_est.T
enc_est /= zhat_covs[0].diagonal()[:, None] ** 0.5
zhat_covs = enc_est @ x_covs @ enc_est.T

# THE BUG: LSCALE's pseudo-inverse computation
U_enc, S_enc, Vh_enc = np.linalg.svd(enc_est, full_matrices=False)
lscale_pinv = U_enc @ np.diagflat(1 / S_enc) @ Vh_enc       # what LSCALE does
correct_pinv = Vh_enc.T @ np.diagflat(1 / S_enc) @ U_enc.T  # correct formula
numpy_pinv = np.linalg.pinv(enc_est)                          # numpy reference

print(f"enc_est shape: {enc_est.shape}")
print(f"\n||LSCALE_pinv - numpy_pinv||_F  = {np.linalg.norm(lscale_pinv - numpy_pinv, 'fro'):.6f}")
print(f"||correct_pinv - numpy_pinv||_F = {np.linalg.norm(correct_pinv - numpy_pinv, 'fro'):.6f}")

# Check identity: pinv @ enc_est should be ~I
lscale_check = lscale_pinv @ enc_est.T  # LSCALE uses enc_est_s_pt.T in rxs transform
correct_check = correct_pinv @ enc_est.T

print(f"\n||LSCALE_pinv @ enc_est.T - I||_F  = {np.linalg.norm(lscale_check - np.eye(n), 'fro'):.6f}")
print(f"||correct_pinv @ enc_est.T - I||_F = {np.linalg.norm(correct_check - np.eye(n), 'fro'):.6f}")

# Check what matters: graph estimation from rzs
rzs_lscale = np.stack([lscale_pinv @ rxs[i] @ lscale_pinv.T for i in range(n)])
rzs_correct = np.stack([correct_pinv @ rxs[i] @ correct_pinv.T for i in range(n)])

print(f"\nGraph estimation edge-weight norms (per intervened node i, target j):")
print(f"{'i->j':>6}  {'LSCALE':>10}  {'correct':>10}  {'diff':>10}")
for i in range(n):
    norms_l = np.linalg.norm(rzs_lscale[i], axis=0)
    norms_c = np.linalg.norm(rzs_correct[i], axis=0)
    for j in range(n):
        if i == j:
            continue
        true_edge = "  <-- TRUE EDGE" if abs(B_true[j, i]) > 1e-10 else ""
        print(f"  {i}->{j}  {norms_l[j]:10.4f}  {norms_c[j]:10.4f}  {abs(norms_l[j] - norms_c[j]):10.4f}{true_edge}")

print("\n>>> VERDICT: If LSCALE and correct differ substantially, the pseudo-inverse")
print("    bug in LSCALE.py may corrupt graph estimation.")


# ============================================================================
# TEST 3: Full LSCALE_i on correct data — encoder, graph, latent recovery
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: LSCALE_i quality on correctly-generated interventional data")
print("=" * 80)

crl = LSCALE_i(
    n=N,
    x_samples=x_samples,
    actions_as_list=actions_list,
    hard_intervention=(INTERVENTION_TYPE == "hard"),
    gamma=GAMMA,
)

enc_est_s, dag_est_s = crl[0]
dag_est_s = dag_est_s.astype(int)

if INTERVENTION_TYPE == "hard" and crl[1] is not None:
    enc_est_h, dag_est_h = crl[1]
    enc_est_h = enc_est_h
    dag_est_h = dag_est_h.astype(int)
else:
    enc_est_h, dag_est_h = enc_est_s, dag_est_s

dag_est = dag_est_s
enc_est_final = enc_est_h if INTERVENTION_TYPE == "hard" else enc_est_s

print(f"\n--- DAG estimation ---")
true_adj = (np.abs(B_true) > 1e-10).astype(int)
print(f"True DAG adjacency:\n{true_adj}")
print(f"\nEstimated DAG (soft):\n{dag_est_s}")
if INTERVENTION_TYPE == "hard" and crl[1] is not None:
    print(f"\nEstimated DAG (hard):\n{dag_est_h}")

dag_match = np.sum(dag_est_s == true_adj)
dag_total = N * N
print(f"\nDAG accuracy (soft): {dag_match}/{dag_total} entries match "
      f"({100*dag_match/dag_total:.1f}%)")

# Check encoder quality: hat_Z = X @ enc.T should approximate Z
print(f"\n--- Encoder / latent recovery quality ---")

sem3 = LinearSEMGenerator(n=N, d=D, seed=SEED, latent_noise_std=1.0)
n_eval = 2000
Z_true_eval = sem3.sample_latents(n_eval, frozenset(), kind="none")
X_eval = Z_true_eval @ sem3.G.T
Z_hat_eval = X_eval @ enc_est_final.T

# Z_hat may be a permuted / sign-flipped / scaled version of Z_true
# Find best alignment by checking correlations
corr_matrix = np.corrcoef(Z_true_eval.T, Z_hat_eval.T)[:N, N:]
print(f"\nCorrelation matrix (true Z_i vs hat Z_j):")
print(f"  Rows = true Z[i], Cols = hat Z[j]")
print(f"{corr_matrix}")

# Best permutation alignment
abs_corr = np.abs(corr_matrix)
print(f"\nPer true latent, best-matching estimated latent:")
for i in range(N):
    best_j = np.argmax(abs_corr[i])
    print(f"  Z_true[{i}] <-> Z_hat[{best_j}]:  corr = {corr_matrix[i, best_j]:+.4f}  "
          f"(|corr| = {abs_corr[i, best_j]:.4f})")

max_per_row = abs_corr.max(axis=1)
print(f"\nMin |correlation| across best matches: {max_per_row.min():.4f}")
if max_per_row.min() > 0.8:
    print(">>> PASS: Encoder recovers latents well (up to permutation/sign).")
else:
    print(">>> FAIL: Encoder does NOT reliably recover latents.")


# ============================================================================
# TEST 4: What ROCRL actually feeds LSCALE (buggy data, kind='none')
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: LSCALE_i quality on data from seed_initial_pools (kind='none' BUG)")
print("=" * 80)

sem_bug = LinearSEMGenerator(n=N, d=D, seed=SEED, latent_noise_std=1.0)
x_samples_bug = []
actions_bug = []
for a in mca0:
    for _ in range(T0):
        Z = sem_bug.sample_latents(1, a)  # BUG: no kind= passed, defaults to "none"
        X = Z @ sem_bug.G.T
        x_samples_bug.append(X.reshape(-1))
        actions_bug.append(a)

x_samples_bug = np.array(x_samples_bug)

crl_bug = LSCALE_i(
    n=N,
    x_samples=x_samples_bug,
    actions_as_list=actions_bug,
    hard_intervention=(INTERVENTION_TYPE == "hard"),
    gamma=GAMMA,
)

enc_bug_s, dag_bug_s = crl_bug[0]
dag_bug_s = dag_bug_s.astype(int)

if INTERVENTION_TYPE == "hard" and crl_bug[1] is not None:
    enc_bug_h, dag_bug_h = crl_bug[1]
    dag_bug_h = dag_bug_h.astype(int)
else:
    enc_bug_h, dag_bug_h = enc_bug_s, dag_bug_s

enc_bug_final = enc_bug_h if INTERVENTION_TYPE == "hard" else enc_bug_s

print(f"\nEstimated DAG (soft, buggy data):\n{dag_bug_s}")
if INTERVENTION_TYPE == "hard" and crl_bug[1] is not None:
    print(f"\nEstimated DAG (hard, buggy data):\n{dag_bug_h}")

dag_match_bug = np.sum(dag_bug_s == true_adj)
print(f"\nDAG accuracy (soft, buggy): {dag_match_bug}/{dag_total} entries match "
      f"({100*dag_match_bug/dag_total:.1f}%)")

# Encoder quality on buggy data
sem_bug2 = LinearSEMGenerator(n=N, d=D, seed=SEED, latent_noise_std=1.0)
Z_hat_bug = X_eval @ enc_bug_final.T

corr_bug = np.corrcoef(Z_true_eval.T, Z_hat_bug.T)[:N, N:]
abs_corr_bug = np.abs(corr_bug)
print(f"\nCorrelation matrix (true Z_i vs hat Z_j) [BUGGY DATA]:")
print(f"{corr_bug}")

max_per_row_bug = abs_corr_bug.max(axis=1)
print(f"\nMin |correlation| across best matches: {max_per_row_bug.min():.4f}")
if max_per_row_bug.min() > 0.8:
    print(">>> PASS: Encoder still recovers latents (unlikely with purely obs data).")
else:
    print(">>> FAIL: Encoder does NOT recover latents — confirms the kind='none' bug.")


# ============================================================================
# TEST 5: Parameter estimation quality (on correct data)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: SEM parameter estimation using ROCRL's build_weight_matrices")
print("=" * 80)

from ROCRL import ROCRLLearner

learner = ROCRLLearner(
    n=N, d=D,
    intervention_type=INTERVENTION_TYPE,
    gamma=GAMMA,
    T0=T0,
    delta=0.05,
    epsilon_max=0.25,
    seed=0,
)

# Manually feed correct data into learner pools (bypassing seed_initial_pools bug)
sem_correct2 = LinearSEMGenerator(n=N, d=D, seed=SEED, latent_noise_std=1.0)
util_correct2 = LinearUtility(n=N, noise_std=0.0, theta_dist="rademacher", seed=1)
env_correct = ROCRLEnvironment(sem=sem_correct2, utility=util_correct2)

forced_actions = (
    [frozenset() for _ in range(T0)]
    + [frozenset({i}) for i in range(N) for _ in range(T0)]
)

for a in forced_actions:
    kind = "none" if len(a) == 0 else INTERVENTION_TYPE
    z = sem_correct2.sample_latents(1, a, kind=kind).reshape(-1)
    u = float(z @ util_correct2.theta)
    learner.apply_action_and_update_pools(a, z, sem_correct2.G, u)

print(f"Learner pools: {len(learner.X_all)} total samples, {len(learner.X_crl)} CRL samples")

# Run the CRL step
X_crl = np.asarray(learner.X_crl, dtype=float)
X_all = np.asarray(learner.X_all, dtype=float)

crl_result = LSCALE_i(
    n=N,
    x_samples=X_crl,
    actions_as_list=learner.A_crl,
    hard_intervention=(INTERVENTION_TYPE == "hard"),
    gamma=GAMMA,
)

enc_s, dag_s = crl_result[0]
dag_s = dag_s.astype(int)

if INTERVENTION_TYPE == "hard" and crl_result[1] is not None:
    enc_h, dag_h = crl_result[1]
else:
    enc_h, dag_h = enc_s, dag_s

enc_final = enc_h if INTERVENTION_TYPE == "hard" else enc_s

pa = [list(np.nonzero(dag_s[:, i])[0]) for i in range(N)]
topo = learner.get_causal_order(dag_s)

print(f"\nEstimated DAG:\n{dag_s}")
print(f"True DAG:\n{true_adj}")
print(f"\nEstimated parents: {pa}")
true_pa = [list(np.nonzero(true_adj[:, i])[0]) for i in range(N)]
print(f"True parents:      {true_pa}")
print(f"Topo order:        {topo}")

# Compute hat_Z
hat_Z = X_all @ enc_final.T
print(f"\nhat_Z shape: {hat_Z.shape}")

# Check hat_Z alignment with true Z
Z_all_true = np.array(learner.Z_all)
corr_final = np.corrcoef(Z_all_true.T, hat_Z.T)[:N, N:]
abs_corr_final = np.abs(corr_final)

print(f"\nCorrelation (true Z vs hat Z):")
for i in range(N):
    best_j = np.argmax(abs_corr_final[i])
    print(f"  Z_true[{i}] <-> Z_hat[{best_j}]:  corr = {corr_final[i, best_j]:+.4f}")

# SEM parameter estimation
delta_t = learner.get_delta_t(len(X_all), learner.delta)
u = learner.compute_u_from_graph(dag_s, pa)
f_val = learner.f_t(u, len(X_all), delta_t, learner.epsilon_max, learner.C_const)
zeta_t = 0.1 * len(X_all) * np.sqrt((D + np.log(1.0 / delta_t)) / f_val)

V, tilde_V, hat_b, g__ = learner.initialize_weight_matrices(pa)
V, tilde_V, hat_b, VV = learner.build_weight_matrices(
    hat_Z=hat_Z, V=V, tilde_V=tilde_V, hat_b=hat_b, g__=g__,
    zeta_t=zeta_t, pa=pa,
)

A_hat, Astar_hat, nu_hat, nu_star_hat = learner.reconstruct_A_from_hat_b(hat_b, pa)

print(f"\n--- SEM coefficient comparison (observational) ---")
print(f"A_hat (estimated):\n{A_hat}")

# True A in ROCRL convention: A_hat[i, pa[i]] should match B_true[i, pa[i]]
# But hat_Z might be permuted/scaled relative to true Z, so direct comparison
# requires finding the alignment first.

# Build true A in the same format
A_true = np.zeros((N, N))
for i in range(N):
    if len(true_pa[i]) > 0:
        A_true[i, true_pa[i]] = B_true[i, true_pa[i]]
print(f"\nA_true (ground truth):\n{A_true}")

print(f"\n--- Intercepts ---")
print(f"nu_hat (obs):       {nu_hat}")
print(f"nu_star_hat (int):  {nu_star_hat}")

# True intercepts: E[eps] = latent_noise_std / 2 for uniform noise
true_nu = (sem.latent_noise_std / 2.0) * np.ones(N)
print(f"True nu (E[eps]):   {true_nu}")

# Theta estimation
U_all = np.asarray(learner.U_all, dtype=float)
theta_hat, V_theta, tilde_V_theta, VV_theta = learner.estimate_theta(
    hat_Z=hat_Z, U=U_all, zeta_t=zeta_t,
)

print(f"\n--- Theta comparison ---")
print(f"theta_hat:  {theta_hat}")
print(f"theta_true: {theta_true}")

# Since hat_Z may be permuted/scaled, theta_hat won't match directly.
# Instead check if theta_hat @ hat_Z predicts U well.
U_pred = hat_Z @ theta_hat
U_actual = U_all
residual = U_actual - U_pred
r2 = 1.0 - np.var(residual) / np.var(U_actual)
print(f"\nR² of theta_hat·hat_Z predicting U: {r2:.4f}")

# Check predicted E[U] under each action
print(f"\n--- Predicted vs true E[U] per action ---")
from ucb import predict_latent_mean_under_action

print(f"{'Action':>10}  {'E[U] true':>10}  {'E[U] pred':>10}  {'error':>10}")
for a in [frozenset()] + [frozenset({i}) for i in range(N)]:
    true_eu = expected_utility_under_action(env_correct, action=set(a), kind=kind_of(a))
    mu_pred = predict_latent_mean_under_action(
        topo=topo, pa=pa, action=a,
        A_hat=A_hat, Astar_hat=Astar_hat,
        nu_hat=nu_hat, nu_star_hat=nu_star_hat,
    )
    pred_eu = float(theta_hat @ mu_pred)
    err = pred_eu - true_eu
    label = "obs" if len(a) == 0 else f"do({list(a)[0]})"
    print(f"  {label:>8}  {true_eu:10.4f}  {pred_eu:10.4f}  {err:+10.4f}")


# ============================================================================
# TEST 6: Quick check of under-sampling threshold behavior
# ============================================================================
print("\n" + "=" * 80)
print("TEST 6: Under-sampling threshold f_t vs actual counts")
print("=" * 80)

counts = learner.current_mca0_counts()
a0_ue = learner.get_a0_UE(dag_s, pa, delta=learner.delta, epsilon_max=learner.epsilon_max)

print(f"f_t threshold: {learner.f_t_val:.1f}")
print(f"Action counts: {dict(counts)}")
print(f"Under-sampled actions: {a0_ue}")
if len(a0_ue) > 0:
    print(">>> Under-sampling is still active — UCB would not run yet.")
    print("    This means the learner would remain in forced exploration mode.")
else:
    print(">>> No under-sampled actions — UCB would run.")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print("""
BUG 1 (CRITICAL): seed_initial_pools in ROCRL.py never passes kind= to
    sem.sample_latents(). All forced-exploration data is generated under the
    OBSERVATIONAL distribution regardless of the action label. This means
    LSCALE receives identical distributions for all actions, making the
    precision differences Theta_i - Theta_0 ≈ 0 (pure noise). The encoder,
    graph, and all downstream estimates are garbage.

    FIX: In seed_initial_pools, change:
        z_sample = env.sem.sample_latents(1, a)
    to:
        kind = "none" if len(a) == 0 else self.intervention_type
        z_sample = env.sem.sample_latents(1, a, kind=kind)

BUG 2 (in LSCALE.py): The pseudo-inverse of enc_est_s is computed incorrectly.
    The code does:
        enc_est_s_pt = U @ diag(1/S) @ Vh
    but the correct pseudo-inverse of A = U @ diag(S) @ Vh is:
        A^+ = Vh.T @ diag(1/S) @ U.T
    
    FIX: In LSCALE.py, change:
        enc_est_s_pt = enc_est_s_svd.U @ np.diagflat(1/enc_est_s_svd.S) @ enc_est_s_svd.Vh
    to:
        enc_est_s_pt = enc_est_s_svd.Vh.T @ np.diagflat(1/enc_est_s_svd.S) @ enc_est_s_svd.U.T
    (or simply use np.linalg.pinv(enc_est_s))
""")
