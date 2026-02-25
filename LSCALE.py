
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
from itertools import permutations

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]


def LSCALE_i(
    n: int,                                  # number of latent variables
    x_samples: FloatArray,                   # (n_samples, d) each row is a d-dimensional observation
    actions_as_list: list[list[int]],        # (n_samples, ) each list is a set of actions
    hard_intervention: bool,                 # whether to perform hard intervention
    gamma: float,                            # graph threshold
    dim_reduction: bool = True,              # optional projection from d to n dimensions
    hard_graph_postprocess: bool = False,    # whether to perform hard graph refinement
) -> Tuple[Tuple[FloatArray, BoolArray], Optional[Tuple[FloatArray, BoolArray]]]:
    """
    Returns baseline recovered H_t, H_t^hard and refined H_t, H_t^hard if hard_intervention is True.
    Also returns the estimated graph Ghat_t.
    """
    assert x_samples.ndim == 2
    n_samples, d = x_samples.shape
    assert len(actions_as_list) == n_samples

    actions = list(map(frozenset, actions_as_list))

    if dim_reduction: # reduce dimensionality from d to n
        x_svd = np.linalg.svd(x_samples[:n+d], full_matrices=False)
        dec_colbt = x_svd.Vh[:n]
        x_samples = x_samples @ dec_colbt.T
    
    # organize samples by actions: x_by_mca0[a] is the stack of x_samples for action a
    mca0 = [frozenset()] + [frozenset([i]) for i in range(n)]
    x_by_mca0 = {a: np.stack([x_samples[i] for i in range(len(actions)) if actions[i] == a]) for a in mca0}

    # compute sample covariance and precision matrices for each action
    x_covs = np.stack([np.cov(x_by_mca0[a], rowvar = False) for a in mca0])
    x_precs = np.stack([np.linalg.pinv(x_covs[i]) for i in range(n + 1)])
    rxs = x_precs[1:] - x_precs[0] # Theta_i - Theta_0 for i = 1, ..., n

    enc_est_s = _get_encoder(rxs) # H_t

    # normalize enc_est_s by observational latent covariance
    zhat_covs = enc_est_s @ x_covs @ enc_est_s.T # Sigma^Z_t = H_t Sigma_t H_t^T
    enc_est_s /= zhat_covs[0].diagonal()[:, None] ** 0.5 # H_t = H_t / sqrt(Sigma^Z_t[0,0])
    
    zhat_covs = enc_est_s @ x_covs @ enc_est_s.T # recomputed Sigma^Z_t = H_t Sigma_t H_t^T

    # manual pinv computation (to get H_t^+)
    enc_est_s_svd = np.linalg.svd(enc_est_s, full_matrices = False)
    enc_est_s_pt = enc_est_s_svd.U @ np.diagflat(1 / enc_est_s_svd.S) @ enc_est_s_svd.Vh
    
    rzs = enc_est_s_pt @ rxs @ enc_est_s_pt.T # Rhat^Z_t = H_t^+ R_t H_t^+

    # get Ghat_t from Rhat^Z_t
    dag_est_s, top_order = _get_graph(rzs, gamma)


    if dim_reduction:
        enc_est_s = enc_est_s @ dec_colbt
    
    soft_results = enc_est_s, dag_est_s
    hard_results = None
    if hard_intervention:
        unmix_mat, dag_est_h = _unmixing_procedure(
            zhat_covs[1:],
            rzs,
            top_order,
            gamma)
        enc_est_h = unmix_mat @ enc_est_s
        hard_results = enc_est_h, dag_est_h
    
    return soft_results, hard_results


def _unmixing_procedure(
    z_covs: FloatArray,
    rzs: FloatArray,
    top_order: IntArray,
    gamma: float = 0.1,
) -> tuple[FloatArray, BoolArray]:
    """
    """
    n = z_covs.shape[0]
    unmix_mat = np.eye(n)
    for k in range(n): # will this throw an error if k = 0?
        ck = unmix_mat @ z_covs[top_order[k]]
        
        a_mat = ck[np.ix_(top_order[:k], top_order[:k])] # submatrix of latent covariances for ancestors (?) of node k
        b_vec = ck[top_order[:k], top_order[k]] # vector of latent covariances for ancestors (?) of node k
        # regress out effect of ancestors on node k
        unmix_mat[top_order[k], top_order[:k]] = np.linalg.solve(a_mat, -b_vec)

    unmix_it = np.linalg.inv(unmix_mat).T # H_t^-T
    rzs = unmix_it @ rzs @ unmix_it.T # Rhat^Z_t = H_t^-T R_t H_t^-1
    # Re-estimate graph with new Rhat^Z_t but without transitive closure
    dag_est, _top_order = _get_graph(rzs, gamma, hard_intervention = True)

    return unmix_mat, dag_est


def _get_encoder(rxs: FloatArray) -> FloatArray:
    """

    """
    n, _, d = rxs.shape
    enc_est = np.zeros((n, d))
    for i in range(n):
        enc_est[i] = np.linalg.svd(rxs[i], full_matrices = False).Vh[0]
    return enc_est


def _get_graph(
    rzs: FloatArray,
    gamma: float = 0.1,
    hard_intervention: bool = False,
) -> Tuple[BoolArray, IntArray]:
    """
    """
    n, _, _ = rzs.shape
    # compute edge weights for each node: || Rhat^Z_i[j,:] ||_2 for j = 1, ..., n
    dag_est_wo_th = np.stack([np.linalg.vector_norm(rzs[i], axis = 0) for i in range(n)]).T
    # threshold with gamma
    dag_est = dag_est_wo_th > gamma
    # remove self-loops
    np.fill_diagonal(dag_est, False)

    # force it to be a DAG
    dag_est, top_order = _closest_dag(dag_est)

    if not hard_intervention:
        dag_est = _transitive_closure(dag_est)
    return dag_est, top_order


def _closest_dag(g: BoolArray) -> Tuple[BoolArray, IntArray]:
    """
    Finds permutation of nodes that maximizes the number of forward edges.
    Builds adjacency matrix under this permutation and enforces acyclicity.
    Re-maps node indices to original order and returns resulting matrix along with best permuation.
    """
    n = g.shape[0]
    np.fill_diagonal(g, False)
    best_perm = np.arange(n)
    
    # reorder rows/cols by best_perm, take strict upper triangle and count ones
    best_perm_trius = np.triu(g[np.ix_(best_perm, best_perm)], 1).sum() 

    for perm in map(np.array, permutations(np.arange(n))):
        # reorder rows/cols by perm, take strict upper triangle and count ones
        perm_trius = np.triu(g[np.ix_(perm, perm)], 1).sum() 
        if perm_trius > best_perm_trius:
            best_perm_trius = perm_trius
            best_perm = perm
    
    # best_perm orders original node indices 
    # i.e. best_perm = [1, 2, 0] means node 1, node 2, then node 0
    # inv_best_perm tracks how the original nodes are reorderd in best_perm
    # i.e. inv_best_perm = [2, 0, 1] means node 0 is in spot 2, node 1 in spot 0, and node 2 in spot 1
    inv_best_perm = np.arange(n)
    inv_best_perm[best_perm] = np.arange(n)

    # reorder rows/cols by best_perm and enforce acyclicity
    g = g[np.ix_(best_perm, best_perm)]
    g = np.triu(g, 1)
    # invert back to original node indices
    g = g[np.ix_(inv_best_perm, inv_best_perm)]
    return g, best_perm


def _transitive_closure(g: BoolArray) -> BoolArray:
    ATOL = 1e-7
    n = g.shape[0]
    # Compute (I_n - G)^-1
    series = np.linalg.inv(np.eye(n) - g.astype(np.floating))
    series_nz = np.abs(series) > ATOL # threshold for some reason
    np.fill_diagonal(series_nz, False)
    return series_nz


if __name__ == "__main__":
    n = 3
    g = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    print(g)
    print(_transitive_closure(g))