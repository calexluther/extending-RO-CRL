
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional

type FloatArray = npt.NDArray[np.floating]
type IntArray = npt.NDArray[np.int_]
type BoolArray = npt.NDArray[np.bool_]



def LSCALE_i(
    n: int,                                  # number of latent variables
    x_samples: FloatArray,                   # (n_samples, d) each row is a d-dimensional observation
    actions_as_list: list[list[int]],        # (n_samples, ) each list is a set of actions
    hard_intervention: bool,                 # whether to perform hard intervention
    gamma: float,                            # graph threshold
    dim_reduction: bool = True,              # optional projection from d to n dimensions
    hard_graph_postprocess: bool = False,    # whether to perform hard graph refinement
) -> Tuple[
    Tuple[FloatArray, BoolArray],
    Optional[Tuple[FloatArray, BoolArray]],
]
"""
Returns baseline recovered hat{Z}, hat{G} and refined hat{Z}, hat{G} if hard_intervention is True.
"""
    assert x_samples.ndim == 2
    n_samples, d = x_samples.shape
    assert len(actions_as_list) == n_samples

    actions = list(map(frozenset, actions_as_list))

    if dim_reduction: # reduce dimensionality from d to n
        x_svd = np.linalg.svd(x_samples[:n+d], full_matrices=False)
        dec_colbt = x.svd.Vh[:n]
        x_samples = x_samples @ dec_colbt.T
    
    # organize samples by actions: x_by_mca0[a] is the stack of x_samples for action a
    mca0 = [frozenset()] + [frozenset([i]) for i in range(n)]
    x_by_mca0 = {a: np.stack([x_samples[i] for i in range(len(actions)) if actions[i] == a]) for a in mca0}

    # compute sample covariance and precision matrices for each action
    x_covs = np.stack([np.cov(x_by_mca0[a], rowvar = False) for a in mca0])
    x_precs = np.stack([np.linalg.pinv(x_covs[i]) for i in range(n + 1)])
    rxs = x_precs[1:] - x_precs[0] # Theta_i - Theta_0 for i = 1, ..., n

    enc_est_s = _get_encoder(rxs)

    # normalize enc_est_s by observational latent covariance
    zhat_covs = enc_est_s @ x_covs @ enc_est_s.T
    enc_est_s /= zhat_covs[0].diagonal()[:, None] ** 0.5
    zhat_covs = enc_est_s @ x_covs @ enc_est_s.T








def _get_encoder(rxs: FloatArray) -> FloatArray:
    """

    """
    n, _, d = rxs.shape
    enc_est = np.zeros((n, d))
    for i in range(n):
        enc_est[i] = np.linalg.svd(rxs[i], full_matrices = False).Vh[0]
    return enc_est




# why might we reduce dimensions?
# why not directly use the principal eigenvector of rxs?
# why do we normalize by the observational latent covariance?
# why compute pinv manually?