import numpy as np
from typing import List, Tuple


def ucb_bonus(
    pi: List[int],
    pa: List[List[int]],
    Ni: List[List[int]],
    hat_b: List[Tuple[np.ndarray, np.ndarray]],
    zeta_t: float,
    delta_t: float,
    n: int,
    VV: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, List[int], List[List[int]]]:

    mean = np.zeros(n)
    int_ = []
    for idx in pi:
        feat = np.concatenate((np.array([1]), mean[pa[idx]])).reshape(-1, 1)
        ys_hat = []
        for r in [0, 1]:
            beta_t = 1+ np.sqrt(len(pa[idx])+1) + np.sqrt(
                2*np.log(n/delta_t)+
                (len(pa[idx])+1) *np.log((1+ Ni[idx][r]/(len(pa[idx])+1) /zeta_t**2))
            )
            beta_t = 0.1 * beta_t

            
            Vinv = np.linalg.inv(VV[idx][r])
            numer = Vinv @ feat
            denom = np.sqrt(feat.T @ Vinv @ feat)
            bonus = numer / denom
            
            ys_hat.append(hat_b[idx][r] + beta_t * bonus)
        
        try:
            mean_obs = ys_hat[0].T @ np.concatenate((np.array([1]), mean[pa[idx]]))
            mean_int = ys_hat[1].T @ np.concatenate((np.array([1]), mean[pa[idx]]))
        except:
            mean_obs = ys_hat[0]
            mean_int = ys_hat[1]

        if mean_obs > mean_int:
            mean[idx] = mean_obs
            Ni[idx][0] += 1
        else:
            mean[idx] = mean_int
            int_.append(idx)
            Ni[idx][1] += 1

    return mean, int_, Ni


    