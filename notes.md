# Implementation tricks
## LSCALE-I:
- Dimension reduction
- SVD Decomposition to compute $H_t$
- Normalization of $H_t$ by observational latent covariance
- Manual pinv computation
- DAG Enforcement
- Transitive closure series approximation
- MMSE Recursive computation
  - After the baseline recovery, the algorithm identifies $H_t$ up to a linear transformation, i.e. $H_t=TG^\dagger$ for some unknown invertible matrix $T\in\mathbb{R}^{n\times n}$ with estimated latents $\hat{Z}_t=XH_t^T=ZT^T$. Hard interventions allow us to identify the latents up to permuations and scaling: $\hat{Z}_t=X()$ 