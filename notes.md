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


## `LSCALE1` Routine:
- Inputs: $n$, $\mathbf{X}_t$, $\mathcal{A}_t$, intervention type, $\gamma$
- Returns: $H_t, H_t^{\text{hard}}$ (if intervention is hard), and $\mathcal{G}_t$

1. Run dimensionality reduction?
2. Compute $\Sigma_{a, t},\Theta_{\{i\},t},R_{i,t}$ for all $a,i$.
3. Estimate encoder $H_t$ via `_get_encoder`: 
   1. Compute SVD of $R_{i,t}$
   2. Assign $[H_t]_i\leftarrow v_0$, the first right singular vector (corresponds to the principle eigenvector of $R_{i,t}^TR_{i,t}$)
4. Compute latent covariances $\hat{\Sigma}^Z_{a,t}=H_t\Sigma_{t}H_t^T$ for all $a$. Normalize rows via $$[H_t]_i \leftarrow [H_t]_i/\sqrt{\text{diag}(\hat{\Sigma}^Z_{0,t})[i]}$$ and re-compute $\hat{\Sigma}^Z_{a,t}$ for all $a$.
5. Manually compute $H_t^\dagger$ and use it to compute $\hat{R}^Z_{i,t}=H_t^\dagger R_{i,t}(H_t^\dagger)^T$
6. Estimate adjacency graph via `_get_graph`:
   1. Compute $||[\hat{R}^Z_{i,t}]_j||_2$ for each row $j$
   2. Assign $\hat{\mathcal{G}}_t[i,j]=1\{||[\hat{R}^Z_{i,t}]_j||_2>\gamma\}$
   3. Find closest DAG with `_closest_dag`:
      1. Find permutation of nodes that maximizes the number of forward edges
      2. Enforce acyclicity under this permutation
      3. Re-map indices to original order
   4. If intervention is hard, set $\hat{\mathcal{G}}_t$ to its transitive closure with `_transitive_closure`:
      1. Compute $(I-\mathcal{G}_t)^{-1}$ and threshold entries
      2. Set diagonals to 0/False
7.  Reduce dimension of $H_t$?
8. If the intervention is hard: 
   1. compute unmixing matrix $D_t$ via `_unmixing_procedure`:
      1. hmm
   2. Assign $H_t^{\text{hard}}=D_tH_t$