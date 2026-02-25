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
- Returns: $H_t, H_t^{\text{hard}}$ (if intervention is hard), and $\hat{\mathcal{G}}_t$

1. Run dimensionality reduction?
2. Compute $\Sigma_{a, t},\Theta_{\{i\},t},R_{i,t}$ for all $a,i$.
3. Estimate encoder $H_t$ via `_get_encoder`: 
   1. Compute SVD of $R_{i,t}$
   2. Assign $[H_t]_i\leftarrow v_0$, the first right singular vector (corresponds to the principle eigenvector of $R_{i,t}^TR_{i,t}$)
4. Compute latent covariances $\hat{\Sigma}^Z_{a,t}=H_t\Sigma_{t}H_t^T$ for all $a$. Normalize rows via $$[H_t]_i \leftarrow [H_t]_i/\sqrt{\text{diag}(\hat{\Sigma}^Z_{0,t})[i]}$$ and re-compute $\hat{\Sigma}^Z_{a,t}$ for all $a$.
5. Manually compute $H_t^\dagger$ and use it to compute $\hat{R}^Z_{i,t}=H_t^\dagger R_{i,t}(H_t^\dagger)^T$
6. Estimate adjacency graph via `_get_graph`:
   1. Compute $||[\hat{R}^Z_{i,t}]_j||_2$ for each column $j$
   2. Assign $\hat{\mathcal{G}}_t[i,j]=1\{||[\hat{R}^Z_{i,t}]_j||_2>\gamma\}$
   3. Find closest DAG with `_closest_dag`:
      1. Find permutation of nodes that maximizes the number of forward edges
      2. Enforce acyclicity under this permutation
      3. Re-map indices to original order
   4. If intervention is not hard, set $\hat{\mathcal{G}}_t$ to its transitive closure with `_transitive_closure`:
      1. Compute $(I-\mathcal{G}_t)^{-1}$ and threshold entries
      2. Set diagonals to 0/False
7.  Reduce dimension of $H_t$?
8. If the intervention is hard, run `_unmixing_procedure`: find an unmixing matrix $D$ such that $\text{Cov}(D\hat{Z}_j,D\hat{Z}_{\text{an}(j)}|do(j))=0$ for all nodes $j$ under given topological order (computed earlier in `_closest_dag`). Equivalently,
   $$\begin{align*}&\text{Cov}(D\hat{Z}_j,D\hat{Z}_{\text{an}(j)}|do(j))=0\\
   \iff& \text{Cov}(\hat{Z}_j-\beta^TD\hat{Z}_{\text{an}(j)},D\hat{Z}_{\text{an}(j)}|do(j))=0\\
   \iff& \text{Cov}(\hat{Z}_j,D\hat{Z}_{\text{an}(j)}|do(j))=\beta^T\text{Cov}(D\hat{Z}_{\text{an}(j)},D\hat{Z}_{\text{an}(j)}|do(j))\\
   \iff&(D\hat{\Sigma}^Z_{\{j\}})_{j,\text{an}(j)}=\beta^T(D\hat{\Sigma}^Z_{\{j\}}D^T)_{\text{an}(j),\text{an}(j)}\end{align*}$$
   This is essentially a Gram-Schmidt process in topological order. 
   1. For node $j$, compute $(D\hat{\Sigma}^Z_{\{j\}})_{j,\text{an}(j)}$ and $(D\hat{\Sigma}^Z_{\{j\}})_{\text{an}(j),\text{an}(j)}$ from current $D$. Note that we don't need to right-multiply the second term by $D$. 
   2. Solve equation for $\beta$ and assign $D_{j,\text{an}(j)}\leftarrow -\beta$
   3. Repeat for all $j$.
   4. Re-compute $\hat{R}^Z_t\leftarrow D^{-T}\hat{R}^Z_tD^{-1}$ and re-run `_get_graph` with new $\hat{R}^Z_t$.
   5. Set $H_t^{\text{hard}}=DH_t$.