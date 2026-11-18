# Sections 1.25-1.26 Instructor Solutions

## 1.25 Review Questions

### 1. Why is minimizing empirical risk not sufficient to guarantee learning?
Empirical-risk minimization controls performance on the observed sample, not future draws from the target distribution. An expressive model can fit noise or sample-specific structure and obtain nearly zero empirical risk while having poor population risk. Generalization therefore requires inductive bias, capacity control or regularization, and an out-of-sample evaluation protocol.

### 2. Compare inductive bias in linear models, CNNs, transformers, and physics-informed models.
Linear models impose affine structure in the chosen representation. CNNs impose locality and translation-equivariant weight sharing. Transformers use token representations with shared attention and feed-forward blocks that learn content-dependent interactions. Physics-informed models add explicit scientific structure such as PDE residuals, conservation laws, symmetries, or boundary conditions.

### 3. Why does hypothesis-space complexity influence sample complexity?
A richer hypothesis class can fit more distinct labelings of a finite sample, so more observations are needed to distinguish genuine predictive structure from accidental fit. VC dimension, Rademacher complexity, margins, and norm-based bounds formalize this dependence. Regularization and architecture can reduce effective complexity.

### 4. Explain the difference between optimization error and estimation error.
Optimization error is the gap between the objective value reached by an algorithm and the best value achievable inside the selected hypothesis class. Estimation error is the finite-sample gap between the empirical best predictor and the population-optimal predictor in that class. Training longer can reduce optimization error but cannot remove finite-sample estimation error.

### 5. Why does stochastic gradient noise sometimes improve generalization?
Mini-batch gradients are noisy estimates of the full gradient. Moderate noise can prevent premature convergence to sharp sample-specific minima and can bias training toward broader low-loss regions. The benefit is not universal: too much noise slows or destabilizes optimization, and the flatness-generalization relationship is parameterization dependent.

### 6. Compare Adam, AdamW, Adafactor, and Lion from memory and regularization perspectives.
Adam stores first and second moments, adding roughly two optimizer-state tensors per parameter. AdamW has the same state cost but decouples weight decay from adaptive scaling. Adafactor factorizes matrix second moments, reducing matrix-state memory from O(mn) to O(m+n). Lion uses approximately one momentum-like state tensor. Decoupled decay is generally easier to interpret than L2 regularization passed through adaptive scaling.

### 7. Why are second-order methods useful for PINNs but difficult for large language models?
PINN losses can be badly conditioned because data, PDE, boundary, and initial-condition terms have different scales. Quasi-Newton or Newton-like curvature information can improve late-stage convergence. Large language models have billions of parameters, making Hessian storage or even approximate curvature operations expensive in memory, communication, and compute.

### 8. Why is learning-rate warmup important for transformer training?
At initialization, normalization/attention statistics and adaptive-optimizer moments are poorly calibrated. A large target learning rate can produce destructive early updates. Warmup gradually increases the rate until the representation and optimizer statistics stabilize, after which cosine or other decay schedules can be applied.

### 9. Compare natural gradient descent and mirror descent geometrically.
Natural gradient uses the Fisher information matrix as a local Riemannian metric, giving the update Δθ=-ηF^{-1}∇L. Mirror descent uses a convex potential and its Bregman divergence to define a non-Euclidean geometry in a dual coordinate system. Both replace Euclidean distance with problem-appropriate geometry.

### 10. Why does reinforcement learning require trust-region or clipped policy updates?
Changing a policy changes the state-action distribution. Large updates can invalidate the advantage estimates used to justify the update and sharply reduce return. Trust-region methods constrain KL divergence; PPO approximates this by clipping the probability ratio so a single batch cannot reward arbitrarily large policy changes.

### 11. Explain the role of automatic differentiation in scientific machine learning.
Automatic differentiation applies the chain rule through a computation graph and provides exact derivatives up to floating-point arithmetic. Reverse-mode AD enables backpropagation. In scientific ML it also supplies spatial and temporal derivatives for PDE residuals, inverse problems, differentiable simulators, and sensitivity analysis.

### 12. Why should optimizer selection depend on objective geometry and systems constraints?
Conditioning, curvature, stochasticity, smoothness, sparsity, and constraints determine which numerical method is effective. Systems constraints are equally important: optimizer state consumes memory, communication can dominate distributed runtime, and some algorithms map poorly to accelerators. Optimizer choice must therefore match both numerical geometry and resource constraints.

## 1.26 Mathematical Problems

### 1. Gradient and Hessian of regularized logistic regression
For labels y_i∈{0,1}, p_i=σ(x_i^T w), and
J(w)=-Σ_i[y_i log p_i+(1-y_i)log(1-p_i)]+(λ/2)||w||²,
the gradient is
∇J(w)=X^T(p-y)+λw.
With W=diag(p_i(1-p_i)), the Hessian is
∇²J(w)=X^T W X+λI.
W is positive semidefinite, so the objective is convex; λ>0 gives strong convexity.

### 2. Convergence of gradient descent for strongly convex smooth functions
Assume f is μ-strongly convex with L-Lipschitz gradient. For x_{k+1}=x_k-η∇f(x_k) and 0<η≤1/L, smoothness and strong convexity imply
f(x_{k+1})-f* ≤ (1-ημ)[f(x_k)-f*].
Thus
f(x_k)-f* ≤ (1-ημ)^k[f(x_0)-f*],
a linear convergence rate. With η=1/L the factor is 1-μ/L.

### 3. Stochastic gradient estimator and unbiasedness
For empirical risk Rhat(w)=n^{-1}Σ_i ℓ_i(w), sample I uniformly and define g(w)=∇ℓ_I(w). Then
E[g(w)]=n^{-1}Σ_i∇ℓ_i(w)=∇Rhat(w),
assuming differentiation and expectation can be interchanged. A uniformly sampled mini-batch mean is likewise unbiased.

### 4. Momentum as an exponentially weighted gradient sum
For v_t=βv_{t-1}+g_t with v_0=0,
v_t=Σ_{k=0}^{t-1}β^k g_{t-k}.
Thus older gradients receive exponentially decreasing weights. The normalized convention v_t=βv_{t-1}+(1-β)g_t gives an exponential moving average.

### 5. AdamW from Adam
Adam applies θ_{t+1}=θ_t-η mhat_t/(sqrt(vhat_t)+ε). If L2 regularization is inserted into the gradient, the shrinkage term is adaptively rescaled coordinate-wise. AdamW instead decouples decay:
θ_{t+1}=(1-ηλ)θ_t-η mhat_t/(sqrt(vhat_t)+ε).
This makes weight decay independent of adaptive gradient scaling.

### 6. Newton update
The second-order Taylor model around x_k is
f(x_k+p)≈f(x_k)+g_k^T p+(1/2)p^T H_k p.
Setting its gradient with respect to p to zero gives H_k p=-g_k, so
x_{k+1}=x_k-H_k^{-1}g_k
when H_k is nonsingular.

### 7. L-BFGS memory complexity
Full BFGS stores O(d²) matrix state. L-BFGS stores m displacement pairs s_k and y_k, each pair containing two d-vectors, so history memory is approximately 2md=O(md), plus a small number of work vectors. The two-loop recursion is also O(md).

### 8. Fisher information for a softmax classifier
For p_k(x;θ) and score s(y,x)=∇_θ log p_y(x;θ),
F(θ)=E_x E_{y~p_θ(.|x)}[s s^T].
For a linear softmax model with logits W^T x, the class covariance is C=diag(p)-pp^T and the per-example Fisher is (xx^T)⊗C.

### 9. Natural-gradient invariance
Under a smooth invertible reparameterization φ=h(θ) with Jacobian J=∂θ/∂φ, gradients satisfy g_φ=J^Tg_θ and Fisher matrices satisfy F_φ=J^T F_θ J. Substitution into Δφ=-ηF_φ^{-1}g_φ and mapping back gives the same infinitesimal model-space update Δθ=-ηF_θ^{-1}g_θ.

### 10. Proximal operator for the L1 norm
Solve min_x (1/2)(x-v)²+λ|x|. The solution is
prox_{λ||.||_1}(v)=sign(v)max(|v|-λ,0),
the soft-thresholding operator.

### 11. ADMM updates for LASSO
Write min_{x,z} (1/2)||Ax-b||²+λ||z||_1 subject to x=z. With scaled dual u and penalty ρ:
x^{k+1}=(A^T A+ρI)^{-1}(A^T b+ρ(z^k-u^k)),
z^{k+1}=S_{λ/ρ}(x^{k+1}+u^k),
u^{k+1}=u^k+x^{k+1}-z^{k+1}.
Primal and dual residuals determine convergence.

### 12. PPO clipped surrogate
Let r_t(θ)=π_θ(a_t|s_t)/π_old(a_t|s_t) and advantage Ahat_t. PPO uses
L^CLIP(θ)=E[min(r_t Ahat_t, clip(r_t,1-ε,1+ε)Ahat_t)].
The clipping prevents the surrogate from rewarding excessively large probability-ratio changes, approximating a trust region with a first-order method.

### 13. Composite PINN loss for Burgers equation
For u_t+u u_x-νu_xx=0, define residual r_θ=u_t+u_θ u_x-νu_xx. A standard objective is
L=λ_d L_data+λ_f L_physics+λ_b L_boundary+λ_i L_initial,
where L_physics is the mean squared residual at collocation points, and the other terms enforce observations, boundary values, and the initial state.

### 14. Ring all-reduce communication complexity
For p workers and N bytes of gradients, ring all-reduce performs reduce-scatter and all-gather, each with p-1 steps and N/p bytes per worker per step. Each worker therefore communicates
2N(p-1)/p
bytes. A latency-bandwidth model gives approximately 2(p-1)α+2N(p-1)β/p.

### 15. Optimizer-state memory complexity
For d parameters: SGD adds no persistent optimizer tensor; Momentum adds d state values; AdamW adds 2d; Lion adds approximately d; Adafactor reduces a matrix second moment from O(mn) to O(m+n) through factorization. Mixed-precision systems may also retain FP32 master weights and gradients.
