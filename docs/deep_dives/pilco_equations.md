# PILCO: Complete Mathematical Derivation

All equations from Deisenroth & Rasmussen (2011), *PILCO: A Model-Based and
Data-Efficient Approach to Policy Search*, ICML.

---

## 1. Problem Setup

### System dynamics (Eq. 1)

$$x_t = f(x_{t-1}, u_{t-1})$$

where $x \in \mathbb{R}^D$ is the state and $u \in \mathbb{R}^F$ is the control.

### Objective (Eq. 2)

$$J^\pi(\theta) = \sum_{t=0}^{T} \mathbb{E}_{x_t}[c(x_t)], \quad x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

Minimize expected cumulative cost under policy $\pi$ parameterized by $\theta$.

---

## 2. GP Dynamics Model (Section 2.1)

### Training data

- Inputs: $\tilde{x}_i = [x_{t-1}^\top, u_{t-1}^\top]^\top \in \mathbb{R}^{D+F}$
- Targets: $\Delta_t = x_t - x_{t-1} + \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, \Sigma_\varepsilon)$

### Predictive distribution (Eqs. 3-5)

$$p(x_t \mid x_{t-1}, u_{t-1}) = \mathcal{N}(x_t \mid \mu_t, \Sigma_t)$$

$$\mu_t = x_{t-1} + \mathbb{E}_f[\Delta_t] \tag{4}$$

$$\Sigma_t = \text{var}_f[\Delta_t] \tag{5}$$

### SE-ARD kernel (Eq. 6)

$$k(\tilde{x}, \tilde{x}') = \alpha^2 \exp\!\left(-\frac{1}{2}(\tilde{x} - \tilde{x}')^\top \Lambda^{-1} (\tilde{x} - \tilde{x}')\right) \tag{6}$$

where $\Lambda = \text{diag}(\ell_1^2, \ldots, \ell_{D+F}^2)$ and $\alpha^2$ is the signal variance.

### GP posterior (Eqs. 7-8)

$$m_f(\tilde{x}_*) = k_*^\top \beta, \quad \beta = (K + \sigma_\varepsilon^2 I)^{-1} y \tag{7}$$

$$\sigma_f^2(\tilde{x}_*) = k_{**} - k_*^\top (K + \sigma_\varepsilon^2 I)^{-1} k_* \tag{8}$$

---

## 3. Moment Matching for Uncertain Inputs (Section 2.2)

### Predictive distribution under uncertainty (Eq. 9)

$$p(\Delta_t) = \int p(f(\tilde{x}_{t-1}) \mid \tilde{x}_{t-1})\, p(\tilde{x}_{t-1})\, d\tilde{x}_{t-1} \tag{9}$$

Approximated as Gaussian by exact moment matching.

### State update (Eqs. 10-12)

$$\mu_t = \mu_{t-1} + \mu_\Delta \tag{10}$$

$$\Sigma_t = \Sigma_{t-1} + \Sigma_\Delta + \text{cov}[x_{t-1}, \Delta_t] + \text{cov}[\Delta_t, x_{t-1}] \tag{11}$$

$$\text{cov}[x_{t-1}, \Delta_t] = \text{cov}[x_{t-1}, u_{t-1}]\, \Sigma_u^{-1}\, \text{cov}[u_{t-1}, \Delta_t] \tag{12}$$

---

### 3a. Mean Prediction (Section 2.2.1)

#### Predictive mean (Eq. 14)

$$\mu_\Delta^a = \beta_a^\top q_a \tag{14}$$

#### Expected kernel evaluations (Eq. 15)

$$q_{ai} = \frac{\alpha_a^2}{\sqrt{|\tilde{\Sigma}_{t-1} \Lambda_a^{-1} + I|}}\, \exp\!\left(-\frac{1}{2} \nu_i^\top (\tilde{\Sigma}_{t-1} + \Lambda_a)^{-1} \nu_i\right) \tag{15}$$

#### Centered training inputs (Eq. 16)

$$\nu_i = \tilde{x}_i - \tilde{\mu}_{t-1} \tag{16}$$

---

### 3b. Covariance Prediction (Section 2.2.2)

#### Diagonal elements (Eq. 17)

$$\sigma_{aa}^2 = \mathbb{E}_{\tilde{x}_{t-1}}[\text{var}_f[\Delta_a \mid \tilde{x}_{t-1}]] + \mathbb{E}_{f, \tilde{x}_{t-1}}[\Delta_a^2] - (\mu_\Delta^a)^2 \tag{17}$$

#### Off-diagonal elements (Eq. 18)

$$\sigma_{ab}^2 = \mathbb{E}_{f, \tilde{x}_{t-1}}[\Delta_a \Delta_b] - \mu_\Delta^a \mu_\Delta^b, \quad a \ne b \tag{18}$$

#### Cross-expectation (Eq. 20)

$$\mathbb{E}_{f, \tilde{x}_{t-1}}[\Delta_a \Delta_b] = \beta_a^\top Q \beta_b \tag{20}$$

#### Q matrix entries (Eq. 22)

$$Q_{ij} = \frac{k_a(\tilde{x}_i, \tilde{\mu})\, k_b(\tilde{x}_j, \tilde{\mu})}{\sqrt{|R|}}\, \exp\!\left(\frac{1}{2} z_{ij}^\top R^{-1} \tilde{\Sigma} z_{ij}\right) \tag{22}$$

where:

$$R = \tilde{\Sigma}_{t-1} (\Lambda_a^{-1} + \Lambda_b^{-1}) + I$$

$$z_{ij} = \Lambda_a^{-1} \nu_i + \Lambda_b^{-1} \nu_j$$

#### Expected predictive variance (Eq. 23)

$$\mathbb{E}_{\tilde{x}_{t-1}}[\text{var}_f[\Delta_a \mid \tilde{x}_{t-1}]] = \alpha_a^2 - \text{tr}\!\left((K_a + \sigma_{\varepsilon,a}^2 I)^{-1} Q\right) \tag{23}$$

---

### Covariance summary

$$\Sigma_\Delta[a, b] = \begin{cases}
\alpha_a^2 - \text{tr}(K_a^{-1} Q) + \beta_a^\top Q \beta_a - (\mu_\Delta^a)^2 & \text{if } a = b \\
\beta_a^\top Q \beta_b - \mu_\Delta^a \mu_\Delta^b & \text{if } a \ne b
\end{cases}$$

---

## 4. Cost Function (Section 2.2, Eq. 25)

### Saturating cost

$$c(x) = 1 - \exp\!\left(-\frac{\|x - x_\text{target}\|^2}{\sigma_c^2}\right) \tag{25}$$

### Expected cost under Gaussian state (Eq. 24)

$$\mathbb{E}_{x_t}[c(x_t)] = 1 - \frac{1}{\sqrt{|\Sigma_t / \sigma_c^2 + I|}}\, \exp\!\left(-\frac{1}{2}(\mu_t - x_\text{target})^\top (\Sigma_t + \sigma_c^2 I)^{-1} (\mu_t - x_\text{target})\right) \tag{24}$$

### Exponential reward (used in implementation)

$$r(x) = \exp\!\left(-\frac{1}{2}(x - x_\text{target})^\top W (x - x_\text{target})\right)$$

$$\mathbb{E}[r(x)] = \frac{\exp\!\left(-\frac{1}{2}(\mu - x_\text{target})^\top (I + \Sigma W)^{-1} W (\mu - x_\text{target})\right)}{\sqrt{|I + \Sigma W|}}$$

---

## 5. Policy Optimization (Section 2.3)

### Gradient chain rule (Eqs. 26-30)

$$\frac{\partial \mathbb{E}_t}{\partial \theta} = \frac{\partial \mathbb{E}_t}{\partial \mu_t} \frac{\partial \mu_t}{\partial \theta} + \frac{\partial \mathbb{E}_t}{\partial \Sigma_t} \frac{\partial \Sigma_t}{\partial \theta} \tag{26}$$

$$\frac{\partial p(x_t)}{\partial \theta} = \frac{\partial p(x_t)}{\partial p(x_{t-1})} \frac{\partial p(x_{t-1})}{\partial \theta} + \frac{\partial p(x_t)}{\partial \theta}\bigg|_{\text{partial}} \tag{27}$$

$$\frac{\partial p(x_t)}{\partial p(x_{t-1})} = \left\{\frac{\partial \mu_t}{\partial p(x_{t-1})},\, \frac{\partial \Sigma_t}{\partial p(x_{t-1})}\right\} \tag{28}$$

$$\frac{\partial \mu_t}{\partial \theta} = \frac{\partial \mu_t}{\partial \mu_{t-1}} \frac{\partial \mu_{t-1}}{\partial \theta} + \frac{\partial \mu_t}{\partial \Sigma_{t-1}} \frac{\partial \Sigma_{t-1}}{\partial \theta} + \frac{\partial \mu_t}{\partial \theta}\bigg|_{\text{partial}} \tag{29}$$

$$\frac{\partial \mu_t}{\partial \theta}\bigg|_{\text{partial}} = \frac{\partial \mu_\Delta}{\partial \mu_u} \frac{\partial \mu_u}{\partial \theta} + \frac{\partial \mu_\Delta}{\partial \Sigma_u} \frac{\partial \Sigma_u}{\partial \theta} \tag{30}$$

In practice, these gradients are computed automatically via JAX autodiff through the entire moment matching computation graph.

---

## 6. Controller Parameterization

### RBF network (Eqs. 31-32)

$$\pi(x, \theta) = \sum_{i=1}^{n} w_i \phi_i(x) \tag{31}$$

$$\phi_i(x) = \exp\!\left(-\frac{1}{2}(x - \mu_i)^\top \Lambda^{-1} (x - \mu_i)\right) \tag{32}$$

### Linear controller

$$\pi(x) = Wx + b$$

For Gaussian inputs $x \sim \mathcal{N}(\mu, \Sigma)$:

$$\mu_u = W\mu + b, \quad \Sigma_u = W \Sigma W^\top, \quad \text{cov}[x, u] = \Sigma W^\top$$

### Sin squashing (bounded controls)

$$u_\text{sat} = u_\text{max} \sin(\pi(x))$$

For Gaussian $u \sim \mathcal{N}(m, s)$:

$$\mathbb{E}[\sin(u)] = \exp(-\text{diag}(s)/2) \odot \sin(m)$$

---

## 7. Full Algorithm (Algorithm 1)

```
Input: Initial state distribution N(mu_0, Sigma_0)
1. Sample theta ~ N(0, I), apply random controls, collect data D
2. repeat
3.   Learn GP dynamics from ALL data D (maximize log marginal likelihood)
4.   repeat (policy search)
5.     Compute J^pi(theta) via Eqs. (10)-(12), (14)-(23), (24)
6.     Compute dJ^pi/dtheta via Eqs. (26)-(30) [or autodiff]
7.     Update theta using CG or L-BFGS [or Adam]
8.   until convergence
9.   Set pi* = pi(theta*)
10.  Apply pi* to real system, record new data -> augment D
11. until task learned
```
