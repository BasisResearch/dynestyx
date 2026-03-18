"""Multi-output Gaussian Process Regression with moment matching.

Implements the GP dynamics model and analytic moment matching for propagating
Gaussian uncertainty through the GP, following Deisenroth & Rasmussen (2011),
Equations 14-23.

The GP model can be converted to a dynestyx ``DiscreteTimeStateEvolution``
via :meth:`MGPR.to_state_evolution`, enabling integration with effectful
handlers (``Simulator``, ``Filter``, etc.).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array

from dynestyx.models.core import DiscreteTimeStateEvolution
from dynestyx.types import Control, State, Time


class MGPR(eqx.Module):
    """Multi-output GP Regression using D independent GPs with SE-ARD kernels.

    Each output dimension has its own GP with independent hyperparameters.
    The GPs predict state deltas: $\\Delta_t = x_t - x_{t-1}$.

    For uncertain (Gaussian) inputs, the exact first and second moments of the
    predictive distribution are computed analytically via moment matching
    (Deisenroth & Rasmussen, 2011, Section 2.2).

    Attributes:
        X: Training inputs, shape ``(n, D+F)``.
        Y: Training targets (state deltas), shape ``(n, D)``.
        log_lengthscales: Log length-scales per GP, shape ``(D, D+F)``.
        log_signal_variance: Log signal variance per GP, shape ``(D,)``.
        log_noise_variance: Log noise variance per GP, shape ``(D,)``.
    """

    X: Array
    Y: Array
    log_lengthscales: Array
    log_signal_variance: Array
    log_noise_variance: Array

    def __init__(self, X: Array, Y: Array):
        n, input_dim = X.shape
        _, output_dim = Y.shape
        self.X = X
        self.Y = Y
        self.log_lengthscales = jnp.zeros((output_dim, input_dim))
        self.log_signal_variance = jnp.zeros(output_dim)
        self.log_noise_variance = jnp.full(output_dim, -2.0)

    @property
    def lengthscales(self) -> Array:
        return jnp.exp(jnp.clip(self.log_lengthscales, -5.0, 5.0))

    @property
    def signal_variance(self) -> Array:
        return jnp.exp(jnp.clip(self.log_signal_variance, -5.0, 5.0))

    @property
    def noise_variance(self) -> Array:
        return jnp.exp(jnp.clip(self.log_noise_variance, -8.0, 2.0))

    @property
    def state_dim(self) -> int:
        return self.Y.shape[1]

    def set_data(self, X: Array, Y: Array) -> "MGPR":
        """Return a new MGPR with updated training data, preserving hyperparameters."""
        return eqx.tree_at(lambda m: (m.X, m.Y), self, (X, Y))

    def _kernel_matrix(self, X1: Array, X2: Array, a: int) -> Array:
        """SE-ARD kernel matrix for output dimension a."""
        ls = self.lengthscales[a]
        sv = self.signal_variance[a]
        diff = (X1[:, None, :] - X2[None, :, :]) / ls[None, None, :]
        return sv * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1))

    def _compute_factorizations(self, a: int) -> tuple[Array, Array]:
        """Compute K_inv and beta for output dimension a."""
        K = self._kernel_matrix(self.X, self.X, a)
        n = K.shape[0]
        noise = jnp.maximum(self.noise_variance[a], 1e-4)
        Ky = K + noise * jnp.eye(n) + 1e-6 * jnp.eye(n)
        L = jnp.linalg.cholesky(Ky)
        iK = jax.scipy.linalg.cho_solve((L, True), jnp.eye(n))
        beta = iK @ self.Y[:, a]
        return iK, beta

    def predict(self, x_star: Array) -> tuple[Array, Array]:
        """GP posterior prediction for a deterministic test input.

        Args:
            x_star: Test input, shape ``(D+F,)``.

        Returns:
            mean: Predictive mean, shape ``(D,)``.
            var: Predictive variance, shape ``(D,)``.
        """
        D = self.Y.shape[1]
        means = jnp.zeros(D)
        variances = jnp.zeros(D)
        for a in range(D):
            iK, beta = self._compute_factorizations(a)
            k_star = self._kernel_matrix(self.X, x_star[None, :], a).squeeze(-1)
            mean_a = k_star @ beta
            var_a = self.signal_variance[a] - k_star @ iK @ k_star
            means = means.at[a].set(mean_a)
            variances = variances.at[a].set(jnp.maximum(var_a, 1e-12))
        return means, variances

    def predict_given_factorizations(
        self, m: Array, s: Array
    ) -> tuple[Array, Array, Array]:
        """Moment matching: predict through GP with uncertain Gaussian input.

        Given $p(\\tilde{x}) = \\mathcal{N}(m, s)$, compute the mean, covariance,
        and input-output cross-covariance of the GP predictive distribution,
        approximated as Gaussian by exact moment matching.

        Implements Eqs. 14-23 of Deisenroth & Rasmussen (2011).

        Args:
            m: Mean of input distribution, shape ``(D+F,)``.
            s: Covariance of input distribution, shape ``(D+F, D+F)``.

        Returns:
            M: Predictive mean, shape ``(D,)``.
            S: Predictive covariance, shape ``(D, D)``.
            V: Input-output cross-covariance, shape ``(D+F, D)``.
        """
        D = self.Y.shape[1]
        input_dim = self.X.shape[1]

        iKs = []
        betas = []
        for a in range(D):
            iK, beta = self._compute_factorizations(a)
            iKs.append(iK)
            betas.append(beta)

        nu = self.X - m[None, :]  # (n, D+F)

        # === Mean prediction (Eq. 14-16) ===
        M = jnp.zeros(D)
        qs = []
        for a in range(D):
            Lambda_a = jnp.diag(self.lengthscales[a] ** 2)
            sL = s + Lambda_a
            sL_inv = jnp.linalg.inv(sL + 1e-8 * jnp.eye(input_dim))
            det_factor = jnp.linalg.det(sL) / jnp.prod(self.lengthscales[a] ** 2)
            quad = jnp.sum(nu @ sL_inv * nu, axis=-1)
            q = (
                self.signal_variance[a]
                / jnp.sqrt(jnp.maximum(det_factor, 1e-12))
                * jnp.exp(-0.5 * quad)
            )
            qs.append(q)
            M = M.at[a].set(betas[a] @ q)

        # === Covariance prediction (Eq. 17-23) ===
        S = jnp.zeros((D, D))
        for a in range(D):
            Lambda_a_inv = jnp.diag(1.0 / self.lengthscales[a] ** 2)
            for b in range(a, D):
                Lambda_b_inv = jnp.diag(1.0 / self.lengthscales[b] ** 2)
                R = s @ (Lambda_a_inv + Lambda_b_inv) + jnp.eye(input_dim)
                R_inv = jnp.linalg.inv(R + 1e-8 * jnp.eye(input_dim))
                det_R = jnp.linalg.det(R)

                nu_La = nu @ Lambda_a_inv
                nu_Lb = nu @ Lambda_b_inv
                k_a_m = self.signal_variance[a] * jnp.exp(
                    -0.5 * jnp.sum(nu * nu_La, axis=-1)
                )
                k_b_m = self.signal_variance[b] * jnp.exp(
                    -0.5 * jnp.sum(nu * nu_Lb, axis=-1)
                )

                R_inv_s = R_inv @ s
                t_a = nu_La @ R_inv_s
                t_b = nu_Lb @ R_inv_s
                q_aa = jnp.sum(nu_La * t_a, axis=-1)
                q_bb = jnp.sum(nu_Lb * t_b, axis=-1)
                q_ab = nu_La @ t_b.T

                Q = (
                    k_a_m[:, None]
                    * k_b_m[None, :]
                    / jnp.sqrt(jnp.maximum(det_R, 1e-12))
                    * jnp.exp(0.5 * (q_aa[:, None] + q_bb[None, :] + 2.0 * q_ab))
                )

                S_ab = betas[a] @ Q @ betas[b] - M[a] * M[b]
                if a == b:
                    S_ab += self.signal_variance[a] - jnp.trace(iKs[a] @ Q)

                S = S.at[a, b].set(S_ab)
                if a != b:
                    S = S.at[b, a].set(S_ab)

        # Ensure diagonal is non-negative (numerical stability)
        S = S + 1e-6 * jnp.eye(D)

        # === Input-output cross-covariance V ===
        V = jnp.zeros((input_dim, D))
        for a in range(D):
            Lambda_a = jnp.diag(self.lengthscales[a] ** 2)
            sL = s + Lambda_a
            sL_inv = jnp.linalg.inv(sL + 1e-8 * jnp.eye(input_dim))
            weighted_nu = (betas[a] * qs[a])[:, None] * nu
            V = V.at[:, a].set(s @ sL_inv @ jnp.sum(weighted_nu, axis=0))

        return M, S, V

    def log_marginal_likelihood(self) -> Array:
        """Log marginal likelihood for hyperparameter optimization."""
        D = self.Y.shape[1]
        n = self.X.shape[0]
        total = 0.0
        for a in range(D):
            K = self._kernel_matrix(self.X, self.X, a)
            Ky = K + self.noise_variance[a] * jnp.eye(n)
            L = jnp.linalg.cholesky(Ky + 1e-6 * jnp.eye(n))
            alpha = jax.scipy.linalg.cho_solve((L, True), self.Y[:, a])
            lml = (
                -0.5 * self.Y[:, a] @ alpha
                - jnp.sum(jnp.log(jnp.diag(L)))
                - 0.5 * n * jnp.log(2.0 * jnp.pi)
            )
            total = total + lml
        return total

    def to_state_evolution(self) -> "GPStateEvolution":
        """Convert GP to a dynestyx ``DiscreteTimeStateEvolution``.

        Returns a ``GPStateEvolution`` whose ``__call__`` returns a
        ``MultivariateNormal`` distribution for the next state, enabling
        this GP model to be used inside a ``DynamicalModel`` with dynestyx
        handlers (``DiscreteTimeSimulator``, ``Filter``, etc.).
        """
        return GPStateEvolution(mgpr=self)


class GPStateEvolution(DiscreteTimeStateEvolution):
    """Dynestyx-compatible discrete-time state evolution wrapping a GP model.

    Given the current state $x$ and action $u$, the GP predicts:
    $$x_{t+1} \\sim \\mathcal{N}(x + \\mu_\\Delta(\\tilde{x}), \\Sigma_\\Delta(\\tilde{x}))$$

    where $\\tilde{x} = [x^\\top, u^\\top]^\\top$ and $\\mu_\\Delta, \\Sigma_\\Delta$
    are the GP posterior mean and variance for the state delta.

    This allows the GP dynamics to be used inside ``DynamicalModel`` with effectful
    handlers like ``DiscreteTimeSimulator`` or ``Filter``.
    """

    def __init__(self, mgpr: MGPR):
        self.mgpr = mgpr

    @property
    def control_dim(self) -> int:
        """Infer control dimension from GP input vs output dimensions."""
        return self.mgpr.X.shape[1] - self.mgpr.state_dim

    def __call__(
        self,
        x: State,
        u: Control | None,
        t_now: Time,
        t_next: Time,
    ):
        # Build GP input: [state, action]
        # If u is None but GP expects controls, pad with zeros
        if u is not None:
            x_tilde = jnp.concatenate([x, jnp.atleast_1d(u)])
        elif self.control_dim > 0:
            x_tilde = jnp.concatenate([x, jnp.zeros(self.control_dim)])
        else:
            x_tilde = x

        mean_delta, var_delta = self.mgpr.predict(x_tilde)
        return dist.MultivariateNormal(
            loc=x + mean_delta,
            covariance_matrix=jnp.diag(var_delta),
        )
