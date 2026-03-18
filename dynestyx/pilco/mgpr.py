"""Multi-output Gaussian Process Regression with moment matching.

Implements the GP dynamics model and analytic moment matching for propagating
Gaussian uncertainty through the GP, following Deisenroth & Rasmussen (2011),
Equations 14-23.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class MGPR(eqx.Module):
    """Multi-output GP Regression using D independent GPs with SE-ARD kernels.

    Each output dimension has its own GP with independent hyperparameters.
    The GPs predict state deltas: Delta_t = x_t - x_{t-1}.

    For uncertain (Gaussian) inputs, the exact first and second moments of the
    predictive distribution are computed analytically via moment matching
    (Deisenroth & Rasmussen, 2011, Section 2.2).

    Attributes:
        X: Training inputs, shape (n, D+F).
        Y: Training targets (state deltas), shape (n, D).
        log_lengthscales: Log length-scales per GP, shape (D, D+F).
        log_signal_variance: Log signal variance per GP, shape (D,).
        log_noise_variance: Log noise variance per GP, shape (D,).
    """

    X: Array
    Y: Array
    log_lengthscales: Array
    log_signal_variance: Array
    log_noise_variance: Array

    def __init__(self, X: Array, Y: Array):
        """Initialize MGPR with training data.

        Args:
            X: Training inputs, shape (n, D+F).
            Y: Training targets (state deltas), shape (n, D).
        """
        n, input_dim = X.shape
        _, output_dim = Y.shape

        self.X = X
        self.Y = Y
        self.log_lengthscales = jnp.zeros((output_dim, input_dim))
        self.log_signal_variance = jnp.zeros(output_dim)
        self.log_noise_variance = jnp.full(output_dim, -2.0)

    @property
    def lengthscales(self) -> Array:
        return jnp.exp(self.log_lengthscales)

    @property
    def signal_variance(self) -> Array:
        return jnp.exp(self.log_signal_variance)

    @property
    def noise_variance(self) -> Array:
        return jnp.exp(self.log_noise_variance)

    def set_data(self, X: Array, Y: Array) -> "MGPR":
        """Return a new MGPR with updated training data, preserving hyperparameters."""
        return eqx.tree_at(lambda m: (m.X, m.Y), self, (X, Y))

    def _kernel_matrix(self, X1: Array, X2: Array, a: int) -> Array:
        """SE-ARD kernel matrix for output dimension a.

        Args:
            X1: shape (n1, D+F)
            X2: shape (n2, D+F)
            a: output dimension index

        Returns:
            Kernel matrix, shape (n1, n2).
        """
        ls = self.lengthscales[a]  # (D+F,)
        sv = self.signal_variance[a]
        diff = (X1[:, None, :] - X2[None, :, :]) / ls[None, None, :]  # (n1, n2, D+F)
        return sv * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1))

    def _compute_factorizations(self, a: int) -> tuple[Array, Array]:
        """Compute K_inv and beta for output dimension a.

        Returns:
            iK: (K + noise*I)^{-1}, shape (n, n).
            beta: iK @ Y[:, a], shape (n,).
        """
        K = self._kernel_matrix(self.X, self.X, a)
        n = K.shape[0]
        Ky = K + self.noise_variance[a] * jnp.eye(n)
        L = jnp.linalg.cholesky(Ky + 1e-6 * jnp.eye(n))
        iK = jax.scipy.linalg.cho_solve((L, True), jnp.eye(n))
        beta = iK @ self.Y[:, a]
        return iK, beta

    def predict(self, x_star: Array) -> tuple[Array, Array]:
        """GP posterior prediction for a deterministic test input.

        Args:
            x_star: Test input, shape (D+F,).

        Returns:
            mean: Predictive mean, shape (D,).
            var: Predictive variance, shape (D,).
        """
        D = self.Y.shape[1]
        means = jnp.zeros(D)
        variances = jnp.zeros(D)
        for a in range(D):
            iK, beta = self._compute_factorizations(a)
            k_star = self._kernel_matrix(self.X, x_star[None, :], a).squeeze(-1)  # (n,)
            mean_a = k_star @ beta
            var_a = self.signal_variance[a] - k_star @ iK @ k_star
            means = means.at[a].set(mean_a)
            variances = variances.at[a].set(jnp.maximum(var_a, 1e-12))
        return means, variances

    def predict_given_factorizations(
        self, m: Array, s: Array
    ) -> tuple[Array, Array, Array]:
        """Moment matching: predict through GP with uncertain Gaussian input.

        Given p(x_tilde) = N(m, s), compute the mean, covariance, and
        input-output cross-covariance of the GP predictive distribution,
        approximated as Gaussian by exact moment matching.

        Implements Eqs. 14-23 of Deisenroth & Rasmussen (2011).

        Args:
            m: Mean of input distribution, shape (D+F,).
            s: Covariance of input distribution, shape (D+F, D+F).

        Returns:
            M: Predictive mean, shape (D,).
            S: Predictive covariance, shape (D, D).
            V: Input-output cross-covariance, shape (D+F, D).
        """
        D = self.Y.shape[1]
        input_dim = self.X.shape[1]
        n = self.X.shape[0]

        # Pre-compute factorizations for all output dims
        iKs = []
        betas = []
        for a in range(D):
            iK, beta = self._compute_factorizations(a)
            iKs.append(iK)
            betas.append(beta)

        # Centralized training inputs: nu_i = X_i - m
        nu = self.X - m[None, :]  # (n, D+F)

        # === Mean prediction (Eq. 14-16) ===
        M = jnp.zeros(D)
        qs = []  # store q vectors for cross-covariance
        for a in range(D):
            Lambda_a = jnp.diag(self.lengthscales[a] ** 2)  # (D+F, D+F)
            # (s + Lambda_a)^{-1}
            sL = s + Lambda_a
            sL_inv = jnp.linalg.inv(sL + 1e-8 * jnp.eye(input_dim))
            # Determinant factor: |s @ Lambda_a^{-1} + I| = |sL| / |Lambda_a|
            det_factor = jnp.linalg.det(sL) / jnp.prod(self.lengthscales[a] ** 2)
            # q_ai = alpha^2 / sqrt(det) * exp(-0.5 * nu_i^T @ sL^{-1} @ nu_i)
            # Vectorized over training points
            quad = jnp.sum(nu @ sL_inv * nu, axis=-1)  # (n,)
            q = self.signal_variance[a] / jnp.sqrt(jnp.maximum(det_factor, 1e-12)) * jnp.exp(-0.5 * quad)
            qs.append(q)
            M = M.at[a].set(betas[a] @ q)

        # === Covariance prediction (Eq. 17-23) ===
        S = jnp.zeros((D, D))
        for a in range(D):
            Lambda_a = jnp.diag(self.lengthscales[a] ** 2)
            Lambda_a_inv = jnp.diag(1.0 / self.lengthscales[a] ** 2)
            for b in range(a, D):
                Lambda_b = jnp.diag(self.lengthscales[b] ** 2)
                Lambda_b_inv = jnp.diag(1.0 / self.lengthscales[b] ** 2)

                # R = s @ (Lambda_a^{-1} + Lambda_b^{-1}) + I  (Eq. 22)
                R = s @ (Lambda_a_inv + Lambda_b_inv) + jnp.eye(input_dim)
                R_inv = jnp.linalg.inv(R + 1e-8 * jnp.eye(input_dim))
                det_R = jnp.linalg.det(R)

                # z_ij = Lambda_a^{-1} @ nu_i + Lambda_b^{-1} @ nu_j
                # Q_ij = k_a(X_i, m) * k_b(X_j, m) / sqrt(|R|) *
                #        exp(0.5 * z_ij^T @ R^{-1} @ s @ z_ij)
                # where k_a(X_i, m) uses only the deterministic kernel eval

                # Efficient vectorized computation of Q
                nu_La = nu @ Lambda_a_inv  # (n, D+F)
                nu_Lb = nu @ Lambda_b_inv  # (n, D+F)

                # k_a(X_i, m) = alpha_a^2 * exp(-0.5 * nu_i^T @ Lambda_a^{-1} @ nu_i)
                k_a_m = self.signal_variance[a] * jnp.exp(
                    -0.5 * jnp.sum(nu * nu_La, axis=-1)
                )  # (n,)
                k_b_m = self.signal_variance[b] * jnp.exp(
                    -0.5 * jnp.sum(nu * nu_Lb, axis=-1)
                )  # (n,)

                # z_ij = nu_La[i] + nu_Lb[j], shape would be (n, n, D+F)
                # z_ij^T @ R^{-1} @ s @ z_ij -- compute efficiently
                R_inv_s = R_inv @ s  # (D+F, D+F)

                # t_a = nu_La @ R_inv_s, t_b = nu_Lb @ R_inv_s
                t_a = nu_La @ R_inv_s  # (n, D+F)
                t_b = nu_Lb @ R_inv_s  # (n, D+F)

                # The quadratic form for z_ij:
                # z_ij^T R^{-1} s z_ij = (nu_La_i + nu_Lb_j)^T R^{-1} s (nu_La_i + nu_Lb_j)
                # = nu_La_i^T t_a_i + 2 * nu_La_i^T t_b_j + nu_Lb_j^T t_b_j
                q_aa = jnp.sum(nu_La * t_a, axis=-1)  # (n,)
                q_bb = jnp.sum(nu_Lb * t_b, axis=-1)  # (n,)
                q_ab = nu_La @ t_b.T  # (n, n)

                Q = (
                    k_a_m[:, None]
                    * k_b_m[None, :]
                    / jnp.sqrt(jnp.maximum(det_R, 1e-12))
                    * jnp.exp(0.5 * (q_aa[:, None] + q_bb[None, :] + 2.0 * q_ab))
                )  # (n, n)

                # E[Delta_a * Delta_b] = beta_a^T @ Q @ beta_b
                S_ab = betas[a] @ Q @ betas[b] - M[a] * M[b]

                if a == b:
                    # Add expected predictive variance (Eq. 23)
                    # E[var_f[Delta_a | x_tilde]] = alpha_a^2 - tr(iK_a @ Q)
                    S_ab += self.signal_variance[a] - jnp.trace(iKs[a] @ Q)

                S = S.at[a, b].set(S_ab)
                if a != b:
                    S = S.at[b, a].set(S_ab)

        # === Input-output cross-covariance V ===
        # V[:, a] = s @ (s + Lambda_a)^{-1} @ sum_i(beta_ai * nu_i * q_ai) / sum(q_ai * beta_ai)
        # Simplified: V = s @ Lambda^{-1} contribution through the q-weighted betas
        V = jnp.zeros((input_dim, D))
        for a in range(D):
            Lambda_a = jnp.diag(self.lengthscales[a] ** 2)
            sL = s + Lambda_a
            sL_inv = jnp.linalg.inv(sL + 1e-8 * jnp.eye(input_dim))
            # cov[x_tilde, Delta_a] = s @ sL^{-1} @ sum_i(beta_ai * q_ai * nu_i)
            weighted_nu = (betas[a] * qs[a])[:, None] * nu  # (n, D+F)
            V = V.at[:, a].set(s @ sL_inv @ jnp.sum(weighted_nu, axis=0))

        return M, S, V

    def log_marginal_likelihood(self) -> Array:
        """Compute the log marginal likelihood for hyperparameter optimization.

        Returns:
            Total log marginal likelihood summed over all output dimensions.
        """
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
