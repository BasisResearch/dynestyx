"""Multi-output GP regression with moment matching (Deisenroth & Rasmussen, 2011)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array

from dynestyx.models.core import DiscreteTimeStateEvolution
from dynestyx.pilco.moment_matching import (
    compute_predictive_moments,
    gp_log_marginal_likelihood,
)
from dynestyx.types import Control, State, Time


class MGPR(eqx.Module):
    """
    Multi-output GP Regression with SE-ARD kernels.

    Trains $D$ independent GPs on inputs $\\tilde{x} = [x^\\top, u^\\top]^\\top$
    and targets $\\Delta = x_{t+1} - x_t$. Converts to a dynestyx
    ``DiscreteTimeStateEvolution`` via ``to_state_evolution()``.
    """

    X: Array
    Y: Array
    log_lengthscales: Array
    log_signal_variance: Array
    log_noise_variance: Array

    def __init__(self, X: Array, Y: Array):
        _, input_dim = X.shape
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
        return eqx.tree_at(lambda m: (m.X, m.Y), self, (X, Y))

    def _kernel_matrix(self, X1: Array, X2: Array, a: int) -> Array:
        ls = self.lengthscales[a]
        diff = (X1[:, None, :] - X2[None, :, :]) / ls[None, None, :]
        return self.signal_variance[a] * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1))

    def _compute_factorizations(self, a: int) -> tuple[Array, Array]:
        K = self._kernel_matrix(self.X, self.X, a)
        n = K.shape[0]
        Ky = (
            K
            + jnp.maximum(self.noise_variance[a], 1e-4) * jnp.eye(n)
            + 1e-6 * jnp.eye(n)
        )
        L = jnp.linalg.cholesky(Ky)
        iK = jax.scipy.linalg.cho_solve((L, True), jnp.eye(n))
        return iK, iK @ self.Y[:, a]

    def _all_factorizations(self) -> tuple[list[Array], list[Array]]:
        iKs, betas = [], []
        for a in range(self.state_dim):
            iK, beta = self._compute_factorizations(a)
            iKs.append(iK)
            betas.append(beta)
        return iKs, betas

    def _predict_deterministic(self, x_star: Array) -> tuple[Array, Array]:
        """GP posterior mean/variance for a deterministic input."""
        D = self.state_dim
        means = jnp.zeros(D)
        variances = jnp.zeros(D)
        for a in range(D):
            iK, beta = self._compute_factorizations(a)
            k_star = self._kernel_matrix(self.X, x_star[None, :], a).squeeze(-1)
            means = means.at[a].set(k_star @ beta)
            variances = variances.at[a].set(
                jnp.maximum(self.signal_variance[a] - k_star @ iK @ k_star, 1e-12)
            )
        return means, variances

    def predict_uncertain(self, m: Array, s: Array) -> tuple[Array, Array, Array]:
        """Moment matching for uncertain Gaussian input (Eqs. 14-23)."""
        iKs, betas = self._all_factorizations()
        nu = self.X - m[None, :]
        return compute_predictive_moments(
            nu, s, self.lengthscales, self.signal_variance, betas, iKs
        )

    def log_marginal_likelihood(self) -> Array:
        return gp_log_marginal_likelihood(
            self._kernel_matrix, self.X, self.Y, self.noise_variance
        )

    def to_state_evolution(self) -> "GPStateEvolution":
        """Convert to a dynestyx ``DiscreteTimeStateEvolution``."""
        return GPStateEvolution(mgpr=self)


def _build_gp_input(x: State, u: Control | None, control_dim: int) -> Array:
    """Build GP input $\\tilde{x} = [x^\\top, u^\\top]^\\top$, padding zeros if needed."""
    if u is not None:
        return jnp.concatenate([x, jnp.atleast_1d(u)])
    if control_dim > 0:
        return jnp.concatenate([x, jnp.zeros(control_dim)])
    return x


class GPStateEvolution(DiscreteTimeStateEvolution):
    """GP dynamics as a dynestyx ``DiscreteTimeStateEvolution``."""

    def __init__(self, mgpr: MGPR):
        self.mgpr = mgpr

    @property
    def control_dim(self) -> int:
        return self.mgpr.X.shape[1] - self.mgpr.state_dim

    def __call__(self, x: State, u: Control | None, t_now: Time, t_next: Time):
        x_tilde = _build_gp_input(x, u, self.control_dim)
        mean_delta, var_delta = self.mgpr._predict_deterministic(x_tilde)
        return dist.MultivariateNormal(
            loc=x + mean_delta, covariance_matrix=jnp.diag(var_delta)
        )
