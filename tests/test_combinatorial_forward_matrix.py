import itertools
import json
import os
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from xml.sax.saxutils import escape as xml_escape
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx.discretizers import euler_maruyama
from dynestyx.dynamical_models import Context, Trajectory
from dynestyx.filters import (
    FilterBasedHMMMarginalLogLikelihood,
    FilterBasedMarginalLogLikelihood,
)
from dynestyx.handlers import Condition, Discretizer
from dynestyx.simulators import DiscreteTimeSimulator, ODESimulator, SDESimulator


FULL_EXHAUSTIVE = os.getenv("DYNESTYX_EXHAUSTIVE_COMBINATORIAL", "0") == "1"
SCORECARD_PATH = os.getenv(
    "DYNESTYX_COMBINATORIAL_SCORECARD_PATH",
    ".pytest_cache/dynestyx_combinatorial_scorecard.json",
)
PREDICTIVE_SCORECARD_PATH = os.getenv(
    "DYNESTYX_PREDICTIVE_SCORECARD_PATH",
    ".pytest_cache/dynestyx_predictive_scorecard.json",
)
SCORECARD_XLS_PATH = os.getenv(
    "DYNESTYX_COMBINATORIAL_SCORECARD_XLS_PATH",
    ".pytest_cache/dynestyx_combinatorial_scorecard.xls",
)
PREDICTIVE_SCORECARD_XLS_PATH = os.getenv(
    "DYNESTYX_PREDICTIVE_SCORECARD_XLS_PATH",
    ".pytest_cache/dynestyx_predictive_scorecard.xls",
)

_USE_COLOR = os.getenv("NO_COLOR") is None
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_CYAN = "\033[36m"


@dataclass(frozen=True)
class DataSpec:
    obs_rank: int  # 1 -> (T,), 2 -> (T, 2)
    timesteps: int
    ctrl_rank: int  # 0 -> none, 1 -> (T,), 2 -> (T,2)


@dataclass(frozen=True)
class ModelSpec:
    family: str  # "discrete_gaussian", "categorical_hmm", "continuous"
    initial_kind: str  # mvn, uniform, categorical
    init_rank: int  # 1 or 2
    state_impl: str  # function, eqx
    uses_control: bool
    transition_kind: str  # linear_mvn, nonlinear_mvn, categorical, linear_ct, zero_ct
    diffusion_coeff: str  # eye, none (continuous only)
    diffusion_cov: str  # eye, none (continuous only)
    observation_kind: str  # linear_gaussian, perfect, poisson
    observation_rank: int  # 1 or 2


@dataclass(frozen=True)
class InferenceSpec:
    runner: str  # discrete_sim, ode_sim, sde_sim, filter, filter_hmm
    filter_type: str | None
    discretizer: str  # none, default, euler


@dataclass
class CaseResult:
    case_id: str
    expected_pass: bool
    expected_error: str | None
    actual_pass: bool
    actual_error: str | None
    model: ModelSpec
    inference: InferenceSpec


class LinearDiscreteEqx(eqx.Module):
    uses_control: bool

    def __call__(self, x, u, t_now, t_next):
        dt = t_next - t_now
        u_term = 0.2 * u if (self.uses_control and u is not None) else 0.0
        loc = x + dt * (0.1 * x + u_term)
        if jnp.ndim(loc) == 0:
            return dist.Normal(loc=loc, scale=0.2)
        return dist.MultivariateNormal(
            loc=loc, covariance_matrix=0.04 * jnp.eye(loc.shape[-1])
        )


class NonlinearDiscreteEqx(eqx.Module):
    uses_control: bool

    def __call__(self, x, u, t_now, t_next):
        dt = t_next - t_now
        nonlin = 0.05 * jnp.sin(x)
        u_term = 0.1 * u if (self.uses_control and u is not None) else 0.0
        loc = x + dt * (nonlin + u_term)
        if jnp.ndim(loc) == 0:
            return dist.Normal(loc=loc, scale=0.2)
        return dist.MultivariateNormal(
            loc=loc, covariance_matrix=0.04 * jnp.eye(loc.shape[-1])
        )


class ContinuousDriftEqx(eqx.Module):
    drift_kind: str
    uses_control: bool

    def __call__(self, x, u, t):
        if self.drift_kind == "zero_ct":
            drift = jnp.zeros_like(x)
        else:
            drift = 0.1 * x
        u_term = 0.1 * u if (self.uses_control and u is not None) else 0.0
        return drift + u_term


def _iterable_samples(case_limit: int, combos):
    if FULL_EXHAUSTIVE:
        return list(combos)
    # Deterministic thinning for CI-friendly default runs.
    combo_list = list(combos)
    if len(combo_list) <= case_limit:
        return combo_list
    stride = max(1, len(combo_list) // case_limit)
    return combo_list[::stride][:case_limit]


def _time_grid(timesteps: int):
    if timesteps == 1:
        return jnp.array([0.0])
    return jnp.linspace(0.0, 1.0, timesteps)


def _make_data(data: DataSpec):
    times = _time_grid(data.timesteps)
    obs_dim = 1 if data.obs_rank == 1 else 2
    if data.obs_rank == 1:
        obs_values = jnp.linspace(0.0, 1.0, data.timesteps)
    else:
        obs_values = jnp.stack(
            [jnp.linspace(0.0, 1.0, data.timesteps), jnp.linspace(1.0, 2.0, data.timesteps)],
            axis=-1,
        )

    if data.ctrl_rank == 0:
        controls = Trajectory()
    elif data.ctrl_rank == 1:
        controls = Trajectory(times=times, values=jnp.ones((data.timesteps,)))
    else:
        controls = Trajectory(times=times, values=jnp.ones((data.timesteps, 2)))
    return (
        times,
        obs_values,
        controls,
        obs_dim,
    )


def _make_initial_condition(kind: str, init_rank: int):
    state_dim = 1 if init_rank == 1 else 2
    if kind == "mvn":
        loc = jnp.zeros((state_dim,))
        cov = 0.25 * jnp.eye(state_dim)
        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov), state_dim
    if kind == "uniform":
        low = -jnp.ones((state_dim,))
        high = jnp.ones((state_dim,))
        return dist.Uniform(low, high).to_event(1), state_dim
    # categorical: vector probs for rank-1, batched categorical for rank-2.
    if init_rank == 1:
        probs = jnp.array([0.4, 0.6])
    else:
        probs = jnp.array([[0.3, 0.7], [0.8, 0.2]])
    return dist.Categorical(probs=probs), state_dim


def _make_observation_model(
    kind: str, obs_rank: int, state_dim: int, categorical_state: bool
) -> Callable:
    obs_dim = 1 if obs_rank == 1 else 2
    if kind == "perfect":
        return dsx.DiracIdentityObservation(), obs_dim
    if kind == "linear_gaussian":
        H = jnp.eye(obs_dim, state_dim)
        R = 0.05 * jnp.eye(obs_dim)
        return dsx.LinearGaussianObservation(H=H, R=R), obs_dim

    def poisson_obs(x, u, t):
        # x = jnp.asarray(x, dtype=jnp.float32)
        # if categorical_state:
        #     x = x.astype(jnp.float32)
        # flat = jnp.atleast_1d(x)
        flat = x
        if obs_dim == 1:
            rate = jnp.exp(0.1 + 0.05 * jnp.sum(flat))
        else:
            if flat.shape[0] >= 2:
                rate = jnp.exp(0.1 + 0.05 * flat[:2])
            else:
                rate = jnp.exp(0.1 + 0.05 * jnp.repeat(flat, 2))
        return dist.Poisson(rate)

    return poisson_obs, obs_dim


def _make_discrete_transition(
    transition_kind: str, state_impl: str, uses_control: bool, categorical_state: bool
):
    if transition_kind == "categorical":

        def fn(x, u, t_now, t_next):
            probs = jnp.array([[0.85, 0.15], [0.2, 0.8]])
            return dist.Categorical(probs=probs[x])

        return fn

    if transition_kind == "linear_mvn":
        if state_impl == "eqx":
            return LinearDiscreteEqx(uses_control=uses_control)

        def fn(x, u, t_now, t_next):
            dt = t_next - t_now
            u_term = 0.2 * u if (uses_control and u is not None) else 0.0
            loc = x + dt * (0.1 * x + u_term)
            if jnp.ndim(loc) == 0:
                return dist.Normal(loc=loc, scale=0.2)
            return dist.MultivariateNormal(
                loc=loc, covariance_matrix=0.04 * jnp.eye(loc.shape[-1])
            )

        return fn

    if state_impl == "eqx":
        return NonlinearDiscreteEqx(uses_control=uses_control)

    def fn(x, u, t_now, t_next):
        dt = t_next - t_now
        nonlin = 0.05 * jnp.sin(x)
        u_term = 0.1 * u if (uses_control and u is not None) else 0.0
        loc = x + dt * (nonlin + u_term)
        if jnp.ndim(loc) == 0:
            return dist.Normal(loc=loc, scale=0.2)
        return dist.MultivariateNormal(
            loc=loc, covariance_matrix=0.04 * jnp.eye(loc.shape[-1])
        )

    return fn


def _make_continuous_transition(
    state_impl: str, uses_control: bool, transition_kind: str, diffusion_coeff: str, diffusion_cov: str
):
    if state_impl == "eqx":
        drift = ContinuousDriftEqx(drift_kind=transition_kind, uses_control=uses_control)
    else:

        def drift(x, u, t):
            base = jnp.zeros_like(x) if transition_kind == "zero_ct" else 0.1 * x
            u_term = 0.1 * u if (uses_control and u is not None) else 0.0
            return base + u_term

    dcoeff = None
    if diffusion_coeff == "eye":
        dcoeff = lambda x, u, t: jnp.eye(jnp.atleast_1d(x).shape[0])
    dcov = None
    if diffusion_cov == "eye":
        dcov = lambda x, u, t: jnp.eye(jnp.atleast_1d(x).shape[0])
    return dsx.ContinuousTimeStateEvolution(
        drift=drift,
        diffusion_coefficient=dcoeff,
        diffusion_covariance=dcov,
    )


def _build_model(spec: ModelSpec):
    initial_condition, state_dim = _make_initial_condition(spec.initial_kind, spec.init_rank)
    categorical_state = spec.initial_kind == "categorical" or spec.transition_kind == "categorical"
    obs_model, obs_dim = _make_observation_model(
        spec.observation_kind, spec.observation_rank, state_dim, categorical_state
    )

    if spec.family == "continuous":
        state_evolution = _make_continuous_transition(
            state_impl=spec.state_impl,
            uses_control=spec.uses_control,
            transition_kind=spec.transition_kind,
            diffusion_coeff=spec.diffusion_coeff,
            diffusion_cov=spec.diffusion_cov,
        )
    else:
        state_evolution = _make_discrete_transition(
            transition_kind=spec.transition_kind,
            state_impl=spec.state_impl,
            uses_control=spec.uses_control,
            categorical_state=categorical_state,
        )

    control_dim = 0 if not spec.uses_control else 2
    dynamics = dsx.DynamicalModel(
        state_dim=state_dim,
        observation_dim=obs_dim if spec.observation_kind != "perfect" else state_dim,
        control_dim=control_dim,
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=obs_model,
    )

    def model():
        dsx.sample("f", dynamics)

    return model, dynamics


def _is_discrete_filter(filter_type: str | None):
    return filter_type is not None and filter_type.lower() in {"default", "taylor_kf", "pf"}


def _is_continuous_filter(filter_type: str | None):
    return filter_type is not None and filter_type.lower() in {
        "default",
        "enkf",
        "ekf",
        "ukf",
        "dpf",
    }


def _expected_outcome(model: ModelSpec, inf: InferenceSpec):
    # Perfect observation needs matching shape.
    if model.observation_kind == "perfect" and model.observation_rank != model.init_rank:
        return False, "shape_mismatch"

    if inf.runner == "filter_hmm":
        ok = model.family == "categorical_hmm"
        return ok, None if ok else "hmm_required"

    if inf.runner == "sde_sim":
        if model.family != "continuous":
            return False, "continuous_required"
        if model.diffusion_coeff != "eye" or model.diffusion_cov != "eye":
            return False, "missing_diffusion"
        return True, None

    if inf.runner == "ode_sim":
        if model.family != "continuous":
            return False, "continuous_required"
        return True, None

    if inf.runner == "discrete_sim":
        if model.family == "continuous" and inf.discretizer == "none":
            return False, "not_discretized"
        return True, None

    if inf.runner == "filter":
        if model.family == "continuous" and inf.discretizer == "none":
            if _is_continuous_filter(inf.filter_type):
                return True, None
            return False, "invalid_filter_type"
        # Discrete model, or continuous + discretizer => discrete filter family.
        if _is_discrete_filter(inf.filter_type):
            return True, None
        return False, "invalid_filter_type"

    return False, "unsupported_runner"


def _run_case(model_spec: ModelSpec, inf_spec: InferenceSpec):
    model, dynamics = _build_model(model_spec)
    times = _time_grid(3)
    obs_dim = dynamics.observation_dim
    obs_values = (
        jnp.linspace(0.0, 1.0, len(times))
        if obs_dim == 1
        else jnp.stack([jnp.linspace(0.0, 1.0, len(times)), jnp.linspace(1.0, 2.0, len(times))], axis=-1)
    )

    controls = Trajectory()
    if model_spec.uses_control:
        controls = Trajectory(times=times, values=jnp.ones((len(times), 2)))

    context = Context(
        observations=Trajectory(times=times, values=obs_values),
        controls=controls,
    )

    with seed(rng_seed=jr.PRNGKey(0)):
        with ExitStack() as stack:
            if inf_spec.runner == "discrete_sim":
                stack.enter_context(DiscreteTimeSimulator())
            elif inf_spec.runner == "ode_sim":
                stack.enter_context(ODESimulator())
            elif inf_spec.runner == "sde_sim":
                stack.enter_context(SDESimulator(key=jr.PRNGKey(1)))
            elif inf_spec.runner == "filter":
                filter_kwargs = {}
                ftype = inf_spec.filter_type or "default"
                if ftype.lower() == "pf":
                    filter_kwargs["n_filter_particles"] = 32
                if ftype.lower() == "dpf":
                    filter_kwargs["N_particles"] = 32
                if ftype.lower() in {"enkf", "default"}:
                    filter_kwargs["enkf_N_particles"] = 16
                stack.enter_context(FilterBasedMarginalLogLikelihood(filter_type=ftype, **filter_kwargs))
            elif inf_spec.runner == "filter_hmm":
                stack.enter_context(FilterBasedHMMMarginalLogLikelihood())

            if inf_spec.discretizer == "default":
                stack.enter_context(Discretizer())
            elif inf_spec.discretizer == "euler":
                stack.enter_context(Discretizer(discretize=euler_maruyama))

            stack.enter_context(Condition(context))
            model()


def _scorecard_table(results: list[CaseResult]):
    def _paint(text: str, color: str) -> str:
        if not _USE_COLOR:
            return text
        return f"{color}{text}{_ANSI_RESET}"

    def _status_text(passed: bool, width: int) -> str:
        label = "pass" if passed else "fail"
        return _paint(label.ljust(width), _ANSI_GREEN if passed else _ANSI_RED)

    def _flag_text(mismatch: bool, width: int) -> str:
        if mismatch:
            return _paint("MISMATCH".ljust(width), _ANSI_YELLOW + _ANSI_BOLD)
        return _paint("ok".ljust(width), _ANSI_DIM)

    headers = [
        "flag",
        "case_id",
        "model_family",
        "transition_kind",
        "observation_kind",
        "runner",
        "filter_type",
        "discretizer",
        "expected",
        "actual",
        "expected_error",
        "actual_error",
    ]
    raw_rows = []
    for r in results:
        mismatch = r.expected_pass != r.actual_pass
        raw_rows.append(
            [
                "MISMATCH" if mismatch else "ok",
                r.case_id,
                r.model.family,
                r.model.transition_kind,
                r.model.observation_kind,
                r.inference.runner,
                r.inference.filter_type or "-",
                r.inference.discretizer,
                "pass" if r.expected_pass else "fail",
                "pass" if r.actual_pass else "fail",
                r.expected_error or "-",
                r.actual_error or "-",
            ]
        )
    raw_rows.sort(key=lambda row: (row[0] != "MISMATCH", row[1]))

    widths = [
        max(len(headers[idx]), max((len(row[idx]) for row in raw_rows), default=0))
        for idx in range(len(headers))
    ]

    header_line = " | ".join(
        _paint(headers[idx].ljust(widths[idx]), _ANSI_CYAN + _ANSI_BOLD)
        for idx in range(len(headers))
    )
    sep_line = "-+-".join("-" * width for width in widths)
    lines = [header_line, sep_line]

    for row in raw_rows:
        is_mismatch = row[0] == "MISMATCH"
        cells = []
        for idx, cell in enumerate(row):
            padded = cell.ljust(widths[idx])
            if idx == 0:
                padded = _flag_text(is_mismatch, widths[idx])
            elif headers[idx] in {"expected", "actual"}:
                padded = _status_text(cell == "pass", widths[idx])
            elif is_mismatch and headers[idx] in {"expected_error", "actual_error"} and cell != "-":
                padded = _paint(padded, _ANSI_YELLOW)
            cells.append(padded)
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _write_scorecard(results: list[CaseResult]):
    os.makedirs(os.path.dirname(SCORECARD_PATH), exist_ok=True)
    payload = [asdict(r) for r in results]
    with open(SCORECARD_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_predictive_scorecard(payload: list[dict]):
    os.makedirs(os.path.dirname(PREDICTIVE_SCORECARD_PATH), exist_ok=True)
    with open(PREDICTIVE_SCORECARD_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_xls_scorecard(
    path: str,
    sheet_name: str,
    headers: list[str],
    rows: list[list[str]],
    mismatch_flags: list[bool],
    expected_col: str = "expected",
    actual_col: str = "actual",
):
    """Write a SpreadsheetML 2003 .xls file with mismatch highlighting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    expected_idx = headers.index(expected_col)
    actual_idx = headers.index(actual_col)

    xml_lines = [
        '<?xml version="1.0"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:x="urn:schemas-microsoft-com:office:excel" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:html="http://www.w3.org/TR/REC-html40">',
        "<Styles>",
        '<Style ss:ID="header"><Font ss:Bold="1" ss:Color="#FFFFFF"/><Interior ss:Color="#1F4E78" ss:Pattern="Solid"/></Style>',
        '<Style ss:ID="normal"/>',
        '<Style ss:ID="mismatch"><Interior ss:Color="#FFF2CC" ss:Pattern="Solid"/></Style>',
        '<Style ss:ID="pass"><Font ss:Color="#006100"/><Interior ss:Color="#C6EFCE" ss:Pattern="Solid"/></Style>',
        '<Style ss:ID="fail"><Font ss:Color="#9C0006"/><Interior ss:Color="#FFC7CE" ss:Pattern="Solid"/></Style>',
        "</Styles>",
        f'<Worksheet ss:Name="{xml_escape(sheet_name)}">',
        "<Table>",
        "<Row>",
    ]
    for h in headers:
        xml_lines.append(
            f'<Cell ss:StyleID="header"><Data ss:Type="String">{xml_escape(h)}</Data></Cell>'
        )
    xml_lines.append("</Row>")

    for row, is_mismatch in zip(rows, mismatch_flags, strict=True):
        xml_lines.append("<Row>")
        for col_idx, cell in enumerate(row):
            style = "mismatch" if is_mismatch else "normal"
            if col_idx in {expected_idx, actual_idx}:
                style = "pass" if str(cell).strip().lower() == "pass" else "fail"
            xml_lines.append(
                f'<Cell ss:StyleID="{style}"><Data ss:Type="String">{xml_escape(str(cell))}</Data></Cell>'
            )
        xml_lines.append("</Row>")

    xml_lines.extend(["</Table>", "</Worksheet>", "</Workbook>"])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(xml_lines))


def _predictive_scorecard_table(results: list[dict]):
    def _paint(text: str, color: str) -> str:
        if not _USE_COLOR:
            return text
        return f"{color}{text}{_ANSI_RESET}"

    def _status_text(passed: bool, width: int) -> str:
        label = "pass" if passed else "fail"
        return _paint(label.ljust(width), _ANSI_GREEN if passed else _ANSI_RED)

    def _flag_text(mismatch: bool, width: int) -> str:
        if mismatch:
            return _paint("MISMATCH".ljust(width), _ANSI_YELLOW + _ANSI_BOLD)
        return _paint("ok".ljust(width), _ANSI_DIM)

    headers = [
        "flag",
        "case_id",
        "model_family",
        "transition_kind",
        "context_mode",
        "expected",
        "actual",
        "expected_error",
        "actual_error",
    ]
    raw_rows = []
    for row in results:
        mismatch = row["expected_pass"] != row["actual_pass"]
        raw_rows.append(
            [
                "MISMATCH" if mismatch else "ok",
                str(row["case_id"]),
                row["model_family"],
                row["transition_kind"],
                row["context_mode"],
                "pass" if row["expected_pass"] else "fail",
                "pass" if row["actual_pass"] else "fail",
                row["expected_error"] or "-",
                row["actual_error"] or "-",
            ]
        )
    raw_rows.sort(key=lambda row: (row[0] != "MISMATCH", row[1]))

    widths = [
        max(len(headers[idx]), max((len(row[idx]) for row in raw_rows), default=0))
        for idx in range(len(headers))
    ]
    header_line = " | ".join(
        _paint(headers[idx].ljust(widths[idx]), _ANSI_CYAN + _ANSI_BOLD)
        for idx in range(len(headers))
    )
    sep_line = "-+-".join("-" * width for width in widths)
    lines = [header_line, sep_line]

    for row in raw_rows:
        is_mismatch = row[0] == "MISMATCH"
        cells = []
        for idx, cell in enumerate(row):
            padded = cell.ljust(widths[idx])
            if idx == 0:
                padded = _flag_text(is_mismatch, widths[idx])
            elif headers[idx] in {"expected", "actual"}:
                padded = _status_text(cell == "pass", widths[idx])
            elif is_mismatch and headers[idx] in {"expected_error", "actual_error"} and cell != "-":
                padded = _paint(padded, _ANSI_YELLOW)
            cells.append(padded)
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def test_data_generation_matrix():
    data_specs = [
        DataSpec(obs_rank=o, timesteps=t, ctrl_rank=c)
        for o, t, c in itertools.product([1, 2], [1, 10], [0, 1, 2])
    ]
    for spec in data_specs:
        times, obs_values, controls, _ = _make_data(spec)
        assert times.shape[0] == spec.timesteps
        if spec.obs_rank == 1:
            assert obs_values.shape == (spec.timesteps,)
        else:
            assert obs_values.shape == (spec.timesteps, 2)
        if spec.ctrl_rank == 0:
            assert controls.values is None
        elif spec.ctrl_rank == 1:
            assert controls.values.shape == (spec.timesteps,)
        else:
            assert controls.values.shape == (spec.timesteps, 2)


def test_forward_pass_combinatorial_scorecard(capsys):
    model_specs = [
        # Discrete Gaussian transitions.
        ModelSpec("discrete_gaussian", ik, ir, si, uc, tk, "none", "none", ok, orank)
        for ik, ir, si, uc, tk, ok, orank in itertools.product(
            ["mvn", "uniform", "categorical"],
            [1, 2],
            ["function", "eqx"],
            [False, True],
            ["linear_mvn", "nonlinear_mvn", "categorical"],
            ["linear_gaussian", "perfect", "poisson"],
            [1, 2],
        )
    ] + [
        # Continuous transitions with full diffusion permutations.
        ModelSpec("continuous", ik, ir, si, uc, tk, dc, dv, ok, orank)
        for ik, ir, si, uc, tk, dc, dv, ok, orank in itertools.product(
            ["mvn", "uniform", "categorical"],
            [1, 2],
            ["function", "eqx"],
            [False, True],
            ["linear_ct", "zero_ct"],
            ["eye", "none"],
            ["eye", "none"],
            ["linear_gaussian", "perfect", "poisson"],
            [1, 2],
        )
    ]

    # Dedicated HMM family for exact HMM filter coverage.
    hmm_specs = [
        ModelSpec(
            "categorical_hmm",
            "categorical",
            ir,
            "function",
            uc,
            "categorical",
            "none",
            "none",
            ok,
            orank,
        )
        for ir, uc, ok, orank in itertools.product([1, 2], [False, True], ["poisson", "perfect"], [1, 2])
    ]
    model_specs.extend(hmm_specs)

    inference_specs = [
        InferenceSpec("discrete_sim", None, "none"),
        InferenceSpec("ode_sim", None, "none"),
        InferenceSpec("sde_sim", None, "none"),
        InferenceSpec("filter_hmm", None, "none"),
    ] + [
        InferenceSpec("filter", ftype, disc)
        for ftype, disc in itertools.product(
            ["EnKF", "EKF", "UKF", "DPF", "taylor_kf", "pf", "kf"],
            ["none", "default", "euler"],
        )
    ]

    matrix = list(itertools.product(model_specs, inference_specs))
    selected = _iterable_samples(case_limit=220, combos=matrix)

    results: list[CaseResult] = []

    for idx, (m_spec, i_spec) in enumerate(selected):
        case_id = f"case_{idx:04d}"
        expected_pass, expected_error = _expected_outcome(m_spec, i_spec)
        actual_pass = True
        actual_error = None
        try:
            _run_case(m_spec, i_spec)
        except Exception as exc:  # noqa: BLE001 - intentional for scorecarding.
            actual_pass = False
            actual_error = type(exc).__name__

        result = CaseResult(
            case_id=case_id,
            expected_pass=expected_pass,
            expected_error=expected_error,
            actual_pass=actual_pass,
            actual_error=actual_error,
            model=m_spec,
            inference=i_spec,
        )
        results.append(result)
    _write_scorecard(results)
    xls_headers = [
        "flag",
        "case_id",
        "model_family",
        "transition_kind",
        "observation_kind",
        "runner",
        "filter_type",
        "discretizer",
        "expected",
        "actual",
        "expected_error",
        "actual_error",
    ]
    xls_rows = []
    xls_mismatch_flags = []
    for r in results:
        mismatch = r.expected_pass != r.actual_pass
        xls_mismatch_flags.append(mismatch)
        xls_rows.append(
            [
                "MISMATCH" if mismatch else "ok",
                r.case_id,
                r.model.family,
                r.model.transition_kind,
                r.model.observation_kind,
                r.inference.runner,
                r.inference.filter_type or "-",
                r.inference.discretizer,
                "pass" if r.expected_pass else "fail",
                "pass" if r.actual_pass else "fail",
                r.expected_error or "-",
                r.actual_error or "-",
            ]
        )
    _write_xls_scorecard(
        path=SCORECARD_XLS_PATH,
        sheet_name="Combinatorial",
        headers=xls_headers,
        rows=xls_rows,
        mismatch_flags=xls_mismatch_flags,
    )
    expected_vs_actual_mismatches = [
        r for r in results if (r.expected_pass and not r.actual_pass) or (not r.expected_pass and r.actual_pass)
    ]
    summary_line = (
        "Combinatorial scorecard summary: "
        f"cases={len(results)} "
        f"expected_pass={sum(r.expected_pass for r in results)} "
        f"actual_pass={sum(r.actual_pass for r in results)} "
        f"mismatches={len(expected_vs_actual_mismatches)}"
    )
    full_table = _scorecard_table(results)
    # Force summary to terminal even when pytest output capture is enabled.
    with capsys.disabled():
        print(summary_line)
        print("Combinatorial scorecard table:")
        print(full_table)
    # Keep this matrix resilient as APIs evolve: the scorecard is the main artifact.
    assert results
    assert any(r.actual_pass for r in results)
    assert any(not r.actual_pass for r in results)


def test_predictive_context_matrix(capsys):
    model_specs = [
        ModelSpec(
            "discrete_gaussian",
            "mvn",
            2,
            "function",
            True,
            "linear_mvn",
            "none",
            "none",
            "linear_gaussian",
            2,
        ),
        ModelSpec(
            "continuous",
            "mvn",
            2,
            "eqx",
            True,
            "linear_ct",
            "eye",
            "eye",
            "linear_gaussian",
            2,
        ),
        ModelSpec(
            "categorical_hmm",
            "categorical",
            1,
            "function",
            False,
            "categorical",
            "none",
            "none",
            "poisson",
            1,
        ),
    ]

    context_modes = [
        "obs_times",
        "obs_times_obs_values",
        "obs_times_obs_values_ctrl_times_ctrl_values",
    ]

    results = []
    for case_idx, (model_spec, context_mode) in enumerate(
        itertools.product(model_specs, context_modes)
    ):
        model, dynamics = _build_model(model_spec)
        predictive = Predictive(model, num_samples=1, exclude_deterministic=False)
        times = _time_grid(3)

        obs_values = None
        if context_mode in {
            "obs_times_obs_values",
            "obs_times_obs_values_ctrl_times_ctrl_values",
        }:
            if dynamics.observation_dim == 1:
                obs_values = jnp.array([0.0, 1.0, 2.0])
            else:
                obs_values = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

        controls = Trajectory()
        if context_mode == "obs_times_obs_values_ctrl_times_ctrl_values":
            controls = Trajectory(times=times, values=jnp.ones((len(times), dynamics.control_dim or 1)))

        context = Context(
            observations=Trajectory(times=times, values=obs_values),
            controls=controls,
        )

        expected_pass = model_spec.family != "continuous"
        expected_error = None if expected_pass else "predictive_continuous_unsupported"
        actual_pass = True
        actual_error = None
        try:
            with seed(rng_seed=jr.PRNGKey(0)):
                with ExitStack() as stack:
                    if model_spec.family == "continuous":
                        stack.enter_context(SDESimulator(key=jr.PRNGKey(1)))
                    else:
                        stack.enter_context(DiscreteTimeSimulator())
                    stack.enter_context(Condition(context))
                    predictive(jr.PRNGKey(2))
        except Exception as exc:  # noqa: BLE001
            actual_pass = False
            actual_error = type(exc).__name__

        results.append(
            {
                "case_id": f"predictive_{case_idx:03d}",
                "model_family": model_spec.family,
                "transition_kind": model_spec.transition_kind,
                "context_mode": context_mode,
                "expected_pass": expected_pass,
                "expected_error": expected_error,
                "actual_pass": actual_pass,
                "actual_error": actual_error,
            }
        )

    _write_predictive_scorecard(results)
    xls_headers = [
        "flag",
        "case_id",
        "model_family",
        "transition_kind",
        "context_mode",
        "expected",
        "actual",
        "expected_error",
        "actual_error",
    ]
    xls_rows = []
    xls_mismatch_flags = []
    for row in results:
        mismatch = row["expected_pass"] != row["actual_pass"]
        xls_mismatch_flags.append(mismatch)
        xls_rows.append(
            [
                "MISMATCH" if mismatch else "ok",
                str(row["case_id"]),
                row["model_family"],
                row["transition_kind"],
                row["context_mode"],
                "pass" if row["expected_pass"] else "fail",
                "pass" if row["actual_pass"] else "fail",
                row["expected_error"] or "-",
                row["actual_error"] or "-",
            ]
        )
    _write_xls_scorecard(
        path=PREDICTIVE_SCORECARD_XLS_PATH,
        sheet_name="Predictive",
        headers=xls_headers,
        rows=xls_rows,
        mismatch_flags=xls_mismatch_flags,
    )
    summary_line = (
        "Predictive scorecard summary: "
        f"cases={len(results)} "
        f"expected_pass={sum(row['expected_pass'] for row in results)} "
        f"actual_pass={sum(row['actual_pass'] for row in results)}"
    )
    full_table = _predictive_scorecard_table(results)
    with capsys.disabled():
        print(summary_line)
        print("Predictive scorecard table:")
        print(full_table)
    assert results
    assert any(row["actual_pass"] for row in results)
    assert any(not row["actual_pass"] for row in results)
