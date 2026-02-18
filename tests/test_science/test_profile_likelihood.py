"""Profile likelihood tests for all filter-based models. Tests each system with all available filters from fixture params."""

from tests.fixtures import (
    data_conditioned_continuous_time_lti_gaussian,  # noqa: F401
    data_conditioned_continuous_time_stochastic_l63,  # noqa: F401
    data_conditioned_discrete_time_l63_filter,  # noqa: F401
    data_conditioned_discrete_time_l63_filter_pf,  # noqa: F401
    data_conditioned_discrete_time_lti_kf,  # noqa: F401
    data_conditioned_hmm,  # noqa: F401
)
from tests.test_utils import get_output_dir, run_profile_likelihood

SAVE_FIG = True


def test_profile_hmm(data_conditioned_hmm):  # noqa: F811
    data_conditioned_model, true_params, synthetic, use_controls = data_conditioned_hmm
    output_name = (
        "profile_hmm_filter_hmm" + ("_controlled" if use_controls else "") + ".png"
    )
    output_dir = get_output_dir("profiles") if SAVE_FIG else None
    run_profile_likelihood(
        model=data_conditioned_model,
        param_name="sigma",
        true_val=float(true_params["sigma"]),
        param_min=0.1,
        param_max=1.0,
        n_grid=11,
        output_dir=output_dir,
        output_name=output_name,
    )


def test_profile_discrete_time_l63(data_conditioned_discrete_time_l63_filter):  # noqa: F811
    (
        data_conditioned_model,
        true_params,
        synthetic,
        use_controls,
    ) = data_conditioned_discrete_time_l63_filter
    output_name = (
        "profile_discrete_time_l63_filter_default"
        + ("_controlled" if use_controls else "")
        + ".png"
    )
    output_dir = get_output_dir("profiles") if SAVE_FIG else None
    run_profile_likelihood(
        model=data_conditioned_model,
        param_name="rho",
        true_val=float(true_params["rho"]),
        param_min=20.0,
        param_max=36.0,
        n_grid=11,
        output_dir=output_dir,
        output_name=output_name,
    )


def test_profile_discrete_time_l63_pf(data_conditioned_discrete_time_l63_filter_pf):  # noqa: F811
    (
        data_conditioned_model,
        true_params,
        synthetic,
        use_controls,
    ) = data_conditioned_discrete_time_l63_filter_pf
    output_name = (
        "profile_discrete_time_l63_filter_pf"
        + ("_controlled" if use_controls else "")
        + ".png"
    )
    output_dir = get_output_dir("profiles") if SAVE_FIG else None
    run_profile_likelihood(
        model=data_conditioned_model,
        param_name="rho",
        true_val=float(true_params["rho"]),
        param_min=20.0,
        param_max=36.0,
        n_grid=11,
        output_dir=output_dir,
        output_name=output_name,
    )


def test_profile_continuous_time_l63_sde(
    data_conditioned_continuous_time_stochastic_l63,  # noqa: F811
):
    (
        data_conditioned_model,
        true_params,
        synthetic,
        use_controls,
        filter_type,
    ) = data_conditioned_continuous_time_stochastic_l63
    output_name = (
        "profile_l63_sde"
        + ("_controlled" if use_controls else "")
        + f"_filter_{filter_type}.png"
    )
    output_dir = get_output_dir("profiles") if SAVE_FIG else None
    run_profile_likelihood(
        model=data_conditioned_model,
        param_name="rho",
        true_val=float(true_params["rho"]),
        param_min=20.0,
        param_max=36.0,
        n_grid=11,
        output_dir=output_dir,
        output_name=output_name,
    )


def test_profile_continuous_time_lti(data_conditioned_continuous_time_lti_gaussian):  # noqa: F811
    (
        data_conditioned_model,
        true_params,
        synthetic,
        use_controls,
        filter_type,
    ) = data_conditioned_continuous_time_lti_gaussian
    output_name = (
        "profile_lti_gaussian"
        + ("_controlled" if use_controls else "")
        + f"_filter_{filter_type}.png"
    )
    output_dir = get_output_dir("profiles") if SAVE_FIG else None
    run_profile_likelihood(
        model=data_conditioned_model,
        param_name="rho",
        true_val=float(true_params["rho"]),
        param_min=0.5,
        param_max=3.5,
        n_grid=11,
        output_dir=output_dir,
        output_name=output_name,
    )


def test_profile_discrete_time_lti(data_conditioned_discrete_time_lti_kf):  # noqa: F811
    (
        data_conditioned_model,
        true_params,
        synthetic,
        use_controls,
        filter_type,
    ) = data_conditioned_discrete_time_lti_kf
    output_name = (
        "profile_discrete_time_lti"
        + ("_controlled" if use_controls else "")
        + f"_filter_{filter_type}.png"
    )
    output_dir = get_output_dir("profiles") if SAVE_FIG else None
    run_profile_likelihood(
        model=data_conditioned_model,
        param_name="alpha",
        true_val=float(true_params["alpha"]),
        param_min=0.0,
        param_max=0.7,
        n_grid=11,
        output_dir=output_dir,
        output_name=output_name,
    )
