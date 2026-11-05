"""Private interpretation inspection and composition."""

from contextlib import ExitStack

import pytest
from effectful.ops.semantics import coproduct, handler
from effectful.ops.syntax import defop

import dynestyx as dsx
from dynestyx.control import DiscreteControlLoopSimulator
from dynestyx.handlers import _dynestyx_stack_kind
from dynestyx.handlers import _DynestyxStackKind as Kind


@pytest.mark.parametrize(
    "interpretation,kind",
    [
        (dsx.plate("members", 2), Kind.PLATE),
        (dsx.Discretizer(), Kind.DISCRETIZER),
        (dsx.GaussianRelaxation(), Kind.GAUSSIAN_RELAXATION),
        (dsx.Filter(), Kind.FILTER),
        (dsx.Smoother(), Kind.SMOOTHER),
        (dsx.LatentPathBuilder(), Kind.LATENT_PATH_BUILDER),
        (dsx.Simulator(), Kind.SIMULATOR),
        (dsx.DiscreteTimeSimulator(), Kind.SIMULATOR),
        (
            DiscreteControlLoopSimulator(
                control_policy=lambda x_hat, t_now, t_next, s: (x_hat.mean, s)
            ),
            Kind.SIMULATOR,
        ),
        (dsx.ODESimulator(), Kind.SIMULATOR),
        (dsx.SDESimulator(), Kind.SIMULATOR),
        (dsx.Evaluation(dsx.ObservationScoringConfig()), Kind.EVALUATION),
    ],
)
def test_interpretation_kind(interpretation, kind):
    with handler(interpretation):
        assert _dynestyx_stack_kind() == [kind]
    assert _dynestyx_stack_kind() == []


def test_stack_order_duplicates_and_restoration():
    with dsx.Filter(), dsx.Discretizer():
        assert _dynestyx_stack_kind() == [Kind.DISCRETIZER, Kind.FILTER]
        with pytest.raises(RuntimeError), ExitStack() as stack:
            stack.enter_context(dsx.plate("outer", 2))
            stack.enter_context(dsx.plate("inner", 3))
            assert _dynestyx_stack_kind() == [
                Kind.PLATE,
                Kind.PLATE,
                Kind.DISCRETIZER,
                Kind.FILTER,
            ]
            raise RuntimeError("restore contexts")
        assert _dynestyx_stack_kind() == [Kind.DISCRETIZER, Kind.FILTER]
    assert _dynestyx_stack_kind() == []


def test_coproduct_and_unrelated_effect():
    unrelated = defop(lambda: "default")
    with handler({unrelated: lambda: "handled"}):
        with handler(coproduct(dsx.Filter(), dsx.Discretizer())):
            assert _dynestyx_stack_kind() == [Kind.DISCRETIZER, Kind.FILTER]
            assert unrelated() == "handled"
        assert unrelated() == "handled"
    empty = _dynestyx_stack_kind()
    empty.append(Kind.FILTER)
    assert _dynestyx_stack_kind() == []
    assert not hasattr(dsx, "_dynestyx_stack_kind")


def test_query_inside_an_operation_with_arguments():
    operation = defop(lambda name, **kwargs: None)
    with dsx.Simulator(), dsx.Filter():
        with handler({operation: lambda name, **kwargs: _dynestyx_stack_kind()}):
            assert operation("site", obs_values="data") == [Kind.FILTER, Kind.SIMULATOR]
