def pytest_addoption(parser):
    group = parser.getgroup("dynestyx combinatorial")
    group.addoption(
        "--dsx-thin-combinatorial",
        action="store_true",
        default=False,
        help="Enable deterministic thinning for combinatorial matrix tests.",
    )
    group.addoption(
        "--dsx-case-limit",
        action="store",
        type=int,
        default=220,
        help="Case limit used when thinning combinatorial matrix tests.",
    )
    group.addoption(
        "--dsx-progress",
        action="store_true",
        default=True,
        help="Show progress updates for long combinatorial loops.",
    )
    group.addoption(
        "--dsx-no-progress",
        action="store_false",
        dest="dsx_progress",
        help="Disable progress updates for combinatorial loops.",
    )
    group.addoption(
        "--dsx-progress-every",
        action="store",
        type=int,
        default=25,
        help="Emit a progress update every N combinatorial cases.",
    )

