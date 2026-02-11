import os

# Default is no thinning (full matrix).
# Set DYNESTYX_THIN_COMBINATORIAL=1 to enable deterministic thinning.
THIN_COMBINATORIAL = os.getenv("DYNESTYX_THIN_COMBINATORIAL", "0") == "1"

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
SCORECARD_MISMATCH_XLS_PATH = os.getenv(
    "DYNESTYX_COMBINATORIAL_SCORECARD_MISMATCH_XLS_PATH",
    ".pytest_cache/dynestyx_combinatorial_scorecard_mismatch_only.xls",
)
PREDICTIVE_SCORECARD_XLS_PATH = os.getenv(
    "DYNESTYX_PREDICTIVE_SCORECARD_XLS_PATH",
    ".pytest_cache/dynestyx_predictive_scorecard.xls",
)
PREDICTIVE_SCORECARD_MISMATCH_XLS_PATH = os.getenv(
    "DYNESTYX_PREDICTIVE_SCORECARD_MISMATCH_XLS_PATH",
    ".pytest_cache/dynestyx_predictive_scorecard_mismatch_only.xls",
)

USE_COLOR = os.getenv("NO_COLOR") is None
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"

