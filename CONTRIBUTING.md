# Contributing to dynestyx

Thanks for contributing. This guide explains the two main ways to contribute: (a) open an issue describing a bug/idea, and/or (b) open a PR with a proposed fix or implementation. We strongly encourage the combined path (A+B): open an issue, then address it yourself and submit a PR.

## 0) Pick a contribution path

Preferred path: **A + B** (open an issue, then open a PR that addresses it).

### Option A: Open an issue

Use this when you want feedback before coding, or to report a bug/request.

1. Go to the [dynestyx Issues page](https://github.com/BasisResearch/dynestyx/issues).
2. Click **New issue**.
3. Use a clear title.
4. Include:
   - expected behavior
   - observed behavior
   - minimal reproduction (if bug)
   - environment details (OS, Python, key package versions)
5. Submit the issue.

### Option B: Open a Pull Request

Use this when you already have a concrete proposed change. Follow the steps below.

## 1) Setup

Fork `dynestyx` to your GitHub account first:

1. Go to [github.com/BasisResearch/dynestyx](https://github.com/BasisResearch/dynestyx).
2. Click **Fork** (top-right).
3. Choose your GitHub account as the destination.

Then in your terminal, clone your fork and add the original repo as `upstream`:

```bash
git clone https://github.com/<your-username>/dynestyx.git
cd dynestyx
git remote add upstream https://github.com/BasisResearch/dynestyx.git
git remote -v
uv venv
source .venv/bin/activate
uv sync --dev --all-extras
```

You should see:
- `origin` -> `https://github.com/<your-username>/dynestyx.git`
- `upstream` -> `https://github.com/BasisResearch/dynestyx.git`

## 2) Create a branch

```bash
git checkout main
git pull upstream main
git checkout -b <type>/<short-description>
```

Examples: `feature/add-ode-example`, `fix/filter-shape-bug`, `docs/update-quickstart`.

## 3) Before opening a PR, run project scripts

From repo root:

```bash
uv run scripts/clean.sh
uv run scripts/lint.sh
uv run scripts/test.sh
```

If you changed a lot or touched core inference code, also run:

```bash
uv run scripts/test_full.sh
```

`scripts/test_full.sh` writes run artifacts and logs under `.output`. To inspect results:

```bash
ls .output
```

Open relevant files in `.output` to review test summaries/logs for failures.

If you edited docs, also build docs locally:

```bash
uv run mkdocs build
```

To preview docs locally while editing:

```bash
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

## 4) Commit and push

```bash
git add .
git commit -m "Short imperative message"
git push -u origin <type>/<short-description>
```

## 5) Open the Pull Request

1. Go to [github.com/BasisResearch/dynestyx](https://github.com/BasisResearch/dynestyx).
2. If prompted with your recently pushed branch, click **Compare & pull request**.
   - If not prompted, go to the **Pull requests** tab and click **New pull request**.
3. Set **base** to `BasisResearch/dynestyx:main`.
4. Set **compare** to your branch (`<type>/<short-description>`).
5. Add title + description (template below), then click **Create pull request**.

Use this PR description format:

- **Summary**: what changed and why
- **Notes**: any caveats, follow-ups, or breaking behavior

Keep PRs focused. If your change is user-facing, include doc updates in the same PR.
