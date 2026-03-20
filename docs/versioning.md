# Versioned documentation

The published site uses **[mike](https://github.com/jimporter/mike)** with **[MkDocs Material’s versioning](https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/)**. Each build is stored under a version id on the `gh-pages` branch; **aliases** point human-friendly names at those builds.

## Aliases

| Alias | What it tracks | When it updates |
|--------|----------------|-----------------|
| **`latest`** | Documentation built from the **`main`** branch (development) | Every push to `main` |
| **`stable`** | Documentation built from the **latest SemVer tag** (e.g. `v0.0.1`) | Every push of a tag matching `v*` |

The **default** landing version (root URL of the site) is set to **`stable`** when a release tag is built, so casual visitors see docs that match the latest **released** package. Use the version menu in the header to switch to **`latest`** for API and tutorials that match `main`.

!!! note "GitHub Pages source"
    The repository must serve Pages from the **`gh-pages` branch** (folder `/` or `/root`), not from the older “GitHub Actions artifact” flow. If you previously deployed with `actions/deploy-pages` + `upload-pages-artifact`, switch **Settings → Pages → Build and deployment → Branch → `gh-pages`**.

## Installing a matching package version

To reproduce examples against a **released** API:

```bash
pip install dynestyx==0.0.1
```

To work against **`main`** (same as **latest** docs):

```bash
pip install git+https://github.com/BasisResearch/dynestyx.git
```

## Notebooks and API drift

Tutorials and deep dives are **`.ipynb` files in this repo**. There is **one notebook tree**: it always reflects the **tip of `main`**. Tagged releases do **not** carry a second copy of notebooks in git; instead, the docs build checks out the requested git ref and builds the site from that snapshot. So:

- **`/stable/`** in the deployed site shows notebooks **as they were at the release tag** (e.g. `v0.0.1`).
- **`/latest/`** shows notebooks **as they are on `main` today** (e.g. updated `predict_times` / `f_observations` conventions).

If you follow a notebook locally, align your install with the doc version you are reading (**stable** vs **latest**).

### Optional: banner cell for authors

When adding or updating a notebook, you can paste a short markdown cell at the top so readers know which branch the source matches:

```markdown
> **Source:** This notebook tracks **`main`**. Published HTML for **`stable`** (releases) and **`latest`** (dev) may differ; use the docs **version menu** and install the matching package (see [Versioned documentation](versioning.md)).
```

For notebooks that are **only** meant for a specific release, prefer **tagging** and building docs from that tag rather than maintaining duplicate files.

## Local builds (maintainers)

For normal docs work:

```bash
uv sync --dev --all-extras
uv run mkdocs build
uv run mkdocs serve
```

If you want to verify the **versioned** layout locally, especially **`stable`** vs **`latest`**, use the helper script:

```bash
scripts/preview_versioned_docs.sh v0.0.1
```

That will:

- create / reuse sibling worktrees under `../dynestyx-docs-preview/`
- build **`stable`** from the supplied ref / tag
- build **`latest`** from `main`
- set the local default to **`stable`**
- serve the combined Mike site at `http://127.0.0.1:8000`

Open `http://127.0.0.1:8000` and use the version switcher to compare **`stable (0.0.1)`** and **`latest`**.

Optional arguments:

```bash
# default latest ref is main
scripts/preview_versioned_docs.sh v0.0.1

# compare the release against your current feature branch
scripts/preview_versioned_docs.sh v0.0.1 "$(git branch --show-current)"

# use a different "latest" ref
scripts/preview_versioned_docs.sh v0.0.1 my-feature-branch

# change the local port
PORT=9000 scripts/preview_versioned_docs.sh v0.0.1
```

!!! note "Why `uv run --with mike`?"
    Older tags may not list `mike` in their `pyproject.toml`. Using `uv run --with mike mike ...` ensures the command is available even when you check out a historical release.

**Do not** force-push to `gh-pages` manually unless you know what you are doing; CI is the source of truth.

## CI

Deployment is defined in `.github/workflows/build_docs.yml`:

- Push to **`main`** → Mike version `dev`, alias **`latest`**, title **`latest`**.
- Push **`v*`** tag → Mike version `<semver>`, alias **`stable`**, title **`stable (<semver>)`**, then `mike set-default stable` so the site root tracks the newest release.

### Manual publish / backfill (`workflow_dispatch`)

Use **Actions → Deploy documentation → Run workflow** when you want to publish docs from an arbitrary git ref without creating a new tag.

This is especially useful for your already-existing historical release tag.

Example: backfill **`stable`** from the existing tag **`v0.0.1`**:

- **target ref:** `v0.0.1`
- **version:** `0.0.1`
- **alias:** `stable`
- **title:** `stable (0.0.1)`
- **set default:** `true`

That publishes the tagged docs as **`stable`** using the **current** docs workflow and toolchain, and the menu entry will read **`stable (0.0.1)`**.
