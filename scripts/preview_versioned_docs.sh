#!/bin/bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
PREVIEW_ROOT="${PREVIEW_ROOT:-$(dirname "$ROOT_DIR")/dynestyx-docs-preview}"
CURRENT_BRANCH="$(git -C "$ROOT_DIR" branch --show-current || true)"

STABLE_REF="${1:-}"
LATEST_REF="${2:-main}"
PORT="${PORT:-8000}"

if [[ -z "$STABLE_REF" ]]; then
  STABLE_REF="$(git -C "$ROOT_DIR" tag --sort=-v:refname | head -n 1)"
fi

if [[ -z "$STABLE_REF" ]]; then
  echo "No git tag found. Pass a stable ref explicitly, e.g.:" >&2
  echo "  scripts/preview_versioned_docs.sh v0.0.1" >&2
  exit 1
fi

STABLE_VERSION="${STABLE_REF#v}"
STABLE_DIR="$PREVIEW_ROOT/stable"
LATEST_DIR="$PREVIEW_ROOT/latest"
LATEST_BUILD_DIR="$LATEST_DIR"

ensure_worktree() {
  local dir="$1"
  local ref="$2"

  if [[ -d "$dir/.git" || -f "$dir/.git" ]]; then
    git -C "$dir" fetch --all --tags --prune || true
    git -C "$dir" checkout --force "$ref"
  else
    mkdir -p "$(dirname "$dir")"
    git -C "$ROOT_DIR" worktree add "$dir" "$ref"
  fi
}

echo "Preparing worktrees in $PREVIEW_ROOT"
ensure_worktree "$STABLE_DIR" "$STABLE_REF"

# If latest ref is the branch currently checked out in the main repo, reuse the
# current working tree instead of trying to create a second worktree for it.
if [[ -n "$CURRENT_BRANCH" && "$LATEST_REF" == "$CURRENT_BRANCH" ]]; then
  LATEST_BUILD_DIR="$ROOT_DIR"
  echo "Reusing current checkout for latest: $LATEST_REF ($LATEST_BUILD_DIR)"
else
  ensure_worktree "$LATEST_DIR" "$LATEST_REF"
fi

echo "Building stable from $STABLE_REF"
(
  cd "$STABLE_DIR"
  uv sync --dev --all-extras
  uv run --with mike mike deploy --title "stable ($STABLE_VERSION)" --update-aliases "$STABLE_VERSION" stable
)

echo "Building latest from $LATEST_REF"
(
  cd "$LATEST_BUILD_DIR"
  uv sync --dev --all-extras
  uv run --with mike mike deploy --title "latest" --update-aliases dev latest
  uv run --with mike mike set-default stable
  echo
  echo "Serving versioned docs at http://127.0.0.1:$PORT"
  echo "Available versions: stable ($STABLE_VERSION), latest"
  uv run --with mike mike serve --dev-addr "127.0.0.1:$PORT"
)
