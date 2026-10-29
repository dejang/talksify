#!/usr/bin/env bash
# Bump version across package.json, Cargo.toml, Cargo.lock, and tauri.conf.json,
# then create a local release commit and vX.Y.Z tag WITHOUT pushing.
#
# Usage:
#   ./scripts/release.sh            # bumps patch version (e.g. 0.2.1 -> 0.2.2)
#   ./scripts/release.sh 0.3.0      # explicit version (v0.3.0 also accepted)
#   ./scripts/release.sh minor      # bumps minor version (e.g. 0.2.1 -> 0.3.0)
#   ./scripts/release.sh major      # bumps major version (e.g. 0.2.1 -> 1.0.0)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BRANCH="$(git branch --show-current)"
if [ "$BRANCH" != "main" ]; then
  echo "error: releases must be cut from 'main' (currently on '$BRANCH')." >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree has uncommitted changes to tracked files. Commit or stash them first." >&2
  exit 1
fi

CURRENT="$(node -p "require('./package.json').version")"
IFS='.' read -r CUR_MAJOR CUR_MINOR CUR_PATCH <<< "$CURRENT"

ARG="${1:-patch}"
case "$ARG" in
  patch)
    VERSION="${CUR_MAJOR}.${CUR_MINOR}.$((CUR_PATCH + 1))"
    ;;
  minor)
    VERSION="${CUR_MAJOR}.$((CUR_MINOR + 1)).0"
    ;;
  major)
    VERSION="$((CUR_MAJOR + 1)).0.0"
    ;;
  *)
    VERSION="${ARG#v}"
    ;;
esac

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: invalid version '$VERSION' (expected X.Y.Z, patch, minor, or major)." >&2
  exit 1
fi

if [ "$VERSION" = "$CURRENT" ]; then
  echo "error: version is already $CURRENT." >&2
  exit 1
fi

TAG="v${VERSION}"
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "error: tag '$TAG' already exists." >&2
  exit 1
fi

echo "Running lint check before release..."
pnpm lint

echo "Bumping Talksify $CURRENT -> $VERSION..."

VERSION="$VERSION" node -e '
const fs = require("fs");
const version = process.env.VERSION;

for (const file of ["package.json", "src-tauri/tauri.conf.json"]) {
  const raw = fs.readFileSync(file, "utf8");
  const updated = raw.replace(/^(\s*"version"\s*:\s*")[^"]+(")/m, `$1${version}$2`);
  if (raw === updated || JSON.parse(updated).version !== version) {
    throw new Error(`Could not update version in ${file}`);
  }
  fs.writeFileSync(file, updated);
}

const cargoPath = "src-tauri/Cargo.toml";
const cargo = fs.readFileSync(cargoPath, "utf8");
const updatedCargo = cargo.replace(
  /^(\[package\][\s\S]*?^version\s*=\s*")[^"]+(")/m,
  `$1${version}$2`
);
if (cargo === updatedCargo) {
  throw new Error("Could not update version in src-tauri/Cargo.toml");
}
fs.writeFileSync(cargoPath, updatedCargo);
'

# Refresh talksify package entry in Cargo.lock without touching dependency versions
cargo metadata --manifest-path src-tauri/Cargo.toml --format-version 1 >/dev/null

git add package.json src-tauri/tauri.conf.json src-tauri/Cargo.toml src-tauri/Cargo.lock
git commit -m "chore(release): ${TAG}"
git tag "$TAG"

echo ""
echo "✓ Created local commit and tag ${TAG} (not pushed)."
echo ""
echo "When ready to trigger the GitHub Release build, run:"
echo "  git push origin main ${TAG}"
echo ""
echo "To undo before pushing:"
echo "  git tag -d ${TAG} && git reset --hard HEAD~1"
