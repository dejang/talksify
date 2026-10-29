---
name: release
description: >-
  Bump the Talksify release version across package.json, Cargo.toml, Cargo.lock,
  and tauri.conf.json, and create a local release commit and vX.Y.Z git tag
  without pushing. Activate when the user invokes /release or asks to cut or
  bump a release version.
---

# Release Skill (`/release`)

Prepares a local release commit (`chore(release): vX.Y.Z`) and git tag (`vX.Y.Z`) using [release.sh](../../../scripts/release.sh), **without pushing to `origin`**.

## Usage

1. Determine the version argument from the user's request:
   - **No version specified** (e.g. `/release`): omit the argument so the script bumps the **patch** version automatically (`0.2.1` -> `0.2.2`).
   - **Explicit version or bump keyword** (e.g. `/release 0.3.0`, `/release v0.3.0`, `/release minor`, `/release major`): pass that single argument to the script.

2. Ensure all prior feature/fix changes are committed (tracked working tree must be clean on `main`), then run:

   ```bash
   ./scripts/release.sh [version]
   ```

3. Verify the commit and tag locally:

   ```bash
   git show --stat HEAD
   ```

4. **Never run `git push` automatically** unless the user explicitly asks to push after reviewing. Show the user the exact push command printed by the script:

   ```bash
   git push origin main vX.Y.Z
   ```
