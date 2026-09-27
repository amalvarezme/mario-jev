#!/usr/bin/env bash
#
# jev-env.sh — hand the Jev API key to one command without ever putting it in a
# file or a commit.
#
# Design choice: the Keychain lookup lives here, not in the library. The package
# must stay portable — it cannot assume macOS or a Keychain — so the
# platform-specific secret store stays in this opt-in shell bridge, and the
# library only ever reads JEV_API_KEY from the environment.
#
# Usage:
#   scripts/jev-env.sh <command> [args...]   # run one command with the key set
#   source scripts/jev-env.sh                # export the key into this shell
#
# If JEV_API_KEY is already set, it is used as-is and the Keychain is never
# touched, so CI and gateway-injected keys keep working.
#
# Works on macOS's stock bash 3.2.

# Sourcing this file is a documented usage, so remember how the caller configured
# its shell before strict mode is applied: strict mode must not leak into the
# caller's shell.
__jev_caller_opts="$-"
if shopt -o -q pipefail; then
  __jev_caller_pipefail=on
else
  __jev_caller_pipefail=off
fi

set -euo pipefail

if [[ -z "${JEV_API_KEY:-}" ]]; then
  __jev_service="${JEV_KEYCHAIN_SERVICE:-jev-api-key}"
  __jev_account="${JEV_KEYCHAIN_ACCOUNT:-${USER:-$(id -un)}}"

  if ! command -v security >/dev/null 2>&1; then
    echo "jev-env: the macOS 'security' tool is not on PATH and JEV_API_KEY is unset." >&2
    echo "jev-env: set JEV_API_KEY, or run this on macOS with the Keychain entry installed." >&2
    exit 1
  fi

  if ! JEV_API_KEY="$(security find-generic-password \
        -s "$__jev_service" -a "$__jev_account" -w 2>/dev/null)"; then
    echo "jev-env: no Keychain entry for service '$__jev_service' and account '$__jev_account'." >&2
    echo "jev-env: add it with: security add-generic-password -s '$__jev_service' -a \"\$USER\" -w" >&2
    exit 1
  fi

  if [[ -z "$JEV_API_KEY" ]]; then
    echo "jev-env: the Keychain entry for service '$__jev_service' is empty." >&2
    echo "jev-env: store a non-empty key, or set JEV_API_KEY directly." >&2
    exit 1
  fi

  unset __jev_service __jev_account
  export JEV_API_KEY
fi

# Sourced: put the key in the caller's shell and stop. Never exec here, because
# that would replace the interactive shell. The caller's shell options are put
# back first, so sourcing this file cannot change how that shell behaves.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  case "$__jev_caller_opts" in *e*) set -e ;; *) set +e ;; esac
  case "$__jev_caller_opts" in *u*) set -u ;; *) set +u ;; esac
  if [[ "$__jev_caller_pipefail" == "on" ]]; then set -o pipefail; else set +o pipefail; fi
  unset __jev_caller_opts __jev_caller_pipefail
  return 0
fi

if [[ "$#" -eq 0 ]]; then
  echo "usage: scripts/jev-env.sh <command> [args...]" >&2
  echo "   or: source scripts/jev-env.sh" >&2
  exit 2
fi

exec "$@"
