#!/usr/bin/env bash
# Lint and format-check the bytes git will actually commit.
#
# core.autocrlf=true means the working tree is CRLF and the index is LF. ruff
# run against the working tree reports every file as needing reformatting,
# which drowns the real findings. Check the index instead.
set -euo pipefail
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for f in "$@"; do
  git show ":$f" > "$tmp/$(basename "$f")"
done
uvx ruff check "$tmp"
uvx ruff format --check "$tmp"
