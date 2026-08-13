#!/usr/bin/env bash
# Local stand-in for CI on kizzasi-io's video backends (`video` / `video-pure`).
#
# This repository's branch policy keeps .github/workflows to publish
# workflows only (see CLAUDE.md), so there is no GitHub Actions job that
# exercises this matrix -- this script is it. It runs the four feature
# configurations that matter for the video code (both backends compiled in,
# each one alone, and neither), plus clippy on the two configurations that
# actually compile video code, plus cargo-deny. Every step must pass; the
# script stops at the first failure (set -e) so a red step is never buried
# under later output.
#
# Usage: scripts/check-video-matrix.sh
# Run from anywhere -- it cd's to the repository root itself.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

step() {
  echo ""
  echo "==> $1"
}

pass() {
  echo "    OK: $1"
}

step "1/9 nextest: video (FFmpeg backend only)"
cargo nextest run -p kizzasi-io --features video
pass "video"

step "2/9 nextest: video-pure (pure-Rust backend only, non-default features)"
cargo nextest run -p kizzasi-io --no-default-features --features std,video-pure
pass "video-pure"

step "3/9 nextest: video,video-pure (both backends compiled in)"
cargo nextest run -p kizzasi-io --features video,video-pure
pass "video,video-pure"

step "4/9 nextest: default features (neither video backend compiled in)"
cargo nextest run -p kizzasi-io
pass "default (no video backend)"

step "5/9 bench --no-run: all four configs compile"
cargo bench -p kizzasi-io --features video --no-run
cargo bench -p kizzasi-io --no-default-features --features std,video-pure --no-run
cargo bench -p kizzasi-io --features video,video-pure --no-run
cargo bench -p kizzasi-io --no-run
pass "benches compile under all four configs"

step "6/9 doc build: RUSTDOCFLAGS=-D warnings, video and video-pure"
RUSTDOCFLAGS="-D warnings" cargo doc -p kizzasi-io --no-deps --features video
RUSTDOCFLAGS="-D warnings" cargo doc -p kizzasi-io --no-deps --no-default-features --features std,video-pure
pass "rustdoc clean on both configs"

step "7/9 doctests: video and video-pure"
cargo test --doc -p kizzasi-io --features video
cargo test --doc -p kizzasi-io --no-default-features --features std,video-pure
pass "doctests pass on both configs"

step "8/9 clippy -D warnings: video,video-pure and video-pure alone"
cargo clippy -p kizzasi-io --all-targets --features video,video-pure -- -D warnings
cargo clippy -p kizzasi-io --all-targets --no-default-features --features std,video-pure -- -D warnings
pass "clippy clean on both configs"

step "9/9 cargo deny check bans"
cargo deny check bans
pass "no banned crates"

echo ""
echo "==> check-video-matrix: all steps passed"
