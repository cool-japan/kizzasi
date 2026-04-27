#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
wasm-pack build crates/kizzasi-tokenizer --target web --features wasm --out-dir ../../examples/web/pkg
echo "Build complete. Serve examples/web/ with a local HTTP server."
