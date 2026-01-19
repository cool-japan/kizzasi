.PHONY: help test bench coverage lint format check-perf clean install-tools ci-local docs

# Default target
help:
	@echo "kizzasi-tokenizer - Make targets"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Run all tests"
	@echo "  make bench         - Run benchmarks"
	@echo "  make coverage      - Generate coverage report"
	@echo "  make lint          - Run clippy"
	@echo "  make format        - Format code with rustfmt"
	@echo "  make docs          - Build documentation"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci-local      - Run CI checks locally"
	@echo "  make check-perf    - Check for performance regressions"
	@echo ""
	@echo "Tools:"
	@echo "  make install-tools - Install required development tools"
	@echo "  make clean         - Clean build artifacts"

# Run all tests
test:
	cargo test --all-features --verbose

# Run benchmarks
bench:
	cargo bench --all-features

# Generate coverage report with tarpaulin
coverage:
	@echo "Generating coverage report..."
	@mkdir -p coverage
	cargo tarpaulin \
		--all-features \
		--workspace \
		--timeout 300 \
		--out Html \
		--out Xml \
		--output-dir coverage \
		--exclude-files 'benches/*' 'examples/*'
	@echo "Coverage report generated in ./coverage/"
	@echo "Open ./coverage/index.html in a browser to view"

# Run clippy lints
lint:
	cargo clippy --all-targets --all-features -- -D warnings

# Format code
format:
	cargo fmt --all

# Check formatting
format-check:
	cargo fmt --all -- --check

# Check for performance regressions
check-perf:
	@./scripts/check_performance.sh master 10

# Build documentation
docs:
	cargo doc --all-features --no-deps --open

# Run all CI checks locally
ci-local: format-check lint test docs
	@echo ""
	@echo "✅ All CI checks passed!"

# Clean build artifacts
clean:
	cargo clean
	rm -rf coverage/
	rm -f performance_report.txt

# Install development tools
install-tools:
	@echo "Installing development tools..."
	cargo install cargo-tarpaulin
	cargo install cargo-criterion
	cargo install critcmp
	cargo install cargo-audit
	cargo install cargo-udeps
	cargo install cargo-minimal-versions
	@echo "✅ All tools installed!"

# Quick check before committing
pre-commit: format lint test
	@echo ""
	@echo "✅ Pre-commit checks passed!"

# Run tests with nextest (faster test runner)
test-nextest:
	@if ! command -v cargo-nextest &> /dev/null; then \
		echo "Installing cargo-nextest..."; \
		cargo install cargo-nextest --locked; \
	fi
	cargo nextest run --all-features

# Run benchmarks and save baseline
bench-baseline:
	cargo bench --all-features -- --save-baseline current

# Compare benchmarks with baseline
bench-compare:
	cargo bench --all-features -- --baseline current

# Run property-based tests only
test-proptest:
	cargo test --all-features --test proptest_suite

# Run unit tests only (no integration tests)
test-unit:
	cargo test --all-features --lib

# Run all checks including miri
check-all: format-check lint test
	@echo "Running Miri (undefined behavior detection)..."
	@if command -v cargo-miri &> /dev/null; then \
		cargo +nightly miri test --all-features; \
	else \
		echo "⚠️  Miri not installed. Install with: rustup +nightly component add miri"; \
	fi

# Memory profiling with valgrind
profile-memory:
	@echo "Building test binary..."
	@cargo test --all-features --no-run
	@TEST_BIN=$$(find target/debug/deps -name 'kizzasi_tokenizer-*' -type f -executable | head -n 1); \
	if [ -n "$$TEST_BIN" ]; then \
		echo "Running Valgrind memory profiling..."; \
		valgrind --tool=massif --massif-out-file=massif.out "$$TEST_BIN" --test-threads=1; \
		ms_print massif.out > memory_profile.txt; \
		echo "Memory profile saved to memory_profile.txt"; \
	else \
		echo "Error: Test binary not found"; \
		exit 1; \
	fi

# Security audit
audit:
	cargo audit

# Check for unused dependencies
check-unused:
	cargo +nightly udeps --all-features

# Build in release mode
release:
	cargo build --release --all-features

# Run examples
examples:
	@echo "Running examples..."
	@cargo run --example basic_quantizers --features vqvae
	@cargo run --example vqvae_tokenizer --features vqvae
	@cargo run --example advanced_features --features vqvae

# Continuous testing (watch mode)
watch:
	@if ! command -v cargo-watch &> /dev/null; then \
		echo "Installing cargo-watch..."; \
		cargo install cargo-watch; \
	fi
	cargo watch -x test

# Generate flamegraph for benchmarks
flamegraph:
	@if ! command -v cargo-flamegraph &> /dev/null; then \
		echo "Installing cargo-flamegraph..."; \
		cargo install flamegraph; \
	fi
	@echo "Generating flamegraph..."
	@cargo flamegraph --bench comprehensive_benchmarks

# Check if crate is ready to publish
publish-check:
	cargo publish --dry-run --allow-dirty

# Full release checklist
release-checklist: clean ci-local check-perf coverage publish-check
	@echo ""
	@echo "✅ Release checklist complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Update version in Cargo.toml"
	@echo "2. Update CHANGELOG.md"
	@echo "3. Commit: git commit -am 'chore: bump version to X.Y.Z'"
	@echo "4. Tag: git tag kizzasi-tokenizer-vX.Y.Z"
	@echo "5. Push: git push --tags"
