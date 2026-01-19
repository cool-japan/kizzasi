# CI/CD Scripts for kizzasi-tokenizer

This directory contains scripts for continuous integration, testing, and performance monitoring.

## Scripts

### `check_performance.sh`

Performance regression detection script that compares benchmark results between branches.

**Usage:**
```bash
./scripts/check_performance.sh [base_branch] [threshold_percent]
```

**Arguments:**
- `base_branch`: Branch to compare against (default: `master`)
- `threshold_percent`: Performance regression threshold in % (default: `10`)

**Examples:**
```bash
# Compare current branch with master, 10% threshold
./scripts/check_performance.sh master 10

# Compare with main branch, 5% threshold
./scripts/check_performance.sh main 5
```

**How it works:**
1. Runs benchmarks on your current branch
2. Switches to the base branch and runs benchmarks
3. Compares results using `critcmp`
4. Reports any regressions exceeding the threshold
5. Generates a detailed performance report

**Requirements:**
- `cargo-criterion` - Install with `cargo install cargo-criterion`
- `critcmp` - Install with `cargo install critcmp`
- Git repository with uncommitted changes will be stashed

**Output:**
- Console output with colored regression/improvement indicators
- `performance_report.txt` - Detailed report saved to project root
- Exit code 0 if no regressions, 1 if regressions detected

## GitHub Actions Workflows

The CI/CD pipeline is automated through GitHub Actions workflows in `.github/workflows/`:

### `ci.yml` - Continuous Integration
- Runs on push and pull requests
- Tests on Ubuntu, macOS, and Windows
- Tests with stable, beta, and nightly Rust
- Runs clippy, rustfmt, and documentation checks
- Includes minimal versions check, Miri, security audit, and unused dependencies

### `benchmark.yml` - Performance Testing
- Automated benchmarking on every push/PR
- Performance comparison between base and PR branches
- Memory profiling with Valgrind
- Throughput analysis
- Benchmark results stored as artifacts

### `coverage.yml` - Code Coverage
- Generates coverage reports with Tarpaulin and grcov
- Uploads to Codecov
- Coverage diff for pull requests
- Line-by-line coverage reporting

### `release.yml` - Release Automation
- Triggered on version tags (e.g., `kizzasi-tokenizer-v0.1.0`)
- Pre-release checks (tests, clippy, docs)
- Multi-platform builds (Linux, macOS, Windows)
- GitHub release creation with changelog
- Automatic publication to crates.io
- Documentation deployment

## Local Testing

You can test GitHub Actions workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash  # Linux

# Test CI workflow
act -j test

# Test benchmark workflow
act -j benchmark

# Test coverage workflow
act -j coverage
```

## Performance Monitoring

### Automated Benchmark Tracking

Benchmarks are automatically tracked over time using `github-action-benchmark`. Results are:
- Stored in the `gh-pages` branch
- Visualized at `https://<username>.github.io/<repo>/kizzasi-tokenizer/bench/`
- Alerted when performance degrades by >150%

### Regression Detection

The `performance-comparison` job in `benchmark.yml` automatically:
1. Runs benchmarks on PR branch
2. Runs benchmarks on base branch
3. Compares results with `critcmp`
4. Posts comparison to GitHub PR summary

## Coverage Reporting

Code coverage is tracked using:
- **Tarpaulin**: Line and branch coverage
- **grcov**: LLVM-based coverage with detailed line-by-line reports
- **Codecov**: Coverage history and PR diffs

Coverage reports are:
- Uploaded to Codecov on every push
- Compared between base and PR branches
- Available as HTML artifacts in workflow runs

## Release Process

### Automatic Release

1. Update version in `Cargo.toml`
2. Commit changes: `git commit -am "chore: bump version to 0.x.0"`
3. Create and push tag: `git tag kizzasi-tokenizer-v0.x.0 && git push --tags`
4. GitHub Actions will:
   - Run all tests and checks
   - Build artifacts for all platforms
   - Create GitHub release with changelog
   - Publish to crates.io
   - Deploy documentation

### Manual Release (via workflow_dispatch)

1. Go to Actions tab in GitHub
2. Select "Release" workflow
3. Click "Run workflow"
4. Specify crate name and version
5. Workflow will handle the rest

## Troubleshooting

### Performance script fails with "Not in a git repository"
Ensure you're running the script from within the git repository.

### Benchmark comparison shows no results
Make sure you have committed changes on both branches. The script requires valid git commits to compare.

### Coverage upload fails
Ensure `CODECOV_TOKEN` is set in GitHub repository secrets.

### Release fails on version mismatch
Ensure the version in `Cargo.toml` matches the version in the git tag (e.g., tag `v0.1.0` requires version `0.1.0` in Cargo.toml).

## Maintenance

### Updating Dependencies in Workflows

GitHub Actions dependencies should be updated regularly:
- `actions/checkout`: Keep at latest v4
- `dtolnay/rust-toolchain`: Pin to specific Rust versions
- `actions/cache`: Update to latest v4
- Other actions: Check for updates monthly

### Benchmark Baseline Management

Benchmarks use Criterion's baseline feature. To reset baselines:
```bash
cargo bench --bench comprehensive_benchmarks -- --save-baseline new-baseline
```

### Coverage Threshold Adjustments

Coverage thresholds can be adjusted in `coverage.yml`:
- Warning threshold for coverage decrease: Currently 1%
- Can be made stricter or more lenient as needed
