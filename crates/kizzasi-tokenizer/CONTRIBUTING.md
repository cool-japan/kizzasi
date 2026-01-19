# Contributing to kizzasi-tokenizer

Thank you for your interest in contributing to kizzasi-tokenizer! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Rust 1.70+ (stable)
- Git
- (Optional) Docker for local CI testing

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/cool-japan/kizzasi.git
cd kizzasi/crates/kizzasi-tokenizer
```

2. Install development tools:
```bash
make install-tools
```

This installs:
- `cargo-tarpaulin` - Code coverage
- `cargo-criterion` - Benchmarking
- `critcmp` - Benchmark comparison
- `cargo-audit` - Security auditing
- `cargo-udeps` - Unused dependency detection
- `cargo-minimal-versions` - Minimum version testing

3. Run tests to verify setup:
```bash
make test
```

## Development Workflow

### Before Making Changes

1. Create a feature branch:
```bash
git checkout -b feature/my-feature
```

2. Ensure all tests pass:
```bash
make test
```

### Making Changes

1. Write code following Rust conventions:
   - Use `cargo fmt` to format code
   - Run `cargo clippy` to catch common mistakes
   - Add documentation for public APIs
   - Write tests for new functionality

2. Run pre-commit checks:
```bash
make pre-commit
```

This runs:
- Code formatting check
- Clippy lints
- All tests

### Testing

We have comprehensive testing requirements:

#### Unit Tests
```bash
make test-unit
```

#### Integration Tests
```bash
make test
```

#### Property-Based Tests
```bash
make test-proptest
```

#### Benchmarks
```bash
make bench
```

### Code Coverage

Generate a coverage report:
```bash
make coverage
```

Open `coverage/index.html` in your browser to view the report.

**Coverage Requirements:**
- New code should have >80% coverage
- Critical paths should have >95% coverage
- Property-based tests are encouraged for mathematical properties

### Performance Testing

Check for performance regressions:
```bash
make check-perf
```

This compares your branch's performance against `master`. Regressions >10% will fail the check.

### Documentation

Build and view documentation:
```bash
make docs
```

**Documentation Requirements:**
- All public APIs must have rustdoc comments
- Include examples in doc comments
- Use `//!` for module-level documentation
- Add usage examples to `examples/` directory

## Pull Request Process

### Before Submitting

1. Ensure all checks pass:
```bash
make ci-local
```

2. Update documentation if needed
3. Add examples for new features
4. Update CHANGELOG.md (if applicable)

### Submitting a PR

1. Push your branch:
```bash
git push origin feature/my-feature
```

2. Create a pull request on GitHub

3. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Breaking changes (if any)

### PR Review Process

Your PR will be automatically checked for:
- ✅ All tests passing on Ubuntu, macOS, and Windows
- ✅ Clippy lints passing
- ✅ Code formatting
- ✅ Documentation building without warnings
- ✅ No performance regressions >150%
- ✅ Code coverage maintained or improved

Manual review will check for:
- Code quality and style
- Test coverage
- Documentation completeness
- API design
- Performance implications

## Code Style Guidelines

### Rust Style

Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/):

```rust
// Good: Clear, documented, tested
/// Encodes a signal into tokens using linear quantization.
///
/// # Arguments
///
/// * `signal` - Input signal to encode
/// * `bits` - Number of bits for quantization (1-16)
///
/// # Returns
///
/// Encoded tokens as a 1D array
///
/// # Examples
///
/// ```
/// use kizzasi_tokenizer::LinearQuantizer;
///
/// let quantizer = LinearQuantizer::new(8, -1.0, 1.0)?;
/// let tokens = quantizer.encode(&signal)?;
/// # Ok::<(), kizzasi_tokenizer::TokenizerError>(())
/// ```
pub fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
    // Implementation
}
```

### Naming Conventions

- **Types**: `PascalCase` (e.g., `LinearQuantizer`)
- **Functions**: `snake_case` (e.g., `encode_batch`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_BITS`)
- **Module files**: `snake_case` (e.g., `linear_quantizer.rs`)

### Error Handling

Use the `TokenizerError` enum for all errors:

```rust
// Good: Specific error with context
if signal.is_empty() {
    return Err(TokenizerError::invalid_input(
        "encode",
        "Signal cannot be empty"
    ));
}

// Bad: Generic error
if signal.is_empty() {
    return Err(TokenizerError::InvalidConfig("empty".into()));
}
```

### Testing Guidelines

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        // Setup
        let quantizer = LinearQuantizer::new(8, -1.0, 1.0).unwrap();
        let signal = Array1::from_vec(vec![0.0, 0.5, -0.5, 1.0]);

        // Execute
        let encoded = quantizer.encode(&signal).unwrap();
        let decoded = quantizer.decode(&encoded).unwrap();

        // Verify
        assert_eq!(decoded.len(), signal.len());
        for (orig, dec) in signal.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01); // Within quantization error
        }
    }
}
```

## Benchmarking

### Writing Benchmarks

Add benchmarks to `benches/comprehensive_benchmarks.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_my_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_feature");

    for size in [64, 256, 1024, 4096] {
        group.bench_function(format!("size_{}", size), |b| {
            let data = vec![0.0f32; size];
            b.iter(|| {
                // Benchmark code
                black_box(my_feature(&data))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_my_feature);
criterion_main!(benches);
```

### Running Benchmarks

```bash
# Run all benchmarks
make bench

# Save baseline
make bench-baseline

# Compare with baseline
make bench-compare
```

## CI/CD Pipeline

### Automated Checks

Every push and PR triggers:

1. **Test Suite** (`ci.yml`)
   - Multi-platform testing (Ubuntu, macOS, Windows)
   - Multiple Rust versions (stable, beta, nightly)
   - Clippy and rustfmt checks
   - Documentation build
   - Miri undefined behavior detection
   - Security audit
   - Minimal versions check
   - Unused dependencies check

2. **Benchmarks** (`benchmark.yml`)
   - Performance benchmarking
   - Regression detection
   - Memory profiling
   - Throughput analysis

3. **Coverage** (`coverage.yml`)
   - Code coverage with Tarpaulin
   - Coverage diff for PRs
   - Line-by-line coverage with grcov

### Local CI Testing

Test workflows locally with [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run CI tests
act -j test

# Run benchmarks
act -j benchmark

# Run coverage
act -j coverage
```

Or use the Makefile:

```bash
make ci-local
```

## Release Process

Releases are automated through GitHub Actions:

### Version Bump

1. Update version in `Cargo.toml`:
```toml
[package]
version = "0.2.0"  # Bump version
```

2. Update CHANGELOG.md with changes

3. Commit and tag:
```bash
git commit -am "chore: bump version to 0.2.0"
git tag kizzasi-tokenizer-v0.2.0
git push --tags
```

4. GitHub Actions will:
   - Run all tests
   - Build for all platforms
   - Create GitHub release
   - Publish to crates.io
   - Deploy documentation

### Pre-Release Checklist

Use the Makefile for a comprehensive check:

```bash
make release-checklist
```

This runs:
- Clean build
- All CI checks
- Performance regression check
- Coverage report
- Publish dry-run

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/cool-japan/kizzasi/discussions)
- **Bugs**: Open an [Issue](https://github.com/cool-japan/kizzasi/issues)
- **Security**: Email security@cool-japan.org

## Code of Conduct

Be respectful, inclusive, and collaborative. We follow the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (check LICENSE file).

## Additional Resources

- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Rust Book](https://doc.rust-lang.org/book/)
- [Criterion.rs Guide](https://bheisler.github.io/criterion.rs/book/)
- [Property-based testing with proptest](https://proptest-rs.github.io/proptest/)

Thank you for contributing! 🦀
