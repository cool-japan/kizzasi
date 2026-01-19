# SCIRS2 Policy Compliance

This document outlines how the `kizzasi-model` crate adheres to the SCIRS2 ecosystem policies.

## Policy Statement

**SCIRS2 POLICY**: Use SciRS2-Core instead of direct `rand` or `ndarray` dependencies.

## Compliance Status

✅ **FULLY COMPLIANT**

## Dependencies

### Core Numerical Libraries

All numerical operations use the COOLJAPAN ecosystem libraries:

```toml
[dependencies]
# COOLJAPAN Ecosystem - Used for all numerical operations
scirs2-core.workspace = true       # Array operations, random number generation
scirs2-linalg.workspace = true     # Linear algebra, BLAS/LAPACK operations
```

### No Direct Dependencies

The following are **NOT** used directly:
- ❌ `rand` - Use `scirs2_core::random` instead
- ❌ `ndarray` - Use `scirs2_core::ndarray` instead
- ❌ `rand_distr` - Use `scirs2_core::random` instead

## Code Usage Patterns

### Array Operations

**Correct (SCIRS2):**
```rust
use scirs2_core::ndarray::{Array1, Array2};

let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let y = Array2::zeros((3, 3));
```

**Incorrect (Direct ndarray):**
```rust
use ndarray::{Array1, Array2}; // ❌ DO NOT USE
```

### Random Number Generation

**Correct (SCIRS2):**
```rust
use scirs2_core::random::{rng, Rng};

let mut rng = rng();
let value = rng.random::<f32>();
```

**Incorrect (Direct rand):**
```rust
use rand::Rng; // ❌ DO NOT USE
```

## Verification

### Automated Checks

Run the following commands to verify compliance:

```bash
# 1. Check for direct rand usage (should return nothing)
grep -r "^use rand" src/

# 2. Check for direct ndarray usage (should return nothing)
grep -r "^use ndarray" src/

# 3. Verify scirs2-core usage (should show many matches)
grep -r "use scirs2_core" src/

# 4. Check Cargo.toml dependencies
grep -E "(rand|ndarray)" Cargo.toml | grep -v "scirs2"
```

### Current Compliance Status

```
Last Verified: 2026-01-18
Status: ✅ COMPLIANT
Checked By: Automated verification
Issues: None
```

## Module-by-Module Compliance

| Module | Arrays | Random | BLAS | Status |
|--------|--------|--------|------|--------|
| `mamba.rs` | scirs2-core | scirs2-core | scirs2-linalg | ✅ |
| `mamba2.rs` | scirs2-core | scirs2-core | scirs2-linalg | ✅ |
| `rwkv.rs` | scirs2-core | scirs2-core | scirs2-linalg | ✅ |
| `rwkv7.rs` | scirs2-core | scirs2-core | - | ✅ |
| `s4.rs` | scirs2-core | scirs2-core | scirs2-linalg | ✅ |
| `s5.rs` | scirs2-core | scirs2-core | - | ✅ |
| `h3.rs` | scirs2-core | scirs2-core | - | ✅ |
| `hybrid.rs` | scirs2-core | scirs2-core | - | ✅ |
| `transformer.rs` | scirs2-core | scirs2-core | - | ✅ |
| `moe.rs` | scirs2-core | scirs2-core | - | ✅ |
| `quantization.rs` | scirs2-core | - | - | ✅ |
| `training.rs` | scirs2-core | - | - | ✅ |
| `batch.rs` | scirs2-core | - | - | ✅ |
| `blas_ops.rs` | scirs2-core | - | scirs2-linalg | ✅ |
| `cache_friendly.rs` | scirs2-core | - | - | ✅ |
| `parallel_multihead.rs` | scirs2-core | - | - | ✅ |
| `simd_ops.rs` | scirs2-core | - | - | ✅ |

## Benefits of SCIRS2 Compliance

1. **Consistency**: Uniform API across all COOLJAPAN ecosystem projects
2. **Performance**: Optimized implementations with SIMD and BLAS acceleration
3. **Compatibility**: Seamless integration with other COOLJAPAN crates
4. **Maintainability**: Centralized numerical computing infrastructure
5. **Quality**: Battle-tested implementations from SciRS2

## Related Documentation

- **SciRS2-Core**: `~/work/scirs/` (reference implementation)
- **NumRS2**: `~/work/numrs/` (numerical algorithms)
- **OptiRS**: `~/work/optirs/` (optimization algorithms)
- **SciRS2-Linalg**: Linear algebra operations with BLAS/LAPACK

## Contact

For questions about SCIRS2 policy compliance:
- Check CLAUDE.md guidelines
- Review reference implementations in `~/work/scirs/`
- Consult COOLJAPAN ecosystem documentation

---

*This document is automatically verified as part of CI/CD pipeline.*
*Last Updated: 2026-01-18*
