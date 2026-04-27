//! Performance regression tests
//!
//! These tests establish performance baselines and catch regressions.
//! Run with: cargo test --test perf_regression --release -p kizzasi-model -- --nocapture
//!
//! Design notes:
//! - `assert_time_budget` is **non-fatal**: it warns via `eprintln!` but does not
//!   panic. This avoids CI flakiness on slow shared runners.
//! - All timings are wall-clock (not CPU) via `std::time::Instant`.
//! - Tests are intentionally small so they run in debug mode too.

use kizzasi_core::SignalPredictor;
use kizzasi_logic::parallel_advanced::check_range_constraints_simd;
use kizzasi_model::mamba::{Mamba, MambaConfig};
use kizzasi_model::quantize::{quantize_to_int8, ModelQuantizer};
use kizzasi_model::AutoregressiveModel;
use kizzasi_tokenizer::{LinearQuantizer, Quantizer};
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Timing harness
// ============================================================================

/// Run `f`, warn (but do **not** panic) if it exceeds `budget`.
///
/// Returns the value produced by `f` so callers can do further assertions.
fn assert_time_budget<F, R>(label: &str, budget: Duration, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();

    if elapsed > budget {
        eprintln!(
            "PERF WARNING: '{}' took {:.2?}, budget was {:.2?} ({:.1}× over budget)",
            label,
            elapsed,
            budget,
            elapsed.as_secs_f64() / budget.as_secs_f64()
        );
    } else {
        eprintln!(
            "PERF OK: '{}' took {:.2?} (budget {:.2?})",
            label, elapsed, budget
        );
    }

    result
}

// ============================================================================
// Mamba model forward-step performance
// ============================================================================

/// Verify that a tiny Mamba model can complete 100 single-step predictions
/// within a generous wall-clock budget. The actual production budget is tighter,
/// but we keep it loose here to avoid flakiness on slow CI runners.
#[test]
fn test_mamba_forward_perf() {
    // Use the `tiny` constructor so the test is fast even in debug mode.
    let config = MambaConfig::tiny(1)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(2);
    let mut model = Mamba::new(config).expect("Mamba::new should succeed with valid tiny config");

    let input = Array1::from_elem(1, 0.1_f32);
    let budget = Duration::from_millis(500);

    assert_time_budget("Mamba step ×100 (tiny 32-dim, 2-layer)", budget, || {
        for _ in 0..100 {
            let _ = model
                .step(&input)
                .expect("Mamba::step should not fail on valid input");
        }
    });
}

// ============================================================================
// LinearQuantizer encode performance
// ============================================================================

/// Verify that encoding 1 000 individual scalar samples through `LinearQuantizer`
/// is fast — the quantizer is on the hot inference path for real-time audio.
#[test]
fn test_linear_quantizer_encode_perf() {
    let q = LinearQuantizer::new(-1.0_f32, 1.0_f32, 8)
        .expect("LinearQuantizer::new with valid args should succeed");

    let budget = Duration::from_millis(100);

    assert_time_budget("LinearQuantizer::quantize ×1000 scalars", budget, || {
        let mut checksum: i32 = 0;
        for i in 0..1_000_i32 {
            // Sweep the full [-1, 1] range
            let sample = (i as f32 / 1_000.0) * 2.0_f32 - 1.0_f32;
            checksum = checksum.wrapping_add(q.quantize(sample));
        }
        // Use the checksum so the compiler cannot elide the loop.
        assert!(checksum.abs() < i32::MAX);
    });
}

/// Verify that encoding a batch of 1 000 samples via `SignalTokenizer::encode`
/// (array API) is fast.
#[test]
fn test_linear_quantizer_batch_encode_perf() {
    use kizzasi_tokenizer::SignalTokenizer;

    let q =
        LinearQuantizer::new(-1.0_f32, 1.0_f32, 8).expect("LinearQuantizer::new should succeed");

    let signal: Array1<f32> =
        Array1::from_iter((0..1_000).map(|i| (i as f32 / 1_000.0) * 2.0_f32 - 1.0_f32));
    let budget = Duration::from_millis(100);

    assert_time_budget(
        "LinearQuantizer::encode (array, 1000 samples)",
        budget,
        || {
            let encoded = q.encode(&signal).expect("encode should not fail");
            assert_eq!(encoded.len(), 1_000);
        },
    );
}

// ============================================================================
// SIMD range constraint check performance
// ============================================================================

/// Verify that checking 10 000 values against a range bound is sub-50 ms
/// even without auto-vectorization (debug builds). In release mode with SIMD
/// this should be < 1 ms.
#[test]
fn test_simd_range_check_perf() {
    let values: Vec<f64> = (0..10_000).map(|i| i as f64 / 10_000.0).collect();
    let budget = Duration::from_millis(50);

    assert_time_budget(
        "check_range_constraints_simd (10 000 f64 values)",
        budget,
        || {
            let (all_ok, violations) = check_range_constraints_simd(&values, 0.0_f64, 1.0_f64);
            // All values are in [0, 1] so there should be no violations.
            assert!(
                all_ok,
                "expected all values in range, got {} violations",
                violations.len()
            );
            assert!(violations.is_empty());
        },
    );
}

/// Check that checking a batch with deliberate violations is also fast.
#[test]
fn test_simd_range_check_with_violations_perf() {
    // Half the values are out of range.
    let values: Vec<f64> = (0..10_000)
        .map(|i| if i % 2 == 0 { 0.5_f64 } else { 2.0_f64 })
        .collect();
    let budget = Duration::from_millis(50);

    assert_time_budget(
        "check_range_constraints_simd (10 000 values, ~50% violations)",
        budget,
        || {
            let (all_ok, violations) = check_range_constraints_simd(&values, 0.0_f64, 1.0_f64);
            assert!(!all_ok, "expected violations");
            assert_eq!(
                violations.len(),
                5_000,
                "expected exactly 5000 out-of-range indices"
            );
        },
    );
}

// ============================================================================
// INT8 post-training quantization performance
// ============================================================================

/// Quantize and dequantize 10 synthetic weight tensors of 1 024 elements each.
/// This exercises the PTQ pipeline that runs once before deployment.
#[test]
fn test_quantize_weights_perf() {
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    for i in 0..10_usize {
        // Realistic weight distribution: near-zero values with occasional outliers.
        let tensor: Vec<f32> = (0..1_024)
            .map(|j| {
                let x = j as f32 / 1_024.0;
                (x * std::f32::consts::TAU).sin() * 0.5_f32
            })
            .collect();
        weights.insert(format!("layer{i}.weight"), tensor);
    }

    let budget = Duration::from_millis(200);

    assert_time_budget(
        "quantize_to_int8 + dequantize_weights (10 × 1024-element tensors)",
        budget,
        || {
            let quantized = quantize_to_int8(&weights).expect("quantize_to_int8 should succeed");
            assert_eq!(quantized.len(), 10, "all 10 tensors should be quantized");

            let recovered = ModelQuantizer::dequantize_weights(&quantized)
                .expect("dequantize_weights should succeed");
            assert_eq!(recovered.len(), 10);

            // Sanity-check round-trip error is bounded (INT8 error ≤ scale ≈ 0.004)
            for (name, orig) in &weights {
                let deq = &recovered[name];
                let max_err = orig
                    .iter()
                    .zip(deq.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_err < 0.02_f32,
                    "tensor '{}': max round-trip error {} exceeds 0.02",
                    name,
                    max_err
                );
            }
        },
    );
}

// ============================================================================
// Mamba state management performance
// ============================================================================

/// Verify that saving and restoring hidden states (used by beam search and
/// speculative decoding) is fast.
#[test]
fn test_mamba_state_save_restore_perf() {
    let config = MambaConfig::tiny(1)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(2);
    let mut model = Mamba::new(config).expect("Mamba::new should succeed");

    let input = Array1::from_elem(1, 0.5_f32);

    // Warm up the state.
    for _ in 0..10 {
        let _ = model.step(&input).expect("step should not fail");
    }

    let budget = Duration::from_millis(50);

    assert_time_budget("get_states + set_states ×100", budget, || {
        for _ in 0..100 {
            let states = model.get_states();
            model
                .set_states(states)
                .expect("set_states should not fail with valid states");
        }
    });
}
