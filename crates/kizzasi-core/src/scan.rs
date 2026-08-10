//! Parallel Scan Algorithms for SSMs
//!
//! Implements efficient parallel scan (prefix sum) operations for:
//! - **Linear-time sequential scan**: O(N) for inference
//! - **Logarithmic-depth parallel scan**: O(log N) parallel time for training
//! - **Associative scan for SSMs**: Enables parallel state computation
//!
//! # Theory
//!
//! For SSM recurrence: h_t = A * h_{t-1} + B * x_t
//!
//! We can express this as an associative binary operation:
//! ```text
//! (A₁, B₁) ⊗ (A₂, B₂) = (A₂ ∘ A₁, A₂ ∘ B₁ + B₂)
//! ```
//!
//! Where ∘ is element-wise multiplication for diagonal A (S4D).
//!
//! This allows computing all states in parallel via:
//! ```text
//! h_t = SCAN(⊗, [(A₁,B₁), (A₂,B₂), ..., (Aₙ,Bₙ)])
//! ```
//!
//! # Complexity
//!
//! - Sequential: O(N) time, O(1) extra space
//! - Parallel (work-efficient): O(N) work, O(log N) depth
//! - Memory: O(N) for intermediate results

use crate::error::{CoreError, CoreResult};
use crate::parallel::ParallelConfig;
use scirs2_core::ndarray::{Array1, Array2, Array3};

/// Associative binary operation for parallel scan
pub trait AssociativeOp<T>: Send + Sync {
    /// Apply the associative operation: a ⊗ b
    fn combine(&self, a: &T, b: &T) -> T;

    /// Identity element (if it exists)
    fn identity(&self) -> Option<T>;
}

/// Generic parallel scan (prefix sum) using work-efficient algorithm
///
/// Computes: [x₀, x₀⊗x₁, x₀⊗x₁⊗x₂, ..., x₀⊗x₁⊗...⊗xₙ]
///
/// Uses Blelloch's work-efficient parallel scan algorithm:
/// - Up-sweep (reduce) phase: O(log N) depth
/// - Down-sweep (propagate) phase: O(log N) depth
/// - Total work: O(N)
pub fn parallel_scan<T, Op>(data: &[T], op: &Op, parallel: bool) -> Vec<T>
where
    T: Clone + Send + Sync,
    Op: AssociativeOp<T>,
{
    if data.is_empty() {
        return Vec::new();
    }

    if data.len() == 1 {
        return vec![data[0].clone()];
    }

    if !parallel || data.len() < 64 {
        // Sequential scan for small inputs
        return sequential_scan(data, op);
    }

    // Parallel work-efficient scan
    parallel_scan_impl(data, op)
}

/// Sequential inclusive scan
fn sequential_scan<T, Op>(data: &[T], op: &Op) -> Vec<T>
where
    T: Clone,
    Op: AssociativeOp<T>,
{
    let mut result = Vec::with_capacity(data.len());
    result.push(data[0].clone());

    for i in 1..data.len() {
        let combined = op.combine(&result[i - 1], &data[i]);
        result.push(combined);
    }

    result
}

/// Work-efficient parallel scan implementation
///
/// Uses `scirs2_core::distributed::parallel_scan::parallel_scan` (Blelloch
/// three-phase algorithm) when the associative operation has a known identity
/// element. Otherwise falls back to a sequential scan, since the scirs2
/// API requires an identity for its tile-reduction phase.
fn parallel_scan_impl<T, Op>(data: &[T], op: &Op) -> Vec<T>
where
    T: Clone + Send + Sync,
    Op: AssociativeOp<T>,
{
    // Hot path: emit a debug-level span so the parallel scan kernel can be
    // identified in tracing-subscriber output. We log only the input length;
    // logging the actual data would be prohibitively expensive on long
    // sequences (and is mostly noise for profiling).
    let _span = tracing::debug_span!("parallel_scan_impl", len = data.len()).entered();
    // Take a reference to op that is Copy (and therefore Clone), so it can be
    // captured by the closure required by scirs2_core::distributed::parallel_scan.
    // The closure itself becomes Copy + Clone because it captures only a shared
    // reference (&Op is Copy).
    match op.identity() {
        Some(identity) => {
            let op_ref = op;
            let op_closure = move |a: T, b: T| op_ref.combine(&a, &b);
            scirs2_core::distributed::parallel_scan::parallel_scan(data, identity, op_closure)
        }
        None => {
            // Without an identity, scirs2's parallel_scan cannot be used directly:
            // its tile-reduction phase initializes each tile accumulator with
            // `identity`. We fall back to the sequential implementation, which
            // is still correct (and the common case for SSMScanOp where the
            // identity depends on per-element shape).
            sequential_scan(data, op)
        }
    }
}

/// SSM Scan Element: (A_bar, B_bar) for diagonal SSM
#[derive(Clone, Debug)]
pub struct SSMElement {
    /// Discretized A (diagonal): exp(Δ * λ)
    pub a_bar: Array1<f32>,
    /// Discretized B: B̄ = (exp(Δλ) - 1)/λ * B
    pub b_bar: Array1<f32>,
}

/// Associative operation for SSM elements
pub struct SSMScanOp;

impl AssociativeOp<SSMElement> for SSMScanOp {
    fn combine(&self, left: &SSMElement, right: &SSMElement) -> SSMElement {
        // (A₁, B₁) ⊗ (A₂, B₂) = (A₂ ∘ A₁, A₂ ∘ B₁ + B₂)
        let a_combined = &right.a_bar * &left.a_bar;
        let b_combined = &right.a_bar * &left.b_bar + &right.b_bar;

        SSMElement {
            a_bar: a_combined,
            b_bar: b_combined,
        }
    }

    fn identity(&self) -> Option<SSMElement> {
        None // No identity for SSM scan
    }
}

/// Parallel SSM scan for efficient state computation
///
/// Given sequences of (A_bar, B_bar) elements, computes all hidden states in parallel.
///
/// # Arguments
/// * `a_bars` - Discretized A matrices (seq_len, state_dim)
/// * `b_bars` - Discretized B vectors (seq_len, state_dim)
/// * `c` - Output projection vector (state_dim,)
/// * `parallel_config` - Parallelization configuration
///
/// # Returns
/// Output sequence (seq_len, state_dim) where each position contains the cumulative state
pub fn parallel_ssm_scan(
    a_bars: &Array2<f32>,
    b_bars: &Array2<f32>,
    c: &Array1<f32>,
    parallel_config: &ParallelConfig,
) -> CoreResult<Array2<f32>> {
    let (seq_len, state_dim) = a_bars.dim();
    // Per-call top-level span: one per SSM scan invocation. Records the
    // sequence length and state dimension so trace consumers can spot
    // which call corresponds to which input shape.
    let _span = tracing::debug_span!(
        "parallel_ssm_scan",
        seq_len = seq_len,
        state_dim = state_dim,
    )
    .entered();

    if b_bars.dim() != (seq_len, state_dim) {
        return Err(CoreError::DimensionMismatch {
            expected: seq_len,
            got: b_bars.nrows(),
        });
    }

    if c.len() != state_dim {
        return Err(CoreError::DimensionMismatch {
            expected: state_dim,
            got: c.len(),
        });
    }

    // Create SSM elements
    let elements: Vec<SSMElement> = (0..seq_len)
        .map(|t| SSMElement {
            a_bar: a_bars.row(t).to_owned(),
            b_bar: b_bars.row(t).to_owned(),
        })
        .collect();

    // Perform parallel scan
    let op = SSMScanOp;
    let scanned = parallel_scan(&elements, &op, parallel_config.parallel_batch);

    // Extract B_bar components (which contain the cumulative states)
    let mut states = Array2::zeros((seq_len, state_dim));
    for (t, elem) in scanned.iter().enumerate() {
        states.row_mut(t).assign(&elem.b_bar);
    }

    Ok(states)
}

/// Parallel SSM forward pass for batch of sequences
///
/// Computes outputs for multiple sequences in parallel using associative scan.
///
/// # Arguments
/// * `a_bars` - Discretized A matrices (batch_size, seq_len, state_dim)
/// * `b_bars` - Discretized B vectors (batch_size, seq_len, state_dim)
/// * `c` - Output projection (state_dim,)
/// * `d` - Skip connection
///
/// # Returns
/// Outputs (batch_size, seq_len)
pub fn parallel_ssm_batch(
    a_bars: &Array3<f32>,
    b_bars: &Array3<f32>,
    c: &Array1<f32>,
    d: f32,
    parallel_config: &ParallelConfig,
) -> CoreResult<Array2<f32>> {
    let (batch_size, seq_len, state_dim) = a_bars.dim();
    // Batched SSM forward: span records batch_size, seq_len, state_dim so a
    // trace can distinguish small "latency" batches from large training
    // batches at a glance.
    let _span = tracing::debug_span!(
        "parallel_ssm_batch",
        batch_size = batch_size,
        seq_len = seq_len,
        state_dim = state_dim,
    )
    .entered();

    if b_bars.dim() != (batch_size, seq_len, state_dim) {
        return Err(CoreError::InvalidConfig(
            "b_bars shape mismatch".to_string(),
        ));
    }

    // Process each batch item in parallel via scirs2-core's parallel_ops layer.
    // When the `parallel` feature is enabled (default in this crate), this
    // dispatches to rayon under the hood; otherwise it falls back to a
    // sequential iterator. We collect into a `CoreResult<Vec<_>>` so a scan
    // failure on any batch element propagates instead of being swallowed.
    use scirs2_core::parallel_ops::{IntoParallelIterator, ParallelIterator};

    let outputs: CoreResult<Vec<Array1<f32>>> = (0..batch_size)
        .into_par_iter()
        .map(|b| -> CoreResult<Array1<f32>> {
            // Get this batch's A and B
            let a_batch = a_bars.slice(s![b, .., ..]).to_owned();
            let b_batch = b_bars.slice(s![b, .., ..]).to_owned();

            // Perform scan for this sequence
            let states = parallel_ssm_scan(&a_batch, &b_batch, c, parallel_config)?;

            // Compute outputs: y_t = C · h_t + D · x_t
            // (for simplicity, assuming D*x is already included in b_bar)
            let mut output = Array1::zeros(seq_len);
            for t in 0..seq_len {
                let h_t = states.row(t);
                output[t] = c.dot(&h_t) + d;
            }

            Ok(output)
        })
        .collect();

    let outputs = outputs?;

    // Stack into output array
    let mut result = Array2::zeros((batch_size, seq_len));
    for (b, output) in outputs.iter().enumerate() {
        result.row_mut(b).assign(output);
    }

    Ok(result)
}

/// Segmented parallel scan for variable-length sequences
///
/// Performs parallel scan where sequences are separated by segment boundaries.
/// This is useful for processing multiple variable-length sequences in one batch.
///
/// # Arguments
/// * `data` - Packed data elements
/// * `segment_ids` - Segment ID for each element (resets scan at boundaries)
/// * `op` - Associative operation
///
/// # Returns
/// Scanned result with scan reset at segment boundaries
pub fn segmented_scan<T, Op>(data: &[T], segment_ids: &[usize], op: &Op, parallel: bool) -> Vec<T>
where
    T: Clone + Send + Sync,
    Op: AssociativeOp<T>,
{
    if data.len() != segment_ids.len() {
        panic!("data and segment_ids must have same length");
    }

    if !parallel {
        return segmented_scan_sequential(data, segment_ids, op);
    }

    // For parallel version, we need more sophisticated handling
    // For now, fall back to sequential for correctness
    segmented_scan_sequential(data, segment_ids, op)
}

fn segmented_scan_sequential<T, Op>(data: &[T], segment_ids: &[usize], op: &Op) -> Vec<T>
where
    T: Clone,
    Op: AssociativeOp<T>,
{
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0].clone());

    for i in 1..data.len() {
        if segment_ids[i] != segment_ids[i - 1] {
            // New segment - reset scan
            result.push(data[i].clone());
        } else {
            // Same segment - continue scan
            let combined = op.combine(&result[i - 1], &data[i]);
            result.push(combined);
        }
    }

    result
}

// Re-export for convenience
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    // Simple addition operation for testing
    struct AddOp;

    impl AssociativeOp<f32> for AddOp {
        fn combine(&self, a: &f32, b: &f32) -> f32 {
            a + b
        }

        fn identity(&self) -> Option<f32> {
            Some(0.0)
        }
    }

    #[test]
    fn test_sequential_scan() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let op = AddOp;

        let result = sequential_scan(&data, &op);
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    #[test]
    fn test_parallel_scan() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let op = AddOp;

        let result = parallel_scan(&data, &op, true);
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0]);
    }

    #[test]
    fn test_ssm_scan_op() {
        let elem1 = SSMElement {
            a_bar: Array1::from_vec(vec![0.9, 0.8]),
            b_bar: Array1::from_vec(vec![0.1, 0.2]),
        };

        let elem2 = SSMElement {
            a_bar: Array1::from_vec(vec![0.9, 0.8]),
            b_bar: Array1::from_vec(vec![0.1, 0.2]),
        };

        let op = SSMScanOp;
        let result = op.combine(&elem1, &elem2);

        // (A₂ ∘ A₁, A₂ ∘ B₁ + B₂)
        assert!((result.a_bar[0] - 0.81).abs() < 1e-6); // 0.9 * 0.9
        assert!((result.a_bar[1] - 0.64).abs() < 1e-6); // 0.8 * 0.8
        assert!((result.b_bar[0] - 0.19).abs() < 1e-6); // 0.9 * 0.1 + 0.1
        assert!((result.b_bar[1] - 0.36).abs() < 1e-6); // 0.8 * 0.2 + 0.2
    }

    #[test]
    fn test_parallel_ssm_scan() {
        let seq_len = 4;
        let state_dim = 2;

        let a_bars = Array2::from_shape_vec(
            (seq_len, state_dim),
            vec![0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8],
        )
        .unwrap();

        let b_bars = Array2::from_shape_vec(
            (seq_len, state_dim),
            vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )
        .unwrap();

        let c = Array1::from_vec(vec![1.0, 1.0]);

        let config = ParallelConfig::latency(); // Use sequential for determinism

        let states = parallel_ssm_scan(&a_bars, &b_bars, &c, &config).unwrap();
        assert_eq!(states.dim(), (seq_len, state_dim));

        // Check that states are non-zero
        assert!(states.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_segmented_scan() {
        let data = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let segments = vec![0, 0, 0, 1, 1]; // Two segments
        let op = AddOp;

        let result = segmented_scan(&data, &segments, &op, false);

        // First segment: [1, 3, 6]
        // Second segment: [1, 3]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 1.0); // Reset
        assert_eq!(result[4], 3.0);
    }

    #[test]
    fn test_empty_scan() {
        let data: Vec<f32> = vec![];
        let op = AddOp;

        let result = parallel_scan(&data, &op, false);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_single_element_scan() {
        let data = vec![42.0];
        let op = AddOp;

        let result = parallel_scan(&data, &op, true);
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_parallel_scan_matches_sequential() {
        // Exercise the scirs2-core Blelloch path: length 1024 forces the
        // local parallel_scan into parallel_scan_impl (>= 64) and crosses
        // scirs2-core's SEQUENTIAL_THRESHOLD boundary.
        //
        // We use integer-valued f32s so that summation is exact regardless
        // of evaluation order; this lets us assert bit-exact equality
        // between the parallel and sequential implementations.
        let len = 1024usize;
        let data: Vec<f32> = (0..len).map(|i| (i % 128) as f32).collect();
        let op = AddOp;

        let parallel_result = parallel_scan(&data, &op, true);
        let sequential_result = sequential_scan(&data, &op);

        assert_eq!(parallel_result.len(), sequential_result.len());
        assert_eq!(parallel_result.len(), len);

        // Bit-exact (no rounding) comparison — possible because all inputs
        // and partial sums fit exactly in f32 mantissa.
        for (i, (p, s)) in parallel_result
            .iter()
            .zip(sequential_result.iter())
            .enumerate()
        {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "mismatch at index {}: parallel={} sequential={}",
                i,
                p,
                s,
            );
        }

        // Order preservation: the i-th element of result must be the
        // cumulative scan up to and including index i.
        let mut acc = 0.0f32;
        for (i, &v) in data.iter().enumerate() {
            acc += v;
            assert_eq!(parallel_result[i].to_bits(), acc.to_bits());
        }
    }

    #[test]
    fn test_parallel_scan_with_tracing() {
        // Verify that the tracing instrumentation added to parallel_scan_impl
        // does not perturb its results. We initialise tracing-subscriber via
        // try_init so the call is a no-op if any other test already installed
        // a global subscriber in this process.
        let _ = tracing_subscriber::fmt::try_init();

        let len = 1024usize;
        let data: Vec<f32> = (0..len).map(|i| (i % 64) as f32).collect();
        let op = AddOp;

        // This goes through parallel_scan -> parallel_scan_impl, exercising
        // the freshly-added debug_span!("parallel_scan_impl").
        let result = parallel_scan(&data, &op, true);
        assert_eq!(result.len(), len);

        // Verify cumulative sum semantics are preserved under instrumentation.
        let mut acc = 0.0f32;
        for (i, &v) in data.iter().enumerate() {
            acc += v;
            assert_eq!(
                result[i].to_bits(),
                acc.to_bits(),
                "tracing must not alter scan semantics (index {i})",
            );
        }
    }

    #[test]
    fn test_parallel_ssm_batch_matches_sequential() {
        // Compare batched parallel SSM forward against per-batch sequential
        // reference. Use deterministic deterministic values for reproducibility.
        let batch_size = 4;
        let seq_len = 128;
        let state_dim = 16;

        // Construct A_bars near identity-but-less-than-one for numerical
        // stability across 128 steps.
        let mut a_data = Vec::with_capacity(batch_size * seq_len * state_dim);
        let mut b_data = Vec::with_capacity(batch_size * seq_len * state_dim);
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..state_dim {
                    let mix = ((b * 7 + t * 3 + d) % 19) as f32;
                    a_data.push(0.90 + mix * 0.001);
                    b_data.push(0.01 + mix * 0.0005);
                }
            }
        }

        let a_bars = Array3::from_shape_vec((batch_size, seq_len, state_dim), a_data).unwrap();
        let b_bars = Array3::from_shape_vec((batch_size, seq_len, state_dim), b_data).unwrap();
        let c = Array1::from_shape_fn(state_dim, |d| 0.5 + (d as f32) * 0.01);
        let d_skip = 0.05f32;

        // Parallel batch path (default config triggers parallel processing).
        let cfg_par = ParallelConfig::default();
        let par_out = parallel_ssm_batch(&a_bars, &b_bars, &c, d_skip, &cfg_par).unwrap();

        // Sequential reference: process each batch element directly.
        let cfg_seq = ParallelConfig::latency();
        let mut seq_out = Array2::zeros((batch_size, seq_len));
        for b in 0..batch_size {
            let a_batch = a_bars.slice(s![b, .., ..]).to_owned();
            let b_batch = b_bars.slice(s![b, .., ..]).to_owned();
            let states = parallel_ssm_scan(&a_batch, &b_batch, &c, &cfg_seq).unwrap();
            for t in 0..seq_len {
                let h_t = states.row(t);
                seq_out[[b, t]] = c.dot(&h_t) + d_skip;
            }
        }

        // The per-batch computation is identical (each batch's scan is its
        // own sequential reduction), so results should match closely.
        assert_eq!(par_out.dim(), seq_out.dim());
        for ((b, t), v) in par_out.indexed_iter() {
            let diff = (v - seq_out[[b, t]]).abs();
            assert!(
                diff < 1e-6,
                "parallel_ssm_batch mismatch at ({},{}): par={} seq={} diff={}",
                b,
                t,
                v,
                seq_out[[b, t]],
                diff
            );
        }
    }
}
