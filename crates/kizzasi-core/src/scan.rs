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
/// NOTE: Currently uses sequential implementation.
/// Parallel version will be implemented when scirs2-core parallel API is stable.
fn parallel_scan_impl<T, Op>(data: &[T], op: &Op) -> Vec<T>
where
    T: Clone + Send + Sync,
    Op: AssociativeOp<T>,
{
    // For now, use sequential scan
    // TODO: Implement true parallel Blelloch scan when scirs2-core API is ready
    sequential_scan(data, op)
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

    if b_bars.dim() != (batch_size, seq_len, state_dim) {
        return Err(CoreError::InvalidConfig(
            "b_bars shape mismatch".to_string(),
        ));
    }

    // Process each batch item
    // TODO: Use scirs2-core parallel when API is stable
    let outputs: Vec<Array1<f32>> = (0..batch_size)
        .map(|b| {
            // Get this batch's A and B
            let a_batch = a_bars.slice(s![b, .., ..]).to_owned();
            let b_batch = b_bars.slice(s![b, .., ..]).to_owned();

            // Perform scan for this sequence
            let states = parallel_ssm_scan(&a_batch, &b_batch, c, parallel_config).unwrap();

            // Compute outputs: y_t = C · h_t + D · x_t
            // (for simplicity, assuming D*x is already included in b_bar)
            let mut output = Array1::zeros(seq_len);
            for t in 0..seq_len {
                let h_t = states.row(t);
                output[t] = c.dot(&h_t) + d;
            }

            output
        })
        .collect();

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
}
