//! Backend abstraction for SSM scan computations.
//!
//! [`SsmBackend`] defines a uniform interface for running SSM prefix scans on
//! different compute backends (CPU, WebGPU). Implementations live here (CPU)
//! and in `kizzasi-webgpu` (GPU).

use crate::error::CoreResult;

/// Abstraction over compute backends for SSM (State Space Model) operations.
///
/// The core operation is the diagonal-A associative SSM scan used by Mamba/S4D.
/// For hidden state recurrence `h_t = A·h_{t-1} + B·x_t`, each step produces
/// an element `(a_t, bu_t)`, and the scan computes all prefix states in parallel.
///
/// # Associative operator
/// `(a₁, bu₁) ⊗ (a₂, bu₂) = (a₂·a₁, a₂·bu₁ + bu₂)`
///
/// Identity element: `(1.0, 0.0)`.
pub trait SsmBackend: Send + Sync {
    /// Execute an inclusive diagonal-A associative SSM scan.
    ///
    /// Given `elements` as `(a_t, bu_t)` pairs of length `seq_len`, returns
    /// `seq_len` cumulative pairs representing the prefix states:
    /// `out[i] = elements[0] ⊗ elements[1] ⊗ … ⊗ elements[i]`.
    ///
    /// An empty input returns an empty output without error.
    fn ssm_scan(&self, elements: &[(f32, f32)]) -> CoreResult<Vec<(f32, f32)>>;

    /// Human-readable backend identifier for diagnostics / logging.
    fn backend_name(&self) -> &str;
}

/// Pure-CPU sequential SSM scan backend.
///
/// Always available; used when no GPU accelerator is compiled in or when the
/// input is small enough that GPU overhead isn't worth paying.
#[derive(Debug, Clone, Default)]
pub struct CpuSsmBackend;

impl SsmBackend for CpuSsmBackend {
    fn ssm_scan(&self, elements: &[(f32, f32)]) -> CoreResult<Vec<(f32, f32)>> {
        if elements.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(elements.len());
        result.push(elements[0]);

        for i in 1..elements.len() {
            let (a_prev, bu_prev) = result[i - 1];
            let (a_cur, bu_cur) = elements[i];
            result.push((a_cur * a_prev, a_cur * bu_prev + bu_cur));
        }

        Ok(result)
    }

    fn backend_name(&self) -> &str {
        "cpu"
    }
}

/// The backend available from *this* crate: always [`CpuSsmBackend`].
///
/// This function cannot return the GPU backend, and that is structural rather
/// than an omission: `WebGpuSsmBackend` lives in `kizzasi-webgpu`, which
/// depends on `kizzasi-core`, so selecting it from here would be a dependency
/// cycle. GPU selection therefore lives one level up, in the `kizzasi` facade
/// crate:
///
/// ```text
/// kizzasi::ssm_backend::select_ssm_backend()   // with --features webgpu
/// ```
///
/// which returns the hybrid GPU backend when a wgpu adapter is reachable and
/// falls back to [`CpuSsmBackend`] otherwise. Use this function when you are
/// inside `kizzasi-core` or deliberately want the CPU implementation.
pub fn default_backend() -> Box<dyn SsmBackend> {
    Box::new(CpuSsmBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_scan() {
        let backend = CpuSsmBackend;
        let result = backend.ssm_scan(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_element() {
        let backend = CpuSsmBackend;
        let elements = [(0.9_f32, 1.0_f32)];
        let result = backend.ssm_scan(&elements).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0].0 - 0.9).abs() < 1e-6);
        assert!((result[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_two_elements_associativity() {
        // (0.9, 1.0) ⊗ (0.8, 2.0) = (0.8*0.9, 0.8*1.0 + 2.0) = (0.72, 2.8)
        let backend = CpuSsmBackend;
        let elements = [(0.9_f32, 1.0_f32), (0.8_f32, 2.0_f32)];
        let result = backend.ssm_scan(&elements).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0].0 - 0.9).abs() < 1e-6);
        assert!((result[0].1 - 1.0).abs() < 1e-6);
        assert!((result[1].0 - 0.72).abs() < 1e-6);
        assert!((result[1].1 - 2.8).abs() < 1e-6);
    }

    #[test]
    fn test_identity_does_not_change_value() {
        // Combining with identity (1.0, 0.0) on the left should leave the element unchanged.
        let backend = CpuSsmBackend;
        let elements = [(1.0_f32, 0.0_f32), (0.5_f32, 3.0_f32)];
        let result = backend.ssm_scan(&elements).unwrap();
        assert_eq!(result.len(), 2);
        // (1.0, 0.0) ⊗ (0.5, 3.0) = (0.5*1.0, 0.5*0.0 + 3.0) = (0.5, 3.0)
        assert!((result[1].0 - 0.5).abs() < 1e-6);
        assert!((result[1].1 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_longer_sequence() {
        // Verify correct cumulative folding over 5 elements.
        let backend = CpuSsmBackend;
        // All elements with a=1.0, bu=1.0: result[i] should be (1.0, i+1.0) by induction
        let elements: Vec<(f32, f32)> = (0..5).map(|_| (1.0_f32, 1.0_f32)).collect();
        let result = backend.ssm_scan(&elements).unwrap();
        for (i, &(a, bu)) in result.iter().enumerate() {
            assert!((a - 1.0).abs() < 1e-6, "a mismatch at {i}");
            assert!((bu - (i + 1) as f32).abs() < 1e-6, "bu mismatch at {i}");
        }
    }

    #[test]
    fn test_default_backend_name() {
        let backend = default_backend();
        assert_eq!(backend.backend_name(), "cpu");
    }

    #[test]
    fn test_backend_send_sync() {
        // Verify that CpuSsmBackend can be sent across threads.
        let backend: Box<dyn SsmBackend> = default_backend();
        std::thread::spawn(move || {
            let _ = backend.backend_name();
        })
        .join()
        .unwrap();
    }
}
