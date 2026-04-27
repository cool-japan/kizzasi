//! # BLAS-Accelerated Operations
//!
//! Hardware-accelerated linear algebra operations using scirs2-linalg.
//! Provides BLAS/LAPACK-backed operations for matrix multiplication,
//! vector operations, and other performance-critical computations.
//!
//! ## Features
//! - **GEMM/GEMV**: Matrix-matrix and matrix-vector multiplication
//! - **AXPY**: Scaled vector addition (y = α*x + y)
//! - **Parallel Operations**: Multi-threaded implementations for large matrices
//! - **SIMD Acceleration**: AVX/AVX2/NEON optimized paths
//! - **Cache-Friendly**: Optimized memory access patterns
//!
//! ## Performance
//! These operations leverage native BLAS libraries (OpenBLAS, Intel MKL, or Apple Accelerate)
//! for peak performance on supported platforms.

use crate::error::{ModelError, ModelResult};
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use tracing::{debug, trace};

/// Configuration for BLAS operations
#[derive(Debug, Clone, Copy)]
pub struct BlasConfig {
    /// Use parallel implementation for matrices larger than this threshold
    pub parallel_threshold: usize,
    /// Number of threads for parallel operations (0 = auto)
    pub num_threads: usize,
    /// Enable SIMD acceleration
    pub enable_simd: bool,
}

impl Default for BlasConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: 1024,
            num_threads: 0, // Auto-detect
            enable_simd: true,
        }
    }
}

/// BLAS-accelerated matrix-vector multiplication: y = A * x
///
/// Uses GEMV from BLAS for optimal performance.
///
/// # Arguments
/// * `matrix` - MxN matrix
/// * `vector` - Vector of length N
///
/// # Returns
/// * Vector of length M
pub fn matmul_vec(matrix: &ArrayView2<f32>, vector: &ArrayView1<f32>) -> ModelResult<Array1<f32>> {
    trace!(
        "BLAS matmul_vec: matrix {:?} × vector {:?}",
        matrix.shape(),
        vector.shape()
    );

    if matrix.ncols() != vector.len() {
        return Err(ModelError::dimension_mismatch(
            "matrix-vector multiplication",
            matrix.ncols(),
            vector.len(),
        ));
    }

    // Use scirs2-linalg's SIMD-optimized matrix-vector product
    use scirs2_linalg::simd_ops::simd_matvec_f32;
    simd_matvec_f32(matrix, vector)
        .map_err(|e| ModelError::numerical_instability("BLAS matvec", e.to_string()))
}

/// BLAS-accelerated matrix-matrix multiplication: C = A * B
///
/// Uses GEMM from BLAS for optimal performance.
///
/// # Arguments
/// * `a` - MxK matrix
/// * `b` - KxN matrix
///
/// # Returns
/// * MxN matrix
pub fn matmul_mat(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> ModelResult<Array2<f32>> {
    trace!("BLAS matmul_mat: {:?} × {:?}", a.shape(), b.shape());

    if a.ncols() != b.nrows() {
        return Err(ModelError::dimension_mismatch(
            "matrix-matrix multiplication",
            a.ncols(),
            b.nrows(),
        ));
    }

    // Use scirs2-linalg's SIMD-accelerated matrix multiplication
    use scirs2_linalg::simd_ops::simd_matmul_optimized_f32;
    simd_matmul_optimized_f32(a, b)
        .map_err(|e| ModelError::numerical_instability("BLAS matmul", e.to_string()))
}

/// BLAS-accelerated scaled vector addition: y = alpha * x + y
///
/// Uses AXPY from BLAS for optimal performance.
///
/// # Arguments
/// * `alpha` - Scalar multiplier
/// * `x` - Input vector
/// * `y` - Vector to add to (modified in-place in the result)
///
/// # Returns
/// * Result vector (alpha * x + y)
pub fn axpy(alpha: f32, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> ModelResult<Array1<f32>> {
    trace!("BLAS axpy: {} * {:?} + {:?}", alpha, x.shape(), y.shape());

    if x.len() != y.len() {
        return Err(ModelError::dimension_mismatch(
            "vector addition",
            x.len(),
            y.len(),
        ));
    }

    // Use scirs2-linalg's SIMD-accelerated AXPY
    // Note: AXPY modifies y in-place, so we need to create a copy
    use scirs2_linalg::simd_ops::simd_axpy_f32;
    let mut result = y.to_owned();
    simd_axpy_f32(alpha, x, &mut result)
        .map_err(|e| ModelError::numerical_instability("BLAS axpy", e.to_string()))?;
    Ok(result)
}

/// Compute dot product of two vectors: x · y
///
/// # Arguments
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
/// * Scalar dot product
pub fn dot(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> ModelResult<f32> {
    trace!("BLAS dot: {:?} · {:?}", x.shape(), y.shape());

    if x.len() != y.len() {
        return Err(ModelError::dimension_mismatch(
            "dot product",
            x.len(),
            y.len(),
        ));
    }

    // Use scirs2-linalg's SIMD-accelerated dot product
    use scirs2_linalg::simd_ops::simd_dot_f32;
    simd_dot_f32(x, y).map_err(|e| ModelError::numerical_instability("BLAS dot", e.to_string()))
}

/// Compute L2 norm (Euclidean norm) of a vector: ||x||₂
///
/// # Arguments
/// * `x` - Input vector
///
/// # Returns
/// * L2 norm
pub fn norm_l2(x: &ArrayView1<f32>) -> ModelResult<f32> {
    trace!("BLAS norm_l2: {:?}", x.shape());

    // Use scirs2-linalg's SIMD-accelerated norm
    use scirs2_linalg::simd_ops::norms::simd_vector_norm_f32;
    let result = simd_vector_norm_f32(x);

    // Check for NaN or infinite values
    if result.is_nan() || result.is_infinite() {
        return Err(ModelError::numerical_instability(
            "BLAS vector norm",
            format!("Invalid norm value: {}", result),
        ));
    }

    Ok(result)
}

/// Compute Frobenius norm of a matrix: ||A||_F
///
/// # Arguments
/// * `matrix` - Input matrix
///
/// # Returns
/// * Frobenius norm
pub fn norm_frobenius(matrix: &ArrayView2<f32>) -> ModelResult<f32> {
    trace!("BLAS norm_frobenius: {:?}", matrix.shape());

    // Use scirs2-linalg's SIMD-accelerated Frobenius norm
    use scirs2_linalg::simd_ops::norms::simd_frobenius_norm_f32;
    let result = simd_frobenius_norm_f32(matrix);

    // Check for NaN or infinite values
    if result.is_nan() || result.is_infinite() {
        return Err(ModelError::numerical_instability(
            "BLAS Frobenius norm",
            format!("Invalid norm value: {}", result),
        ));
    }

    Ok(result)
}

/// Cache-friendly matrix transpose
///
/// Uses blocked algorithm for better cache utilization.
///
/// # Arguments
/// * `matrix` - Input matrix to transpose
///
/// # Returns
/// * Transposed matrix
pub fn transpose(matrix: &ArrayView2<f32>) -> ModelResult<Array2<f32>> {
    trace!("BLAS transpose: {:?}", matrix.shape());

    // Use scirs2-linalg's cache-friendly transpose
    use scirs2_linalg::simd_ops::simd_transpose_f32;
    simd_transpose_f32(matrix)
        .map_err(|e| ModelError::numerical_instability("BLAS transpose", e.to_string()))
}

/// Batch matrix-vector multiplication: `Y[i] = A[i] * x[i]` for all i
///
/// Processes multiple matrix-vector products in parallel when beneficial.
///
/// # Arguments
/// * `matrices` - Slice of matrices
/// * `vectors` - Slice of vectors (same length as matrices)
///
/// # Returns
/// * Vector of result vectors
pub fn batch_matmul_vec(
    matrices: &[ArrayView2<f32>],
    vectors: &[ArrayView1<f32>],
) -> ModelResult<Vec<Array1<f32>>> {
    debug!("BLAS batch_matmul_vec: {} operations", matrices.len());

    if matrices.len() != vectors.len() {
        return Err(ModelError::dimension_mismatch(
            "batch size",
            matrices.len(),
            vectors.len(),
        ));
    }

    // Process in parallel using rayon
    let results: Result<Vec<_>, _> = matrices
        .par_iter()
        .zip(vectors.par_iter())
        .map(|(mat, vec)| matmul_vec(mat, vec))
        .collect();

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_matmul_vec() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let vector = array![1.0, 2.0];

        let result = matmul_vec(&matrix.view(), &vector.view()).expect("matmul_vec failed");

        assert_eq!(result.len(), 2);
        assert!((result[0] - 5.0).abs() < 1e-5); // 1*1 + 2*2 = 5
        assert!((result[1] - 11.0).abs() < 1e-5); // 3*1 + 4*2 = 11
    }

    #[test]
    fn test_matmul_mat() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[2.0, 0.0], [1.0, 2.0]];

        let result = matmul_mat(&a.view(), &b.view()).expect("matmul_mat failed");

        assert_eq!(result.shape(), &[2, 2]);
        assert!((result[[0, 0]] - 4.0).abs() < 1e-5); // 1*2 + 2*1 = 4
        assert!((result[[0, 1]] - 4.0).abs() < 1e-5); // 1*0 + 2*2 = 4
        assert!((result[[1, 0]] - 10.0).abs() < 1e-5); // 3*2 + 4*1 = 10
        assert!((result[[1, 1]] - 8.0).abs() < 1e-5); // 3*0 + 4*2 = 8
    }

    #[test]
    fn test_axpy() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];
        let alpha = 2.0;

        let result = axpy(alpha, &x.view(), &y.view()).expect("axpy failed");

        assert_eq!(result.len(), 3);
        assert!((result[0] - 6.0).abs() < 1e-5); // 2*1 + 4 = 6
        assert!((result[1] - 9.0).abs() < 1e-5); // 2*2 + 5 = 9
        assert!((result[2] - 12.0).abs() < 1e-5); // 2*3 + 6 = 12
    }

    #[test]
    fn test_dot() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];

        let result = dot(&x.view(), &y.view()).expect("dot failed");

        assert!((result - 32.0).abs() < 1e-5); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_norm_l2() {
        let x = array![3.0, 4.0];

        let result = norm_l2(&x.view()).expect("norm_l2 failed");

        assert!((result - 5.0).abs() < 1e-5); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_norm_frobenius() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];

        let result = norm_frobenius(&matrix.view()).expect("norm_frobenius failed");

        let expected = (1.0f32 + 4.0 + 9.0 + 16.0).sqrt(); // sqrt(30)
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_transpose() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let result = transpose(&matrix.view()).expect("transpose failed");

        assert_eq!(result.shape(), &[3, 2]);
        assert!((result[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((result[[1, 0]] - 2.0).abs() < 1e-5);
        assert!((result[[2, 0]] - 3.0).abs() < 1e-5);
        assert!((result[[0, 1]] - 4.0).abs() < 1e-5);
        assert!((result[[1, 1]] - 5.0).abs() < 1e-5);
        assert!((result[[2, 1]] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_dimension_mismatch_matmul_vec() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let wrong_vector = array![1.0, 2.0, 3.0];

        let result = matmul_vec(&matrix.view(), &wrong_vector.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_matmul_mat() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let wrong_b = array![[1.0], [2.0], [3.0]];

        let result = matmul_mat(&a.view(), &wrong_b.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_matmul_vec() {
        let mat1 = array![[1.0, 2.0], [3.0, 4.0]];
        let mat2 = array![[2.0, 1.0], [4.0, 3.0]];
        let vec1 = array![1.0, 1.0];
        let vec2 = array![2.0, 1.0];

        let matrices = vec![mat1.view(), mat2.view()];
        let vectors = vec![vec1.view(), vec2.view()];

        let results = batch_matmul_vec(&matrices, &vectors).expect("batch_matmul_vec failed");

        assert_eq!(results.len(), 2);
        assert!((results[0][0] - 3.0).abs() < 1e-5);
        assert!((results[0][1] - 7.0).abs() < 1e-5);
        assert!((results[1][0] - 5.0).abs() < 1e-5);
        assert!((results[1][1] - 11.0).abs() < 1e-5);
    }
}
