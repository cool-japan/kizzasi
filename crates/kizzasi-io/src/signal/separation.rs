//! Blind source separation algorithms
//!
//! This module provides state-of-the-art techniques for separating mixed signals:
//! - FastICA (Independent Component Analysis)
//! - NMF (Non-negative Matrix Factorization)
//! - PCA (Principal Component Analysis)
//! - Temporal decorrelation methods

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;

/// FastICA algorithm for Independent Component Analysis
///
/// Separates mixed signals into statistically independent components
/// using higher-order statistics (non-Gaussianity).
pub struct FastICA {
    /// Number of components to extract
    n_components: usize,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: f32,
    /// Nonlinearity function type
    nonlinearity: Nonlinearity,
}

/// Nonlinearity function for ICA
#[derive(Debug, Clone, Copy)]
pub enum Nonlinearity {
    /// Logcosh (default, robust)
    LogCosh,
    /// Exponential (fast convergence)
    Exp,
    /// Cubic (simple, less robust)
    Cube,
}

impl FastICA {
    /// Create a new FastICA analyzer
    ///
    /// # Arguments
    /// * `n_components` - Number of independent components to extract
    /// * `max_iter` - Maximum iterations (default 200)
    /// * `tolerance` - Convergence tolerance (default 1e-4)
    pub fn new(n_components: usize, max_iter: Option<usize>, tolerance: Option<f32>) -> Self {
        Self {
            n_components,
            max_iter: max_iter.unwrap_or(200),
            tolerance: tolerance.unwrap_or(1e-4),
            nonlinearity: Nonlinearity::LogCosh,
        }
    }

    /// Set nonlinearity function
    pub fn with_nonlinearity(mut self, nonlinearity: Nonlinearity) -> Self {
        self.nonlinearity = nonlinearity;
        self
    }

    /// Fit and transform mixed signals to independent components
    ///
    /// # Arguments
    /// * `mixed` - Mixed signals (n_samples × n_signals)
    ///
    /// # Returns
    /// * Independent components (n_samples × n_components)
    /// * Unmixing matrix (n_components × n_signals)
    pub fn fit_transform(&self, mixed: &Array2<f32>) -> IoResult<(Array2<f32>, Array2<f32>)> {
        let (n_samples, n_signals) = mixed.dim();

        if n_samples < 2 || n_signals < 2 {
            return Err(IoError::SignalError(
                "Need at least 2 samples and 2 signals".into(),
            ));
        }

        if self.n_components > n_signals {
            return Err(IoError::SignalError(
                "n_components cannot exceed n_signals".into(),
            ));
        }

        // 1. Center the data
        let mean = mixed
            .mean_axis(Axis(0))
            .expect("Mean axis computation must succeed");
        let centered = mixed - &mean.view().insert_axis(Axis(0));

        // 2. Whiten the data using PCA
        let (whitened, whitening_matrix) = Self::whiten(&centered)?;

        // 3. FastICA algorithm
        let unmixing = self.fastica_core(&whitened)?;

        // 4. Compute sources
        let sources = whitened.dot(&unmixing.t());

        // 5. Compute full unmixing matrix
        let full_unmixing = unmixing.dot(&whitening_matrix);

        Ok((sources, full_unmixing))
    }

    /// Core FastICA algorithm
    fn fastica_core(&self, whitened: &Array2<f32>) -> IoResult<Array2<f32>> {
        let n_components = self.n_components;
        let mut rng = thread_rng();

        // Initialize unmixing matrix randomly
        let mut w = Array2::from_shape_fn((n_components, whitened.ncols()), |_| {
            rng.gen_range(-1.0..1.0)
        });

        // Orthogonalize
        Self::gram_schmidt(&mut w);

        // Iterate until convergence
        for _iter in 0..self.max_iter {
            let w_old = w.clone();

            // Update each component
            for i in 0..n_components {
                let mut w_i = w.row(i).to_owned();

                // Compute E{x g(w^T x)} and E{g'(w^T x)}
                let wx = whitened.dot(&w_i);
                let (g_wx, gp_wx) = self.apply_nonlinearity(&wx);

                let eg = whitened.t().dot(&g_wx) / whitened.nrows() as f32;
                let egp = gp_wx.mean().unwrap_or(0.0);

                // Newton update: w = E{x g(w^T x)} - E{g'(w^T x)} w
                w_i = eg - &w_i * egp;

                // Store updated component
                for (j, val) in w_i.iter().enumerate() {
                    w[[i, j]] = *val;
                }

                // Orthogonalize against previous components
                for j in 0..i {
                    let w_j = w.row(j).to_owned();
                    let dot = w_i.dot(&w_j);
                    w_i = w_i - &w_j * dot;
                }

                // Normalize
                let norm = w_i.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    w_i /= norm;
                }

                // Update row
                for (j, val) in w_i.iter().enumerate() {
                    w[[i, j]] = *val;
                }
            }

            // Check convergence
            let mut max_diff = 0.0f32;
            for i in 0..n_components {
                for j in 0..w.ncols() {
                    let diff = (w[[i, j]] - w_old[[i, j]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            if max_diff < self.tolerance {
                break;
            }
        }

        Ok(w)
    }

    /// Apply nonlinearity function and its derivative
    fn apply_nonlinearity(&self, x: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        match self.nonlinearity {
            Nonlinearity::LogCosh => {
                let alpha = 1.0;
                let g = x.mapv(|v| (alpha * v).tanh());
                let gp = x.mapv(|v| alpha * (1.0 - (alpha * v).tanh().powi(2)));
                (g, gp)
            }
            Nonlinearity::Exp => {
                let g = x.mapv(|v| v * (-v * v / 2.0).exp());
                let gp = x.mapv(|v| (1.0 - v * v) * (-v * v / 2.0).exp());
                (g, gp)
            }
            Nonlinearity::Cube => {
                let g = x.mapv(|v| v.powi(3));
                let gp = x.mapv(|v| 3.0 * v * v);
                (g, gp)
            }
        }
    }

    /// Whiten data using PCA
    fn whiten(data: &Array2<f32>) -> IoResult<(Array2<f32>, Array2<f32>)> {
        let n_samples = data.nrows();

        // Compute covariance matrix
        let cov = data.t().dot(data) / n_samples as f32;

        // Simplified eigenvalue decomposition (for small matrices)
        // In production, use proper eigendecomposition from linalg crate
        let (eigenvalues, eigenvectors) = Self::simple_eig(&cov)?;

        // Compute whitening matrix: W = D^{-1/2} E^T
        let mut whitening = eigenvectors.t().to_owned();
        for i in 0..eigenvalues.len() {
            let scale = 1.0 / (eigenvalues[i].max(1e-10).sqrt());
            for j in 0..whitening.ncols() {
                whitening[[i, j]] *= scale;
            }
        }

        // Whiten data
        let whitened = data.dot(&whitening.t());

        Ok((whitened, whitening))
    }

    /// Simplified eigenvalue decomposition (power iteration)
    fn simple_eig(matrix: &Array2<f32>) -> IoResult<(Vec<f32>, Array2<f32>)> {
        let n = matrix.nrows();
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Array2::zeros((n, n));
        let mut remaining = matrix.clone();

        for k in 0..n {
            // Power iteration to find largest eigenvalue/eigenvector
            let mut v = Array1::from_shape_fn(n, |_| thread_rng().gen_range(-1.0..1.0));
            v = &v / v.iter().map(|x| x * x).sum::<f32>().sqrt();

            for _ in 0..100 {
                let v_new = remaining.dot(&v);
                let norm = v_new.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                v = &v_new / norm;
            }

            // Compute eigenvalue
            let av = remaining.dot(&v);
            let eigenvalue = av.dot(&v);
            eigenvalues.push(eigenvalue);

            // Store eigenvector
            for i in 0..n {
                eigenvectors[[i, k]] = v[i];
            }

            // Deflate matrix
            let vv = v
                .clone()
                .insert_axis(Axis(1))
                .dot(&v.clone().insert_axis(Axis(0)));
            remaining = &remaining - &(&vv * eigenvalue);
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Gram-Schmidt orthogonalization
    fn gram_schmidt(matrix: &mut Array2<f32>) {
        let n_rows = matrix.nrows();

        for i in 0..n_rows {
            // Orthogonalize against previous rows
            for j in 0..i {
                let dot: f32 = (0..matrix.ncols())
                    .map(|k| matrix[[i, k]] * matrix[[j, k]])
                    .sum();

                for k in 0..matrix.ncols() {
                    matrix[[i, k]] -= dot * matrix[[j, k]];
                }
            }

            // Normalize
            let norm: f32 = (0..matrix.ncols())
                .map(|k| matrix[[i, k]] * matrix[[i, k]])
                .sum::<f32>()
                .sqrt();

            if norm > 1e-10 {
                for k in 0..matrix.ncols() {
                    matrix[[i, k]] /= norm;
                }
            }
        }
    }
}

/// Non-negative Matrix Factorization
///
/// Factorizes a non-negative matrix into two non-negative matrices
/// using multiplicative update rules.
pub struct NMF {
    /// Number of components
    n_components: usize,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: f32,
}

impl NMF {
    /// Create a new NMF analyzer
    ///
    /// # Arguments
    /// * `n_components` - Number of components (rank)
    /// * `max_iter` - Maximum iterations (default 200)
    /// * `tolerance` - Convergence tolerance (default 1e-4)
    pub fn new(n_components: usize, max_iter: Option<usize>, tolerance: Option<f32>) -> Self {
        Self {
            n_components,
            max_iter: max_iter.unwrap_or(200),
            tolerance: tolerance.unwrap_or(1e-4),
        }
    }

    /// Factorize matrix V ≈ W H
    ///
    /// # Arguments
    /// * `v` - Non-negative matrix (n_samples × n_features)
    ///
    /// # Returns
    /// * W - Basis matrix (n_samples × n_components)
    /// * H - Coefficient matrix (n_components × n_features)
    pub fn fit_transform(&self, v: &Array2<f32>) -> IoResult<(Array2<f32>, Array2<f32>)> {
        let (n_samples, n_features) = v.dim();

        if n_samples < 1 || n_features < 1 {
            return Err(IoError::SignalError("Empty matrix".into()));
        }

        // Check non-negativity
        if v.iter().any(|&x| x < 0.0) {
            return Err(IoError::SignalError("Matrix must be non-negative".into()));
        }

        // Initialize W and H randomly
        let mut rng = thread_rng();
        let mut w =
            Array2::from_shape_fn((n_samples, self.n_components), |_| rng.gen_range(0.0..1.0));
        let mut h =
            Array2::from_shape_fn((self.n_components, n_features), |_| rng.gen_range(0.0..1.0));

        let eps = 1e-10;
        let mut prev_error = f32::MAX;

        // Multiplicative update rules
        for _iter in 0..self.max_iter {
            // Update H: H = H .* (W^T V) ./ (W^T W H + eps)
            let wt_v = w.t().dot(v);
            let wt_w_h = w.t().dot(&w).dot(&h);

            for i in 0..h.nrows() {
                for j in 0..h.ncols() {
                    h[[i, j]] *= wt_v[[i, j]] / (wt_w_h[[i, j]] + eps);
                }
            }

            // Update W: W = W .* (V H^T) ./ (W H H^T + eps)
            let v_ht = v.dot(&h.t());
            let w_h_ht = w.dot(&h).dot(&h.t());

            for i in 0..w.nrows() {
                for j in 0..w.ncols() {
                    w[[i, j]] *= v_ht[[i, j]] / (w_h_ht[[i, j]] + eps);
                }
            }

            // Compute reconstruction error
            let wh = w.dot(&h);
            let error: f32 = v
                .iter()
                .zip(wh.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            // Check convergence
            if (prev_error - error).abs() < self.tolerance {
                break;
            }
            prev_error = error;
        }

        Ok((w, h))
    }

    /// Transform new data using fitted model
    ///
    /// # Arguments
    /// * `v` - New data matrix
    /// * `h` - Previously fitted coefficient matrix
    pub fn transform(&self, v: &Array2<f32>, h: &Array2<f32>) -> IoResult<Array2<f32>> {
        let (n_samples, _n_features) = v.dim();
        let mut rng = thread_rng();

        // Initialize W randomly
        let mut w =
            Array2::from_shape_fn((n_samples, self.n_components), |_| rng.gen_range(0.0..1.0));

        let eps = 1e-10;

        // Fix H, update only W
        for _ in 0..self.max_iter {
            let v_ht = v.dot(&h.t());
            let w_h_ht = w.dot(h).dot(&h.t());

            for i in 0..w.nrows() {
                for j in 0..w.ncols() {
                    w[[i, j]] *= v_ht[[i, j]] / (w_h_ht[[i, j]] + eps);
                }
            }
        }

        Ok(w)
    }
}

/// Principal Component Analysis
///
/// Reduces dimensionality by projecting onto principal components
/// (directions of maximum variance).
pub struct PCA {
    /// Number of components to keep
    n_components: usize,
}

impl PCA {
    /// Create a new PCA analyzer
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Fit and transform data to principal components
    ///
    /// # Arguments
    /// * `data` - Input data (n_samples × n_features)
    ///
    /// # Returns
    /// * Transformed data (n_samples × n_components)
    /// * Principal components (n_components × n_features)
    /// * Explained variance
    pub fn fit_transform(
        &self,
        data: &Array2<f32>,
    ) -> IoResult<(Array2<f32>, Array2<f32>, Vec<f32>)> {
        let (n_samples, n_features) = data.dim();

        if self.n_components > n_features {
            return Err(IoError::SignalError(
                "n_components cannot exceed n_features".into(),
            ));
        }

        // Center the data
        let mean = data
            .mean_axis(Axis(0))
            .expect("Mean axis computation must succeed");
        let centered = data - &mean.view().insert_axis(Axis(0));

        // Compute covariance matrix
        let cov = centered.t().dot(&centered) / n_samples as f32;

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = FastICA::simple_eig(&cov)?;

        // Sort by eigenvalue (descending)
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top n_components
        let mut components = Array2::zeros((self.n_components, n_features));
        let mut explained_var = Vec::new();

        for i in 0..self.n_components {
            let idx = indices[i];
            explained_var.push(eigenvalues[idx]);

            for j in 0..n_features {
                components[[i, j]] = eigenvectors[[j, idx]];
            }
        }

        // Transform data
        let transformed = centered.dot(&components.t());

        Ok((transformed, components, explained_var))
    }

    /// Transform new data using fitted components
    pub fn transform(&self, data: &Array2<f32>, components: &Array2<f32>) -> Array2<f32> {
        // Center (ideally use mean from training)
        let mean = data
            .mean_axis(Axis(0))
            .expect("Mean axis computation must succeed");
        let centered = data - &mean.view().insert_axis(Axis(0));

        // Project onto components
        centered.dot(&components.t())
    }

    /// Inverse transform (reconstruct from components)
    pub fn inverse_transform(
        &self,
        transformed: &Array2<f32>,
        components: &Array2<f32>,
    ) -> Array2<f32> {
        transformed.dot(components)
    }
}

/// Temporal decorrelation for source separation
pub struct TemporalDecorrelation {
    /// Time delay for decorrelation
    tau: usize,
}

impl TemporalDecorrelation {
    /// Create a new temporal decorrelation analyzer
    pub fn new(tau: usize) -> Self {
        Self { tau }
    }

    /// Separate sources using temporal structure
    ///
    /// Exploits temporal correlations in source signals
    pub fn separate(&self, mixed: &Array2<f32>) -> IoResult<Array2<f32>> {
        let (n_samples, n_channels) = mixed.dim();

        if n_samples <= self.tau {
            return Err(IoError::SignalError("Insufficient samples".into()));
        }

        // Compute time-delayed covariance
        let mut cov_delay = Array2::zeros((n_channels, n_channels));

        for i in 0..(n_samples - self.tau) {
            for j in 0..n_channels {
                for k in 0..n_channels {
                    cov_delay[[j, k]] += mixed[[i, j]] * mixed[[i + self.tau, k]];
                }
            }
        }

        cov_delay /= (n_samples - self.tau) as f32;

        // Eigendecomposition of time-delayed covariance
        let (_, eigenvectors) = FastICA::simple_eig(&cov_delay)?;

        // Separate sources
        let separated = mixed.dot(&eigenvectors);

        Ok(separated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_fastica_basic() {
        // Create simple mixed signals
        let mixed = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]);

        let ica = FastICA::new(2, None, None);
        let result = ica.fit_transform(&mixed);

        assert!(result.is_ok());
        let (sources, unmixing) = result.unwrap();
        assert_eq!(sources.dim(), (4, 2));
        assert_eq!(unmixing.dim(), (2, 2));
    }

    #[test]
    fn test_nmf_basic() {
        // Create non-negative matrix
        let v = arr2(&[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]);

        let nmf = NMF::new(2, Some(50), None);
        let result = nmf.fit_transform(&v);

        assert!(result.is_ok());
        let (w, h) = result.unwrap();
        assert_eq!(w.dim(), (3, 2));
        assert_eq!(h.dim(), (2, 3));

        // Check non-negativity
        assert!(w.iter().all(|&x| x >= 0.0));
        assert!(h.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_pca_basic() {
        let data = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ]);

        let pca = PCA::new(2);
        let result = pca.fit_transform(&data);

        assert!(result.is_ok());
        let (transformed, components, explained_var) = result.unwrap();
        assert_eq!(transformed.dim(), (4, 2));
        assert_eq!(components.dim(), (2, 3));
        assert_eq!(explained_var.len(), 2);
    }

    #[test]
    fn test_temporal_decorrelation() {
        let mixed = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]);

        let td = TemporalDecorrelation::new(1);
        let result = td.separate(&mixed);

        assert!(result.is_ok());
        let separated = result.unwrap();
        assert_eq!(separated.dim(), (5, 2));
    }

    #[test]
    fn test_nmf_negative_input() {
        let v = arr2(&[[1.0, -2.0], [2.0, 3.0]]);

        let nmf = NMF::new(2, None, None);
        let result = nmf.fit_transform(&v);

        assert!(result.is_err());
    }

    #[test]
    fn test_pca_reconstruction() {
        let data = arr2(&[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]);

        let pca = PCA::new(2);
        let (transformed, components, _) = pca.fit_transform(&data).unwrap();

        let reconstructed = pca.inverse_transform(&transformed, &components);
        assert_eq!(reconstructed.nrows(), 3);
    }
}
