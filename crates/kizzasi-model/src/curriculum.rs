//! Curriculum Learning for kizzasi-model
//!
//! Implements curriculum learning strategies that control the order and
//! difficulty of training samples presented to the model. This follows the
//! insight from Bengio et al. (2009) that training on easier examples first
//! and gradually introducing harder ones can improve convergence and
//! generalization.
//!
//! # Strategies
//!
//! - **Competence-based pacing**: linearly increases a competence threshold
//!   each epoch; only samples with difficulty <= competence are included.
//! - **Self-Paced Learning (SPL)**: uses a soft weight function
//!   `w = max(0, 1 - difficulty/lambda)` where lambda grows over time,
//!   gradually admitting harder samples.
//! - **Annealing**: linearly interpolates the difficulty gate from an easy
//!   start threshold to a harder end threshold across epochs.
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::curriculum::{CurriculumScheduler, CurriculumStrategy, CurriculumDataProvider};
//! use kizzasi_model::training_loop::ArrayDataProvider;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! let features = Array2::<f32>::zeros((100, 4));
//! let targets = Array1::<f32>::zeros(100);
//! let data = ArrayDataProvider::new(features, targets);
//!
//! // Per-sample difficulty scores in [0, 1]
//! let difficulties = Array1::from_vec(vec![0.1; 100]);
//! let strategy = CurriculumStrategy::Competence { initial: 0.2, increment: 0.1 };
//!
//! let mut provider = CurriculumDataProvider::new(data, difficulties, strategy);
//! provider.advance_epoch();
//! let active = provider.active_indices();
//! ```

use crate::error::{ModelError, ModelResult};
use crate::training_loop::{ArrayDataProvider, DataProvider};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// CurriculumStrategy
// ---------------------------------------------------------------------------

/// Strategy for controlling which samples are presented during training.
#[derive(Debug, Clone)]
pub enum CurriculumStrategy {
    /// Competence-based pacing.
    ///
    /// The competence threshold starts at `initial` and increases by
    /// `increment` each time `step()` is called, clamped to [0, 1].
    /// A sample is included if its difficulty <= current competence.
    Competence {
        /// Starting competence level in [0, 1].
        initial: f32,
        /// Amount competence grows per epoch.
        increment: f32,
    },

    /// Self-Paced Learning (SPL).
    ///
    /// Uses a soft weight function: `w(d) = max(0, 1 - d / lambda)`.
    /// A sample is included when `w > 0.5`, i.e. `d < lambda / 2`.
    /// `lambda` grows by a multiplicative factor each epoch, starting at
    /// the provided initial value.
    SelfPaced {
        /// Initial pace parameter controlling the difficulty boundary.
        /// Higher lambda admits more samples.
        lambda: f32,
    },

    /// Annealing strategy.
    ///
    /// Linearly interpolates the difficulty gate from `start_difficulty`
    /// to `end` over successive epochs. The gate value at epoch `e` is:
    ///
    /// `gate = start + (end - start) * min(1, e / ramp_epochs)`
    ///
    /// where `ramp_epochs` is derived from the number of `step()` calls.
    Annealing {
        /// Starting difficulty threshold (easy end).
        start_difficulty: f32,
        /// Ending difficulty threshold (hard end, typically 1.0).
        end: f32,
    },
}

// ---------------------------------------------------------------------------
// CurriculumScheduler
// ---------------------------------------------------------------------------

/// Drives the curriculum pacing across training epochs.
///
/// Each call to [`step()`](CurriculumScheduler::step) advances the internal
/// epoch counter and recomputes the current competence / difficulty gate.
#[derive(Debug, Clone)]
pub struct CurriculumScheduler {
    strategy: CurriculumStrategy,
    /// Current competence level in [0, 1] — determines which samples pass.
    current_competence: f32,
    /// Number of times `step()` has been called.
    epoch: usize,
    /// For SPL: current lambda (grows each epoch).
    spl_lambda: f32,
    /// For SPL: multiplicative growth factor for lambda each epoch.
    spl_growth: f32,
}

impl CurriculumScheduler {
    /// Create a new scheduler for the given strategy.
    ///
    /// The initial competence is derived from the strategy parameters:
    /// - Competence: `initial`
    /// - SelfPaced: derived from `lambda` → `lambda / 2`
    /// - Annealing: `start_difficulty`
    pub fn new(strategy: CurriculumStrategy) -> Self {
        let (initial_competence, spl_lambda) = match &strategy {
            CurriculumStrategy::Competence { initial, .. } => (*initial, 0.0),
            CurriculumStrategy::SelfPaced { lambda } => {
                // SPL: include sample if difficulty < lambda/2
                // So effective competence = lambda/2 clamped to [0,1]
                ((*lambda * 0.5).clamp(0.0, 1.0), *lambda)
            }
            CurriculumStrategy::Annealing {
                start_difficulty, ..
            } => (*start_difficulty, 0.0),
        };

        Self {
            strategy,
            current_competence: initial_competence.clamp(0.0, 1.0),
            epoch: 0,
            spl_lambda,
            // Lambda grows by 20% each epoch by default — a reasonable pace
            // that ensures convergence to including all samples.
            spl_growth: 1.2,
        }
    }

    /// Override the SPL growth factor (default 1.2).
    ///
    /// Only meaningful for [`CurriculumStrategy::SelfPaced`].
    pub fn with_spl_growth(mut self, growth: f32) -> Self {
        self.spl_growth = growth.max(1.0);
        self
    }

    /// Advance one epoch and return the updated competence in [0, 1].
    pub fn step(&mut self) -> f32 {
        self.epoch += 1;

        match &self.strategy {
            CurriculumStrategy::Competence {
                initial, increment, ..
            } => {
                // Linear ramp: competence = initial + epoch * increment
                self.current_competence =
                    (*initial + self.epoch as f32 * *increment).clamp(0.0, 1.0);
            }
            CurriculumStrategy::SelfPaced { .. } => {
                // Grow lambda each epoch
                self.spl_lambda *= self.spl_growth;
                // Effective competence boundary = lambda / 2, clamped
                self.current_competence = (self.spl_lambda * 0.5).clamp(0.0, 1.0);
            }
            CurriculumStrategy::Annealing {
                start_difficulty,
                end,
            } => {
                // We use a fixed ramp of 100 epochs for the interpolation.
                // After 100 epochs the gate is fully at `end`.
                let ramp_epochs = 100.0_f32;
                let t = (self.epoch as f32 / ramp_epochs).clamp(0.0, 1.0);
                self.current_competence =
                    (*start_difficulty + (*end - *start_difficulty) * t).clamp(0.0, 1.0);
            }
        }

        self.current_competence
    }

    /// Return the current competence level without advancing.
    pub fn current_competence(&self) -> f32 {
        self.current_competence
    }

    /// Return how many epochs have been stepped.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Whether a sample with the given difficulty should be included at the
    /// current competence level.
    pub fn should_include(&self, difficulty: f32) -> bool {
        match &self.strategy {
            CurriculumStrategy::Competence { .. } | CurriculumStrategy::Annealing { .. } => {
                difficulty <= self.current_competence
            }
            CurriculumStrategy::SelfPaced { .. } => {
                // SPL weight: w = max(0, 1 - difficulty / lambda)
                // Include if w > 0.5, i.e. difficulty < lambda / 2
                if self.spl_lambda <= 0.0 {
                    return false;
                }
                let w = (1.0 - difficulty / self.spl_lambda).max(0.0);
                w > 0.5
            }
        }
    }

    /// Return the indices of samples whose difficulty passes the current gate.
    pub fn filter_indices(&self, difficulties: &[f32]) -> Vec<usize> {
        difficulties
            .iter()
            .enumerate()
            .filter(|(_, &d)| self.should_include(d))
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute SPL weight for a given difficulty.
    ///
    /// Returns a value in [0, 1] where 1 = easy/fully included and 0 = too hard.
    /// For non-SPL strategies this returns 1.0 if included, 0.0 otherwise.
    pub fn spl_weight(&self, difficulty: f32) -> f32 {
        match &self.strategy {
            CurriculumStrategy::SelfPaced { .. } => {
                if self.spl_lambda <= 0.0 {
                    return 0.0;
                }
                (1.0 - difficulty / self.spl_lambda).clamp(0.0, 1.0)
            }
            _ => {
                if self.should_include(difficulty) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CurriculumDataProvider
// ---------------------------------------------------------------------------

/// A data provider that wraps [`ArrayDataProvider`] and filters samples
/// according to a [`CurriculumScheduler`].
///
/// Each sample has an associated difficulty score in [0, 1]. The scheduler
/// controls which samples are visible at each epoch, starting with easy
/// samples and progressively including harder ones.
pub struct CurriculumDataProvider {
    /// The underlying data.
    inner: ArrayDataProvider,
    /// Per-sample difficulty scores, length == `inner.num_samples()`.
    difficulties: Array1<f32>,
    /// The curriculum scheduler driving the pacing.
    scheduler: CurriculumScheduler,
    /// Cached active indices (recomputed on `advance_epoch`).
    cached_active: Vec<usize>,
}

impl CurriculumDataProvider {
    /// Create a new curriculum data provider.
    ///
    /// # Errors
    ///
    /// Returns an error if `difficulties.len() != data.num_samples()`.
    pub fn new(
        data: ArrayDataProvider,
        difficulties: Array1<f32>,
        strategy: CurriculumStrategy,
    ) -> ModelResult<Self> {
        if difficulties.len() != data.num_samples() {
            return Err(ModelError::dimension_mismatch(
                "CurriculumDataProvider::new",
                data.num_samples(),
                difficulties.len(),
            ));
        }

        let scheduler = CurriculumScheduler::new(strategy);

        // Compute initial active indices.
        let cached_active = scheduler.filter_indices(difficulties.as_slice().unwrap_or(&[]));

        Ok(Self {
            inner: data,
            difficulties,
            scheduler,
            cached_active,
        })
    }

    /// Advance one epoch: steps the scheduler and recomputes active indices.
    pub fn advance_epoch(&mut self) {
        self.scheduler.step();
        self.recompute_active();
    }

    /// Recompute the cached active indices from the current scheduler state.
    fn recompute_active(&mut self) {
        let diff_slice = self.difficulties.as_slice().unwrap_or(&[]);
        self.cached_active = self.scheduler.filter_indices(diff_slice);
    }

    /// Return the indices of currently active (included) samples.
    pub fn active_indices(&self) -> Vec<usize> {
        self.cached_active.clone()
    }

    /// Number of currently active samples.
    pub fn active_count(&self) -> usize {
        self.cached_active.len()
    }

    /// Total samples in the underlying dataset (not just active ones).
    pub fn total_samples(&self) -> usize {
        self.inner.num_samples()
    }

    /// Fraction of the dataset currently active.
    pub fn active_fraction(&self) -> f32 {
        if self.inner.num_samples() == 0 {
            return 0.0;
        }
        self.cached_active.len() as f32 / self.inner.num_samples() as f32
    }

    /// Access the scheduler (read-only).
    pub fn scheduler(&self) -> &CurriculumScheduler {
        &self.scheduler
    }

    /// Access the difficulty scores.
    pub fn difficulties(&self) -> &Array1<f32> {
        &self.difficulties
    }

    /// Get SPL weights for all active samples.
    ///
    /// Returns a vector of `(sample_index, weight)` pairs for all currently
    /// active samples.
    pub fn active_weights(&self) -> Vec<(usize, f32)> {
        self.cached_active
            .iter()
            .map(|&idx| {
                let d = self.difficulties[idx];
                (idx, self.scheduler.spl_weight(d))
            })
            .collect()
    }
}

impl DataProvider for CurriculumDataProvider {
    fn num_samples(&self) -> usize {
        // Report the number of *active* samples so training loops
        // naturally iterate over the curriculum-filtered subset.
        self.cached_active.len()
    }

    fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    fn get_batch(&self, indices: &[usize]) -> (Array2<f32>, Array1<f32>) {
        // `indices` are positions within the *active* set.
        // Map them to the original dataset indices.
        let mapped: Vec<usize> = indices
            .iter()
            .map(|&i| {
                if i < self.cached_active.len() {
                    self.cached_active[i]
                } else {
                    // Clamp to last active index to avoid panic.
                    self.cached_active.last().copied().unwrap_or(0)
                }
            })
            .collect();

        self.inner.get_batch(&mapped)
    }

    fn shuffle_indices(&self, rng_seed: u64) -> Vec<usize> {
        // Shuffle over the active set only.
        let n = self.cached_active.len();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut state = rng_seed.wrapping_add(1);
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
        indices
    }
}

// ---------------------------------------------------------------------------
// Difficulty estimators
// ---------------------------------------------------------------------------

/// Estimate per-sample difficulty from loss values.
///
/// Given a vector of per-sample losses, normalises them to [0, 1] by
/// mapping the minimum loss to 0 and the maximum to 1. Samples with
/// higher loss are considered harder.
///
/// Returns an `Array1<f32>` of difficulty scores.
pub fn estimate_difficulty_from_loss(losses: &[f32]) -> ModelResult<Array1<f32>> {
    if losses.is_empty() {
        return Err(ModelError::invalid_config(
            "Cannot estimate difficulty from empty loss vector",
        ));
    }

    let min_loss = losses.iter().copied().fold(f32::INFINITY, f32::min);
    let max_loss = losses.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let range = max_loss - min_loss;

    let difficulties = if range.abs() < f32::EPSILON {
        // All losses are the same — assign uniform difficulty 0.5.
        Array1::from_elem(losses.len(), 0.5)
    } else {
        Array1::from_vec(
            losses
                .iter()
                .map(|&l| ((l - min_loss) / range).clamp(0.0, 1.0))
                .collect(),
        )
    };

    Ok(difficulties)
}

/// Estimate difficulty based on feature variance.
///
/// Samples whose features have higher variance (further from the dataset
/// mean) are considered harder. Normalised to [0, 1].
pub fn estimate_difficulty_from_variance(features: &Array2<f32>) -> ModelResult<Array1<f32>> {
    let n = features.nrows();
    if n == 0 {
        return Err(ModelError::invalid_config(
            "Cannot estimate difficulty from empty feature matrix",
        ));
    }

    let ncols = features.ncols();

    // Compute per-sample variance of features.
    let mut variances = Vec::with_capacity(n);
    for row in features.rows() {
        let mean: f32 = row.iter().sum::<f32>() / ncols.max(1) as f32;
        let var: f32 =
            row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / ncols.max(1) as f32;
        variances.push(var);
    }

    estimate_difficulty_from_loss(&variances)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    /// Helper: create an ArrayDataProvider with n samples, nf features.
    fn make_provider(n: usize, nf: usize) -> ArrayDataProvider {
        let features = Array2::<f32>::zeros((n, nf));
        let targets = Array1::<f32>::zeros(n);
        ArrayDataProvider::new(features, targets)
    }

    // 1. Competence-based pacing
    #[test]
    fn test_curriculum_competence_pacing() {
        let strategy = CurriculumStrategy::Competence {
            initial: 0.2,
            increment: 0.1,
        };
        let mut sched = CurriculumScheduler::new(strategy);

        // Before any step: competence = 0.2
        assert!((sched.current_competence() - 0.2).abs() < 1e-6);

        // After step 1: competence = 0.2 + 0.1 = 0.3
        let c1 = sched.step();
        assert!((c1 - 0.3).abs() < 1e-6);

        // After step 2: 0.4
        let c2 = sched.step();
        assert!((c2 - 0.4).abs() < 1e-6);

        // Filter test: difficulties [0.1, 0.35, 0.5, 0.9]
        let difficulties = [0.1, 0.35, 0.5, 0.9];
        let indices = sched.filter_indices(&difficulties);
        // Competence = 0.4 → should include 0.1 and 0.35
        assert_eq!(indices, vec![0, 1]);

        // Step a lot — should clamp at 1.0
        for _ in 0..20 {
            sched.step();
        }
        assert!((sched.current_competence() - 1.0).abs() < 1e-6);
        // Now all should be included
        let all = sched.filter_indices(&difficulties);
        assert_eq!(all.len(), 4);
    }

    // 2. Self-Paced Learning
    #[test]
    fn test_curriculum_self_paced() {
        let strategy = CurriculumStrategy::SelfPaced { lambda: 0.4 };
        let mut sched = CurriculumScheduler::new(strategy);

        // Initial: lambda = 0.4, boundary = 0.2
        // Samples below 0.2 are included
        let difficulties = [0.1, 0.19, 0.3, 0.5, 0.8];

        let easy_first = sched.filter_indices(&difficulties);
        // 0.1 and 0.19 are < 0.2
        assert_eq!(easy_first, vec![0, 1]);

        // After several steps, lambda grows and more are included
        for _ in 0..5 {
            sched.step();
        }

        let more = sched.filter_indices(&difficulties);
        // Lambda has grown: 0.4 * 1.2^5 = ~0.995
        // Boundary = ~0.497, so 0.1, 0.19, 0.3 should now be included
        assert!(
            more.len() >= 3,
            "expected >= 3 included, got {}",
            more.len()
        );

        // After many steps, all should be included (lambda large enough)
        for _ in 0..20 {
            sched.step();
        }
        let all = sched.filter_indices(&difficulties);
        assert_eq!(all.len(), difficulties.len());
    }

    // 3. Annealing
    #[test]
    fn test_curriculum_annealing() {
        let strategy = CurriculumStrategy::Annealing {
            start_difficulty: 0.1,
            end: 1.0,
        };
        let mut sched = CurriculumScheduler::new(strategy);

        // Initial gate = start_difficulty = 0.1
        assert!((sched.current_competence() - 0.1).abs() < 1e-6);

        // After 50 steps (halfway through 100 ramp): gate = 0.1 + 0.9*0.5 = 0.55
        for _ in 0..50 {
            sched.step();
        }
        assert!(
            (sched.current_competence() - 0.55).abs() < 0.02,
            "expected ~0.55, got {}",
            sched.current_competence()
        );

        // After 100 steps: gate = 1.0
        for _ in 0..50 {
            sched.step();
        }
        assert!(
            (sched.current_competence() - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            sched.current_competence()
        );

        // All samples should be included at gate = 1.0
        let diffs = [0.0, 0.3, 0.7, 1.0];
        let included = sched.filter_indices(&diffs);
        assert_eq!(included.len(), 4);
    }

    // 4. CurriculumDataProvider active indices grow
    #[test]
    fn test_curriculum_provider_active_indices() {
        let provider = make_provider(10, 2);
        // Difficulties: 0.0, 0.1, 0.2, ..., 0.9
        let difficulties = Array1::from_vec((0..10).map(|i| i as f32 * 0.1).collect());

        let strategy = CurriculumStrategy::Competence {
            initial: 0.0,
            increment: 0.2,
        };

        let mut cp = CurriculumDataProvider::new(provider, difficulties, strategy)
            .expect("construction should succeed");

        // Initial competence = 0.0 → only difficulty == 0.0 passes (index 0)
        let a0 = cp.active_indices();
        assert_eq!(a0, vec![0]);

        // After 1 epoch: competence = 0.2 → indices 0,1,2 (difficulties 0.0, 0.1, 0.2)
        cp.advance_epoch();
        let a1 = cp.active_indices();
        assert_eq!(a1, vec![0, 1, 2]);

        // After 2 epochs: competence = 0.4 → indices 0..=4
        cp.advance_epoch();
        let a2 = cp.active_indices();
        assert_eq!(a2, vec![0, 1, 2, 3, 4]);

        // Set grows monotonically
        assert!(a2.len() > a1.len());
        assert!(a1.len() > a0.len());
    }

    // 5. CurriculumDataProvider implements DataProvider correctly
    #[test]
    fn test_curriculum_provider_implements_data_provider() {
        let features = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 2.0, // easy
                3.0, 4.0, // easy
                5.0, 6.0, // medium
                7.0, 8.0, // hard
                9.0, 10.0, // hard
            ],
        )
        .expect("shape ok");
        let targets = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let data = ArrayDataProvider::new(features, targets);

        let difficulties = Array1::from_vec(vec![0.1, 0.15, 0.5, 0.8, 0.95]);
        let strategy = CurriculumStrategy::Competence {
            initial: 0.2,
            increment: 0.1,
        };

        let cp = CurriculumDataProvider::new(data, difficulties, strategy)
            .expect("construction should succeed");

        // At initial competence 0.2: indices 0,1 are active (difficulties 0.1, 0.15)
        assert_eq!(cp.num_samples(), 2);
        assert_eq!(cp.num_features(), 2);

        // get_batch with active-set indices [0, 1] → original indices [0, 1]
        let (feat, tgt) = cp.get_batch(&[0, 1]);
        assert_eq!(feat.shape(), &[2, 2]);
        assert_eq!(tgt.len(), 2);

        // Verify actual values match original samples 0 and 1
        assert!((feat[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((feat[[1, 0]] - 3.0).abs() < 1e-6);
        assert!((tgt[0] - 10.0).abs() < 1e-6);
        assert!((tgt[1] - 20.0).abs() < 1e-6);
    }
}
