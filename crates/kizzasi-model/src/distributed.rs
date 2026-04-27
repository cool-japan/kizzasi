//! Distributed Training Support for kizzasi-model
//!
//! Provides gradient synchronization primitives for single-node and multi-threaded
//! distributed training simulation. The design follows an extensible trait-based
//! architecture so that real network-based all-reduce can be plugged in later.
//!
//! # Architecture
//!
//! - [`GradientSync`]: Core trait for gradient synchronization strategies.
//! - [`LocalGradientSync`]: No-op implementation for single-node training.
//! - [`ThreadedGradientSync`]: `Arc<Mutex>`-based all-reduce for multi-threaded simulation.
//! - [`run_parallel_workers`]: Helper to run closure-per-worker in parallel threads.

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::Array1;
use std::sync::{Arc, Condvar, Mutex};

// ---------------------------------------------------------------------------
// GradientSync trait
// ---------------------------------------------------------------------------

/// Trait for gradient synchronization strategies.
///
/// Implementations are responsible for aggregating gradients across workers
/// (e.g., averaging in all-reduce) and writing the result back in-place.
pub trait GradientSync: Send {
    /// Synchronize (aggregate) gradients across all workers.
    ///
    /// On return `gradients` holds the post-synchronization values.
    fn sync_gradients(&self, gradients: &mut Array1<f32>) -> ModelResult<()>;

    /// Returns `true` if this sync implementation involves multiple workers.
    fn is_distributed(&self) -> bool {
        false
    }

    /// Number of workers participating in synchronization.
    fn num_workers(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// LocalGradientSync — no-op for single-node training
// ---------------------------------------------------------------------------

/// No-op gradient sync for single-node / single-threaded training.
///
/// `sync_gradients` is a pure identity operation; it leaves the gradient
/// array untouched and never allocates.
#[derive(Debug, Clone, Default)]
pub struct LocalGradientSync;

impl LocalGradientSync {
    /// Create a new `LocalGradientSync`.
    pub fn new() -> Self {
        Self
    }
}

impl GradientSync for LocalGradientSync {
    #[inline]
    fn sync_gradients(&self, _gradients: &mut Array1<f32>) -> ModelResult<()> {
        // Single-node: nothing to do.
        Ok(())
    }

    fn is_distributed(&self) -> bool {
        false
    }

    fn num_workers(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// ThreadedGradientSync — barrier + all-reduce over Arc<Mutex<>>
// ---------------------------------------------------------------------------

/// Shared state for a group of [`ThreadedGradientSync`] workers.
///
/// All workers in the same group share a single `SharedState` instance.
/// The barrier uses a single `Mutex<BarrierState>` and a `Condvar` so that
/// accumulation, averaging, read-back, and reset all happen under coordinated
/// locking with no races.
#[derive(Debug)]
struct BarrierState {
    /// Accumulated gradient sum; `None` before the first worker deposits.
    accumulator: Option<Vec<f32>>,
    /// Averaged result available for all workers to read back.
    result: Option<Vec<f32>>,
    /// How many workers have deposited their gradients this round.
    arrived: usize,
    /// How many workers have finished reading back the result.
    departed: usize,
    /// Generation counter — incremented when the averaging is done so waiters
    /// can distinguish this round from the next.
    generation: usize,
}

impl BarrierState {
    fn new() -> Self {
        Self {
            accumulator: None,
            result: None,
            arrived: 0,
            departed: 0,
            generation: 0,
        }
    }
}

#[derive(Debug)]
struct SharedState {
    inner: Mutex<BarrierState>,
    all_arrived: Condvar,
    all_departed: Condvar,
    num_workers: usize,
}

impl SharedState {
    fn new(num_workers: usize) -> Self {
        Self {
            inner: Mutex::new(BarrierState::new()),
            all_arrived: Condvar::new(),
            all_departed: Condvar::new(),
            num_workers,
        }
    }
}

/// All-reduce gradient synchronizer backed by `Arc<Mutex<>>` for multi-threaded
/// training simulation within a single process.
///
/// All workers that share the same underlying `SharedState` barrier must call
/// [`GradientSync::sync_gradients`] with arrays of the same length, otherwise an
/// error is returned. The synchronization algorithm is:
///
/// 1. Worker adds its gradients into the shared accumulator.
/// 2. The last arriving worker computes the element-wise mean, stores it as the
///    result, and signals all waiters.
/// 3. All workers copy the averaged result back into their local gradient buffer.
/// 4. The last departing worker resets state for the next round.
#[derive(Debug, Clone)]
pub struct ThreadedGradientSync {
    shared: Arc<SharedState>,
    worker_id: usize,
}

impl ThreadedGradientSync {
    /// Create `num_workers` sync objects that share the same barrier state.
    ///
    /// # Panics
    ///
    /// Panics if `num_workers == 0`.
    pub fn new_workers(num_workers: usize) -> Vec<Self> {
        assert!(num_workers > 0, "num_workers must be at least 1");
        let shared = Arc::new(SharedState::new(num_workers));
        (0..num_workers)
            .map(|id| Self {
                shared: Arc::clone(&shared),
                worker_id: id,
            })
            .collect()
    }

    /// Return the worker index (0-based) for this instance.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }
}

impl GradientSync for ThreadedGradientSync {
    fn sync_gradients(&self, gradients: &mut Array1<f32>) -> ModelResult<()> {
        let n = gradients.len();
        let num_workers = self.shared.num_workers;

        // ----------------------------------------------------------------
        // Phase 1: deposit gradients into the shared accumulator.
        // ----------------------------------------------------------------
        {
            let mut state =
                self.shared.inner.lock().map_err(|_| {
                    ModelError::load_error("gradient sync", "barrier mutex poisoned")
                })?;

            match state.accumulator.as_mut() {
                None => {
                    state.accumulator = Some(gradients.iter().copied().collect());
                }
                Some(acc) => {
                    if acc.len() != n {
                        return Err(ModelError::dimension_mismatch(
                            "gradient sync",
                            acc.len(),
                            n,
                        ));
                    }
                    for (a, &g) in acc.iter_mut().zip(gradients.iter()) {
                        *a += g;
                    }
                }
            }
            state.arrived += 1;
        }

        // ----------------------------------------------------------------
        // Phase 2: barrier — wait until all workers have deposited; the
        // last worker computes the mean and signals everyone.
        // ----------------------------------------------------------------
        {
            let mut state =
                self.shared.inner.lock().map_err(|_| {
                    ModelError::load_error("gradient sync", "barrier mutex poisoned")
                })?;

            if state.arrived == num_workers {
                // Last worker: compute average and publish result.
                if let Some(acc) = state.accumulator.take() {
                    let scale = 1.0 / num_workers as f32;
                    state.result = Some(acc.iter().map(|&x| x * scale).collect());
                }
                state.generation = state.generation.wrapping_add(1);
                self.shared.all_arrived.notify_all();
            } else {
                let gen_before = state.generation;
                // Release lock and wait.
                let state = self
                    .shared
                    .all_arrived
                    .wait_while(state, |s| s.generation == gen_before)
                    .map_err(|_| {
                        ModelError::load_error("gradient sync", "condvar wait failed (arrived)")
                    })?;
                // Keep `state` alive until end of block so the guard is dropped.
                drop(state);
            }
        }

        // ----------------------------------------------------------------
        // Phase 3: read back the averaged result (result is now published).
        // ----------------------------------------------------------------
        {
            let state =
                self.shared.inner.lock().map_err(|_| {
                    ModelError::load_error("gradient sync", "barrier mutex poisoned")
                })?;
            if let Some(result) = state.result.as_ref() {
                for (g, &r) in gradients.iter_mut().zip(result.iter()) {
                    *g = r;
                }
            }
        }

        // ----------------------------------------------------------------
        // Phase 4: depart barrier — the last departing worker resets state
        // so the next round can begin. Earlier departing workers wait until
        // reset is complete to prevent fast workers from lapping.
        // ----------------------------------------------------------------
        let should_wait;
        {
            let mut state =
                self.shared.inner.lock().map_err(|_| {
                    ModelError::load_error("gradient sync", "barrier mutex poisoned")
                })?;

            state.departed += 1;
            if state.departed == num_workers {
                state.accumulator = None;
                state.result = None;
                state.arrived = 0;
                state.departed = 0;
                self.shared.all_departed.notify_all();
                should_wait = false;
            } else {
                should_wait = true;
            }
        }

        if should_wait {
            let state =
                self.shared.inner.lock().map_err(|_| {
                    ModelError::load_error("gradient sync", "barrier mutex poisoned")
                })?;
            let _guard = self
                .shared
                .all_departed
                .wait_while(state, |s| s.departed != 0)
                .map_err(|_| {
                    ModelError::load_error("gradient sync", "condvar wait failed (departed)")
                })?;
        }

        Ok(())
    }

    fn is_distributed(&self) -> bool {
        true
    }

    fn num_workers(&self) -> usize {
        self.shared.num_workers
    }
}

// ---------------------------------------------------------------------------
// run_parallel_workers helper
// ---------------------------------------------------------------------------

/// Run a closure on each of `num_workers` [`ThreadedGradientSync`] instances
/// in parallel threads, collecting the resulting gradient arrays.
///
/// This is primarily useful for testing all-reduce correctness:
///
/// ```rust,ignore
/// let results = run_parallel_workers(2, |sync| {
///     let mut grad = Array1::from_vec(vec![1.0, 2.0]);
///     sync.sync_gradients(&mut grad).unwrap();
///     grad
/// });
/// ```
///
/// # Type bounds
///
/// `F` must be `Send + Sync + Clone` so it can be cloned per worker and sent
/// across thread boundaries.
pub fn run_parallel_workers<F>(num_workers: usize, f: F) -> Vec<Array1<f32>>
where
    F: Fn(ThreadedGradientSync) -> Array1<f32> + Send + Sync + Clone + 'static,
{
    let syncs = ThreadedGradientSync::new_workers(num_workers);
    let f = Arc::new(f);

    let handles: Vec<_> = syncs
        .into_iter()
        .map(|sync| {
            let f_clone = Arc::clone(&f);
            std::thread::spawn(move || f_clone(sync))
        })
        .collect();

    handles
        .into_iter()
        .map(|h| h.join().expect("worker thread panicked"))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_local_gradient_sync_noop() {
        let sync = LocalGradientSync::new();
        let original = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut gradients = Array1::from_vec(original.clone());

        sync.sync_gradients(&mut gradients)
            .expect("local sync should not fail");

        for (g, o) in gradients.iter().zip(original.iter()) {
            assert!(
                (g - o).abs() < 1e-7,
                "LocalGradientSync must not modify gradients: got {g} expected {o}"
            );
        }

        assert!(!sync.is_distributed());
        assert_eq!(sync.num_workers(), 1);
    }

    #[test]
    fn test_threaded_gradient_sync_averaging() {
        // Worker 0 has gradients [2.0, 4.0], worker 1 has [4.0, 8.0].
        // Expected average: [3.0, 6.0].
        let worker_grads = [vec![2.0_f32, 4.0], vec![4.0_f32, 8.0]];
        let expected = [3.0_f32, 6.0];

        let results = run_parallel_workers(2, move |sync| {
            let id = sync.worker_id();
            let mut grad = Array1::from_vec(worker_grads[id].clone());
            sync.sync_gradients(&mut grad)
                .expect("threaded sync should not fail");
            grad
        });

        for result in &results {
            for (r, e) in result.iter().zip(expected.iter()) {
                assert!(
                    (r - e).abs() < 1e-5,
                    "averaged gradient mismatch: got {r} expected {e}"
                );
            }
        }
    }

    #[test]
    fn test_checkpoint_save_load_weights() {
        use crate::checkpoint::CheckpointManager;
        use std::env::temp_dir;

        let dir = temp_dir().join(format!(
            "kizzasi_weights_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        let manager = CheckpointManager::new(&dir);

        let weights = Array1::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]);
        let bias = 0.42_f32;
        let step = 100_usize;

        let path = manager
            .save_weights(&weights, bias, step)
            .expect("save_weights should succeed");

        let (loaded_weights, loaded_bias) =
            CheckpointManager::load_weights(&path).expect("load_weights should succeed");

        assert_eq!(loaded_weights.len(), weights.len());
        for (l, w) in loaded_weights.iter().zip(weights.iter()) {
            assert!((l - w).abs() < 1e-6, "weight mismatch: {l} vs {w}");
        }
        assert!((loaded_bias - bias).abs() < 1e-6, "bias mismatch");
    }
}

// ---------------------------------------------------------------------------
// Data-Parallel Infrastructure
// ---------------------------------------------------------------------------

/// Gradient averaging strategy for distributed training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientStrategy {
    /// Average gradients across all workers (AllReduce).
    AllReduce,
    /// Reduce to rank 0 only.
    ReduceToRoot,
    /// No gradient sync (for inference).
    NoSync,
}

/// Communication backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommBackend {
    /// In-process simulation — Pure Rust, no networking.
    InProcess,
    /// Placeholder for future external (NCCL/MPI) backend (C dependency, feature-gated).
    #[allow(dead_code)]
    External,
}

/// Configuration for distributed (data-parallel) training or inference.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of data-parallel workers.
    pub world_size: usize,
    /// This worker's rank (0..world_size).
    pub rank: usize,
    /// How gradients are aggregated across workers.
    pub grad_strategy: GradientStrategy,
    /// Communication backend (always InProcess for Pure Rust).
    pub backend: CommBackend,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            grad_strategy: GradientStrategy::AllReduce,
            backend: CommBackend::InProcess,
        }
    }
}

/// Named gradient buffer for a single parameter tensor.
#[derive(Debug, Clone)]
pub struct GradientBuffer {
    /// Parameter name (must match the weight key in the model's weight map).
    pub name: String,
    /// Gradient values, same length as the corresponding weight tensor.
    pub gradients: Vec<f32>,
}

// ---------------------------------------------------------------------------
// SharedGradientStore
// ---------------------------------------------------------------------------

/// Thread-safe gradient store that simulates AllReduce across `world_size` ranks.
///
/// Each rank pushes its local gradients via [`SharedGradientStore::push`].
/// Once all ranks have pushed, any rank can call [`SharedGradientStore::all_reduce_mean`]
/// to obtain the element-wise average. Call [`SharedGradientStore::clear`] after
/// each optimiser step to reset state for the next iteration.
pub struct SharedGradientStore {
    buffers: Arc<Mutex<Vec<Option<Vec<GradientBuffer>>>>>,
    world_size: usize,
}

impl SharedGradientStore {
    /// Create a new store for `world_size` ranks.
    pub fn new(world_size: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(vec![None; world_size])),
            world_size,
        }
    }

    /// Submit gradient buffers from `rank`.
    ///
    /// # Errors
    /// Returns an error if the mutex is poisoned or `rank >= world_size`.
    pub fn push(&self, rank: usize, grads: Vec<GradientBuffer>) -> ModelResult<()> {
        if rank >= self.world_size {
            return Err(ModelError::load_error(
                "distributed",
                format!(
                    "rank {rank} out of bounds for world_size {}",
                    self.world_size
                ),
            ));
        }
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| ModelError::load_error("distributed", "lock poisoned"))?;
        guard[rank] = Some(grads);
        Ok(())
    }

    /// Wait until all ranks have pushed, then return the element-wise mean.
    ///
    /// In tests all ranks run in the same process/thread, so all buffers will
    /// already be filled before this is called.
    ///
    /// # Errors
    /// Returns an error if not all ranks have submitted yet, or on lock failure.
    pub fn all_reduce_mean(&self, _rank: usize) -> ModelResult<Vec<GradientBuffer>> {
        let guard = self
            .buffers
            .lock()
            .map_err(|_| ModelError::load_error("distributed", "lock poisoned"))?;
        let all_filled = guard.iter().all(|b| b.is_some());
        if !all_filled {
            return Err(ModelError::load_error(
                "distributed",
                "not all ranks have submitted gradients",
            ));
        }
        let grad_lists: Vec<Vec<GradientBuffer>> = guard.iter().filter_map(|b| b.clone()).collect();
        drop(guard);
        average_gradients(&grad_lists)
    }

    /// Clear all gradient buffers — call after each optimiser step.
    ///
    /// # Errors
    /// Returns an error on lock failure.
    pub fn clear(&self) -> ModelResult<()> {
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| ModelError::load_error("distributed", "lock poisoned"))?;
        for slot in guard.iter_mut() {
            *slot = None;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DataParallelModel
// ---------------------------------------------------------------------------

/// Data-parallel wrapper around a named weight map.
///
/// Simulates splitting a mini-batch across `world_size` workers, each
/// computing local gradients, then performing an AllReduce followed by an
/// SGD update. Because all workers live in the same process they share a
/// single `Arc<RwLock<HashMap>>` so weight broadcasts are free.
pub struct DataParallelModel {
    config: DistributedConfig,
    weights: Arc<std::sync::RwLock<std::collections::HashMap<String, Vec<f32>>>>,
    grad_store: Option<SharedGradientStore>,
}

impl DataParallelModel {
    /// Create a new data-parallel model with the given weight map and config.
    pub fn new(
        weights: std::collections::HashMap<String, Vec<f32>>,
        config: DistributedConfig,
    ) -> Self {
        let grad_store =
            if config.grad_strategy == GradientStrategy::AllReduce && config.world_size > 1 {
                Some(SharedGradientStore::new(config.world_size))
            } else {
                None
            };
        Self {
            config,
            weights: Arc::new(std::sync::RwLock::new(weights)),
            grad_store,
        }
    }

    /// Return a snapshot of the current weight map.
    pub fn weights(&self) -> std::collections::HashMap<String, Vec<f32>> {
        self.weights.read().map(|g| g.clone()).unwrap_or_default()
    }

    /// Apply a gradient update using the configured strategy.
    ///
    /// For `AllReduce` with `world_size > 1` this pushes local gradients to the
    /// [`SharedGradientStore`] and then applies the averaged result. For single-
    /// worker or `NoSync` modes the update is applied directly.
    ///
    /// # Errors
    /// Propagates gradient-store and weight-lock errors.
    pub fn step(&self, local_grads: Vec<GradientBuffer>, learning_rate: f32) -> ModelResult<()> {
        let effective_grads = match &self.grad_store {
            Some(store) => {
                store.push(self.config.rank, local_grads)?;
                store.all_reduce_mean(self.config.rank)?
            }
            None => local_grads,
        };

        let mut guard = self
            .weights
            .write()
            .map_err(|_| ModelError::load_error("distributed", "weight RwLock poisoned"))?;
        sgd_step(&mut guard, &effective_grads, learning_rate)
    }

    /// Broadcast weights from rank 0 to all ranks.
    ///
    /// In-process: all workers already share the same `Arc`, so this is a
    /// no-op that succeeds immediately.
    pub fn broadcast_weights(&self) -> ModelResult<()> {
        // In-process: shared Arc means all workers see the same data.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Partition `total` sample indices across `world_size` workers using round-robin.
///
/// Returns the indices owned by `rank`.
pub fn partition_indices(total: usize, world_size: usize, rank: usize) -> Vec<usize> {
    let step = world_size.max(1);
    (rank..total).step_by(step).collect()
}

/// Compute the element-wise mean of multiple gradient-buffer lists.
///
/// All lists must contain the same number of buffers, each with the same
/// gradient length.
///
/// # Errors
/// Returns an error if buffer lists are mismatched in length or gradient sizes differ.
pub fn average_gradients(grad_lists: &[Vec<GradientBuffer>]) -> ModelResult<Vec<GradientBuffer>> {
    if grad_lists.is_empty() {
        return Ok(vec![]);
    }
    let n = grad_lists.len() as f32;
    let template = &grad_lists[0];
    let mut result = template.clone();
    for (i, res_buf) in result.iter_mut().enumerate() {
        for list in grad_lists.iter().skip(1) {
            let other = list.get(i).ok_or_else(|| {
                ModelError::load_error("distributed", "gradient list length mismatch")
            })?;
            if other.gradients.len() != res_buf.gradients.len() {
                return Err(ModelError::dimension_mismatch(
                    "average_gradients",
                    res_buf.gradients.len(),
                    other.gradients.len(),
                ));
            }
            for (r, o) in res_buf.gradients.iter_mut().zip(other.gradients.iter()) {
                *r += o;
            }
        }
        for v in res_buf.gradients.iter_mut() {
            *v /= n;
        }
    }
    Ok(result)
}

/// Apply a vanilla SGD update: `weight -= lr * gradient`.
///
/// Only weights that appear in `gradients` are updated; missing parameter
/// names are silently skipped (sparse gradient support).
///
/// # Errors
/// Returns an error if gradient and weight lengths differ for any parameter.
pub fn sgd_step(
    weights: &mut std::collections::HashMap<String, Vec<f32>>,
    gradients: &[GradientBuffer],
    lr: f32,
) -> ModelResult<()> {
    for grad_buf in gradients {
        if let Some(w) = weights.get_mut(&grad_buf.name) {
            if w.len() != grad_buf.gradients.len() {
                return Err(ModelError::dimension_mismatch(
                    "sgd_step",
                    w.len(),
                    grad_buf.gradients.len(),
                ));
            }
            for (wi, &gi) in w.iter_mut().zip(grad_buf.gradients.iter()) {
                *wi -= lr * gi;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Data-parallel tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod dp_tests {
    use super::*;

    #[test]
    fn test_partition_indices_basic() {
        let idx = partition_indices(10, 3, 0);
        assert_eq!(idx, vec![0, 3, 6, 9]);
        let idx1 = partition_indices(10, 3, 1);
        assert_eq!(idx1, vec![1, 4, 7]);
        let idx2 = partition_indices(10, 3, 2);
        assert_eq!(idx2, vec![2, 5, 8]);
    }

    #[test]
    fn test_average_gradients_two_workers() {
        let grads1 = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![1.0_f32, 2.0],
        }];
        let grads2 = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![3.0_f32, 4.0],
        }];
        let avg = average_gradients(&[grads1, grads2]).expect("average should succeed");
        assert!((avg[0].gradients[0] - 2.0).abs() < 1e-6);
        assert!((avg[0].gradients[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_step_updates_weights() {
        let mut weights = std::collections::HashMap::new();
        weights.insert("w".to_string(), vec![1.0_f32, 2.0, 3.0]);
        let grads = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![0.1_f32, 0.2, 0.3],
        }];
        sgd_step(&mut weights, &grads, 1.0).expect("sgd_step should succeed");
        assert!((weights["w"][0] - 0.9).abs() < 1e-6);
        assert!((weights["w"][1] - 1.8).abs() < 1e-6);
        assert!((weights["w"][2] - 2.7).abs() < 1e-6);
    }

    #[test]
    fn test_shared_gradient_store_all_reduce() {
        let store = SharedGradientStore::new(2);
        let grads0 = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![1.0_f32, 2.0],
        }];
        let grads1 = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![3.0_f32, 4.0],
        }];
        store.push(0, grads0).expect("push rank 0");
        store.push(1, grads1).expect("push rank 1");
        let avg = store.all_reduce_mean(0).expect("all_reduce_mean");
        assert!((avg[0].gradients[0] - 2.0).abs() < 1e-6);
        assert!((avg[0].gradients[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_data_parallel_model_weights_shared() {
        let mut weights = std::collections::HashMap::new();
        weights.insert("embed".to_string(), vec![0.1_f32; 16]);
        let model = DataParallelModel::new(weights, DistributedConfig::default());
        let w = model.weights();
        assert!(w.contains_key("embed"));
        assert_eq!(w["embed"].len(), 16);
    }

    #[test]
    fn test_distributed_config_default() {
        let cfg = DistributedConfig::default();
        assert_eq!(cfg.world_size, 1);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.grad_strategy, GradientStrategy::AllReduce);
        assert_eq!(cfg.backend, CommBackend::InProcess);
    }

    #[test]
    fn test_partition_indices_single_worker() {
        let idx = partition_indices(5, 1, 0);
        assert_eq!(idx, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_average_gradients_single() {
        let grads = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![2.0_f32, 4.0],
        }];
        let avg = average_gradients(&[grads]).expect("single-list average");
        assert_eq!(avg[0].gradients, vec![2.0_f32, 4.0]);
    }

    #[test]
    fn test_data_parallel_model_step_single_worker() {
        let mut weights = std::collections::HashMap::new();
        weights.insert("w".to_string(), vec![1.0_f32, 2.0]);
        let model = DataParallelModel::new(weights, DistributedConfig::default());
        let grads = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![0.5_f32, 0.5],
        }];
        model.step(grads, 0.1).expect("step should succeed");
        let w = model.weights();
        assert!((w["w"][0] - 0.95).abs() < 1e-6);
        assert!((w["w"][1] - 1.95).abs() < 1e-6);
    }

    #[test]
    fn test_broadcast_weights_noop() {
        let weights = std::collections::HashMap::new();
        let model = DataParallelModel::new(weights, DistributedConfig::default());
        assert!(model.broadcast_weights().is_ok());
    }

    #[test]
    fn test_shared_gradient_store_clear() {
        let store = SharedGradientStore::new(1);
        let grads = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![1.0_f32],
        }];
        store.push(0, grads).expect("push");
        store.clear().expect("clear");
        // After clear, all_reduce_mean should fail (not all submitted)
        assert!(store.all_reduce_mean(0).is_err());
    }
}
