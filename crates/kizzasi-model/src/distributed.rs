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
    /// Set to `Some((expected_len, got_len))` when a worker's Phase-1
    /// deposit detects a gradient-length mismatch.
    ///
    /// Once set, the round still completes the full arrive/depart barrier —
    /// it just skips computing/publishing an averaged result — so every
    /// worker (including ones that already deposited successfully) wakes up
    /// and returns the *same* `DimensionMismatch` error instead of hanging
    /// forever waiting for a generation bump nobody will publish, and the
    /// barrier resets cleanly for the next round exactly like the success
    /// path does.
    aborted: Option<(usize, usize)>,
}

impl BarrierState {
    fn new() -> Self {
        Self {
            accumulator: None,
            result: None,
            arrived: 0,
            departed: 0,
            generation: 0,
            aborted: None,
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
    /// # Errors
    /// Returns [`ModelError::InvalidConfig`] if `num_workers == 0` — a
    /// barrier for zero workers could never complete a round, and library
    /// code must report that as a typed error rather than panicking on a
    /// caller-supplied value.
    pub fn new_workers(num_workers: usize) -> ModelResult<Vec<Self>> {
        if num_workers == 0 {
            return Err(ModelError::invalid_config(
                "ThreadedGradientSync::new_workers: num_workers must be at least 1",
            ));
        }
        let shared = Arc::new(SharedState::new(num_workers));
        Ok((0..num_workers)
            .map(|id| Self {
                shared: Arc::clone(&shared),
                worker_id: id,
            })
            .collect())
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
        // Phase 1+2: deposit gradients into the shared accumulator, then
        // barrier — wait until all workers have deposited. The last worker
        // to arrive computes the mean (or, if any deposit this round hit a
        // dimension mismatch, records the abort) and wakes everyone else.
        //
        // A dimension mismatch is recorded in `state.aborted` rather than
        // returned immediately: an early `return` here (before `arrived` is
        // incremented and before any wake-up) is exactly what left every
        // other worker blocked in `wait_while` forever, since the
        // generation counter they're waiting on would never advance.
        // ----------------------------------------------------------------
        let mut state = self
            .shared
            .inner
            .lock()
            .map_err(|_| ModelError::load_error("gradient sync", "barrier mutex poisoned"))?;

        if state.aborted.is_none() {
            match state.accumulator.as_mut() {
                None => {
                    state.accumulator = Some(gradients.iter().copied().collect());
                }
                Some(acc) => {
                    if acc.len() != n {
                        state.aborted = Some((acc.len(), n));
                    } else {
                        for (a, &g) in acc.iter_mut().zip(gradients.iter()) {
                            *a += g;
                        }
                    }
                }
            }
        }
        state.arrived += 1;

        if state.arrived == num_workers {
            // Last worker: compute average and publish result (skipped if
            // this round was aborted — there is nothing valid to average).
            if state.aborted.is_none() {
                if let Some(acc) = state.accumulator.take() {
                    let scale = 1.0 / num_workers as f32;
                    state.result = Some(acc.iter().map(|&x| x * scale).collect());
                }
            }
            state.generation = state.generation.wrapping_add(1);
            self.shared.all_arrived.notify_all();
        } else {
            let gen_before = state.generation;
            // Release lock and wait.
            state = self
                .shared
                .all_arrived
                .wait_while(state, |s| s.generation == gen_before)
                .map_err(|_| {
                    ModelError::load_error("gradient sync", "condvar wait failed (arrived)")
                })?;
        }

        let abort = state.aborted;
        drop(state);

        // ----------------------------------------------------------------
        // Phase 3: read back the averaged result (result is now published) —
        // skipped when the round was aborted, since no result was computed.
        // ----------------------------------------------------------------
        if abort.is_none() {
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
        // (including `aborted`) so the next round can begin. Earlier
        // departing workers wait until reset is complete to prevent fast
        // workers from lapping. This runs identically whether or not the
        // round was aborted, so a mismatched-length round never leaves the
        // barrier's arrived/departed counters desynchronized for the round
        // after it.
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
                state.aborted = None;
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

        match abort {
            Some((expected, got)) => Err(ModelError::dimension_mismatch(
                "gradient sync",
                expected,
                got,
            )),
            None => Ok(()),
        }
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
/// })?;
/// ```
///
/// # Type bounds
///
/// `F` must be `Send + Sync + Clone` so it can be cloned per worker and sent
/// across thread boundaries.
///
/// # Errors
/// Returns [`ModelError::InvalidConfig`] if `num_workers == 0`. Returns
/// [`ModelError::LoadError`] identifying the worker index and (when
/// recoverable as a string) the panic message if any worker closure panics,
/// instead of propagating the panic to the calling thread.
pub fn run_parallel_workers<F>(num_workers: usize, f: F) -> ModelResult<Vec<Array1<f32>>>
where
    F: Fn(ThreadedGradientSync) -> Array1<f32> + Send + Sync + Clone + 'static,
{
    let syncs = ThreadedGradientSync::new_workers(num_workers)?;
    let f = Arc::new(f);

    let handles: Vec<_> = syncs
        .into_iter()
        .enumerate()
        .map(|(worker_idx, sync)| {
            let f_clone = Arc::clone(&f);
            (worker_idx, std::thread::spawn(move || f_clone(sync)))
        })
        .collect();

    handles
        .into_iter()
        .map(|(worker_idx, h)| {
            h.join().map_err(|panic_payload| {
                let message = panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                ModelError::load_error(
                    "run_parallel_workers",
                    format!("worker {worker_idx} panicked: {message}"),
                )
            })
        })
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
    fn test_new_workers_zero_returns_err_not_panic() {
        // Regression test: `new_workers` used to `assert!(num_workers > 0)`,
        // aborting the process on a caller-supplied value instead of
        // returning a typed error like every other fallible constructor in
        // this module.
        let result = ThreadedGradientSync::new_workers(0);
        assert!(result.is_err(), "num_workers == 0 must be a typed Err");
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
        })
        .expect("run_parallel_workers should not report a worker panic");

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
    fn test_threaded_gradient_sync_dimension_mismatch_does_not_deadlock() {
        // Regression test for the Phase-1 early-return deadlock: a
        // dimension mismatch used to `return Err` before incrementing
        // `arrived` or notifying anyone, leaving every other worker blocked
        // in `wait_while` forever. Reports back through an `mpsc::channel`
        // with a bounded `recv_timeout` (rather than a bare `thread::join`)
        // so that if the fix regresses, this test *fails* instead of
        // hanging the whole test run.
        use std::sync::mpsc;
        use std::time::Duration;

        let syncs = ThreadedGradientSync::new_workers(3).expect("3 workers is a valid count");
        let (tx, rx) = mpsc::channel();

        for (idx, sync) in syncs.into_iter().enumerate() {
            let tx = tx.clone();
            std::thread::spawn(move || {
                // Workers 0 and 2 agree on length 4; worker 1 sends length 7,
                // guaranteeing a Phase-1 dimension mismatch.
                let len = if idx == 1 { 7 } else { 4 };
                let mut grad = Array1::from_vec(vec![1.0_f32; len]);
                let result = sync.sync_gradients(&mut grad);
                let _ = tx.send(result.is_err());
            });
        }
        drop(tx);

        let mut reports = 0;
        for _ in 0..3 {
            match rx.recv_timeout(Duration::from_secs(15)) {
                Ok(is_err) => {
                    reports += 1;
                    assert!(
                        is_err,
                        "a mismatched-length sync_gradients call must return Err"
                    );
                }
                Err(_) => {
                    panic!("sync_gradients deadlocked: a worker did not report back within 15s")
                }
            }
        }
        assert_eq!(reports, 3, "all three workers must report back");
    }

    #[test]
    fn test_threaded_gradient_sync_recovers_after_aborted_round() {
        // After an aborted (dimension-mismatch) round, the barrier must
        // reset cleanly so a subsequent, well-formed round still works —
        // not leave `arrived`/`departed` desynchronized.
        use std::sync::mpsc;
        use std::time::Duration;

        let syncs = ThreadedGradientSync::new_workers(2).expect("2 workers is a valid count");
        let (tx, rx) = mpsc::channel();
        for (idx, sync) in syncs.into_iter().enumerate() {
            let tx = tx.clone();
            std::thread::spawn(move || {
                // Round 1: mismatched lengths -> both must see Err.
                let bad_len = if idx == 0 { 2 } else { 5 };
                let mut bad_grad = Array1::from_vec(vec![1.0_f32; bad_len]);
                let round1_err = sync.sync_gradients(&mut bad_grad).is_err();

                // Round 2: matching lengths -> must succeed and average
                // correctly, proving the barrier recovered.
                let mut grad = Array1::from_vec(vec![(idx as f32 + 1.0) * 2.0, 4.0]);
                let round2_ok = sync.sync_gradients(&mut grad).is_ok();
                let round2_value = grad[0];

                let _ = tx.send((round1_err, round2_ok, round2_value));
            });
        }
        drop(tx);

        let mut received = 0;
        for _ in 0..2 {
            match rx.recv_timeout(Duration::from_secs(15)) {
                Ok((round1_err, round2_ok, round2_value)) => {
                    received += 1;
                    assert!(round1_err, "round 1 (mismatched lengths) must error");
                    assert!(round2_ok, "round 2 (matched lengths) must succeed");
                    // worker 0 sends 2.0, worker 1 sends 4.0 -> mean 3.0
                    assert!(
                        (round2_value - 3.0).abs() < 1e-5,
                        "round 2 must average fresh gradients, got {round2_value}"
                    );
                }
                Err(_) => panic!("recovery round deadlocked: no report within 15s"),
            }
        }
        assert_eq!(received, 2);
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

/// Internal round state for [`SharedGradientStore`], guarded by one `Mutex`.
struct GradStoreRound {
    /// One slot per rank; `Some` once that rank has pushed this round.
    buffers: Vec<Option<Vec<GradientBuffer>>>,
    /// The computed mean, published once every slot is filled.
    result: Option<Vec<GradientBuffer>>,
    /// Set instead of `result` if averaging itself fails (e.g. ranks pushed
    /// mismatched gradient shapes) — recorded so every rank observes the
    /// *same* failure and the round still drains, rather than the erroring
    /// rank returning early and leaving the others waiting on a `result`
    /// that will never arrive.
    failed: Option<String>,
    /// How many ranks have read back `result`/`failed` this round.
    departed: usize,
}

/// Thread-safe gradient store that simulates AllReduce across `world_size` ranks.
///
/// Each rank pushes its local gradients via [`SharedGradientStore::push`],
/// then calls [`SharedGradientStore::all_reduce_mean`] to block until every
/// rank has pushed and receive the element-wise average. This is a genuine
/// barrier — unlike a version that just checks "has everyone pushed yet?"
/// and errors if not, `all_reduce_mean` here *waits*, so the first rank to
/// call it does not simply fail.
///
/// The store is self-draining: once every rank has read back the result for
/// a round, that round's buffers are cleared automatically, so the next
/// round always starts from a clean slate — [`SharedGradientStore::clear`]
/// is only needed to abort a round manually (e.g. on an unrecoverable
/// error), not after every ordinary optimiser step.
///
/// [`SharedGradientStore::push`] blocks if the calling rank's slot is still
/// occupied by a previous round that hasn't fully drained yet, which is what
/// prevents a fast rank from racing ahead into the next round and either
/// overwriting data a slow rank hasn't averaged yet, or having its own
/// fresh push wiped out by that slow rank's delayed round-reset.
pub struct SharedGradientStore {
    inner: Mutex<GradStoreRound>,
    condvar: Condvar,
    world_size: usize,
}

impl SharedGradientStore {
    /// Create a new store for `world_size` ranks. `world_size` is clamped to
    /// at least 1 (a store for zero ranks could never complete a round).
    pub fn new(world_size: usize) -> Self {
        let world_size = world_size.max(1);
        Self {
            inner: Mutex::new(GradStoreRound {
                buffers: vec![None; world_size],
                result: None,
                failed: None,
                departed: 0,
            }),
            condvar: Condvar::new(),
            world_size,
        }
    }

    /// Submit gradient buffers from `rank`.
    ///
    /// Blocks until `rank`'s slot from a previous round has been fully
    /// drained (i.e. every rank has read that round's result), so a rank
    /// can never race more than one round ahead of the others.
    ///
    /// # Errors
    /// Returns an error if `rank >= world_size` or the mutex is poisoned.
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
            .inner
            .lock()
            .map_err(|_| ModelError::load_error("distributed", "lock poisoned"))?;
        while guard.buffers[rank].is_some() {
            guard = self
                .condvar
                .wait(guard)
                .map_err(|_| ModelError::load_error("distributed", "lock poisoned"))?;
        }
        guard.buffers[rank] = Some(grads);
        self.condvar.notify_all();
        Ok(())
    }

    /// Block until every rank has [`push`](Self::push)ed this round, then
    /// return the element-wise mean.
    ///
    /// The mean is computed once — by whichever call observes the last slot
    /// fill in — cached, and handed back to every rank that asks (including
    /// ones that arrive after it was already computed). Once every rank has
    /// read it back, the round's buffers reset automatically for next time.
    ///
    /// # Errors
    /// Returns an error if `rank >= world_size`, the mutex is poisoned, or
    /// averaging fails (e.g. ranks pushed mismatched gradient shapes) — in
    /// which case every rank in the round observes the same error.
    pub fn all_reduce_mean(&self, rank: usize) -> ModelResult<Vec<GradientBuffer>> {
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
            .inner
            .lock()
            .map_err(|_| ModelError::load_error("distributed", "lock poisoned"))?;

        loop {
            if let Some(failed) = guard.failed.clone() {
                Self::depart(&mut guard, self.world_size, &self.condvar);
                return Err(ModelError::load_error("distributed", failed));
            }
            if let Some(result) = guard.result.clone() {
                Self::depart(&mut guard, self.world_size, &self.condvar);
                return Ok(result);
            }
            if guard.buffers.iter().all(|b| b.is_some()) {
                let grad_lists: Vec<Vec<GradientBuffer>> =
                    guard.buffers.iter().filter_map(|b| b.clone()).collect();
                match average_gradients(&grad_lists) {
                    Ok(mean) => guard.result = Some(mean),
                    Err(e) => guard.failed = Some(e.to_string()),
                }
                self.condvar.notify_all();
                continue;
            }
            guard = self
                .condvar
                .wait(guard)
                .map_err(|_| ModelError::load_error("distributed", "lock poisoned"))?;
        }
    }

    /// Record this rank's departure; the last departer resets the round.
    fn depart(guard: &mut GradStoreRound, world_size: usize, condvar: &Condvar) {
        guard.departed += 1;
        if guard.departed >= world_size {
            for slot in guard.buffers.iter_mut() {
                *slot = None;
            }
            guard.result = None;
            guard.failed = None;
            guard.departed = 0;
            condvar.notify_all();
        }
    }

    /// Forcibly reset all gradient buffers and any pending result — an
    /// escape hatch for aborting a round manually (e.g. error recovery).
    /// Not needed for ordinary use: [`all_reduce_mean`](Self::all_reduce_mean)
    /// already drains and resets the round automatically once every rank has
    /// read the result.
    ///
    /// # Errors
    /// Returns an error on lock failure.
    pub fn clear(&self) -> ModelResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| ModelError::load_error("distributed", "lock poisoned"))?;
        for slot in guard.buffers.iter_mut() {
            *slot = None;
        }
        guard.result = None;
        guard.failed = None;
        guard.departed = 0;
        self.condvar.notify_all();
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
///
/// [`Self::new`] gives each instance its *own* private weight map and
/// gradient store — useful for single-rank use (`world_size <= 1`, or
/// `NoSync`), but two `DataParallelModel`s built this way never share
/// anything, so `step` cannot actually synchronize across them: a real
/// multi-rank simulation needs every rank's instance to share one weight
/// map and one [`SharedGradientStore`]. Use [`Self::new_ranks`] to build a
/// `world_size`-length `Vec<Self>` that does that correctly — one instance
/// per rank, each pointed at the *same* underlying weights and store.
pub struct DataParallelModel {
    config: DistributedConfig,
    weights: Arc<std::sync::RwLock<std::collections::HashMap<String, Vec<f32>>>>,
    grad_store: Option<Arc<SharedGradientStore>>,
}

impl DataParallelModel {
    /// Create a new data-parallel model with the given weight map and config.
    ///
    /// This instance's weight map and gradient store are private to it — see
    /// the struct docs and [`Self::new_ranks`] for building an actual
    /// multi-rank group that shares both.
    pub fn new(
        weights: std::collections::HashMap<String, Vec<f32>>,
        config: DistributedConfig,
    ) -> Self {
        let grad_store =
            if config.grad_strategy == GradientStrategy::AllReduce && config.world_size > 1 {
                Some(Arc::new(SharedGradientStore::new(config.world_size)))
            } else {
                None
            };
        Self {
            config,
            weights: Arc::new(std::sync::RwLock::new(weights)),
            grad_store,
        }
    }

    /// Build `config.world_size` rank instances that share one weight map
    /// and one [`SharedGradientStore`], so `step` genuinely synchronizes
    /// across them — this is the constructor a real multi-rank simulation
    /// (e.g. one `DataParallelModel` per worker thread) should use instead
    /// of calling [`Self::new`] once per rank.
    ///
    /// `config.rank` is overridden per returned instance (`0..world_size`);
    /// every other field is shared from `config`. `world_size` is clamped to
    /// at least 1.
    pub fn new_ranks(
        weights: std::collections::HashMap<String, Vec<f32>>,
        config: DistributedConfig,
    ) -> Vec<Self> {
        let world_size = config.world_size.max(1);
        let shared_weights = Arc::new(std::sync::RwLock::new(weights));
        let grad_store = if config.grad_strategy == GradientStrategy::AllReduce && world_size > 1 {
            Some(Arc::new(SharedGradientStore::new(world_size)))
        } else {
            None
        };

        (0..world_size)
            .map(|rank| {
                let mut rank_config = config.clone();
                rank_config.rank = rank;
                rank_config.world_size = world_size;
                Self {
                    config: rank_config,
                    weights: Arc::clone(&shared_weights),
                    grad_store: grad_store.clone(),
                }
            })
            .collect()
    }

    /// Return a snapshot of the current weight map.
    ///
    /// # Errors
    /// Returns an error if the internal `RwLock` is poisoned (a worker thread
    /// panicked while holding the write lock) — the same failure mode every
    /// other method on this type reports via `ModelResult`, rather than
    /// silently handing back an empty map indistinguishable from "no
    /// parameters".
    pub fn weights(&self) -> ModelResult<std::collections::HashMap<String, Vec<f32>>> {
        self.weights
            .read()
            .map(|g| g.clone())
            .map_err(|_| ModelError::load_error("distributed", "weight RwLock poisoned"))
    }

    /// Apply a gradient update using the configured strategy.
    ///
    /// For `AllReduce` with `world_size > 1` this pushes local gradients to the
    /// [`SharedGradientStore`], blocks until every rank has pushed and the
    /// average is ready, and then — on rank 0 only — applies it. For single-
    /// worker or `NoSync` modes the update is applied directly.
    ///
    /// Only rank 0 writes because ranks built via [`Self::new_ranks`] share
    /// one `Arc<RwLock<..>>` weight map (unlike real distributed training,
    /// where every rank holds its own copy): if every rank applied the same
    /// averaged gradient to that one shared map, the update would land
    /// `world_size` times per round — silently scaling the effective
    /// learning rate by `world_size`. Every rank still calls (and is
    /// blocked by) the all-reduce barrier above, so all ranks stay in lock
    /// step round over round; only the write itself is elected to rank 0.
    ///
    /// # Errors
    /// Propagates gradient-store and weight-lock errors.
    pub fn step(&self, local_grads: Vec<GradientBuffer>, learning_rate: f32) -> ModelResult<()> {
        let (effective_grads, should_apply) = match &self.grad_store {
            Some(store) => {
                store.push(self.config.rank, local_grads)?;
                let grads = store.all_reduce_mean(self.config.rank)?;
                (grads, self.config.rank == 0)
            }
            None => (local_grads, true),
        };

        if !should_apply {
            return Ok(());
        }

        let mut guard = self
            .weights
            .write()
            .map_err(|_| ModelError::load_error("distributed", "weight RwLock poisoned"))?;
        sgd_step(&mut guard, &effective_grads, learning_rate)
    }

    /// Broadcast weights from rank 0 to all ranks.
    ///
    /// In-process: ranks built via [`Self::new_ranks`] already share the
    /// same `Arc<RwLock<..>>`, so this is a no-op that succeeds immediately.
    /// This is *not* true of independently-constructed [`Self::new`]
    /// instances, which have their own private weight maps — this method
    /// cannot broadcast between those, since it has no reference to any
    /// other rank's storage.
    pub fn broadcast_weights(&self) -> ModelResult<()> {
        // In-process: shared Arc (when constructed via `new_ranks`) means
        // every rank instance already sees the same data.
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
        let w = model
            .weights()
            .expect("weights() should succeed on a fresh model");
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
        let w = model
            .weights()
            .expect("weights() should succeed after step");
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
        // `all_reduce_mean` is now a real barrier (it *waits* for missing
        // ranks rather than erroring immediately), so the old assertion
        // ("after clear, all_reduce_mean should fail because not all ranks
        // submitted") no longer makes sense for a world_size=1 store — with
        // only one rank, there is nothing left to wait for, so it would
        // simply recompute from whatever is pushed next. This test instead
        // checks what `clear` is actually for: removing a pushed-but-not-yet
        // consumed value so a later push starts from a clean slate, not a
        // stale accumulation.
        let store = SharedGradientStore::new(1);
        let stale = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![1.0_f32],
        }];
        store.push(0, stale).expect("push");
        store.clear().expect("clear");

        let fresh = vec![GradientBuffer {
            name: "w".to_string(),
            gradients: vec![9.0_f32],
        }];
        store.push(0, fresh).expect("push after clear");
        let avg = store
            .all_reduce_mean(0)
            .expect("all_reduce_mean after clear");
        assert!(
            (avg[0].gradients[0] - 9.0).abs() < 1e-6,
            "clear() must have discarded the stale push, not merged it with the fresh one"
        );
    }

    #[test]
    fn test_shared_gradient_store_all_reduce_is_a_real_barrier() {
        // Regression test for the "not a barrier" bug: the FIRST rank to
        // call `all_reduce_mean` used to fail immediately with "not all
        // ranks have submitted gradients" instead of waiting. Rank 0 here
        // calls it before rank 1 has pushed anything, so if the barrier
        // regressed to the old check-and-fail behavior this test's
        // `expect` fails; if it deadlocked, `recv_timeout` catches that
        // instead of hanging the run.
        use std::sync::mpsc;
        use std::time::Duration;

        let store = Arc::new(SharedGradientStore::new(2));
        let (tx, rx) = mpsc::channel();

        let store0 = Arc::clone(&store);
        let tx0 = tx.clone();
        std::thread::spawn(move || {
            let grads0 = vec![GradientBuffer {
                name: "w".to_string(),
                gradients: vec![1.0_f32, 2.0],
            }];
            store0.push(0, grads0).expect("push rank 0");
            // Rank 0 calls all_reduce_mean FIRST, before rank 1 has pushed.
            let result = store0.all_reduce_mean(0);
            let _ = tx0.send(result.map(|r| r[0].gradients.clone()));
        });

        // Give rank 0 a head start so it is very likely to reach
        // all_reduce_mean before rank 1 pushes (not required for
        // correctness, just makes the barrier property the point of the
        // test rather than incidental).
        std::thread::sleep(Duration::from_millis(50));

        let store1 = Arc::clone(&store);
        let tx1 = tx.clone();
        std::thread::spawn(move || {
            let grads1 = vec![GradientBuffer {
                name: "w".to_string(),
                gradients: vec![3.0_f32, 4.0],
            }];
            store1.push(1, grads1).expect("push rank 1");
            let result = store1.all_reduce_mean(1);
            let _ = tx1.send(result.map(|r| r[0].gradients.clone()));
        });
        drop(tx);

        for _ in 0..2 {
            match rx.recv_timeout(Duration::from_secs(15)) {
                Ok(Ok(grads)) => {
                    assert!((grads[0] - 2.0).abs() < 1e-6);
                    assert!((grads[1] - 3.0).abs() < 1e-6);
                }
                Ok(Err(e)) => panic!("all_reduce_mean must not fail: {e}"),
                Err(_) => panic!("all_reduce_mean deadlocked: no report within 15s"),
            }
        }
    }

    #[test]
    fn test_data_parallel_model_new_ranks_two_consecutive_steps_use_fresh_gradients() {
        // Regression test for id105: two ranks built independently via
        // `DataParallelModel::new` never shared a store, so `step` always
        // errored for the first caller; and even sharing a store, a missing
        // `clear()` after each step let stale gradients leak into the next
        // round. `new_ranks` fixes the sharing; the self-draining barrier
        // (see `SharedGradientStore`) fixes the staleness without needing an
        // explicit `clear()` call at all. Runs two full rounds across two
        // real worker threads and checks the second round's weight update
        // reflects only the second round's gradients (and only a *single*
        // application of each round's averaged gradient — see `step`'s docs
        // on why only rank 0 writes to the shared weight map).
        //
        // Only rank 0's message is used for the final weight assertion:
        // rank 0's `step()` call is what performs the write, so waiting on
        // rank 0's own completion message (sent right after its `step()`
        // calls return) is what actually guarantees the write has landed —
        // rank 1 finishing its (non-writing) `step()` calls says nothing
        // about whether rank 0 has finished writing yet.
        use std::sync::mpsc;
        use std::time::Duration;

        let mut weights = std::collections::HashMap::new();
        weights.insert("w".to_string(), vec![100.0_f32]);
        let config = DistributedConfig {
            world_size: 2,
            rank: 0, // overridden per rank by new_ranks
            grad_strategy: GradientStrategy::AllReduce,
            backend: CommBackend::InProcess,
        };
        let ranks = DataParallelModel::new_ranks(weights, config);
        assert_eq!(ranks.len(), 2);

        let (tx, rx) = mpsc::channel();
        // Round 1 gradients: rank0=2.0, rank1=4.0 -> mean 3.0.
        // Round 2 gradients: rank0=10.0, rank1=20.0 -> mean 15.0.
        let round1 = [2.0_f32, 4.0];
        let round2 = [10.0_f32, 20.0];
        let lr = 1.0;

        for model in ranks {
            let tx = tx.clone();
            std::thread::spawn(move || {
                let rank = model.config.rank;
                let step1 = vec![GradientBuffer {
                    name: "w".to_string(),
                    gradients: vec![round1[rank]],
                }];
                let step1_ok = model.step(step1, lr).is_ok();

                let step2 = vec![GradientBuffer {
                    name: "w".to_string(),
                    gradients: vec![round2[rank]],
                }];
                let step2_ok = model.step(step2, lr).is_ok();

                // Only meaningful from rank 0 (the writer); rank 1 may race
                // ahead of rank 0's write, so it reports `None` instead of a
                // value that could spuriously pass or fail.
                let final_weight = if rank == 0 {
                    model.weights().ok().map(|w| w["w"][0])
                } else {
                    None
                };
                let _ = tx.send((rank, step1_ok, step2_ok, final_weight));
            });
        }
        drop(tx);

        let mut received = 0;
        let mut checked_rank0 = false;
        for _ in 0..2 {
            match rx.recv_timeout(Duration::from_secs(15)) {
                Ok((rank, step1_ok, step2_ok, final_weight)) => {
                    received += 1;
                    assert!(step1_ok, "round 1 step must succeed");
                    assert!(step2_ok, "round 2 step must succeed");
                    if rank == 0 {
                        // weight = 100 - lr*mean(round1) - lr*mean(round2)
                        //        = 100 - 3.0 - 15.0 = 82.0
                        // If round 2 had averaged in stale round-1
                        // gradients, or if the update were (incorrectly)
                        // applied once per rank instead of once per round,
                        // this would not equal 82.0.
                        let w = final_weight.expect("rank 0's weights() should succeed");
                        assert!(
                            (w - 82.0).abs() < 1e-4,
                            "expected weight 82.0 after two fresh, once-applied rounds, got {w}"
                        );
                        checked_rank0 = true;
                    }
                }
                Err(_) => panic!("multi-round step deadlocked: no report within 15s"),
            }
        }
        assert_eq!(received, 2);
        assert!(checked_rank0, "rank 0's report must have been received");
    }
}
