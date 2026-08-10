# Training Guide

**Kizzasi v0.2.2** | Training Autoregressive Signal Models

---

## 1. Overview

Training a Kizzasi AGSP means fitting the parameters of a Selective State Space Model (SSM) to a time-series dataset. The SSM maintains a fixed-size hidden state that is updated in O(1) time per step, so the forward pass through the recurrence is always constant-cost regardless of sequence length. Gradient-based learning, however, requires Backpropagation Through Time (BPTT): the autoregressive recurrence is unrolled across the sequence dimension and gradients flow backward through every timestep.

Kizzasi provides two complementary training subsystems:

- **`kizzasi-core`** — the primary path for training `TrainableSSM` models. It uses [Candle](https://github.com/huggingface/candle) for automatic differentiation. The `Trainer` struct drives the full loop: forward pass, backward pass, gradient clipping, optimizer step, LR scheduling, validation, early stopping, and checkpointing.
- **`kizzasi-model`** — a lighter, pure-scirs2 path built around `TrainingLoop`. It targets simple regression tasks (linear models over `Array1<f32>`) and is the entry point for curriculum learning, `ArrayDataProvider`-backed datasets, and in-process distributed gradient sync.

Most production workflows use `kizzasi-core`'s `Trainer` + `TrainableSSM`. The `kizzasi-model` training API is appropriate when you need fine-grained control over the data pipeline or curriculum strategies.

---

## 2. The Training Loop

### 2.1 `kizzasi-core`: `Trainer`

`Trainer` is the main orchestrator for SSM training. It wraps a `TrainableSSM`, an `AdamW` optimizer, an optional `LRScheduler`, a `TrainingMetrics` recorder, and a `MetricsLogger`.

```rust
use kizzasi_core::{
    KizzasiConfig, TrainableSSM, Trainer, TrainingConfig,
    SchedulerType, Loss, TimeSeriesDataLoader, DataLoaderConfig,
};

let model_config = KizzasiConfig::new()
    .input_dim(3)
    .output_dim(3)
    .hidden_dim(128)
    .state_dim(16)
    .num_layers(4);

let training_config = TrainingConfig {
    learning_rate: 1e-3,
    epochs: 20,
    batch_size: 16,
    grad_clip: Some(1.0),
    ..Default::default()
}
.with_scheduler(SchedulerType::Cosine { warmup_steps: 50, min_lr: 1e-6 })
.with_early_stopping(5);

let model = TrainableSSM::new(model_config, training_config.clone())?;
let mut trainer = Trainer::new(model, training_config)?;

// Full training loop with validation
trainer.fit(train_loader, Some(val_loader), Loss::mse)?;
```

#### Core methods

| Method | Description |
|---|---|
| `Trainer::new(model, config)` | Constructs the trainer; creates optimizer and scheduler from `config` |
| `trainer.fit(train_loader, val_loader, loss_fn)` | Runs all epochs with validation and early stopping |
| `trainer.train_epoch(batches, loss_fn)` | Single-epoch forward+backward+step loop; returns average loss |
| `trainer.evaluate(batches, loss_fn)` | Validation pass (no gradient tracking) |
| `trainer.model()` / `trainer.model_mut()` | Access the underlying `TrainableSSM` |
| `trainer.metrics()` | Read `TrainingMetrics` (losses, LRs, grad norms) |
| `trainer.current_step()` | Total optimizer steps taken |

The `train_epoch` implementation:
1. Calls `model.forward(inputs)` — full BPTT graph is built.
2. Evaluates `loss_fn(&predictions, targets)`.
3. Calls `loss.backward()` to materialise a `GradStore`.
4. Clips the `GradStore` by global L2 norm if `config.grad_clip` is set.
5. Optionally records the gradient norm via `TrainingMetrics`.
6. Calls `optimizer.step(&grads)`.

### 2.2 `kizzasi-model`: `TrainingLoop`

`TrainingLoop` from `kizzasi-model` is designed for array-backed datasets and simple linear or SSM-shaped models.

```rust
use kizzasi_model::training_loop::{
    ArrayDataProvider, TrainingConfig, TrainingLoop,
    AdamOptimizer, ExponentialScheduler,
};
use kizzasi_model::checkpoint::EarlyStopping;
use scirs2_core::ndarray::{Array1, Array2};

let data = ArrayDataProvider::new(features, targets);
let config = TrainingConfig {
    max_epochs: 200,
    batch_size: 32,
    learning_rate: 1e-3,
    val_fraction: 0.1,
    rng_seed: 42,
    log_every_n_epochs: 10,
};

let mut optimizer   = AdamOptimizer::new(config.learning_rate);
let mut scheduler   = ExponentialScheduler::new(config.learning_rate, 0.96, 0.0001);
let mut weights     = Array1::<f32>::zeros(num_features);
let mut bias        = 0.0_f32;
let mut early_stop  = EarlyStopping::new(10, 1e-4);

let mut training_loop = TrainingLoop::new(config);
let result = training_loop.run(
    &data,
    &mut optimizer,
    &mut scheduler,
    Some(&mut early_stop),
    &mut weights,
    &mut bias,
)?;

println!("Finished at epoch {} — final loss: {:.6}", result.epochs_trained, result.final_train_loss);
```

#### `TrainingResult` fields

| Field | Type | Description |
|---|---|---|
| `train_losses` | `Vec<f32>` | Per-epoch average training loss |
| `val_losses` | `Vec<Option<f32>>` | Per-epoch validation loss (`None` when `val_fraction == 0.0`) |
| `best_epoch` | `usize` | 0-based index of the epoch with lowest validation loss |
| `best_val_loss` | `Option<f32>` | Lowest validation loss seen |
| `epochs_trained` | `usize` | Actual epochs run (may be less than `max_epochs` if early stopped) |
| `final_train_loss` | `f32` | Training loss on the last completed epoch |

---

## 3. Training Configuration

### 3.1 `kizzasi-core`: `TrainingConfig`

`TrainingConfig` (from `kizzasi_core::training_core`) is the primary configuration type for `TrainableSSM` + `Trainer`.

```rust
use kizzasi_core::{TrainingConfig, SchedulerType, MixedPrecision};

let config = TrainingConfig {
    learning_rate: 1e-4,
    batch_size: 32,
    epochs: 10,
    weight_decay: 1e-2,
    grad_clip: Some(1.0),
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    scheduler: None,
    track_metrics: true,
    log_interval: 10,
    validation_split: 0.2,
    early_stopping_patience: Some(5),
    use_gradient_checkpointing: false,
    checkpoint_segment_size: Some(2),
    mixed_precision: MixedPrecision::None,
    loss_scale: 1.0,
    ..Default::default()
};
```

Builder methods are available for common overrides:

```rust
let config = TrainingConfig::default()
    .with_scheduler(SchedulerType::Cosine { warmup_steps: 100, min_lr: 1e-6 })
    .with_validation_split(0.15)
    .with_early_stopping(10)
    .without_metrics()
    .with_gradient_checkpointing(Some(4))   // checkpoint every 4 layers
    .with_bf16();                           // BF16 mixed precision
```

### 3.2 `kizzasi-model`: `TrainingConfig`

`TrainingConfig` in `kizzasi_model::training` is lighter and used with the `Optimizer` struct (not `TrainingLoop`):

```rust
use kizzasi_model::training::{TrainingConfig, LossFunction, OptimizerType};

let config = TrainingConfig::new()
    .num_epochs(50)
    .batch_size(16)
    .learning_rate(0.001)
    .loss_function(LossFunction::Huber)
    .optimizer(OptimizerType::AdamW)
    .gradient_clipping(Some(1.0));
```

For `TrainingLoop`-based workflows, use `kizzasi_model::training_loop::TrainingConfig` (which has `max_epochs`, `batch_size`, `learning_rate`, `val_fraction`, `rng_seed`, `log_every_n_epochs`).

---

## 4. Loss Functions

### 4.1 Candle-based losses (`kizzasi-core`)

`Loss` (from `kizzasi_core::training_loop`) operates directly on Candle `Tensor` values and supports automatic differentiation:

| Method | Formula | Notes |
|---|---|---|
| `Loss::mse(&pred, &target)` | `mean((pred - target)^2)` | Default; smooth everywhere |
| `Loss::mae(&pred, &target)` | `mean(\|pred - target\|)` | Robust to outliers |
| `Loss::huber(&pred, &target, delta)` | Smooth L1 with threshold `delta` | `delta: f64` parameter |
| `Loss::cross_entropy(&logits, &target)` | Log-softmax NLL | Classification tasks |

```rust
use kizzasi_core::Loss;

let loss = Loss::mse(&predictions, &targets)?;
// or
let loss = Loss::huber(&predictions, &targets, 1.0)?;
```

### 4.2 Array-based losses (`kizzasi-model`)

`LossFunction` (from `kizzasi_model::training`) is an enum with `compute` and `gradient` methods for use in non-Candle workflows:

```rust
use kizzasi_model::training::LossFunction;
use scirs2_core::ndarray::Array1;

let loss_fn = LossFunction::Huber;
let loss = loss_fn.compute(&predictions, &targets)?;
let grad = loss_fn.gradient(&predictions, &targets)?;
```

Available variants: `MSE`, `MAE`, `Huber` (delta = 1.0, fixed), `CrossEntropy`.

### 4.3 Constraint-aware loss

`ConstraintLoss` (from `kizzasi_core::training_loop`) wraps any task loss with a soft constraint penalty:

```rust
use kizzasi_core::{ConstraintLoss, Loss};

let constraint_loss = ConstraintLoss::new(0.1); // constraint_weight

let task_loss = Loss::mse(&predictions, &targets)?;
let total_loss = constraint_loss.compute(&task_loss, &predictions, |pred| {
    // Return constraint violation as f32.
    // Example: penalise predictions outside [-1, 1].
    let vals = pred.flatten_all()?.to_vec1::<f32>()?;
    let violation: f32 = vals.iter()
        .map(|&v| if v.abs() > 1.0 { v.abs() - 1.0 } else { 0.0 })
        .sum::<f32>() / vals.len() as f32;
    Ok(violation)
})?;
```

The combined loss is: `total = task_loss + constraint_weight * violation`. The constraint closure receives the raw prediction tensor and returns a scalar violation. See `crates/kizzasi-logic/examples/training_integration.rs` for the `PenaltyFunction` and `DifferentiableProjection` types from `kizzasi-logic` that complement this pattern.

---

## 5. Optimizers

### 5.1 `kizzasi-core` optimizer

`Trainer` internally uses `candle_nn::AdamW` created via `TrainableSSM::create_optimizer()`. The hyperparameters are taken from `TrainingConfig`:

```rust
TrainingConfig {
    learning_rate: 1e-4,   // initial LR
    weight_decay: 1e-2,    // decoupled L2 (AdamW)
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    grad_clip: Some(1.0),  // global-norm clip threshold
    ..Default::default()
}
```

Gradient clipping is applied by `Trainer::clip_gradients` before `optimizer.step`. The clip is a global L2 norm clip: if the norm of all parameter gradients exceeds `max_norm`, every gradient tensor is scaled down by `max_norm / norm`. This matches PyTorch's `torch.nn.utils.clip_grad_norm_`.

### 5.2 `kizzasi-model` optimizers

For `TrainingLoop`-based workflows, optimizers implement the `Optimizer` trait:

```rust
pub trait Optimizer: Send {
    fn step(&mut self, weights: &mut Array1<f32>, bias: &mut f32,
            weight_grad: &Array1<f32>, bias_grad: f32);
    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, lr: f32);
}
```

Built-in implementations:

```rust
use kizzasi_model::training_loop::{SgdOptimizer, AdamOptimizer};

// Vanilla SGD
let mut sgd = SgdOptimizer::new(0.01);

// Adam with bias correction (β₁=0.9, β₂=0.999, ε=1e-8 by default)
let mut adam = AdamOptimizer::new(0.001);
```

`AdamOptimizer` initialises moment vectors lazily on the first `step` call, so no pre-sizing is needed.

For matrix-parameter models, `kizzasi_model::training::Optimizer` works with `Parameter` objects that carry both data and accumulated gradients:

```rust
use kizzasi_model::training::{Optimizer, OptimizerConfig, OptimizerType};

let config = OptimizerConfig {
    optimizer_type: OptimizerType::AdamW,
    learning_rate: 1e-3,
    weight_decay: 1e-4,
    ..Default::default()
};
let mut optimizer = Optimizer::new(config);
optimizer.step("embedding.weight", &mut param)?;
```

Available `OptimizerType` variants: `SGD`, `SGDMomentum`, `Adam`, `AdamW`.

Standalone gradient clipping utilities:

```rust
use kizzasi_model::training::{clip_gradients, clip_gradients_by_norm};

clip_gradients(&mut grad, 1.0);          // element-wise clamp to ±threshold
clip_gradients_by_norm(&mut grad, 1.0);  // global L2-norm clip
```

---

## 6. Learning Rate Schedulers

Kizzasi provides two families of schedulers depending on which subsystem you are using.

### 6.1 `kizzasi-core` schedulers (`scheduler.rs`)

These implement the `LRScheduler` trait (`get_lr(step: usize) -> f64`). They are stateless step-functions — call `get_lr(current_step)` at any point without mutating the scheduler.

| Scheduler | Constructor | Behavior |
|---|---|---|
| `ConstantScheduler` | `ConstantScheduler::new(lr)` | Fixed LR for all steps |
| `LinearScheduler` | `LinearScheduler::new(initial, final_lr, total_steps, warmup_steps)` | Linear warmup then linear decay to `final_lr` |
| `CosineScheduler` | `CosineScheduler::new(max_lr, total_steps, warmup_steps).with_min_lr(min)` | Linear warmup then cosine annealing to `min_lr` |
| `StepScheduler` | `StepScheduler::new(initial_lr, decay_factor, milestones)` | Multiply LR by `decay_factor` at each milestone step |
| `ExponentialScheduler` | `ExponentialScheduler::new(initial_lr, decay_rate, decay_steps)` | Decay by `decay_rate` every `decay_steps` steps |
| `OneCycleScheduler` | `OneCycleScheduler::new(max_lr, total_steps).with_warmup_pct(0.3)` | Warmup to `max_lr` then cosine decay (super-convergence) |
| `PolynomialScheduler` | `PolynomialScheduler::new(initial_lr, final_lr, total_steps, power)` | Polynomial decay from `initial_lr` to `final_lr` |

Schedulers are attached to `TrainingConfig` via the `SchedulerType` enum:

```rust
use kizzasi_core::SchedulerType;

let config = TrainingConfig::default()
    .with_scheduler(SchedulerType::Cosine { warmup_steps: 100, min_lr: 1e-6 });
// or
    .with_scheduler(SchedulerType::Linear { warmup_steps: 50, final_lr: 1e-6 });
// or
    .with_scheduler(SchedulerType::Step {
        milestones: vec![500, 1000, 1500],
        decay_factor: 0.1,
    });
// or
    .with_scheduler(SchedulerType::Exponential { decay_rate: 0.96, decay_steps: 100 });
// or
    .with_scheduler(SchedulerType::OneCycle { warmup_pct: 0.3 });
// or
    .with_scheduler(SchedulerType::Polynomial { final_lr: 1e-6, power: 2.0 });
```

Direct use of a scheduler:

```rust
use kizzasi_core::scheduler::{CosineScheduler, LRScheduler};

let scheduler = CosineScheduler::new(1e-3, 10_000, 500).with_min_lr(1e-6);
for step in 0..10_000 {
    let lr = scheduler.get_lr(step);
    // Apply lr to optimizer manually or let Trainer handle it.
}
```

**Guidance on scheduler choice:**

| Scenario | Recommended scheduler |
|---|---|
| Quick experiments, stable baseline | `Constant` |
| Fine-tuning a pretrained model | `Cosine` with warmup (avoids large initial updates) |
| Long runs with known milestones | `Step` (LR drops are sharp and predictable) |
| Training from scratch with known budget | `OneCycle` (fast convergence, single pass) |
| Physics-constrained runs requiring smooth LR | `Polynomial` (power >= 2 for gentle decay) |
| Continuous deployment / uncertain budget | `Exponential` (no total_steps required) |

### 6.2 `kizzasi-model` schedulers (`training_loop.rs`)

These implement `LrScheduler` (`step(epoch, val_loss) -> f32`) — they are mutated each epoch. Use them with `TrainingLoop::run`.

| Type | Constructor | Behavior |
|---|---|---|
| `ConstantScheduler` | `ConstantScheduler::new(lr)` | Fixed |
| `ExponentialScheduler` | `ExponentialScheduler::new(initial, decay_rate, min_lr)` | Multiplies by `decay_rate` each epoch, floors at `min_lr` |
| `StepDecayScheduler` | `StepDecayScheduler::new(initial, step_size, gamma)` | Multiplies by `gamma` every `step_size` epochs |

Additionally, `kizzasi_model::training::LearningRateScheduler` wraps `SchedulerConfig` and supports: `Constant`, `Linear`, `Exponential`, `Cosine`, `CosineWarmRestarts`, `WarmupConstant`, `WarmupCosine`. It is step-indexed rather than epoch-indexed:

```rust
use kizzasi_model::training::{LearningRateScheduler, SchedulerConfig, SchedulerType};

let config = SchedulerConfig::new(SchedulerType::WarmupCosine)
    .initial_lr(0.1)
    .min_lr(0.0)
    .warmup_steps(100)
    .total_steps(1000);

let mut scheduler = LearningRateScheduler::new(config);
for _ in 0..1000 {
    let lr = scheduler.get_lr();
    // apply lr ...
    scheduler.step();
}
scheduler.reset(); // restart from step 0 if needed
```

---

## 7. Curriculum Learning

`kizzasi-model` provides a complete curriculum learning subsystem in `curriculum.rs`. The core idea (Bengio et al., 2009) is to expose easier training examples first and gradually admit harder ones, which often improves convergence and generalisation.

### 7.1 `CurriculumStrategy`

```rust
use kizzasi_model::curriculum::CurriculumStrategy;

// Competence-based: include samples with difficulty <= competence.
// Competence starts at `initial` and grows by `increment` each epoch.
let strategy = CurriculumStrategy::Competence { initial: 0.2, increment: 0.1 };

// Self-Paced Learning: soft weight w = max(0, 1 - difficulty/lambda).
// lambda grows by a multiplicative factor each epoch (default 1.2).
let strategy = CurriculumStrategy::SelfPaced { lambda: 0.4 };

// Annealing: linearly interpolate the difficulty gate from start to end
// over 100 epochs.
let strategy = CurriculumStrategy::Annealing { start_difficulty: 0.1, end: 1.0 };
```

### 7.2 `CurriculumScheduler`

```rust
use kizzasi_model::curriculum::{CurriculumScheduler, CurriculumStrategy};

let mut sched = CurriculumScheduler::new(
    CurriculumStrategy::Competence { initial: 0.2, increment: 0.1 }
);

let competence = sched.step();          // advance one epoch; returns current competence
let c = sched.current_competence();     // query without advancing
let included = sched.filter_indices(&difficulty_scores); // -> Vec<usize>
let weight = sched.spl_weight(0.3);    // SPL soft weight for a given difficulty score
```

For SPL, `with_spl_growth` overrides the default growth factor of 1.2:

```rust
let mut sched = CurriculumScheduler::new(CurriculumStrategy::SelfPaced { lambda: 0.5 })
    .with_spl_growth(1.5);
```

### 7.3 `CurriculumDataProvider`

`CurriculumDataProvider` wraps `ArrayDataProvider` and implements `DataProvider`, so it can be dropped directly into `TrainingLoop::run`:

```rust
use kizzasi_model::curriculum::{
    CurriculumDataProvider, CurriculumStrategy,
    estimate_difficulty_from_loss,
    estimate_difficulty_from_variance,
};
use kizzasi_model::training_loop::ArrayDataProvider;
use scirs2_core::ndarray::{Array1, Array2};

let base_data = ArrayDataProvider::new(features, targets);

// Estimate difficulty from per-sample losses (higher loss = harder).
let per_sample_losses: Vec<f32> = vec![/* ... */];
let difficulties = estimate_difficulty_from_loss(&per_sample_losses)?;

// Or estimate from feature variance.
let difficulties = estimate_difficulty_from_variance(&features)?;

let strategy = CurriculumStrategy::Competence { initial: 0.3, increment: 0.05 };
let mut provider = CurriculumDataProvider::new(base_data, difficulties, strategy)?;

for epoch in 0..max_epochs {
    // Step the curriculum before each epoch.
    provider.advance_epoch();

    println!(
        "Epoch {}: {}/{} samples active ({:.1}%)",
        epoch,
        provider.active_count(),
        provider.total_samples(),
        provider.active_fraction() * 100.0
    );

    let result = training_loop.run(
        &provider,
        &mut optimizer,
        &mut scheduler,
        None,
        &mut weights,
        &mut bias,
    )?;
}
```

`active_weights()` returns `(sample_index, weight)` pairs for SPL-weighted sampling.

---

## 8. Checkpointing

### 8.1 `kizzasi-core` checkpointing (`Trainer`)

`Trainer` provides integrated checkpoint save/load backed by SafeTensors (weights) and JSON (metadata):

```rust
// Save after each epoch.
trainer.save_checkpoint("checkpoints/run_1", "epoch_10")?;
// Creates:
//   checkpoints/run_1/epoch_10.safetensors  -- model weights
//   checkpoints/run_1/epoch_10.json         -- CheckpointMetadata

// Convenience: auto-named by epoch.
trainer.save_checkpoint_auto("checkpoints/run_1")?;
// Creates: checkpoints/run_1/checkpoint_epoch_5.safetensors + .json

// Save only when validation loss improves.
trainer.save_best_checkpoint("checkpoints/run_1")?;
// Creates: checkpoints/run_1/best.safetensors + best.json
```

Resume training from a checkpoint:

```rust
use kizzasi_core::{KizzasiConfig, Trainer};

let model_config = KizzasiConfig::new()
    .input_dim(3).output_dim(3).hidden_dim(128).state_dim(16).num_layers(4);

let trainer = Trainer::load_checkpoint("checkpoints/run_1", "epoch_10", model_config)?;
// trainer.current_step() and trainer.metrics() are restored from the checkpoint.
// Continue training:
trainer.fit(train_loader, Some(val_loader), Loss::mse)?;
```

`CheckpointMetadata` (from `kizzasi_core::training_loop`) contains:

| Field | Type | Description |
|---|---|---|
| `version` | `String` | Crate version when checkpoint was written |
| `timestamp` | `String` | ISO 8601 creation time |
| `current_step` | `usize` | Total optimizer steps at save time |
| `current_epoch` | `usize` | Epoch count at save time |
| `config` | `TrainingConfig` | Full training hyperparameters |
| `metrics` | `TrainingMetrics` | Full loss/LR/gradient-norm history |

### 8.2 `kizzasi-model` checkpointing (`CheckpointManager`)

For `TrainingLoop`-backed workflows, `CheckpointManager` serialises `Array1<f32>` weights to JSON:

```rust
use kizzasi_model::checkpoint::{CheckpointManager, EarlyStopping};

let manager = CheckpointManager::new("checkpoints/").max_checkpoints(5);

// Save weights at step 100.
let path = manager.save_weights(&weights, bias, 100)?;

// List checkpoints sorted by step.
for (step, path) in manager.list_weight_checkpoints()? {
    println!("step {}: {}", step, path.display());
}

// Load weights.
let (weights_loaded, bias_loaded) = CheckpointManager::load_weights(&path)?;
```

`EarlyStopping` can be passed directly to `TrainingLoop::run`:

```rust
let mut es = EarlyStopping::new(/*patience=*/5, /*min_delta=*/1e-4);
let result = training_loop.run(&data, &mut opt, &mut sched, Some(&mut es), &mut w, &mut b)?;
```

### 8.3 Gradient checkpointing (activation checkpointing)

`ActivationCheckpointer` (from `kizzasi_model::gradient_checkpoint`) reduces peak memory during the forward pass by saving activations only at designated layer boundaries and recomputing intermediate ones during backprop (Chen et al., 2016).

```rust
use kizzasi_model::gradient_checkpoint::{ActivationCheckpointer, CheckpointConfig};

let config = CheckpointConfig {
    checkpoint_every_n_layers: 4,  // save every 4th layer
    max_checkpoints: 64,
    use_mixed_precision: false,    // simulate fp16 truncation for stored activations
};
let mut cp = ActivationCheckpointer::new(config);

let output = cp.checkpointed_forward(&input, &layer_indices, |act, layer_idx| {
    // Your layer forward function.
    Ok(act.mapv(|x| x.tanh()))
})?;

// During backward: recompute from nearest checkpoint.
let act_at_5 = cp.recompute_from_checkpoint(5, &layer_indices, |act, l| {
    Ok(act.mapv(|x| x.tanh()))
})?;

println!("Memory saved: {} bytes", cp.memory_saved_bytes());
println!("Memory stored: {} bytes", cp.memory_stored_bytes());
```

`TrainingConfig` also controls gradient checkpointing at the `Trainer` level:

```rust
let config = TrainingConfig::default()
    .with_gradient_checkpointing(Some(2)); // checkpoint every 2 layers
```

---

## 9. Distributed Training

### 9.1 `GradientSync` trait

`TrainingLoop` accepts any `GradientSync` implementation via `with_gradient_sync`. The default is `LocalGradientSync` (a no-op):

```rust
pub trait GradientSync: Send {
    fn sync_gradients(&self, gradients: &mut Array1<f32>) -> ModelResult<()>;
    fn is_distributed(&self) -> bool { false }
    fn num_workers(&self) -> usize { 1 }
}
```

### 9.2 In-process multi-threaded sync

`ThreadedGradientSync` implements a full barrier + all-reduce in pure Rust using `Arc<Mutex>` and `Condvar`. Workers deposit gradients, the last to arrive computes the element-wise mean, then all workers read the averaged result back.

```rust
use kizzasi_model::distributed::{ThreadedGradientSync, run_parallel_workers};
use scirs2_core::ndarray::Array1;

// Create 4 sync objects sharing the same barrier.
let syncs = ThreadedGradientSync::new_workers(4);

// run_parallel_workers is a convenience wrapper for thread::spawn.
let results = run_parallel_workers(4, |sync| {
    let mut grad = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    sync.sync_gradients(&mut grad).expect("sync failed");
    grad
});
```

### 9.3 Data-parallel infrastructure

For named-parameter models, `DataParallelModel` wraps a `HashMap<String, Vec<f32>>` weight store and a `SharedGradientStore`:

```rust
use kizzasi_model::distributed::{
    DataParallelModel, DistributedConfig, GradientBuffer, GradientStrategy,
    partition_indices,
};

let config = DistributedConfig {
    world_size: 4,
    rank: 0,
    grad_strategy: GradientStrategy::AllReduce,
    ..Default::default()
};

let model = DataParallelModel::new(weights_map, config);

// Partition the dataset across workers.
let my_indices = partition_indices(total_samples, world_size, rank);

// After computing local gradients:
let local_grads = vec![GradientBuffer {
    name: "embedding.weight".to_string(),
    gradients: vec![/* ... */],
}];
model.step(local_grads, learning_rate)?;

// Broadcast weights back to all ranks (no-op in in-process mode).
model.broadcast_weights()?;
```

`DistributedConfig` defaults to `world_size=1, rank=0, AllReduce, InProcess`.

### 9.4 Integrating sync with `TrainingLoop`

```rust
use kizzasi_model::distributed::ThreadedGradientSync;
use kizzasi_model::training_loop::{
    TrainingLoop, TrainingConfig, SgdOptimizer, ConstantScheduler,
};
use scirs2_core::ndarray::Array1;

let syncs = ThreadedGradientSync::new_workers(2);

let handles: Vec<_> = syncs.into_iter().enumerate().map(|(rank, sync)| {
    let config = TrainingConfig::default();
    std::thread::spawn(move || {
        let mut loop_ = TrainingLoop::new(config).with_gradient_sync(Box::new(sync));
        let mut w = Array1::<f32>::zeros(num_features);
        let mut b = 0.0_f32;
        let mut opt = SgdOptimizer::new(0.01);
        let mut sched = ConstantScheduler::new(0.01);
        loop_.run(&data_shard, &mut opt, &mut sched, None, &mut w, &mut b)
    })
}).collect();
```

---

## 10. End-to-End Example

The following is a condensed version of `crates/kizzasi-core/examples/train_ssm.rs`, showing a complete training run.

### Cargo.toml

```toml
[dependencies]
kizzasi-core  = { version = "0.2" }
kizzasi-model = { version = "0.2" }
scirs2-core   = { version = "0.4" }
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

### Complete training run

```rust
use kizzasi_core::{
    ConstraintLoss, DataLoaderConfig, KizzasiConfig, Loss,
    SchedulerType, TimeSeriesDataLoader, TrainableSSM, Trainer, TrainingConfig,
};
use scirs2_core::ndarray::Array2;
use scirs2_core::convenience::uniform;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // 1. Generate synthetic time-series data.
    let n_steps = 1_000;
    let n_features = 3;
    let data = Array2::from_shape_fn((n_steps, n_features), |_| {
        (uniform() * 2.0 - 1.0) as f32
    });
    let val_data = Array2::from_shape_fn((200, n_features), |_| {
        (uniform() * 2.0 - 1.0) as f32
    });

    // 2. Configure data loaders.
    let dl_config = DataLoaderConfig {
        window_size: 32,
        horizon: 8,
        batch_size: 16,
        shuffle: true,
        overlap: 0.5,
        drop_last: false,
        num_workers: 1,
    };
    let train_loader = TimeSeriesDataLoader::new(data, dl_config.clone())?;
    let val_loader   = TimeSeriesDataLoader::new(val_data, dl_config)?;

    // 3. Build model.
    let model_config = KizzasiConfig::new()
        .input_dim(n_features)
        .output_dim(n_features)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(4);

    // 4. Configure training.
    let training_config = TrainingConfig {
        learning_rate: 1e-3,
        epochs: 20,
        batch_size: 16,
        grad_clip: Some(1.0),
        ..Default::default()
    }
    .with_scheduler(SchedulerType::Cosine { warmup_steps: 50, min_lr: 1e-6 })
    .with_early_stopping(5);

    // 5. Create trainer.
    let model = TrainableSSM::new(model_config, training_config.clone())?;
    let mut trainer = Trainer::new(model, training_config)?;

    // 6. Train with constraint-aware loss.
    //    The closure adds a penalty for predictions outside [-1, 1].
    let constraint_loss = ConstraintLoss::new(0.1);
    trainer.fit(train_loader, Some(val_loader), |pred, tgt| {
        let task_loss = Loss::mse(pred, tgt)?;
        constraint_loss.compute(&task_loss, pred, |p| {
            let vals = p.flatten_all()?.to_vec1::<f32>()?;
            Ok(vals.iter()
               .map(|&v| if v.abs() > 1.0 { v.abs() - 1.0 } else { 0.0 })
               .sum::<f32>() / vals.len() as f32)
        })
    })?;

    // 7. Save the best checkpoint.
    trainer.save_best_checkpoint("checkpoints/")?;

    // 8. Inspect metrics.
    let summary = trainer.metrics().summary();
    println!("Trained {} epochs", summary.total_epochs);
    if let Some(best) = summary.best_val_loss {
        println!("Best val loss: {:.6}", best);
    }

    Ok(())
}
```

---

## 11. Further Reading

- [Architecture Overview](architecture_overview.md) — crate structure and core traits
- [PyTorch Migration Guide](pytorch_migration.md) — loading pre-trained weights from GGUF / SafeTensors
- [Performance Tuning Guide](performance_tuning.md) — SIMD, CUDA, mixed precision
- [Constraints and Guardrails](cross_modal_integration.md) — constraint-aware training loss
- `crates/kizzasi-core/examples/train_ssm.rs` — runnable end-to-end SSM training example
- `crates/kizzasi/examples/fine_tuning.rs` — fine-tuning a pretrained Mamba model
- `crates/kizzasi-logic/examples/training_integration.rs` — `PenaltyFunction` and `DifferentiableProjection` in training loops
