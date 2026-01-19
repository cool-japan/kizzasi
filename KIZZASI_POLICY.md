# Kizzasi Ecosystem Policy

## Core Architectural Principles

This document establishes the foundational policies for the Kizzasi (兆候) Autoregressive General-Purpose Signal Predictor ecosystem to ensure consistency, maintainability, and architectural integrity across all crates.

## Table of Contents

### Part I: Ecosystem Architecture
1. [Overview](#overview)
2. [Dependency Abstraction Policy](#dependency-abstraction-policy)
3. [COOLJAPAN Ecosystem Integration](#cooljapan-ecosystem-integration)
4. [Implementation Guidelines](#implementation-guidelines)

### Part II: Technical Policies
5. [Array Operations Policy](#array-operations-policy)
6. [Random Number Generation Policy](#random-number-generation-policy)
7. [Signal Processing Policy](#signal-processing-policy)
8. [State Space Model Policy](#state-space-model-policy)
9. [Constraint Logic Policy](#constraint-logic-policy)
10. [Error Handling Policy](#error-handling-policy)

### Part III: Implementation
11. [Crate Structure](#crate-structure)
12. [Examples](#examples)
13. [Enforcement](#enforcement)

---

## Part I: Ecosystem Architecture

## Overview

Kizzasi is a Rust-native Autoregressive General-Purpose Signal Predictor (AGSP) designed for continuous signal streams. It is built on the **COOLJAPAN Ecosystem** foundation, leveraging:

- **scirs2-core**: Scientific computing primitives (arrays, random, numeric types)
- **tensorlogic**: Neuro-symbolic constraint enforcement

### Core Philosophy: "Takumi" Engineering

- **High Context**: Capturing long-term dependencies in time-series data
- **Strict Logic**: Enforcing physical constraints and safety rules
- **Edge Native**: Designed for low-latency inference on robotics and IoT devices

## Dependency Abstraction Policy

### Policy: Use COOLJAPAN Ecosystem Dependencies

**Applies to:** All Kizzasi crates
- `kizzasi`, `kizzasi-core`, `kizzasi-logic`, `kizzasi-io`
- All tests, examples, benchmarks

#### Prohibited Direct Dependencies:
```toml
# ❌ FORBIDDEN in Kizzasi crates
[dependencies]
ndarray = { version = "..." }         # ❌ Use scirs2-core instead
rand = { version = "..." }            # ❌ Use scirs2-core instead
rand_distr = { version = "..." }      # ❌ Use scirs2-core instead
num-traits = { version = "..." }      # ❌ Use scirs2-core instead
num-complex = { version = "..." }     # ❌ Use scirs2-core instead
rayon = { version = "..." }           # ❌ Use scirs2-core parallel instead
```

#### Required Dependencies:
```toml
# ✅ REQUIRED in Kizzasi crates
[dependencies]
scirs2-core = { version = "0.1", features = ["array", "random", "simd"] }
tensorlogic = { version = "0.1" }  # For constraint logic
```

#### Prohibited Imports:
```rust
// ❌ FORBIDDEN in Kizzasi crates
use ndarray::*;
use rand::*;
use num_traits::*;
```

#### Required Imports:
```rust
// ✅ REQUIRED in Kizzasi crates
use scirs2_core::ndarray::*;        // Array types, macros
use scirs2_core::random::*;         // Random number generation
use scirs2_core::numeric::*;        // Numeric traits
use scirs2_core::simd_ops::*;       // SIMD operations
use tensorlogic::*;                 // Constraint logic
```

## COOLJAPAN Ecosystem Integration

### Dependency Mapping

| External Crate | COOLJAPAN Module | Note |
|----------------|------------------|------|
| `ndarray` | `scirs2_core::ndarray` | Full functionality |
| `rand` | `scirs2_core::random` | All distributions |
| `rayon` | `scirs2_core::parallel` | Parallel iteration |
| `num-traits` | `scirs2_core::numeric` | All traits |
| `num-complex` | `scirs2_core::numeric` | Complex numbers |

### Signal Processing

For signal processing operations, Kizzasi builds on:
- **scirs2-signal**: DSP algorithms (FFT, filtering, etc.)
- **scirs2-fft**: Fast Fourier Transform

---

## Part II: Technical Policies

## Array Operations Policy

### Mandatory Rules

1. **ALWAYS use `scirs2_core::ndarray`** for all array operations
2. **NEVER use direct `ndarray` dependency**
3. **Use scirs2-core macros**: `array!`, `s!`, `azip!`

### Usage Pattern

```rust
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, array, s};

// CORRECT - Uses scirs2-core arrays
let signal: Array1<f32> = array![1.0, 2.0, 3.0];
let slice = signal.slice(s![0..2]);

// INCORRECT - Direct ndarray
// use ndarray::Array1;  // FORBIDDEN
```

## Random Number Generation Policy

### Mandatory Rules

1. **ALWAYS use `scirs2_core::random`** for random number generation
2. **NEVER use direct `rand` or `rand_distr` dependencies**

### Usage Pattern

```rust
use scirs2_core::random::{thread_rng, Rng, Normal};

// CORRECT - Uses scirs2-core random
let mut rng = thread_rng();
let dist = Normal::new(0.0, 1.0).unwrap();
let sample: f64 = dist.sample(&mut rng);

// INCORRECT - Direct rand usage
// use rand::thread_rng;  // FORBIDDEN
```

## Signal Processing Policy

### Mandatory Rules

1. **Use `kizzasi-io` for signal acquisition**
2. **Use SciRS2 signal processing** when available
3. **Implement custom DSP in `kizzasi-io`** using scirs2-core primitives

### The Inference Loop (The "Pulse")

1. **Input**: Raw signal stream (e.g., 100Hz vibration sensor)
2. **Filter**: Signal preprocessing (low-pass, Kalman, etc.)
3. **Predict**: SSM engine predicts next signal vector
4. **Constrain**: TensorLogic verifies against safety rules
5. **Act**: Sanitized signal sent to actuator

## State Space Model Policy

### Mandatory Rules

1. **Implement SSM in `kizzasi-core`**
2. **Use Candle or Burn for neural backends**
3. **Support O(1) inference step complexity**

### Supported Architectures

- **Mamba/Mamba2**: Selective State Space Models
- **S4**: Structured State Spaces
- **RWKV**: Linear attention

### Backend Configuration

GPU acceleration (CUDA, Metal) should be enabled through scirs2-core features:

```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["scirs2-core/cuda"]      # Uses scirs2-core CUDA backend
metal = ["scirs2-core/metal"]    # Uses scirs2-core Metal backend
parallel = ["scirs2-core/parallel"]  # Parallel processing via scirs2-core
```

**Note:** Do NOT use `candle-core/cuda` or `candle-core/metal` directly. Use scirs2-core abstractions.

## Constraint Logic Policy

### Mandatory Rules

1. **Use `tensorlogic` for all constraint definitions**
2. **Implement constraint enforcement in `kizzasi-logic`**
3. **Convert constraints to differentiable loss for training**

### Usage Pattern

```rust
use tensorlogic::prelude::*;

// Define physical constraints
let physics = logic!(
    forall t: abs(velocity(t)) <= MAX_SPEED
);

// Apply as guardrails
predictor.set_guardrails(physics);
```

## Error Handling Policy

### Mandatory Rules

1. **Define module-specific errors deriving from core**
2. **Provide proper error conversions**
3. **Use thiserror for error definitions**

### Error Hierarchy

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KizzasiError {
    #[error("Core error: {0}")]
    Core(#[from] kizzasi_core::CoreError),

    #[error("Logic error: {0}")]
    Logic(#[from] kizzasi_logic::LogicError),

    #[error("IO error: {0}")]
    Io(#[from] kizzasi_io::IoError),
}
```

---

## Part III: Implementation

## Crate Structure

```
kizzasi/
├── Cargo.toml              # Workspace root
├── KIZZASI_POLICY.md       # This document
├── crates/
│   ├── kizzasi/            # Facade crate (user API)
│   ├── kizzasi-core/       # SSM engine
│   ├── kizzasi-logic/      # TensorLogic bridge
│   └── kizzasi-io/         # Physical world connectors
```

### Crate Responsibilities

| Crate | Role | Dependencies |
|-------|------|--------------|
| `kizzasi` | User-facing API | All internal crates |
| `kizzasi-core` | SSM implementation | scirs2-core, candle |
| `kizzasi-logic` | Constraint enforcement | scirs2-core, tensorlogic |
| `kizzasi-io` | Signal I/O | scirs2-core, rumqttc, cpal |

## Examples

### Example 1: Basic Inference

```rust
use kizzasi::prelude::*;

fn main() -> Result<()> {
    // Initialize predictor
    let config = KizzasiConfig::new()
        .model_type(ModelType::Mamba2)
        .context_window(8192);

    let mut predictor = Kizzasi::new(config)?;

    // Single step prediction
    let input = array![0.1, 0.2, 0.3];
    let output = predictor.step(&input)?;

    Ok(())
}
```

### Example 2: Constrained Inference

```rust
use kizzasi::prelude::*;
use tensorlogic::prelude::*;

fn main() -> Result<()> {
    // Define constraints
    let physics = logic!(
        forall t: abs(pos(t) - pos(t-1)) <= MAX_SPEED
    );

    let mut predictor = Kizzasi::new(config)?;
    predictor.set_guardrails(physics);

    // Predictions are automatically constrained
    for signal in sensor_stream {
        let next = predictor.step(&signal)?;
        // `next` is guaranteed to satisfy physics constraints
    }

    Ok(())
}
```

### Example 3: Signal Processing

```rust
use kizzasi::prelude::*;
use kizzasi_io::{SignalProcessor, Filter};

fn main() -> Result<()> {
    let mut processor = SignalProcessor::new(1024)
        .with_sample_rate(44100.0);

    // Apply filtering
    let filtered = processor.apply_filter(
        &raw_signal,
        Filter::LowPass { cutoff: 1000.0, order: 4 }
    )?;

    Ok(())
}
```

## Enforcement

### Code Review Checklist

- [ ] No direct `ndarray`, `rand`, `num-*` imports
- [ ] Uses `scirs2_core::*` for all array/numeric operations
- [ ] Uses `tensorlogic` for constraint definitions
- [ ] Error types derive from module-specific errors
- [ ] Tests follow the same import policies

### CI Checks (Planned)

1. Dependency audit for prohibited crates
2. Import pattern validation
3. Policy compliance scoring

## Benefits

By following these policies:

1. **Ecosystem Consistency**: Integrates seamlessly with SciRS2/COOLJAPAN
2. **Maintainability**: Updates to scirs2-core benefit all modules
3. **Performance**: Leverage scirs2-core SIMD optimizations
4. **Type Safety**: Consistent types across the ecosystem
5. **Portability**: Platform-specific code isolated in dependencies

---

## Policy Version

- **Version**: 1.0.0
- **Effective Date**: Kizzasi v0.1.0
- **Last Updated**: 2026-01-18
- **Status**: Active

---

*This policy ensures Kizzasi integrates seamlessly with the COOLJAPAN scientific computing ecosystem while maintaining high performance and safety guarantees.*
