# kizzasi-macros

Procedural macros for the Kizzasi AGSP ecosystem.

## Overview

This crate provides derive macros and procedural macro helpers for Kizzasi, enabling ergonomic configuration patterns and reducing boilerplate code.

## Features

- `#[derive(KizzasiConfig)]` - Automatic builder pattern generation for configurations
- `#[derive(Preset)]` - Generate preset constructor functions
- `#[derive(Instrumented)]` - Automatic metrics instrumentation

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
kizzasi-macros = "0.2.1"
```

Example:

```rust
use kizzasi_macros::KizzasiConfig;

#[derive(KizzasiConfig)]
struct MyConfig {
    learning_rate: f32,
    batch_size: usize,
}

// Automatically generates builder pattern
let config = MyConfig::builder()
    .learning_rate(0.001)
    .batch_size(32)
    .build()?;
```

## Documentation

- [API Documentation](https://docs.rs/kizzasi-macros)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
