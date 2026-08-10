//! Demonstrate the `#[derive(KizzasiConfig)]` procedural macro.
//!
//! Shows all three `#[config(...)]` features:
//!
//! - `#[config(default = EXPR)]` — field becomes optional in builder; uses the expression when unset
//! - `#[config(validate = "path")]` — field value is passed to a fn(&T) -> Result<(), String> after build
//! - `#[config(skip)]` — field is entirely absent from builder; filled by `Default::default()` or
//!   by a `default = EXPR` expression
//!
//! Run with:
//! ```bash
//! cargo run --example kizzasi_config_derive -p kizzasi-macros
//! ```

use kizzasi_macros::KizzasiConfig;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_divisible_by_64(d: &usize) -> Result<(), String> {
    if (*d).is_multiple_of(64) {
        Ok(())
    } else {
        Err(format!("{} is not divisible by 64", d))
    }
}

fn validate_positive_rate(r: &f64) -> Result<(), String> {
    if *r > 0.0 && *r < 1.0 {
        Ok(())
    } else {
        Err(format!("learning_rate must be in (0, 1), got {}", r))
    }
}

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

/// Full-featured config: required fields, defaults, validation, and skip.
#[derive(KizzasiConfig, Debug)]
#[allow(dead_code)]
struct AudioConfig {
    /// Required — no default, not skipped.
    num_layers: usize,

    /// Optional in builder — uses 4096 if not set.
    #[config(default = 4096)]
    context_window: usize,

    /// Optional in builder with a string-expression default.
    #[config(default = "256_usize")]
    hidden_dim: usize,

    /// Required AND validated: must be in (0, 1).
    #[config(validate = "validate_positive_rate")]
    learning_rate: f64,

    /// Optional in builder AND validated: must be divisible by 64.
    #[config(default = 64, validate = "validate_divisible_by_64")]
    state_dim: usize,

    /// Skipped — never appears as a setter; filled by `Default::default()`.
    #[config(skip)]
    internal_cache: Vec<u8>,

    /// Skipped with an explicit default expression.
    #[config(skip, default = 3_usize)]
    retry_limit: usize,
}

/// Minimal config used for a second shape demo.
#[derive(KizzasiConfig, Debug)]
#[allow(dead_code)]
struct TokenizerConfig {
    codebook_size: usize,
    embed_dim: usize,
    commitment_beta: f32,
}

// ---------------------------------------------------------------------------

fn main() {
    println!("=== KizzasiConfig derive demo ===\n");

    // --- Default fields use their default when unset ---
    let cfg = AudioConfig::builder()
        .num_layers(8)
        // context_window and hidden_dim use defaults (4096, 256)
        .learning_rate(3e-4)
        // state_dim uses default 64; internal_cache and retry_limit are skipped
        .build()
        .expect("all required fields supplied, defaults fill the rest");

    println!("AudioConfig with defaults:\n  {:#?}", cfg);
    assert_eq!(cfg.context_window, 4096);
    assert_eq!(cfg.hidden_dim, 256);
    assert_eq!(cfg.state_dim, 64);
    assert!(cfg.internal_cache.is_empty());
    assert_eq!(cfg.retry_limit, 3);

    // --- Explicit overrides replace defaults ---
    let cfg2 = AudioConfig::builder()
        .num_layers(4)
        .context_window(8192)
        .hidden_dim(512)
        .learning_rate(1e-3)
        .state_dim(128)
        .build()
        .expect("valid overrides");

    println!("\nAudioConfig with explicit overrides:\n  {:#?}", cfg2);
    assert_eq!(cfg2.context_window, 8192);
    assert_eq!(cfg2.hidden_dim, 512);
    assert_eq!(cfg2.state_dim, 128);

    // --- Validation failure ---
    let bad_rate = AudioConfig::builder()
        .num_layers(2)
        .learning_rate(1.5) // out-of-range
        .build();
    match bad_rate {
        Ok(_) => panic!("should have failed validation"),
        Err(e) => println!("\nValidation rejected learning_rate=1.5: {}", e),
    }

    let bad_dim = AudioConfig::builder()
        .num_layers(2)
        .learning_rate(1e-4)
        .state_dim(100) // not divisible by 64
        .build();
    match bad_dim {
        Ok(_) => panic!("should have failed validation"),
        Err(e) => println!("Validation rejected state_dim=100: {}", e),
    }

    // --- Missing required field returns Err ---
    let missing = AudioConfig::builder()
        // num_layers intentionally omitted
        .learning_rate(1e-3)
        .build();
    match missing {
        Ok(_) => panic!("should have caught missing field"),
        Err(e) => println!("\nMissing required field caught: {}", e),
    }

    // --- A second, simpler config type ---
    let tok = TokenizerConfig::builder()
        .codebook_size(512)
        .embed_dim(64)
        .commitment_beta(0.25_f32)
        .build()
        .expect("all fields supplied");

    println!("\nTokenizerConfig:\n  {:#?}", tok);

    println!("\nAll demo assertions passed.");
}
