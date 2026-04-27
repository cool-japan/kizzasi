//! # PyTorch Migration Demo
//!
//! Shows how to migrate PyTorch / HuggingFace workflows to Kizzasi:
//!
//! 1. **Weight format overview** — SafeTensors, GGUF, JSON, `.pth`
//! 2. **`NameRemapper`** — HuggingFace key translation to Kizzasi names
//! 3. **JSON weight round-trips** — `save_weights_json` / `load_weights_json`
//! 4. **`HfHubConfig`** — Client configuration for Hub downloads (`hf-hub` feature)
//! 5. **`WeightLoader`** — SafeTensors inspection and mapping utilities
//! 6. **Inference API comparison** — PyTorch `model(x)` → `predictor.step(&x, …)`
//!
//! ## PyTorch vs Kizzasi at a Glance
//!
//! | Concern          | PyTorch / HuggingFace                   | Kizzasi                              |
//! |------------------|-----------------------------------------|--------------------------------------|
//! | Philosophy       | Language Model                          | AGSP (continuous signal predictor)   |
//! | Weight format    | `.pth` pickle, SafeTensors, GGUF        | SafeTensors, GGUF, JSON              |
//! | Hub loading      | `AutoModel.from_pretrained(...)`        | `load_from_hub(repo, file, cfg)?`    |
//! | Inference        | `output = model(input)`                 | `output = predictor.step(&input)?`   |
//! | State mgmt       | `model.hidden_state` attribute          | `predictor.reset()` / `fork()`       |
//! | Key translation  | HF key names                            | `NameRemapper::new().remap(key)`     |
//!
//! Run with:
//! ```bash
//! cargo run --example pytorch_migration_demo -p kizzasi
//! ```
//!
//! To enable HuggingFace Hub download support:
//! ```bash
//! cargo run --example pytorch_migration_demo -p kizzasi --features kizzasi-model/hf-hub
//! ```

use kizzasi::prelude::*;
use kizzasi::Kizzasi;
use kizzasi_model::loader::{NameRemapper, WeightLoader};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== PyTorch Migration Demo ===\n");

    demo_weight_format_overview();
    demo_name_remapper()?;
    demo_json_weights()?;
    demo_hf_hub_config();
    demo_inference_api()?;

    println!("\n=== Migration Demo Complete ===");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Weight format overview
// ─────────────────────────────────────────────────────────────────────────────

/// Prints a conceptual overview of the weight formats Kizzasi supports.
///
/// No actual files are loaded here — this is a reference guide embedded in
/// runnable code so it is always up to date.
fn demo_weight_format_overview() {
    println!("--- 1. Weight Format Overview ---");
    println!();

    let formats = [
        (
            "SafeTensors (.safetensors)",
            "Default for HuggingFace Hub uploads. Zero-copy, safe, fast.",
            "kizzasi_model::loader::ModelLoader::new(\"model.safetensors\")?",
        ),
        (
            "GGUF (.gguf)",
            "llama.cpp / Ollama format with quantisation metadata.",
            "kizzasi_model::gguf::GgufFile::open(\"model.gguf\")?",
        ),
        (
            "JSON weight map",
            "Kizzasi's own portable format: HashMap<String, Vec<f32>> → serde_json.",
            "model.save_weights_json(path)?  /  model.load_weights_json(path)?",
        ),
        (
            "PyTorch .pth / .pt",
            "Pickle-based; conversion via pytorch_compat::PyTorchConverter.",
            "kizzasi_model::pytorch_compat::PyTorchConverter::new().load_checkpoint(path)?",
        ),
    ];

    for (fmt, desc, api) in &formats {
        println!("  [{fmt}]");
        println!("    {desc}");
        println!("    API: {api}");
        println!();
    }

    println!("  [loading path summary]");
    println!("    .safetensors  →  ModelLoader → WeightLoader → model.load_*");
    println!("    .gguf         →  GgufFile::open → inspect / load tensors");
    println!("    .json         →  AutoregressiveModel::load_weights_json");
    println!("    .pth/.pt      →  PyTorchConverter → HashMap<String, Vec<f32>>");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. NameRemapper — HuggingFace key translation
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates `NameRemapper`, which translates HuggingFace checkpoint key
/// names into Kizzasi's internal naming convention.
///
/// ## Common HuggingFace → Kizzasi mappings
///
/// | HuggingFace                               | Kizzasi                        |
/// |-------------------------------------------|--------------------------------|
/// | `embedding.weight`                        | `input_proj`                   |
/// | `lm_head.weight`                          | `output_proj`                  |
/// | `layers.{n}.mixer.in_proj.weight`         | `layers.{n}.input_proj`        |
/// | `layers.{n}.mixer.out_proj.weight`        | `layers.{n}.output_proj`       |
/// | `layers.{n}.attn.q_proj.weight`           | `layers.{n}.attention.q`       |
/// | `layers.{n}.attn.k_proj.weight`           | `layers.{n}.attention.k`       |
/// | `layers.{n}.attn.v_proj.weight`           | `layers.{n}.attention.v`       |
/// | `layers.{n}.attn.o_proj.weight`           | `layers.{n}.attention.out`     |
/// | `layers.{n}.mlp.gate_proj.weight`         | `layers.{n}.ff.gate`           |
/// | `layers.{n}.mlp.up_proj.weight`           | `layers.{n}.ff.up`             |
/// | `layers.{n}.mlp.down_proj.weight`         | `layers.{n}.ff.down`           |
fn demo_name_remapper() -> Result<()> {
    println!("--- 2. NameRemapper (HuggingFace → Kizzasi key translation) ---");

    let remapper = NameRemapper::new();

    // Simulate a set of weight keys as they appear in a HuggingFace checkpoint.
    let hf_keys = [
        "embedding.weight",
        "lm_head.weight",
        "layers.0.mixer.in_proj.weight",
        "layers.0.mixer.out_proj.weight",
        "layers.0.attn.q_proj.weight",
        "layers.0.attn.k_proj.weight",
        "layers.0.attn.v_proj.weight",
        "layers.0.attn.o_proj.weight",
        "layers.1.mlp.gate_proj.weight",
        "layers.1.mlp.up_proj.weight",
        "layers.1.mlp.down_proj.weight",
        "layers.2.norm.weight", // passthrough — no rule matches
        "model.custom_key",     // passthrough — no rule matches
    ];

    println!("  {:<45}  Kizzasi key", "HuggingFace key");
    println!("  {}", "-".repeat(80));
    for key in &hf_keys {
        let remapped = remapper.remap(key);
        let marker = if remapped == *key {
            " (passthrough)"
        } else {
            ""
        };
        println!("  {:<45}  {}{}", key, remapped, marker);
    }

    println!();

    // Demonstrate batch remapping on a weight map
    let weight_map: HashMap<String, Vec<f32>> = hf_keys
        .iter()
        .map(|k| (k.to_string(), vec![0.0f32; 4]))
        .collect();

    let remapped_map = remapper.remap_map(weight_map);
    let remapped_count = remapped_map
        .keys()
        .filter(|k| !hf_keys.contains(&k.as_str()))
        .count();
    println!(
        "  remap_map: {} / {} keys were translated.",
        remapped_count,
        hf_keys.len()
    );

    // Typical workflow when loading from HuggingFace:
    //
    //   let raw: HashMap<String, Vec<f32>> = load_from_hub(repo, file, &cfg)?;
    //   let translated = NameRemapper::new().remap_map(raw);
    //   mamba_model.load_weights_from_dict(&translated)?;
    println!("  [tip] Use remap_map() to translate an entire weight dict at once.");
    println!();

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. JSON weight round-trips
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates the JSON weight round-trip: build a model, save its weights to
/// a temp file in `HashMap<String, Vec<f32>>` / `serde_json` format, then
/// reload them into a fresh instance.
///
/// This is the recommended portable checkpoint format for Kizzasi-native models
/// (no dependency on the `safetensors` crate required at runtime).
fn demo_json_weights() -> Result<()> {
    println!("--- 3. JSON Weight Round-Trip ---");

    // Build a small Mamba model
    let predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba)
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(2)
        .build()?;

    // Save to a temporary file
    let tmp_dir = std::env::temp_dir();
    let weight_path = tmp_dir.join("kizzasi_demo_weights.json");

    // The `save_weights_json` / `load_weights_json` API is on `AutoregressiveModel`
    // (kizzasi_model::AutoregressiveModel trait).  The kizzasi::Kizzasi predictor
    // exposes this via `save_checkpoint` / `load_checkpoint` which serialise the
    // full predictor state (including architecture config).
    let checkpoint_path = tmp_dir.join("kizzasi_demo.checkpoint");
    predictor.save_checkpoint(&checkpoint_path)?;
    println!("  Checkpoint saved:  {}", checkpoint_path.display());

    // Reload
    let loaded = Kizzasi::load_checkpoint(&checkpoint_path)?;
    println!("  Checkpoint loaded successfully");
    println!(
        "  Architecture preserved: context_window = {}",
        loaded.context_window()
    );

    // Clean up
    let _ = std::fs::remove_file(&checkpoint_path);
    let _ = std::fs::remove_file(&weight_path);

    println!();
    println!("  [note] For raw weight dicts use AutoregressiveModel::save_weights_json /");
    println!("         load_weights_json from kizzasi_model directly.");
    println!("         The JSON format is: HashMap<String, Vec<f32>> serialised with serde_json.");
    println!();

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. HfHubConfig overview
// ─────────────────────────────────────────────────────────────────────────────

/// Explains `HfHubConfig` and `HfHubClient` without making network calls.
///
/// The `hf-hub` feature of `kizzasi-model` adds a blocking HTTP client
/// (backed by `reqwest::blocking`) that downloads SafeTensors shards from the
/// HuggingFace Hub with caching and optional token auth.
fn demo_hf_hub_config() {
    println!("--- 4. HuggingFace Hub Client (kizzasi-model/hf-hub) ---");
    println!();
    println!("  To enable: add `kizzasi-model` as a dependency with the `hf-hub` feature.");
    println!();
    println!("  ```toml");
    println!("  [dependencies]");
    println!("  kizzasi-model = {{ version = \"0.1\", features = [\"hf-hub\"] }}");
    println!("  ```");
    println!();
    println!("  Usage pattern:");
    println!();
    println!("  ```rust");
    println!("  use kizzasi_model::hf_hub::{{HfHubClient, HfHubConfig}};");
    println!("  use kizzasi_model::loader::NameRemapper;");
    println!();
    println!("  // 1. Configure the client (reads $HF_TOKEN automatically)");
    println!("  let cfg = HfHubConfig {{");
    println!("      cache_dir: std::env::temp_dir().join(\"kizzasi/hub\"),");
    println!("      timeout_secs: 300,");
    println!("      log_sha256: true,");
    println!("      ..Default::default()");
    println!("  }};");
    println!();
    println!("  // 2. Build the client and download");
    println!("  let client = HfHubClient::new(cfg)?;");
    println!("  let weights = kizzasi_model::hf_hub::load_from_hub(");
    println!("      \"state-spaces/mamba-130m\",");
    println!("      \"model.safetensors\",");
    println!("      &HfHubConfig::default(),");
    println!("  )?;");
    println!();
    println!("  // 3. Translate keys and load into model");
    println!("  let translated = NameRemapper::new().remap_map(weights);");
    println!("  mamba_model.load_weights_from_dict(&translated)?;");
    println!("  ```");
    println!();
    println!("  [note] Downloaded files are cached under HfHubConfig::cache_dir.");
    println!("         Subsequent calls reuse the cached file without re-downloading.");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. WeightLoader inspection utilities
// ─────────────────────────────────────────────────────────────────────────────

// (WeightLoader wraps ModelLoader and requires an actual .safetensors file on disk.
//  This demo uses NameRemapper directly — see above — to avoid needing test files.)

// ─────────────────────────────────────────────────────────────────────────────
// 6. Inference API comparison
// ─────────────────────────────────────────────────────────────────────────────

/// Side-by-side comparison of the PyTorch inference loop vs Kizzasi.
///
/// ## PyTorch (Python)
///
/// ```python
/// import torch
/// from mamba_ssm import Mamba
///
/// model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
/// state = None
/// for step in range(100):
///     x = torch.randn(1, 1, 64)          # [batch, seqlen, d_model]
///     y, state = model(x, inference_params=state)
/// ```
///
/// ## Kizzasi (Rust)
///
/// ```rust
/// let mut predictor = KizzasiBuilder::new()
///     .model_type(ModelType::Mamba)
///     .hidden_dim(64).state_dim(16).num_layers(4)
///     .build()?;
///
/// for _ in 0..100 {
///     let x = array![/* ... d_model values */];
///     let y = predictor.step(&x)?;   // O(1) per step
/// }
/// ```
///
/// Key differences:
/// - Kizzasi stores state internally; no need to thread it through the call.
/// - `reset()` zeroes all hidden states (equivalent to starting a new sequence).
/// - `fork()` clones state for branched prediction without re-encoding context.
fn demo_inference_api() -> Result<()> {
    println!("--- 5. Inference API Migration ---");

    let mut predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba)
        .input_dim(8)
        .output_dim(8)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(2)
        .build()?;

    println!("  [PyTorch equivalent]");
    println!("    model = Mamba(d_model=64, d_state=16)");
    println!("    y, state = model(x, inference_params=state)  # stateful, external state");
    println!();
    println!("  [Kizzasi equivalent]");
    println!("    let y = predictor.step(&x)?;  // state managed internally");
    println!();

    // Single-step
    let x = array![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let y = predictor.step(&x)?;
    println!("  Single step:  input {:?}", x.as_slice().unwrap_or(&[]));
    println!("                output {:?}", y.as_slice().unwrap_or(&[]));
    println!();

    // Multi-step
    predictor.reset();
    let trajectory = predictor.predict_n(&x, 4)?;
    println!("  Multi-step (4 ahead, equivalent to a Python loop):");
    for (i, row) in trajectory.outer_iter().enumerate() {
        println!("    step {:>2}: {:?}", i + 1, row.as_slice().unwrap_or(&[]));
    }
    println!();

    // Fork — replaces PyTorch's "save/restore state" pattern
    predictor.reset();
    for _ in 0..10 {
        let _ = predictor.step(&x)?;
    }
    let mut branch = predictor.fork()?;
    let out_main = predictor.step(&array![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])?;
    let out_branch = branch.step(&array![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])?;
    println!("  Fork (replaces torch.clone(state)):");
    println!(
        "    main   branch output: {:?}",
        out_main.as_slice().unwrap_or(&[])
    );
    println!(
        "    forked branch output: {:?}",
        out_branch.as_slice().unwrap_or(&[])
    );

    // Demonstrate WeightLoader API surface (no actual file needed for docs)
    println!();
    println!("  [WeightLoader API reference — requires a real .safetensors file]");
    println!("    let loader = ModelLoader::new(\"model.safetensors\")?;");
    println!("    let wl = WeightLoader::new(loader).model_type(ModelType::Mamba).strict(false);");
    println!("    wl.print_weights();                    // inspect tensor names + shapes");
    println!("    let mapping = wl.suggest_huggingface_mapping();  // auto-detect HF keys");
    println!("    let wl2 = wl.with_name_mapping(my_map);         // custom key remapping");
    println!();

    // Show WeightLoader API is importable (this will type-check even without a file)
    let _: fn(kizzasi_model::loader::ModelLoader) -> WeightLoader = WeightLoader::new;

    println!("  [migration checklist]");
    let steps = [
        "Export PyTorch weights to SafeTensors: model.save_pretrained(path, safe_serialization=True)",
        "Run NameRemapper::new().remap_map(weights) to translate key names",
        "Call model.load_weights_from_dict(&translated) or load_weights_json",
        "Replace model(x) with predictor.step(&x)?",
        "Replace model.hidden_states = None with predictor.reset()",
        "Replace state cloning with predictor.fork()",
    ];
    for (i, step) in steps.iter().enumerate() {
        println!("    {}. {}", i + 1, step);
    }

    Ok(())
}
