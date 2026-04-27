//! Integration tests for PyTorch `.pth` pickle loader.
//!
//! These tests exercise `PthIndex` parsing, `split_x_proj` correctness, and
//! error paths in `load_checkpoint_raw`.  They do not require network access
//! or actual PyTorch files.

use kizzasi_model::pytorch_compat::PyTorchConverter;
use scirs2_core::ndarray::{concatenate, Axis, IxDyn};

// ---------------------------------------------------------------------------
// PthIndex JSON parsing
// ---------------------------------------------------------------------------

/// Exercise the JSON parsing and shard-grouping logic that would be used when
/// reading a sharded `pytorch_model.bin.index.json`.  The struct is private, so
/// we test it indirectly through the converter — but since `PthIndex` is only
/// used inside the crate, the real assertion is that `load_pth_sharded` would
/// return the correct error when the index points to non-existent shard files.
///
/// We test the JSON format directly by calling `load_pth_sharded` on a temp
/// directory that only contains the index file (shards are absent → error).
#[test]
fn test_pth_index_load_error_on_missing_shard() {
    let tmp = std::env::temp_dir().join("kizzasi_pth_index_test");
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    let index_json = r#"{
        "weight_map": {
            "backbone.layers.0.mixer.in_proj.weight": "pytorch_model-00001-of-00002.bin",
            "backbone.layers.0.mixer.out_proj.weight": "pytorch_model-00001-of-00002.bin",
            "backbone.layers.1.mixer.in_proj.weight": "pytorch_model-00002-of-00002.bin"
        },
        "metadata": {"total_size": 1000}
    }"#;
    let index_path = tmp.join("pytorch_model.bin.index.json");
    std::fs::write(&index_path, index_json).expect("write index");

    let converter = PyTorchConverter::new();
    let result = converter.load_pth_sharded(&tmp);

    // The shard files don't exist, so we expect an error.
    assert!(
        result.is_err(),
        "expected Err when shard files are absent, got Ok"
    );

    // Clean up.
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// split_x_proj
// ---------------------------------------------------------------------------

#[test]
fn test_split_x_proj_shapes() {
    let converter = PyTorchConverter::new();
    let dt_rank = 4usize;
    let d_state = 8usize;
    let inner_dim = 16usize;

    let rows = dt_rank + 2 * d_state;
    let data: Vec<f32> = (0..rows * inner_dim).map(|i| i as f32).collect();
    let fused = scirs2_core::ndarray::ArrayD::from_shape_vec(IxDyn(&[rows, inner_dim]), data)
        .expect("build fused tensor");

    let (dt, b, c) = converter
        .split_x_proj(fused, dt_rank, d_state)
        .expect("split_x_proj");

    assert_eq!(dt.shape(), &[dt_rank, inner_dim], "dt_proj shape mismatch");
    assert_eq!(b.shape(), &[d_state, inner_dim], "b_proj shape mismatch");
    assert_eq!(c.shape(), &[d_state, inner_dim], "c_proj shape mismatch");
}

#[test]
fn test_split_x_proj_roundtrip() {
    let converter = PyTorchConverter::new();
    let dt_rank = 4usize;
    let d_state = 8usize;
    let inner_dim = 16usize;

    let rows = dt_rank + 2 * d_state;
    let data: Vec<f32> = (0..rows * inner_dim).map(|i| i as f32).collect();
    let fused =
        scirs2_core::ndarray::ArrayD::from_shape_vec(IxDyn(&[rows, inner_dim]), data.clone())
            .expect("build fused");

    // Keep a 2-D reference of the original for comparison.
    let orig_2d = scirs2_core::ndarray::ArrayD::from_shape_vec(IxDyn(&[rows, inner_dim]), data)
        .expect("build orig")
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .expect("cast orig to Ix2");

    let (dt, b, c) = converter
        .split_x_proj(fused, dt_rank, d_state)
        .expect("split_x_proj");

    let recovered = concatenate(Axis(0), &[dt.view(), b.view(), c.view()]).expect("concatenate");

    assert_eq!(
        recovered, orig_2d,
        "concatenated slices do not match original"
    );
}

#[test]
fn test_split_x_proj_wrong_rank() {
    let converter = PyTorchConverter::new();
    // 1-D tensor: should fail with a descriptive error.
    let bad = scirs2_core::ndarray::ArrayD::from_shape_vec(IxDyn(&[20]), vec![0f32; 20])
        .expect("build 1D");
    let result = converter.split_x_proj(bad, 4, 8);
    assert!(result.is_err(), "expected Err for 1-D input");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("2-D") || msg.contains("2D") || msg.contains("rank"),
        "error message should mention rank: {msg}"
    );
}

#[test]
fn test_split_x_proj_wrong_row_count() {
    let converter = PyTorchConverter::new();
    // Rows don't match dt_rank + 2*d_state.
    let bad = scirs2_core::ndarray::ArrayD::from_shape_vec(IxDyn(&[5, 16]), vec![0f32; 80])
        .expect("build 2D with wrong rows");
    let result = converter.split_x_proj(bad, 4, 8); // expects 4+16=20 rows, got 5
    assert!(result.is_err(), "expected Err for wrong row count");
}

// ---------------------------------------------------------------------------
// load_checkpoint_raw error path
// ---------------------------------------------------------------------------

#[test]
fn test_load_checkpoint_raw_nonexistent_path() {
    let converter = PyTorchConverter::new();
    let result = converter.load_checkpoint_raw("/nonexistent/path/model.pth");
    assert!(result.is_err(), "expected Err for missing file");
    let msg = format!("{}", result.unwrap_err());
    // The error should mention the path or a filesystem error.
    assert!(
        msg.contains("nonexistent")
            || msg.contains("No such file")
            || msg.contains("path")
            || msg.contains("failed to read"),
        "error message should reference the path: {msg}"
    );
}
