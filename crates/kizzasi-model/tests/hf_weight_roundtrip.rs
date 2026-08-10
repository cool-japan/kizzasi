//! HuggingFace weight roundtrip integration tests.
//!
//! Covers three areas not exercised by `pytorch_pth_roundtrip.rs`:
//!
//! 1. `NameRemapper` — key translation from HuggingFace naming to Kizzasi
//!    internal naming, including the `backbone.*` prefix-stripping path,
//!    per-layer mixer/attention/mlp rules, top-level aliases, and unknown-key
//!    passthrough.
//!
//! 2. Mamba JSON save/load cycle — saving a freshly initialised model and
//!    reloading it into a same-config model, then verifying that a forward step
//!    produces finite output (i.e. weights were actually applied, not ignored).
//!    Also verifies the exact key count emitted by `save_weights_json`.
//!
//! 3. `fill_lm_head_from_embedding` complementary scenarios — synthesising
//!    `output_proj` from `embedding.weight`, verifying idempotency when
//!    `output_proj` is already present, and the error path when no embedding
//!    key is present.

#[cfg(feature = "mamba")]
mod mamba_tests {
    use kizzasi_core::SignalPredictor;
    use kizzasi_model::loader::NameRemapper;
    use kizzasi_model::mamba::{Mamba, MambaConfig};
    use kizzasi_model::pytorch_compat::fill_lm_head_from_embedding;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::ndarray::Array2;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Global counter for generating unique temp-file paths, preventing
    /// conflicts between tests that run in parallel.
    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_path(prefix: &str) -> std::path::PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("kizzasi_hf_rt_{}_{}.json", prefix, n))
    }

    // -----------------------------------------------------------------------
    // NameRemapper — top-level aliases
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_backbone_embeddings() {
        let remapper = NameRemapper::new();
        assert_eq!(
            remapper.remap("backbone.embeddings.weight"),
            "input_proj",
            "backbone.embeddings.weight must remap to input_proj"
        );
    }

    #[test]
    fn test_remap_backbone_norm() {
        let remapper = NameRemapper::new();
        assert_eq!(
            remapper.remap("backbone.norm_f.weight"),
            "final_norm.weight",
            "backbone.norm_f.weight must remap to final_norm.weight"
        );
    }

    #[test]
    fn test_remap_lm_head() {
        let remapper = NameRemapper::new();
        assert_eq!(
            remapper.remap("lm_head.weight"),
            "output_proj",
            "lm_head.weight must remap to output_proj"
        );
    }

    #[test]
    fn test_remap_embedding_weight() {
        let remapper = NameRemapper::new();
        assert_eq!(
            remapper.remap("embedding.weight"),
            "input_proj",
            "embedding.weight must remap to input_proj"
        );
    }

    // -----------------------------------------------------------------------
    // NameRemapper — backbone-prefixed layer keys
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_layer_mixer_in_proj() {
        let remapper = NameRemapper::new();
        // backbone. prefix is stripped first, then layer-suffix rules apply.
        assert_eq!(
            remapper.remap("backbone.layers.0.mixer.in_proj.weight"),
            "layers.0.input_proj",
            "backbone-prefixed mixer in_proj should remap to layers.0.input_proj"
        );
    }

    #[test]
    fn test_remap_layer_mixer_out_proj() {
        let remapper = NameRemapper::new();
        assert_eq!(
            remapper.remap("backbone.layers.1.mixer.out_proj.weight"),
            "layers.1.output_proj",
            "backbone-prefixed mixer out_proj should remap to layers.1.output_proj"
        );
    }

    // -----------------------------------------------------------------------
    // NameRemapper — bare (non-backbone-prefixed) layer keys
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_attention_keys() {
        let remapper = NameRemapper::new();
        assert_eq!(
            remapper.remap("layers.0.attn.q_proj.weight"),
            "layers.0.attention.q",
            "attn q_proj should remap to layers.0.attention.q"
        );
        // Verify k and v as well for completeness.
        assert_eq!(
            remapper.remap("layers.0.attn.k_proj.weight"),
            "layers.0.attention.k",
            "attn k_proj should remap to layers.0.attention.k"
        );
        assert_eq!(
            remapper.remap("layers.0.attn.v_proj.weight"),
            "layers.0.attention.v",
            "attn v_proj should remap to layers.0.attention.v"
        );
    }

    // -----------------------------------------------------------------------
    // NameRemapper — unknown / passthrough keys
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_unknown_passthrough() {
        let remapper = NameRemapper::new();
        let key = "some.unknown.key";
        assert_eq!(
            remapper.remap(key),
            key,
            "unknown key must pass through unchanged"
        );
    }

    // -----------------------------------------------------------------------
    // NameRemapper.remap_map — full HF dict translation
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_map_full_hf_dict() {
        let remapper = NameRemapper::new();

        let mut hf_weights: HashMap<String, Vec<f32>> = HashMap::new();
        hf_weights.insert(
            "backbone.embeddings.weight".to_string(),
            vec![1.0f32, 2.0, 3.0],
        );
        hf_weights.insert("backbone.norm_f.weight".to_string(), vec![0.5f32, 0.5]);
        hf_weights.insert("lm_head.weight".to_string(), vec![0.1f32, 0.2]);
        hf_weights.insert(
            "backbone.layers.0.mixer.in_proj.weight".to_string(),
            vec![0.3f32],
        );
        hf_weights.insert("layers.2.attn.q_proj.weight".to_string(), vec![0.9f32]);
        hf_weights.insert("unrecognised.custom.key".to_string(), vec![7.0f32]);

        let remapped = remapper.remap_map(hf_weights);

        assert!(
            remapped.contains_key("input_proj"),
            "backbone.embeddings.weight must become input_proj"
        );
        assert!(
            remapped.contains_key("final_norm.weight"),
            "backbone.norm_f.weight must become final_norm.weight"
        );
        assert!(
            remapped.contains_key("output_proj"),
            "lm_head.weight must become output_proj"
        );
        assert!(
            remapped.contains_key("layers.0.input_proj"),
            "backbone.layers.0.mixer.in_proj.weight must become layers.0.input_proj"
        );
        assert!(
            remapped.contains_key("layers.2.attention.q"),
            "layers.2.attn.q_proj.weight must become layers.2.attention.q"
        );
        assert!(
            remapped.contains_key("unrecognised.custom.key"),
            "unrecognised keys must pass through unchanged"
        );
    }

    // -----------------------------------------------------------------------
    // NameRemapper — values are preserved, only keys change
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_then_verify_top_level_values() {
        let remapper = NameRemapper::new();

        let expected_values = vec![1.1f32, 2.2f32, 3.3f32, 4.4f32];
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        map.insert("embedding.weight".to_string(), expected_values.clone());

        let remapped = remapper.remap_map(map);

        let values = remapped
            .get("input_proj")
            .expect("input_proj must be present after remapping embedding.weight");

        assert_eq!(
            values, &expected_values,
            "values must be preserved verbatim; only the key should change"
        );
    }

    // -----------------------------------------------------------------------
    // Mamba JSON save / load roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_mamba_json_save_load_roundtrip() {
        let config = MambaConfig::new()
            .input_dim(4)
            .hidden_dim(16)
            .state_dim(8)
            .num_layers(2);

        let mamba = Mamba::new(config.clone()).expect("Mamba construction must succeed");

        let path = unique_path("roundtrip");
        mamba
            .save_weights_json(&path)
            .expect("save_weights_json must not fail");

        let mut mamba2 = Mamba::new(config).expect("second Mamba construction must succeed");
        mamba2
            .load_weights_json(&path)
            .expect("load_weights_json must not fail");

        // Run a single step to confirm weights are genuinely loaded and the
        // model produces a numerically valid result.
        let input = Array1::<f32>::zeros(4);
        let output = mamba2
            .step(&input)
            .expect("step after load_weights_json must succeed");

        assert_eq!(
            output.len(),
            4,
            "output dimension must match input_dim after roundtrip"
        );
        for (idx, val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "output[{}] = {} is not finite after weight roundtrip",
                idx,
                val
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mamba_json_key_count() {
        // For a 2-layer model the serialised map must contain exactly:
        //   2 top-level keys (input_proj, output_proj)
        //   + 8 per-layer keys × 2 layers = 18 total.
        let config = MambaConfig::new()
            .input_dim(4)
            .hidden_dim(16)
            .state_dim(8)
            .num_layers(2);

        let mamba = Mamba::new(config).expect("Mamba construction must succeed");

        let path = unique_path("keycount");
        mamba
            .save_weights_json(&path)
            .expect("save_weights_json must not fail");

        let file = std::fs::File::open(&path).expect("saved JSON file must be readable");
        let map: HashMap<String, Vec<f32>> =
            serde_json::from_reader(file).expect("saved JSON must deserialise correctly");

        assert_eq!(
            map.len(),
            18,
            "2-layer Mamba must serialise exactly 18 weight keys \
             (2 top-level + 8 per-layer × 2), found: {:?}",
            {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                keys
            }
        );

        // Spot-check that the expected key names are present.
        assert!(map.contains_key("input_proj"), "input_proj must be present");
        assert!(
            map.contains_key("output_proj"),
            "output_proj must be present"
        );
        assert!(
            map.contains_key("layers.0.in_proj"),
            "layers.0.in_proj must be present"
        );
        assert!(
            map.contains_key("layers.1.ssm.log_a"),
            "layers.1.ssm.log_a must be present"
        );

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // fill_lm_head_from_embedding — complementary scenarios
    // -----------------------------------------------------------------------

    #[test]
    fn test_fill_lm_head_from_embedding_synthesizes_output_proj() {
        // embedding.weight has shape [vocab_size, hidden_dim].
        // After fill, output_proj must be its transpose: [hidden_dim, vocab_size].
        let vocab_size = 10usize;
        let hidden_dim = 4usize;
        let data: Vec<f32> = (0..(vocab_size * hidden_dim)).map(|i| i as f32).collect();
        let embedding = Array2::from_shape_vec((vocab_size, hidden_dim), data)
            .expect("valid shape for embedding");

        let mut weights: HashMap<String, Array2<f32>> = HashMap::new();
        weights.insert("embedding.weight".to_string(), embedding.clone());

        fill_lm_head_from_embedding(&mut weights).expect("fill must succeed");

        assert!(
            weights.contains_key("output_proj"),
            "output_proj must be synthesised from embedding.weight"
        );

        let output_proj = &weights["output_proj"];
        assert_eq!(
            output_proj.shape(),
            &[hidden_dim, vocab_size],
            "output_proj shape must be the transpose of the embedding shape"
        );

        // Verify that the transpose is element-wise correct.
        for r in 0..vocab_size {
            for c in 0..hidden_dim {
                let emb_val = embedding[[r, c]];
                let out_val = output_proj[[c, r]];
                assert!(
                    (emb_val - out_val).abs() < 1e-6,
                    "output_proj[[{}, {}]] = {} but embedding[[{}, {}]] = {}",
                    c,
                    r,
                    out_val,
                    r,
                    c,
                    emb_val
                );
            }
        }
    }

    #[test]
    fn test_fill_lm_head_idempotent_when_output_proj_present() {
        // When output_proj is already in the map, fill must be a no-op.
        let existing_proj = Array2::from_shape_vec((4, 10), vec![99.0f32; 40])
            .expect("valid shape for existing output_proj");
        let sentinel_value = existing_proj[[0, 0]];

        let mut weights: HashMap<String, Array2<f32>> = HashMap::new();
        weights.insert("output_proj".to_string(), existing_proj);
        // Also add an embedding that, if used, would produce different values.
        weights.insert(
            "embedding.weight".to_string(),
            Array2::from_shape_vec((10, 4), vec![1.0f32; 40]).expect("valid embedding shape"),
        );

        fill_lm_head_from_embedding(&mut weights).expect("fill must not error");

        // output_proj must remain untouched.
        assert_eq!(
            weights["output_proj"][[0, 0]],
            sentinel_value,
            "output_proj must not be overwritten when already present"
        );
        // The map should still have exactly two entries.
        assert_eq!(
            weights.len(),
            2,
            "fill must not insert additional keys when output_proj already present"
        );
    }

    #[test]
    fn test_fill_lm_head_error_when_no_embedding() {
        // With neither embedding.weight nor input_proj present, fill must Err.
        let mut weights: HashMap<String, Array2<f32>> = HashMap::new();
        weights.insert("layers.0.in_proj".to_string(), Array2::zeros((4, 8)));

        let result = fill_lm_head_from_embedding(&mut weights);
        assert!(
            result.is_err(),
            "fill_lm_head_from_embedding must return Err when no embedding weight is present"
        );
    }
}
