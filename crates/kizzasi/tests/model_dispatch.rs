//! Integration tests for real `ModelType` dispatch and weight persistence.
//!
//! These cover the two defects that the crate's own unit tests could not see:
//! every `ModelType` used to build the identical `SelectiveSSM`, and
//! `weights_path` was stored but never read.

use kizzasi::optimization::{OptimizationConfig, OptimizedPredictor};
use kizzasi::{Kizzasi, KizzasiBuilder, KizzasiConfig, KizzasiError, ModelType};
use scirs2_core::ndarray::Array1;

fn config_for(model_type: ModelType) -> KizzasiConfig {
    KizzasiConfig::new()
        .model_type(model_type)
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(2)
        .context_window(256)
}

fn run_sequence(predictor: &mut Kizzasi, steps: usize) -> Vec<Array1<f32>> {
    let mut outputs = Vec::with_capacity(steps);
    for i in 0..steps {
        let phase = i as f32 * 0.25;
        let input = Array1::from_vec(vec![phase.sin(), phase.cos(), 0.1, -0.2]);
        outputs.push(predictor.step(&input).expect("step must succeed"));
    }
    outputs
}

#[test]
fn every_model_type_builds_and_runs() {
    for model_type in [
        ModelType::Mamba,
        ModelType::Mamba2,
        ModelType::S4,
        ModelType::Rwkv,
    ] {
        let mut predictor = Kizzasi::new(config_for(model_type))
            .unwrap_or_else(|e| panic!("{model_type:?} failed to build: {e}"));
        assert_eq!(predictor.model_type(), model_type);

        let outputs = run_sequence(&mut predictor, 6);
        assert_eq!(outputs.len(), 6);
        for output in &outputs {
            assert_eq!(output.len(), 4);
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn different_model_types_compute_different_functions() {
    // The load-bearing assertion: before real dispatch existed, all four
    // ModelType values produced a bit-identical SelectiveSSM, so architecture
    // selection had no observable effect at all.
    let mut engines = Vec::new();
    for model_type in [
        ModelType::Mamba,
        ModelType::Mamba2,
        ModelType::S4,
        ModelType::Rwkv,
    ] {
        let mut predictor = Kizzasi::new(config_for(model_type)).expect("build");
        let outputs = run_sequence(&mut predictor, 8);
        engines.push((model_type, outputs, predictor.ssm().is_some()));
    }

    // Exactly one variant is served by the SelectiveSSM engine.
    let selective_count = engines.iter().filter(|(_, _, is_ssm)| *is_ssm).count();
    assert_eq!(
        selective_count, 1,
        "only ModelType::Mamba2 may use the SelectiveSSM engine"
    );

    // Every pair of architectures must differ somewhere in the trajectory.
    for i in 0..engines.len() {
        for j in (i + 1)..engines.len() {
            let (type_a, outputs_a, _) = &engines[i];
            let (type_b, outputs_b, _) = &engines[j];
            let identical = outputs_a
                .iter()
                .zip(outputs_b.iter())
                .all(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9));
            assert!(
                !identical,
                "{type_a:?} and {type_b:?} produced identical trajectories"
            );
        }
    }
}

#[test]
fn weights_round_trip_for_every_model_type() {
    for model_type in [
        ModelType::Mamba,
        ModelType::Mamba2,
        ModelType::S4,
        ModelType::Rwkv,
    ] {
        let config = config_for(model_type);
        let mut trained = Kizzasi::new(config.clone()).expect("build");

        // Both models are compared from their initial state; stepping does
        // not change the weights, so this isolates weight persistence from
        // any per-architecture differences in what `reset()` clears.
        let reference = run_sequence(&mut trained, 5);

        let path = std::env::temp_dir().join(format!("kizzasi_weights_{model_type:?}.json"));
        trained.save_weights(&path).expect("save_weights");

        let mut restored = Kizzasi::new(config.load_weights(&path.to_string_lossy()))
            .unwrap_or_else(|e| panic!("{model_type:?} weights_path load failed: {e}"));
        let replayed = run_sequence(&mut restored, 5);

        for (step, (expected, actual)) in reference.iter().zip(replayed.iter()).enumerate() {
            for (a, b) in expected.iter().zip(actual.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "{model_type:?} step {step}: weights_path did not restore the model \
                     ({b} vs {a})"
                );
            }
        }

        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn weights_round_trip_with_asymmetric_dims() {
    // The facade-owned readout projection must be persisted too, otherwise a
    // 12-in / 6-out model reloads with a fresh random readout.
    let config = KizzasiConfig::new()
        .model_type(ModelType::Rwkv)
        .input_dim(12)
        .output_dim(6)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(1);

    let mut trained = Kizzasi::new(config.clone()).expect("build");
    let input = Array1::from_vec((0..12).map(|i| i as f32 * 0.05).collect::<Vec<_>>());
    let expected = trained.step(&input).expect("step");

    let path = std::env::temp_dir().join("kizzasi_weights_asymmetric.json");
    trained.save_weights(&path).expect("save_weights");

    let mut restored =
        Kizzasi::new(config.load_weights(&path.to_string_lossy())).expect("weights_path load");
    let actual = restored.step(&input).expect("step");

    assert_eq!(actual.len(), 6);
    for (a, b) in expected.iter().zip(actual.iter()) {
        assert!((a - b).abs() < 1e-5, "readout projection was not restored");
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn weights_path_pointing_at_the_wrong_model_type_errors() {
    let saved_config = config_for(ModelType::Rwkv);
    let predictor = Kizzasi::new(saved_config.clone()).expect("build");
    let path = std::env::temp_dir().join("kizzasi_weights_wrong_type.json");
    predictor.save_weights(&path).expect("save_weights");

    let mismatched = config_for(ModelType::S4).load_weights(&path.to_string_lossy());
    assert!(
        matches!(
            Kizzasi::new(mismatched),
            Err(KizzasiError::ModelNotReady { .. })
        ),
        "loading Rwkv weights into an S4 model must fail loudly"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn weights_path_pointing_at_garbage_errors() {
    let path = std::env::temp_dir().join("kizzasi_weights_garbage.json");
    std::fs::write(&path, "{\"not\": \"a weight file\"}").expect("write");

    let result = KizzasiBuilder::new()
        .input_dim(2)
        .output_dim(2)
        .hidden_dim(16)
        .weights_path(&path.to_string_lossy())
        .build();

    assert!(
        result.is_err(),
        "a file that matches nothing must not silently leave the model random"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn presets_still_build_and_run() {
    let mut audio = KizzasiBuilder::audio_preset().build().expect("audio");
    assert_eq!(audio.step(&Array1::from_vec(vec![0.5])).unwrap().len(), 1);

    let mut robotics = KizzasiBuilder::robotics_preset(6)
        .build()
        .expect("robotics");
    assert_eq!(
        robotics
            .step(&Array1::from_vec(vec![0.1; 6]))
            .unwrap()
            .len(),
        6
    );

    let mut sensor = KizzasiBuilder::sensor_preset(10).build().expect("sensor");
    assert_eq!(
        sensor.step(&Array1::from_vec(vec![0.2; 10])).unwrap().len(),
        10
    );

    let mut lightweight = KizzasiBuilder::lightweight_preset(3, 3)
        .build()
        .expect("lightweight");
    assert_eq!(
        lightweight
            .step(&Array1::from_vec(vec![0.3; 3]))
            .unwrap()
            .len(),
        3
    );
    // Every preset must remain forkable and checkpointable.
    assert!(lightweight.fork().is_ok());

    let mut control = KizzasiBuilder::control_preset(8, 4)
        .build()
        .expect("control");
    assert_eq!(
        control.step(&Array1::from_vec(vec![0.4; 8])).unwrap().len(),
        4
    );

    let mut video = KizzasiBuilder::video_preset(16).build().expect("video");
    assert_eq!(
        video.step(&Array1::from_vec(vec![0.1; 16])).unwrap().len(),
        16
    );

    let mut custom = KizzasiBuilder::custom_preset()
        .input_dim(5)
        .output_dim(5)
        .build()
        .expect("custom");
    assert_eq!(
        custom.step(&Array1::from_vec(vec![0.6; 5])).unwrap().len(),
        5
    );
}

#[test]
fn hot_swap_actually_changes_the_engine() {
    let mut predictor = Kizzasi::new(config_for(ModelType::Mamba2)).expect("build");
    assert!(predictor.ssm().is_some());

    predictor
        .hot_swap(config_for(ModelType::Rwkv), false)
        .expect("hot swap");

    assert_eq!(predictor.model_type(), ModelType::Rwkv);
    assert!(
        predictor.ssm().is_none(),
        "hot_swap must build the newly selected architecture"
    );
    assert_eq!(
        predictor
            .step(&Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]))
            .unwrap()
            .len(),
        4
    );
}

#[test]
fn stateless_evaluation_preserves_step_count_on_architecture_backends() {
    // Regression: the architecture backends track `step_count` in the facade,
    // not inside `HiddenState`, so a snapshot that carried only the layer
    // states let `predict_stateless` (snapshot -> reset -> step -> restore)
    // reset the caller's counter to 1.
    for model_type in [ModelType::Mamba, ModelType::S4, ModelType::Rwkv] {
        let predictor = Kizzasi::new(config_for(model_type)).expect("build");
        let mut optimized = OptimizedPredictor::new(predictor, OptimizationConfig::default());

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        for _ in 0..3 {
            optimized.step(&input).expect("step");
        }
        assert_eq!(optimized.inner().step_count(), 3);

        optimized.predict_stateless(&input).expect("stateless");

        assert_eq!(
            optimized.inner().step_count(),
            3,
            "{model_type:?}: stateless evaluation must not disturb the live counter"
        );
    }
}

#[test]
fn weights_from_a_different_head_split_are_rejected() {
    // Every admissible (num_heads, head_dim) split of the same hidden_dim
    // produces identical per-tensor element counts, so a shape check alone
    // would happily load an 8x4 checkpoint into a 4x8 layout.
    let base = KizzasiConfig::new()
        .model_type(ModelType::Rwkv)
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(1);

    // Saved with the derived split (8 heads x 4).
    let saved = Kizzasi::new(base.clone()).expect("build");
    let path = std::env::temp_dir().join("kizzasi_weights_head_split.json");
    saved.save_weights(&path).expect("save_weights");

    // Loaded into an explicitly different split (4 heads x 8).
    let reshaped = base
        .clone()
        .num_heads(4)
        .head_dim(8)
        .load_weights(&path.to_string_lossy());

    assert!(
        Kizzasi::new(reshaped).is_err(),
        "a checkpoint from a different head split must not load silently"
    );

    // The matching split still loads.
    assert!(Kizzasi::new(base.load_weights(&path.to_string_lossy())).is_ok());

    let _ = std::fs::remove_file(&path);
}
