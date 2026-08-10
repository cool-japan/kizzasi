use kizzasi_macros::Preset;

// clippy::duplicated_attributes fires a false positive here because it sees
// `context_window` and `hidden_dim` appearing in both `#[preset(...)]` attrs.
// These are intentional: each preset independently specifies all its field overrides.
#[allow(clippy::duplicated_attributes)]
#[derive(Preset, Debug, PartialEq, Default)]
#[preset(name = "audio", context_window = 8192_usize, hidden_dim = 256_usize)]
#[preset(name = "video", context_window = 16384_usize, hidden_dim = 512_usize)]
struct ModelConfig {
    context_window: usize,
    hidden_dim: usize,
    dropout: f64,
}

#[derive(Preset, Default, Debug, PartialEq)]
#[preset(name = "full", x = 1_usize, y = 2_usize)]
struct FullCoverage {
    x: usize,
    y: usize,
}

#[test]
fn test_audio_preset_sets_fields_rest_default() {
    let c = ModelConfig::audio_preset();
    assert_eq!(c.context_window, 8192);
    assert_eq!(c.hidden_dim, 256);
    assert_eq!(c.dropout, 0.0); // from Default
}

#[test]
fn test_video_preset_sets_fields() {
    let c = ModelConfig::video_preset();
    assert_eq!(c.context_window, 16384);
    assert_eq!(c.hidden_dim, 512);
}

#[test]
fn test_multiple_presets_coexist_and_differ() {
    assert_ne!(ModelConfig::audio_preset(), ModelConfig::video_preset());
}

#[test]
fn test_full_coverage_preset_no_struct_update() {
    // All fields covered — struct update syntax must not be emitted
    let c = FullCoverage::full_preset();
    assert_eq!(c.x, 1);
    assert_eq!(c.y, 2);
}
