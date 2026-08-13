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

// ---------------------------------------------------------------------
// id185: generic structs (lifetimes and type parameters) are supported.
// ---------------------------------------------------------------------

#[derive(Preset, Debug, PartialEq)]
#[preset(name = "named", label = "x")]
struct WithLifetime<'a> {
    label: &'a str,
}

#[test]
fn test_lifetime_generic_preset() {
    // Full coverage (the only field is set), so no `Default` bound needed —
    // `WithLifetime` deliberately does not derive `Default`.
    let c = WithLifetime::named_preset();
    assert_eq!(c.label, "x");
}

/// Deliberately does not implement `Default`, to prove a full-coverage
/// preset does not force an unnecessary `Self: Default` bound (the
/// type-parameter case for this same point is exercised at the codegen
/// level by `full_coverage_preset_on_generic_struct_needs_no_default_bound`
/// in `src/preset.rs`, since a preset field's value expression cannot itself
/// be generic over an unconstrained `T`).
#[derive(Debug, PartialEq)]
struct NotDefaultValue(i32);

#[derive(Preset, Debug, PartialEq)]
#[preset(name = "full", value = NotDefaultValue(5))]
struct FullCoverageNoDefault {
    value: NotDefaultValue,
}

#[test]
fn test_full_coverage_preset_does_not_require_default() {
    // The point of this test is that it compiles at all: a full-coverage
    // preset must not force a `Self: Default` bound the way an unconditional
    // one would — `FullCoverageNoDefault` never derives `Default`.
    let c = FullCoverageNoDefault::full_preset();
    assert_eq!(c.value, NotDefaultValue(5));
}

#[derive(Preset, Default, Debug, PartialEq)]
#[preset(name = "partial", value = 5_i32)]
struct WithTypeParamPartialCoverage<T> {
    value: i32,
    extra: T,
}

#[test]
fn test_type_param_partial_coverage_preset_uses_default_for_rest() {
    let c = WithTypeParamPartialCoverage::<u8>::partial_preset();
    assert_eq!(c.value, 5);
    assert_eq!(c.extra, 0u8);
}
