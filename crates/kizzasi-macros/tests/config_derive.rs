use kizzasi_macros::KizzasiConfig;

fn validate_positive(d: &usize) -> Result<(), String> {
    if *d > 0 {
        Ok(())
    } else {
        Err("must be positive".into())
    }
}

fn validate_divisible_by_64(d: &usize) -> Result<(), String> {
    if (*d).is_multiple_of(64) {
        Ok(())
    } else {
        Err("must be divisible by 64".into())
    }
}

#[derive(KizzasiConfig, Debug)]
struct AllRequired {
    a: usize,
    b: String,
}

#[derive(KizzasiConfig, Debug)]
struct WithDefaults {
    #[config(default = 4096)]
    context_window: usize,
    #[config(default = "256_usize")]
    hidden_dim: usize,
    learning_rate: f64,
}

#[derive(KizzasiConfig, Debug)]
#[allow(dead_code)]
struct WithValidate {
    #[config(validate = "validate_positive")]
    count: usize,
    name: String,
}

#[derive(KizzasiConfig, Debug)]
#[allow(dead_code)]
struct WithSkip {
    name: String,
    #[config(skip)]
    cache: Vec<u8>,
    #[config(skip, default = 7_usize)]
    retries: usize,
}

#[derive(KizzasiConfig, Debug)]
struct WithValidateAndDefault {
    #[config(default = 64, validate = "validate_divisible_by_64")]
    dim: usize,
}

#[test]
fn test_all_required_ok() {
    let c = AllRequired::builder()
        .a(1)
        .b("x".into())
        .build()
        .expect("should build");
    assert_eq!(c.a, 1);
    assert_eq!(c.b, "x");
}

#[test]
fn test_all_required_missing_field_err() {
    let result = AllRequired::builder().a(1).build();
    assert!(result.is_err());
    let err = result.unwrap_err();
    // Regression for id188: `build()` now returns a typed error rather than
    // a bare `String`, so callers can match on it programmatically.
    assert_eq!(err, AllRequiredBuilderError::MissingField("b"));
    assert!(
        err.to_string().contains('b'),
        "error should mention the missing field"
    );
}

#[test]
fn test_default_used_when_unset() {
    let c = WithDefaults::builder()
        .learning_rate(1e-3)
        .build()
        .expect("should build");
    assert_eq!(c.context_window, 4096);
    assert_eq!(c.hidden_dim, 256);
    assert!((c.learning_rate - 1e-3).abs() < 1e-12);
}

#[test]
fn test_default_overridden_when_set() {
    let c = WithDefaults::builder()
        .context_window(8192)
        .hidden_dim(512)
        .learning_rate(1e-3)
        .build()
        .expect("should build");
    assert_eq!(c.context_window, 8192);
    assert_eq!(c.hidden_dim, 512);
}

#[test]
fn test_default_required_still_required() {
    let result = WithDefaults::builder().build();
    assert!(result.is_err(), "learning_rate is still required");
}

#[test]
fn test_validate_passes() {
    let c = WithValidate::builder()
        .count(5)
        .name("ok".into())
        .build()
        .expect("should build");
    assert_eq!(c.count, 5);
}

#[test]
fn test_validate_fails() {
    let err = WithValidate::builder()
        .count(0)
        .name("bad".into())
        .build()
        .unwrap_err();
    // Regression for id188: message now comes through `Display` on the
    // typed `WithValidateBuilderError`, and includes the field name.
    assert_eq!(
        err.to_string(),
        "validation failed for field `count`: must be positive"
    );
}

#[test]
fn test_skip_excluded_from_builder_filled_by_default() {
    // `cache` and `retries` are not setter methods — just don't call them
    let c = WithSkip::builder()
        .name("x".into())
        .build()
        .expect("should build");
    assert!(
        c.cache.is_empty(),
        "skip without default => Default::default()"
    );
    assert_eq!(c.retries, 7, "skip with default => default expr");
}

#[test]
fn test_default_and_validate_compose() {
    // Default 64 passes validate_divisible_by_64
    let c = WithValidateAndDefault::builder()
        .build()
        .expect("should build with default 64");
    assert_eq!(c.dim, 64);
    // Override with invalid value -> Err
    let err = WithValidateAndDefault::builder()
        .dim(100)
        .build()
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "validation failed for field `dim`: must be divisible by 64"
    );
    // Override with valid value -> Ok
    let c = WithValidateAndDefault::builder()
        .dim(128)
        .build()
        .expect("128 is divisible by 64");
    assert_eq!(c.dim, 128);
}

// ---------------------------------------------------------------------
// id188: typed builder error composes with `?` in a `Result<_, String>`
// function via the generated `impl From<BuilderError> for String`.
// ---------------------------------------------------------------------

fn build_all_required(a: Option<usize>) -> Result<AllRequired, String> {
    let mut builder = AllRequired::builder().b("x".into());
    if let Some(a) = a {
        builder = builder.a(a);
    }
    let config = builder.build()?;
    Ok(config)
}

#[test]
fn test_builder_error_converts_to_string_via_from() {
    assert!(build_all_required(Some(1)).is_ok());
    let err = build_all_required(None).unwrap_err();
    assert!(err.contains('a'));
}

// ---------------------------------------------------------------------
// id179: `Option<T>` fields are implicitly optional in the builder.
// ---------------------------------------------------------------------

#[derive(KizzasiConfig, Debug)]
struct WithOptionField {
    name: String,
    timeout: Option<u64>,
}

#[test]
fn test_option_field_defaults_to_none_when_unset() {
    let c = WithOptionField::builder()
        .name("x".into())
        .build()
        .expect("should build even without setting `timeout`");
    assert_eq!(c.name, "x");
    assert_eq!(c.timeout, None);
}

#[test]
fn test_option_field_setter_takes_unwrapped_value() {
    // Note: `.timeout(30)`, not the old, unnatural `.timeout(Some(30))`.
    let c = WithOptionField::builder()
        .name("x".into())
        .timeout(30)
        .build()
        .expect("should build");
    assert_eq!(c.name, "x");
    assert_eq!(c.timeout, Some(30));
}

// ---------------------------------------------------------------------
// id185: generic structs (lifetimes and type parameters) are supported.
// ---------------------------------------------------------------------

#[derive(KizzasiConfig, Debug)]
struct WithLifetime<'a> {
    name: &'a str,
    #[config(default = 10)]
    retries: usize,
}

#[test]
fn test_generic_lifetime_config_builds() {
    let c = WithLifetime::builder()
        .name("hello")
        .build()
        .expect("should build");
    assert_eq!(c.name, "hello");
    assert_eq!(c.retries, 10);
}

/// Deliberately does not implement `Default`, to prove the generated
/// builder's hand-written `impl Default` (needed to keep working for
/// generic structs — see id185/id186) does not impose an unnecessary
/// `T: Default` bound the way a naive `#[derive(Default)]` would.
#[derive(Debug, PartialEq)]
struct NotDefault(i32);

#[derive(KizzasiConfig, Debug)]
struct WithTypeParam<T> {
    value: T,
}

#[test]
fn test_generic_type_param_without_default_builds() {
    let c = WithTypeParam::builder()
        .value(NotDefault(7))
        .build()
        .expect("should build");
    assert_eq!(c.value, NotDefault(7));
}

// ---------------------------------------------------------------------
// id176: the generated builder mirrors the config struct's own visibility
// (module-private here, same as every other struct in this file) rather
// than always being fully `pub`. This module boundary plus a successful
// build is the behavioral proof; exact codegen is unit-tested in
// `src/config.rs`.
// ---------------------------------------------------------------------

mod inner {
    use kizzasi_macros::KizzasiConfig;

    #[derive(KizzasiConfig, Debug)]
    pub(crate) struct InnerConfig {
        pub(crate) label: String,
    }
}

#[test]
fn test_pub_crate_visibility_config_builds_across_module() {
    let c = inner::InnerConfig::builder()
        .label("ok".into())
        .build()
        .expect("should build");
    assert_eq!(c.label, "ok");
}
