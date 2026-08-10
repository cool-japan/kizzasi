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
    let msg = result.unwrap_err();
    assert!(msg.contains('b'), "error should mention the missing field");
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
    assert_eq!(err, "must be positive");
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
    assert_eq!(err, "must be divisible by 64");
    // Override with valid value -> Ok
    let c = WithValidateAndDefault::builder()
        .dim(128)
        .build()
        .expect("128 is divisible by 64");
    assert_eq!(c.dim, 128);
}
