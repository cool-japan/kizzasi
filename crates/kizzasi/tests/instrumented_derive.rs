//! Integration tests for the `#[derive(Instrumented)]` macro.
//!
//! Must live in the `kizzasi` crate (not kizzasi-macros) because the macro
//! generates `impl kizzasi::telemetry::Instrumented for ...`, requiring `kizzasi`
//! as a dependency — which kizzasi-macros cannot have (circular dep).
use kizzasi::telemetry::{Instrumented, MetricsCollector};
use std::sync::Arc;

// Using a non-`collector` field name to prove the #[metrics] annotation works
#[derive(kizzasi_macros::Instrumented)]
struct AnnotatedPredictor {
    #[metrics]
    coll: Arc<MetricsCollector>,
    _data: u32,
}

// Using the name-based fallback (field literally named `collector`)
#[derive(kizzasi_macros::Instrumented)]
struct FallbackPredictor {
    collector: Arc<MetricsCollector>,
    _other: String,
}

#[test]
fn test_instrumented_annotated_field() {
    let mc = Arc::new(MetricsCollector::new("annotated-test"));
    let p = AnnotatedPredictor {
        coll: mc.clone(),
        _data: 42,
    };
    // `metrics()` returns the Arc — it must be the same pointer
    let returned: Arc<MetricsCollector> = p.metrics();
    assert!(
        Arc::ptr_eq(&mc, &returned),
        "metrics() must return the #[metrics]-annotated field"
    );
}

#[test]
fn test_instrumented_collector_fallback() {
    let mc = Arc::new(MetricsCollector::new("fallback-test"));
    let p = FallbackPredictor {
        collector: mc.clone(),
        _other: "hello".into(),
    };
    let returned = p.metrics();
    assert!(
        Arc::ptr_eq(&mc, &returned),
        "metrics() must return the `collector` field"
    );
}
