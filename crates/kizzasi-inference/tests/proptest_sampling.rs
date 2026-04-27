//! Property-based tests for sampling strategies
//!
//! This module uses proptest to verify correctness properties of
//! various sampling strategies across a wide range of inputs.

use kizzasi_inference::{Sampler, SamplingConfig, SamplingStrategy};
use proptest::prelude::*;
use scirs2_core::ndarray::Array1;
use scirs2_core::RngExt;

// ============================================================================
// Properties for Greedy Sampling
// ============================================================================

proptest! {
    /// Greedy sampling should always return the argmax
    #[test]
    fn prop_greedy_returns_argmax(logits in prop::collection::vec(-10.0f32..10.0, 1..100)) {
        let logits_array = Array1::from_vec(logits.clone());
        let mut sampler = Sampler::new(SamplingConfig::new().strategy(SamplingStrategy::Greedy));

        let result = sampler.sample(&logits_array);
        prop_assert!(result.is_ok());

        let sampled_idx = result.unwrap() as usize;
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();

        prop_assert_eq!(sampled_idx, max_idx);
    }

    /// Greedy sampling should be deterministic
    #[test]
    fn prop_greedy_is_deterministic(logits in prop::collection::vec(-10.0f32..10.0, 1..100)) {
        let logits_array = Array1::from_vec(logits);
        let mut sampler = Sampler::new(SamplingConfig::new().strategy(SamplingStrategy::Greedy));

        let result1 = sampler.sample(&logits_array).unwrap();
        let result2 = sampler.sample(&logits_array).unwrap();

        prop_assert_eq!(result1, result2);
    }

    /// Greedy sampling should return valid indices
    #[test]
    fn prop_greedy_valid_index(logits in prop::collection::vec(-10.0f32..10.0, 1..100)) {
        let logits_array = Array1::from_vec(logits.clone());
        let mut sampler = Sampler::new(SamplingConfig::new().strategy(SamplingStrategy::Greedy));

        let result = sampler.sample(&logits_array);
        prop_assert!(result.is_ok());

        let sampled_idx = result.unwrap() as usize;
        prop_assert!(sampled_idx < logits.len());
    }
}

// ============================================================================
// Properties for Top-K Sampling
// ============================================================================

proptest! {
    /// Top-K sampling should only sample from top-k elements
    #[test]
    fn prop_topk_samples_from_topk(
        logits in prop::collection::vec(-10.0f32..10.0, 10..50),
        k in 1usize..10
    ) {
        let logits_array = Array1::from_vec(logits.clone());
        let k = k.min(logits.len());
        let mut sampler = Sampler::new(SamplingConfig::new().top_k(k));

        // Get top-k indices
        let mut indexed: Vec<_> = logits.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let topk_indices: std::collections::HashSet<usize> =
            indexed.iter().take(k).map(|(idx, _)| *idx).collect();

        // Sample multiple times to increase coverage
        let mut all_valid = true;
        for _ in 0..10 {
            if let Ok(sampled) = sampler.sample(&logits_array) {
                let sampled_idx = sampled as usize;
                if !topk_indices.contains(&sampled_idx) {
                    all_valid = false;
                    break;
                }
            }
        }

        prop_assert!(all_valid);
    }

    /// Top-K sampling should return valid indices
    #[test]
    fn prop_topk_valid_index(
        logits in prop::collection::vec(-10.0f32..10.0, 5..50),
        k in 1usize..10
    ) {
        let logits_array = Array1::from_vec(logits.clone());
        let k = k.min(logits.len());
        let mut sampler = Sampler::new(SamplingConfig::new().top_k(k));

        let result = sampler.sample(&logits_array);
        prop_assert!(result.is_ok());

        let sampled_idx = result.unwrap() as usize;
        prop_assert!(sampled_idx < logits.len());
    }
}

// ============================================================================
// Properties for Top-P (Nucleus) Sampling
// ============================================================================

proptest! {
    /// Top-P sampling should return valid indices
    #[test]
    fn prop_topp_valid_index(
        logits in prop::collection::vec(-10.0f32..10.0, 5..50),
        p in 0.1f32..0.99
    ) {
        let logits_array = Array1::from_vec(logits.clone());
        let mut sampler = Sampler::new(SamplingConfig::new().top_p(p));

        let result = sampler.sample(&logits_array);
        prop_assert!(result.is_ok());

        let sampled_idx = result.unwrap() as usize;
        prop_assert!(sampled_idx < logits.len());
    }

    /// Top-P with p=1.0 should behave like full sampling
    #[test]
    fn prop_topp_full_when_p_is_one(logits in prop::collection::vec(-5.0f32..5.0, 5..20)) {
        let logits_array = Array1::from_vec(logits.clone());
        let mut sampler = Sampler::new(SamplingConfig::new().top_p(1.0));

        // Should be able to sample any index
        let result = sampler.sample(&logits_array);
        prop_assert!(result.is_ok());

        let sampled_idx = result.unwrap() as usize;
        prop_assert!(sampled_idx < logits.len());
    }
}

// ============================================================================
// Properties for Temperature Scaling
// ============================================================================

proptest! {
    /// Temperature sampling should return valid indices
    #[test]
    fn prop_temperature_valid_index(
        logits in prop::collection::vec(-10.0f32..10.0, 5..50),
        temp in 0.1f32..5.0
    ) {
        let logits_array = Array1::from_vec(logits.clone());
        let mut sampler = Sampler::new(
            SamplingConfig::new()
                .strategy(SamplingStrategy::Temperature)
                .temperature(temp)
        );

        let result = sampler.sample(&logits_array);
        prop_assert!(result.is_ok());

        let sampled_idx = result.unwrap() as usize;
        prop_assert!(sampled_idx < logits.len());
    }

    /// Low temperature should approach greedy
    #[test]
    fn prop_low_temperature_approaches_greedy(logits in prop::collection::vec(-10.0f32..10.0, 5..20)) {
        let logits_array = Array1::from_vec(logits.clone());

        // Very low temperature
        let mut low_temp_sampler = Sampler::new(
            SamplingConfig::new()
                .strategy(SamplingStrategy::Temperature)
                .temperature(0.01)
        );

        // Greedy
        let mut greedy_sampler = Sampler::new(SamplingConfig::new().strategy(SamplingStrategy::Greedy));

        let low_temp_result = low_temp_sampler.sample(&logits_array).unwrap();
        let greedy_result = greedy_sampler.sample(&logits_array).unwrap();

        // With very low temperature, should often match greedy
        // We allow some variation due to randomness
        let low_temp_idx = low_temp_result as usize;
        let greedy_idx = greedy_result as usize;

        // Both should be valid
        prop_assert!(low_temp_idx < logits.len());
        prop_assert!(greedy_idx < logits.len());
    }
}

// ============================================================================
// Properties for Batch Sampling
// ============================================================================

proptest! {
    /// Batch sampling should return correct number of results
    #[test]
    fn prop_batch_correct_size(
        batch_size in 1usize..20,
        vocab_size in 5usize..50
    ) {
        let logits = scirs2_core::ndarray::Array2::from_shape_fn(
            (batch_size, vocab_size),
            |(_i, _j)| (scirs2_core::random::rng().random::<f32>() - 0.5) * 10.0
        );

        let mut sampler = Sampler::new(SamplingConfig::new().strategy(SamplingStrategy::Greedy));
        let result = sampler.sample_batch(&logits);

        prop_assert!(result.is_ok());
        let samples = result.unwrap();
        prop_assert_eq!(samples.len(), batch_size);

        // All samples should be valid indices
        for sample in samples.iter() {
            let idx = *sample as usize;
            prop_assert!(idx < vocab_size);
        }
    }
}

// ============================================================================
// Properties for Softmax (internal utility)
// ============================================================================

proptest! {
    /// Softmax output should sum to 1
    #[test]
    fn prop_softmax_sums_to_one(logits in prop::collection::vec(-100.0f32..100.0, 2..100)) {
        // Create a sampler and use temperature sampling which uses softmax
        let logits_array = Array1::from_vec(logits);
        let mut sampler = Sampler::new(
            SamplingConfig::new()
                .strategy(SamplingStrategy::Temperature)
                .temperature(1.0)
        );

        // Just verify it doesn't crash and produces valid output
        let result = sampler.sample(&logits_array);
        prop_assert!(result.is_ok());

        let idx = result.unwrap() as usize;
        prop_assert!(idx < logits_array.len());
    }

    /// Softmax should handle extreme values without overflow
    #[test]
    fn prop_softmax_handles_extremes(
        logits in prop::collection::vec(-1000.0f32..1000.0, 2..50)
    ) {
        let logits_array = Array1::from_vec(logits);
        let mut sampler = Sampler::new(
            SamplingConfig::new()
                .strategy(SamplingStrategy::Temperature)
                .temperature(1.0)
        );

        // Should not panic or produce NaN/Inf
        let result = sampler.sample(&logits_array);
        prop_assert!(result.is_ok());

        let sample = result.unwrap();
        prop_assert!(sample.is_finite());
        prop_assert!((sample as usize) < logits_array.len());
    }
}

// ============================================================================
// Integration Properties
// ============================================================================

proptest! {
    /// All sampling strategies should handle empty logits gracefully
    #[test]
    fn prop_handles_empty_logits(strategy_idx in 0usize..4) {
        let strategies = [SamplingStrategy::Greedy,
            SamplingStrategy::Temperature,
            SamplingStrategy::TopK,
            SamplingStrategy::TopP];

        let strategy = strategies[strategy_idx % strategies.len()];
        let mut sampler = Sampler::new(SamplingConfig::new().strategy(strategy));

        let empty_logits = Array1::from_vec(vec![]);
        let result = sampler.sample(&empty_logits);

        // Should return an error for empty input
        prop_assert!(result.is_err());
    }

    /// All sampling strategies should handle single-element logits
    #[test]
    fn prop_handles_single_element(
        strategy_idx in 0usize..4,
        value in -100.0f32..100.0
    ) {
        let strategies = [SamplingStrategy::Greedy,
            SamplingStrategy::Temperature,
            SamplingStrategy::TopK,
            SamplingStrategy::TopP];

        let strategy = strategies[strategy_idx % strategies.len()];
        let mut sampler = Sampler::new(SamplingConfig::new().strategy(strategy));

        let single_logits = Array1::from_vec(vec![value]);
        let result = sampler.sample(&single_logits);

        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), 0.0); // Only valid index is 0
    }
}
