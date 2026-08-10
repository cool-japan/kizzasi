//! PEAQ neural network: 11→3→1 MLP mapping MOVs to Distortion Index (DI).
//!
//! Weight provenance: when BS.1387-1 Annex 2 Tables B.11/B.12 are obtained
//! from the ITU standards document (paid), set `WEIGHTS_VERIFIED = true` and
//! replace the placeholder constants. Until then, a seeded-Gaussian placeholder
//! is used. ODG/DI values produced with `WEIGHTS_VERIFIED = false` should be
//! treated as qualitative trend indicators, not certified PEAQ scores.

/// `true` once the BS.1387-1 Annex 2 Table B.11/B.12 weights are sourced
/// and verified against the paid ITU standards document.
pub const WEIGHTS_VERIFIED: bool = false;

// [amin, amax] per MOV — normalize to [0,1] via (x - amin) / (amax - amin)
// Order: BandwidthRefB, BandwidthTestB, TotalNmrB, WinModDiff1B, AdbB,
//        EhsB, AvgModDiff1B, AvgModDiff2B, RmsNoiseLoudB, MfpdB, RelDistFramesB
const MOV_AMIN: [f32; 11] = [0.0, 0.0, -20.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
const MOV_AMAX: [f32; 11] = [
    15232.0, 15232.0, 130.0, 100.0, 100.0, 0.9, 100.0, 100.0, 3.0, 1.0, 1.0,
];

/// LCG random number generator (Knuth parameters) producing values in [-0.2, 0.2].
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 32) as u32;
    let x = (bits as f32 / u32::MAX as f32) - 0.5;
    x * 0.4 // [-0.2, 0.2]
}

/// PEAQ 11→3→1 MLP neural network.
///
/// Weights are placeholder values seeded from LCG(42) when
/// `WEIGHTS_VERIFIED == false`. Replace with BS.1387-1 Annex 2 Table B.11/B.12
/// values and set `WEIGHTS_VERIFIED = true` once the paid ITU document is available.
pub struct PeaqNeuralNet {
    // Hidden layer: 11 inputs → 3 neurons
    // w_hidden[h][i] = weight from input i to hidden neuron h
    w_hidden: [[f32; 11]; 3],
    b_hidden: [f32; 3],
    // Output layer: 3 hidden → 1 output
    w_out: [f32; 3],
    b_out: f32,
}

impl PeaqNeuralNet {
    /// Build a new network with placeholder weights seeded from LCG(42).
    pub fn new() -> Self {
        let mut state: u64 = 42;
        let mut net = PeaqNeuralNet {
            w_hidden: [[0.0; 11]; 3],
            b_hidden: [0.0; 3],
            w_out: [0.0; 3],
            b_out: 0.0,
        };
        for h in 0..3 {
            for i in 0..11 {
                net.w_hidden[h][i] = lcg_next(&mut state);
            }
            net.b_hidden[h] = lcg_next(&mut state);
        }
        for h in 0..3 {
            net.w_out[h] = lcg_next(&mut state);
        }
        net.b_out = lcg_next(&mut state);
        net
    }

    /// Run inference: 11 MOV values → Distortion Index (DI).
    ///
    /// Inputs are normalized per BS.1387-1 Table B.11 ranges before inference.
    pub fn infer(&self, movs: &[f32; 11]) -> f32 {
        // Normalize each MOV to [0,1]
        let normed: [f32; 11] = core::array::from_fn(|i| {
            let range = MOV_AMAX[i] - MOV_AMIN[i];
            if range > 1e-30 {
                ((movs[i] - MOV_AMIN[i]) / range).clamp(0.0, 1.0)
            } else {
                0.0
            }
        });

        // Hidden layer (sigmoid activation)
        let hidden: [f32; 3] = core::array::from_fn(|h| {
            let z: f32 = self.w_hidden[h]
                .iter()
                .zip(normed.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>()
                + self.b_hidden[h];
            sigmoid(z)
        });

        // Output neuron (sigmoid activation → DI)
        let z_out: f32 = self
            .w_out
            .iter()
            .zip(hidden.iter())
            .map(|(w, h)| w * h)
            .sum::<f32>()
            + self.b_out;
        sigmoid(z_out)
    }
}

impl Default for PeaqNeuralNet {
    fn default() -> Self {
        Self::new()
    }
}

/// Sigmoid activation function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Map Distortion Index (DI) to Objective Difference Grade (ODG).
///
/// Formula: `ODG = -3.98 + 4.2 / (1 + exp(-DI))`, clamped to `[-4, 0]`.
///
/// ODG is monotonically increasing in DI:
/// - Low DI (highly distorted) → ODG near -3.98
/// - High DI (high quality) → ODG near 0.22, clamped to 0.0
pub fn di_to_odg(di: f32) -> f32 {
    (-3.98_f32 + 4.2_f32 / (1.0_f32 + (-di).exp())).clamp(-4.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_di_to_odg_monotonically_increasing() {
        let dis = [-10.0_f32, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0];
        let odgs: Vec<f32> = dis.iter().map(|&d| di_to_odg(d)).collect();
        for w in odgs.windows(2) {
            assert!(
                w[1] >= w[0],
                "ODG should be non-decreasing: {:.4} >= {:.4} failed",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn test_di_to_odg_extremes() {
        // At very large positive DI: ODG approaches 0.22, clamped to 0.0
        let odg_high = di_to_odg(100.0);
        assert!(
            (odg_high - 0.0).abs() < 0.01,
            "Expected ~0.0, got {odg_high}"
        );

        // At very large negative DI: ODG approaches -3.98 (NOT -4.0)
        let odg_low = di_to_odg(-100.0);
        assert!(
            (odg_low - (-3.98)).abs() < 0.01,
            "Expected ~-3.98, got {odg_low}"
        );
    }

    #[test]
    fn test_di_to_odg_clamping() {
        // Verify return values are always in [-4, 0]
        for di in [-100.0_f32, -10.0, 0.0, 10.0, 100.0] {
            let odg = di_to_odg(di);
            assert!(odg >= -4.0, "ODG {odg} below -4.0 for DI {di}");
            assert!(odg <= 0.0, "ODG {odg} above 0.0 for DI {di}");
        }
    }

    #[test]
    fn test_neural_net_input_normalization_in_unit_range() {
        let net = PeaqNeuralNet::new();
        // MOV_AMIN values (lower boundary)
        let movs_min: [f32; 11] = core::array::from_fn(|i| MOV_AMIN[i]);
        let di_min = net.infer(&movs_min);
        assert!(
            di_min.is_finite(),
            "DI should be finite for min inputs, got {di_min}"
        );

        // MOV_AMAX values (upper boundary)
        let movs_max: [f32; 11] = core::array::from_fn(|i| MOV_AMAX[i]);
        let di_max = net.infer(&movs_max);
        assert!(
            di_max.is_finite(),
            "DI should be finite for max inputs, got {di_max}"
        );

        // Sigmoid output is always in (0, 1)
        assert!(di_min > 0.0 && di_min < 1.0);
        assert!(di_max > 0.0 && di_max < 1.0);
    }

    #[test]
    fn test_neural_net_output_in_odg_range() {
        let net = PeaqNeuralNet::new();
        // Typical MOV values
        let mov_test: [f32; 11] = [
            8000.0, 7500.0, 10.0, 5.0, 5.0, 0.2, 10.0, 10.0, 0.5, 0.3, 0.1,
        ];
        let di = net.infer(&mov_test);
        let odg = di_to_odg(di);
        assert!(
            (-4.0..=0.0).contains(&odg),
            "ODG {odg} out of range [-4, 0]"
        );
        assert!(odg.is_finite(), "ODG must be finite, got {odg}");
    }

    #[test]
    fn test_neural_net_deterministic() {
        let net1 = PeaqNeuralNet::new();
        let net2 = PeaqNeuralNet::new();
        let movs: [f32; 11] = [1.0; 11];
        assert_eq!(
            net1.infer(&movs),
            net2.infer(&movs),
            "Neural net must be deterministic"
        );
    }
}
