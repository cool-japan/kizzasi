//! Signal generators for testing and development
//!
//! Provides various signal generators:
//! - Sine waves, square waves, sawtooth
//! - White noise, pink noise
//! - Impulse, step functions
//! - Chirp (frequency sweep)

use scirs2_core::ndarray::Array1;
use scirs2_core::random::thread_rng;
use std::f32::consts::PI;

/// Signal generator trait
pub trait SignalGenerator {
    /// Generate the next sample
    fn next_sample(&mut self) -> f32;

    /// Generate n samples
    fn generate(&mut self, n: usize) -> Array1<f32> {
        let samples: Vec<f32> = (0..n).map(|_| self.next_sample()).collect();
        Array1::from_vec(samples)
    }

    /// Reset the generator to initial state
    fn reset(&mut self);
}

/// Sine wave generator
#[derive(Debug, Clone)]
pub struct SineGenerator {
    frequency: f32,
    amplitude: f32,
    phase: f32,
    sample_rate: f32,
    sample_index: usize,
}

impl SineGenerator {
    /// Create a new sine generator
    pub fn new(frequency: f32, amplitude: f32, sample_rate: f32) -> Self {
        Self {
            frequency,
            amplitude,
            phase: 0.0,
            sample_rate,
            sample_index: 0,
        }
    }

    /// Set initial phase (in radians)
    pub fn with_phase(mut self, phase: f32) -> Self {
        self.phase = phase;
        self
    }
}

impl SignalGenerator for SineGenerator {
    fn next_sample(&mut self) -> f32 {
        let t = self.sample_index as f32 / self.sample_rate;
        let sample = self.amplitude * (2.0 * PI * self.frequency * t + self.phase).sin();
        self.sample_index += 1;
        sample
    }

    fn reset(&mut self) {
        self.sample_index = 0;
    }
}

/// Square wave generator
#[derive(Debug, Clone)]
pub struct SquareGenerator {
    frequency: f32,
    amplitude: f32,
    duty_cycle: f32,
    sample_rate: f32,
    sample_index: usize,
}

impl SquareGenerator {
    /// Create a new square wave generator
    pub fn new(frequency: f32, amplitude: f32, sample_rate: f32) -> Self {
        Self {
            frequency,
            amplitude,
            duty_cycle: 0.5,
            sample_rate,
            sample_index: 0,
        }
    }

    /// Set duty cycle (0.0 to 1.0)
    pub fn with_duty_cycle(mut self, duty: f32) -> Self {
        self.duty_cycle = duty.clamp(0.0, 1.0);
        self
    }
}

impl SignalGenerator for SquareGenerator {
    fn next_sample(&mut self) -> f32 {
        let t = self.sample_index as f32 / self.sample_rate;
        let period = 1.0 / self.frequency;
        let phase = (t % period) / period;
        let sample = if phase < self.duty_cycle {
            self.amplitude
        } else {
            -self.amplitude
        };
        self.sample_index += 1;
        sample
    }

    fn reset(&mut self) {
        self.sample_index = 0;
    }
}

/// Sawtooth wave generator
#[derive(Debug, Clone)]
pub struct SawtoothGenerator {
    frequency: f32,
    amplitude: f32,
    sample_rate: f32,
    sample_index: usize,
}

impl SawtoothGenerator {
    /// Create a new sawtooth generator
    pub fn new(frequency: f32, amplitude: f32, sample_rate: f32) -> Self {
        Self {
            frequency,
            amplitude,
            sample_rate,
            sample_index: 0,
        }
    }
}

impl SignalGenerator for SawtoothGenerator {
    fn next_sample(&mut self) -> f32 {
        let t = self.sample_index as f32 / self.sample_rate;
        let period = 1.0 / self.frequency;
        let phase = (t % period) / period;
        let sample = self.amplitude * (2.0 * phase - 1.0);
        self.sample_index += 1;
        sample
    }

    fn reset(&mut self) {
        self.sample_index = 0;
    }
}

/// Triangle wave generator
#[derive(Debug, Clone)]
pub struct TriangleGenerator {
    frequency: f32,
    amplitude: f32,
    sample_rate: f32,
    sample_index: usize,
}

impl TriangleGenerator {
    /// Create a new triangle wave generator
    pub fn new(frequency: f32, amplitude: f32, sample_rate: f32) -> Self {
        Self {
            frequency,
            amplitude,
            sample_rate,
            sample_index: 0,
        }
    }
}

impl SignalGenerator for TriangleGenerator {
    fn next_sample(&mut self) -> f32 {
        let t = self.sample_index as f32 / self.sample_rate;
        let period = 1.0 / self.frequency;
        let phase = (t % period) / period;
        let sample = self.amplitude * (4.0 * (phase - (phase + 0.5).floor()).abs() - 1.0);
        self.sample_index += 1;
        sample
    }

    fn reset(&mut self) {
        self.sample_index = 0;
    }
}

/// White noise generator using a simple LCG for reproducibility
#[derive(Debug, Clone)]
pub struct WhiteNoiseGenerator {
    amplitude: f32,
    seed: u64,
    state: u64,
}

impl WhiteNoiseGenerator {
    /// Create a new white noise generator with random seed
    pub fn new(amplitude: f32) -> Self {
        let seed = thread_rng().random::<u64>();
        Self {
            amplitude,
            seed,
            state: seed,
        }
    }

    /// Create with a specific seed for reproducibility
    pub fn with_seed(amplitude: f32, seed: u64) -> Self {
        Self {
            amplitude,
            seed,
            state: seed,
        }
    }

    /// Simple LCG random number generator
    fn next_random(&mut self) -> f32 {
        // LCG parameters (same as glibc)
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        // Extract bits and normalize to [0, 1)
        ((self.state >> 16) & 0x7fff) as f32 / 32768.0
    }
}

impl SignalGenerator for WhiteNoiseGenerator {
    fn next_sample(&mut self) -> f32 {
        self.amplitude * (self.next_random() * 2.0 - 1.0)
    }

    fn reset(&mut self) {
        self.state = self.seed;
    }
}

/// Pink noise generator (1/f spectrum)
#[derive(Debug, Clone)]
pub struct PinkNoiseGenerator {
    amplitude: f32,
    seed: u64,
    state: u64,
    // Voss-McCartney algorithm state
    rows: [f32; 16],
    running_sum: f32,
    index: usize,
}

impl PinkNoiseGenerator {
    /// Create a new pink noise generator
    pub fn new(amplitude: f32) -> Self {
        let seed = thread_rng().random::<u64>();
        Self {
            amplitude,
            seed,
            state: seed,
            rows: [0.0; 16],
            running_sum: 0.0,
            index: 0,
        }
    }

    /// Create with a specific seed for reproducibility
    pub fn with_seed(amplitude: f32, seed: u64) -> Self {
        Self {
            amplitude,
            seed,
            state: seed,
            rows: [0.0; 16],
            running_sum: 0.0,
            index: 0,
        }
    }

    /// Simple LCG random number generator
    fn next_random(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state >> 16) & 0x7fff) as f32 / 32768.0
    }
}

impl SignalGenerator for PinkNoiseGenerator {
    fn next_sample(&mut self) -> f32 {
        // Voss-McCartney algorithm
        let last_index = self.index;
        self.index = (self.index + 1) % (1 << 16);

        // Find which rows to update (trailing zeros)
        let diff = self.index ^ last_index;
        for i in 0..16 {
            if (diff >> i) & 1 == 1 {
                self.running_sum -= self.rows[i];
                self.rows[i] = self.next_random() * 2.0 - 1.0;
                self.running_sum += self.rows[i];
            }
        }

        self.amplitude * self.running_sum / 16.0
    }

    fn reset(&mut self) {
        self.state = self.seed;
        self.rows = [0.0; 16];
        self.running_sum = 0.0;
        self.index = 0;
    }
}

/// Impulse generator (single spike)
#[derive(Debug, Clone)]
pub struct ImpulseGenerator {
    amplitude: f32,
    delay_samples: usize,
    sample_index: usize,
}

impl ImpulseGenerator {
    /// Create a new impulse generator
    pub fn new(amplitude: f32, delay_samples: usize) -> Self {
        Self {
            amplitude,
            delay_samples,
            sample_index: 0,
        }
    }
}

impl SignalGenerator for ImpulseGenerator {
    fn next_sample(&mut self) -> f32 {
        let sample = if self.sample_index == self.delay_samples {
            self.amplitude
        } else {
            0.0
        };
        self.sample_index += 1;
        sample
    }

    fn reset(&mut self) {
        self.sample_index = 0;
    }
}

/// Step function generator
#[derive(Debug, Clone)]
pub struct StepGenerator {
    amplitude: f32,
    step_sample: usize,
    sample_index: usize,
}

impl StepGenerator {
    /// Create a new step function generator
    pub fn new(amplitude: f32, step_sample: usize) -> Self {
        Self {
            amplitude,
            step_sample,
            sample_index: 0,
        }
    }
}

impl SignalGenerator for StepGenerator {
    fn next_sample(&mut self) -> f32 {
        let sample = if self.sample_index >= self.step_sample {
            self.amplitude
        } else {
            0.0
        };
        self.sample_index += 1;
        sample
    }

    fn reset(&mut self) {
        self.sample_index = 0;
    }
}

/// Chirp (frequency sweep) generator
#[derive(Debug, Clone)]
pub struct ChirpGenerator {
    start_freq: f32,
    end_freq: f32,
    amplitude: f32,
    duration: f32,
    sample_rate: f32,
    sample_index: usize,
}

impl ChirpGenerator {
    /// Create a new chirp generator
    pub fn new(
        start_freq: f32,
        end_freq: f32,
        amplitude: f32,
        duration: f32,
        sample_rate: f32,
    ) -> Self {
        Self {
            start_freq,
            end_freq,
            amplitude,
            duration,
            sample_rate,
            sample_index: 0,
        }
    }
}

impl SignalGenerator for ChirpGenerator {
    fn next_sample(&mut self) -> f32 {
        let t = self.sample_index as f32 / self.sample_rate;
        if t > self.duration {
            return 0.0;
        }

        // Linear frequency sweep
        let k = (self.end_freq - self.start_freq) / self.duration;
        // Instantaneous frequency = start_freq + k * t
        // Phase integral = start_freq * t + 0.5 * k * t^2
        let phase = 2.0 * PI * (self.start_freq * t + 0.5 * k * t * t);
        let sample = self.amplitude * phase.sin();

        self.sample_index += 1;
        sample
    }

    fn reset(&mut self) {
        self.sample_index = 0;
    }
}

/// Multi-tone generator (sum of multiple sine waves)
#[derive(Debug, Clone)]
pub struct MultiToneGenerator {
    frequencies: Vec<f32>,
    amplitudes: Vec<f32>,
    sample_rate: f32,
    sample_index: usize,
}

impl MultiToneGenerator {
    /// Create a new multi-tone generator
    pub fn new(frequencies: Vec<f32>, amplitudes: Vec<f32>, sample_rate: f32) -> Self {
        assert_eq!(
            frequencies.len(),
            amplitudes.len(),
            "Frequencies and amplitudes must have same length"
        );
        Self {
            frequencies,
            amplitudes,
            sample_rate,
            sample_index: 0,
        }
    }

    /// Create with uniform amplitude
    pub fn uniform(frequencies: Vec<f32>, amplitude: f32, sample_rate: f32) -> Self {
        let amplitudes = vec![amplitude / frequencies.len() as f32; frequencies.len()];
        Self::new(frequencies, amplitudes, sample_rate)
    }
}

impl SignalGenerator for MultiToneGenerator {
    fn next_sample(&mut self) -> f32 {
        let t = self.sample_index as f32 / self.sample_rate;
        let sample: f32 = self
            .frequencies
            .iter()
            .zip(self.amplitudes.iter())
            .map(|(&f, &a)| a * (2.0 * PI * f * t).sin())
            .sum();
        self.sample_index += 1;
        sample
    }

    fn reset(&mut self) {
        self.sample_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_generator() {
        let mut gen = SineGenerator::new(1.0, 1.0, 4.0);
        let samples = gen.generate(4);
        assert_eq!(samples.len(), 4);
        // At t=0, sin(0) = 0
        assert!(samples[0].abs() < 0.01);
        // At t=0.25, sin(pi/2) = 1
        assert!((samples[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_square_generator() {
        let mut gen = SquareGenerator::new(1.0, 1.0, 4.0);
        let samples = gen.generate(4);
        assert_eq!(samples.len(), 4);
        // First half of period should be positive
        assert_eq!(samples[0], 1.0);
        assert_eq!(samples[1], 1.0);
        // Second half should be negative
        assert_eq!(samples[2], -1.0);
        assert_eq!(samples[3], -1.0);
    }

    #[test]
    fn test_white_noise() {
        let mut gen = WhiteNoiseGenerator::new(1.0);
        let samples = gen.generate(1000);
        assert_eq!(samples.len(), 1000);
        // Mean should be close to 0
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_impulse_generator() {
        let mut gen = ImpulseGenerator::new(1.0, 2);
        let samples = gen.generate(5);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 0.0);
        assert_eq!(samples[2], 1.0);
        assert_eq!(samples[3], 0.0);
    }

    #[test]
    fn test_step_generator() {
        let mut gen = StepGenerator::new(1.0, 2);
        let samples = gen.generate(5);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 0.0);
        assert_eq!(samples[2], 1.0);
        assert_eq!(samples[3], 1.0);
    }

    #[test]
    fn test_chirp_generator() {
        let mut gen = ChirpGenerator::new(1.0, 10.0, 1.0, 1.0, 100.0);
        let samples = gen.generate(100);
        assert_eq!(samples.len(), 100);
        // All samples should be in [-1, 1]
        for s in samples.iter() {
            assert!(*s >= -1.0 && *s <= 1.0);
        }
    }

    #[test]
    fn test_multi_tone() {
        let mut gen = MultiToneGenerator::uniform(vec![1.0, 2.0], 1.0, 100.0);
        let samples = gen.generate(100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_reset() {
        let mut gen = SineGenerator::new(1.0, 1.0, 4.0);
        let first = gen.generate(4);
        gen.reset();
        let second = gen.generate(4);

        for i in 0..4 {
            assert!((first[i] - second[i]).abs() < 1e-6);
        }
    }
}
