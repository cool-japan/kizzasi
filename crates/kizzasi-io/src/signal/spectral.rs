//! Spectral analysis and window functions
//!
//! This module provides tools for frequency-domain analysis including:
//! - Spectrogram computation (Short-Time Fourier Transform)
//! - Various window functions for spectral analysis
//! - Time-frequency representations

/// Spectrogram data structure
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Magnitude values (flattened: num_frames * num_bins)
    pub magnitudes: Vec<f32>,
    /// Phase values (flattened: num_frames * num_bins)
    pub phases: Vec<f32>,
    /// Number of time frames
    pub num_frames: usize,
    /// Number of frequency bins
    pub num_bins: usize,
    /// Hop length in samples
    pub hop_length: usize,
    /// Sample rate
    pub sample_rate: f32,
}

impl Spectrogram {
    /// Get magnitude at a specific frame and bin
    pub fn magnitude(&self, frame: usize, bin: usize) -> f32 {
        self.magnitudes[frame * self.num_bins + bin]
    }

    /// Get phase at a specific frame and bin
    pub fn phase(&self, frame: usize, bin: usize) -> f32 {
        self.phases[frame * self.num_bins + bin]
    }

    /// Get power spectrum (magnitude squared)
    pub fn power(&self, frame: usize, bin: usize) -> f32 {
        let mag = self.magnitude(frame, bin);
        mag * mag
    }

    /// Convert bin index to frequency in Hz
    pub fn bin_to_hz(&self, bin: usize, n_fft: usize) -> f32 {
        bin as f32 * self.sample_rate / n_fft as f32
    }

    /// Convert frame index to time in seconds
    pub fn frame_to_time(&self, frame: usize) -> f32 {
        frame as f32 * self.hop_length as f32 / self.sample_rate
    }

    /// Convert to dB scale
    pub fn to_db(&self, ref_value: f32, min_db: f32) -> Vec<f32> {
        self.magnitudes
            .iter()
            .map(|&m| {
                let db = 20.0 * (m / ref_value + 1e-10).log10();
                db.max(min_db)
            })
            .collect()
    }
}

/// Window function types for spectral analysis
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Bartlett (triangular) window
    Bartlett,
    /// Kaiser window with specified beta parameter
    Kaiser { beta: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spectrogram() -> Spectrogram {
        // 2 frames x 3 bins, row-major.
        Spectrogram {
            magnitudes: vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            phases: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            num_frames: 2,
            num_bins: 3,
            hop_length: 256,
            sample_rate: 16_000.0,
        }
    }

    #[test]
    fn test_magnitude_and_phase_are_row_major() {
        let spectrogram = sample_spectrogram();
        assert_eq!(spectrogram.magnitude(0, 0), 1.0);
        assert_eq!(spectrogram.magnitude(0, 2), 4.0);
        assert_eq!(spectrogram.magnitude(1, 0), 8.0);
        assert_eq!(spectrogram.magnitude(1, 2), 32.0);
        assert_eq!(spectrogram.phase(1, 1), 0.4);
    }

    #[test]
    fn test_power_is_squared_magnitude() {
        let spectrogram = sample_spectrogram();
        assert_eq!(spectrogram.power(1, 2), 32.0 * 32.0);
    }

    #[test]
    fn test_bin_to_hz_and_frame_to_time() {
        let spectrogram = sample_spectrogram();
        // Bin k of an n_fft-point transform sits at k * fs / n_fft.
        assert!((spectrogram.bin_to_hz(0, 512) - 0.0).abs() < 1e-6);
        assert!((spectrogram.bin_to_hz(256, 512) - 8000.0).abs() < 1e-3);
        // Frame m starts at m * hop / fs seconds.
        assert!((spectrogram.frame_to_time(0)).abs() < 1e-9);
        assert!((spectrogram.frame_to_time(1) - 0.016).abs() < 1e-6);
    }

    #[test]
    fn test_to_db_scales_and_floors() {
        let spectrogram = sample_spectrogram();
        let db = spectrogram.to_db(1.0, -20.0);
        assert_eq!(db.len(), spectrogram.magnitudes.len());
        // 20*log10(2) ~= 6.02 dB, and doubling adds ~6 dB each step.
        let first = db.first().copied().unwrap_or(f32::NAN);
        let second = db.get(1).copied().unwrap_or(f32::NAN);
        assert!(
            (second - first - 6.0206).abs() < 1e-2,
            "{first} -> {second}"
        );
        assert!(db.iter().all(|&v| v >= -20.0), "floor not applied");

        // Everything below the floor is clamped to it.
        let quiet = Spectrogram {
            magnitudes: vec![0.0; 4],
            ..sample_spectrogram()
        };
        assert!(quiet.to_db(1.0, -80.0).iter().all(|&v| v == -80.0));
    }

    #[test]
    fn test_window_type_variants_are_copyable() {
        let windows = [
            WindowType::Rectangular,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::Bartlett,
            WindowType::Kaiser { beta: 8.6 },
        ];
        for window in windows {
            let copied = window;
            assert!(!format!("{copied:?}").is_empty());
        }
    }
}
