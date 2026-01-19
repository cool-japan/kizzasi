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
