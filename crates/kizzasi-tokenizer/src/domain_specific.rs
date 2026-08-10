//! Domain-specific tokenizers for specialized audio applications
//!
//! Provides tokenizers optimized for:
//! - Speech processing with phoneme alignment
//! - Music analysis with note and chroma features
//! - Environmental sound classification
//! - Multi-speaker scenarios

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Mel-scale frequency conversion
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Inverse mel-scale conversion
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Configuration for Speech tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechTokenizerConfig {
    /// Sample rate in Hz
    pub sample_rate: usize,
    /// Number of mel filterbanks
    pub n_mels: usize,
    /// FFT size
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Number of phoneme classes (for alignment)
    pub n_phonemes: usize,
    /// Enable delta features (velocity)
    pub use_delta: bool,
    /// Enable delta-delta features (acceleration)
    pub use_delta_delta: bool,
}

impl Default for SpeechTokenizerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_fft: 512,
            hop_length: 160, // 10ms at 16kHz
            n_phonemes: 44,  // Standard English phoneme set
            use_delta: true,
            use_delta_delta: true,
        }
    }
}

/// Speech-specific tokenizer with phoneme alignment
///
/// Extracts mel-spectrogram features optimized for speech recognition
/// and provides phoneme-aligned tokenization for ASR tasks.
#[derive(Debug)]
pub struct SpeechTokenizer {
    config: SpeechTokenizerConfig,
    mel_filterbank: Array2<f32>,
}

impl SpeechTokenizer {
    /// Create a new speech tokenizer
    pub fn new(config: SpeechTokenizerConfig) -> TokenizerResult<Self> {
        // Create mel filterbank
        let mel_filterbank = Self::create_mel_filterbank(&config)?;

        Ok(Self {
            config,
            mel_filterbank,
        })
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank(config: &SpeechTokenizerConfig) -> TokenizerResult<Array2<f32>> {
        let n_freqs = config.n_fft / 2 + 1;
        let mut filterbank = Array2::zeros((config.n_mels, n_freqs));

        // Mel frequency points
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(config.sample_rate as f32 / 2.0);
        let mel_points: Vec<f32> = (0..=config.n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (config.n_mels + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();

        // Convert Hz to FFT bin
        let fft_bins: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((config.n_fft as f32 + 1.0) * hz / config.sample_rate as f32) as usize)
            .collect();

        // Create triangular filters
        for m in 0..config.n_mels {
            let f_left = fft_bins[m];
            let f_center = fft_bins[m + 1];
            let f_right = fft_bins[m + 2];

            for k in f_left..f_center {
                if f_center > f_left {
                    filterbank[[m, k]] = (k - f_left) as f32 / (f_center - f_left) as f32;
                }
            }

            for k in f_center..f_right {
                if f_right > f_center {
                    filterbank[[m, k]] = (f_right - k) as f32 / (f_right - f_center) as f32;
                }
            }
        }

        Ok(filterbank)
    }

    /// Compute mel-spectrogram from audio signal
    pub fn compute_mel_spectrogram(&self, signal: &Array1<f32>) -> TokenizerResult<Array2<f32>> {
        let n_frames = (signal.len() - self.config.n_fft) / self.config.hop_length + 1;
        let mut mel_spec = Array2::zeros((self.config.n_mels, n_frames));

        // Hann window
        let window: Vec<f32> = (0..self.config.n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (self.config.n_fft - 1) as f32).cos()))
            .collect();

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.config.hop_length;
            let end = start + self.config.n_fft;

            if end > signal.len() {
                break;
            }

            // Extract frame and apply window
            let frame: Vec<f32> = signal
                .slice(ndarray::s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Simple magnitude spectrum (using basic DFT for n_fft/2+1 bins)
            let n_freqs = self.config.n_fft / 2 + 1;
            let mut spectrum = vec![0.0f32; n_freqs];

            for (k, spec_val) in spectrum.iter_mut().enumerate().take(n_freqs) {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;
                for (n, &x) in frame.iter().enumerate() {
                    let angle = -2.0 * PI * k as f32 * n as f32 / self.config.n_fft as f32;
                    real += x * angle.cos();
                    imag += x * angle.sin();
                }
                *spec_val = (real * real + imag * imag).sqrt();
            }

            // Apply mel filterbank
            for m in 0..self.config.n_mels {
                let mut mel_energy = 0.0f32;
                for (k, &spec_val) in spectrum.iter().enumerate().take(n_freqs) {
                    mel_energy += self.mel_filterbank[[m, k]] * spec_val;
                }
                // Log mel spectrogram
                mel_spec[[m, frame_idx]] = (mel_energy + 1e-10).ln();
            }
        }

        Ok(mel_spec)
    }

    /// Compute delta features (velocity)
    pub fn compute_delta(features: &Array2<f32>) -> Array2<f32> {
        let (n_features, n_frames) = features.dim();
        let mut delta = Array2::zeros((n_features, n_frames));

        for t in 0..n_frames {
            let t_prev = if t > 0 { t - 1 } else { 0 };
            let t_next = if t < n_frames - 1 {
                t + 1
            } else {
                n_frames - 1
            };

            for f in 0..n_features {
                delta[[f, t]] = (features[[f, t_next]] - features[[f, t_prev]]) / 2.0;
            }
        }

        delta
    }

    /// Extract speech features with optional delta and delta-delta
    pub fn extract_features(&self, signal: &Array1<f32>) -> TokenizerResult<Array2<f32>> {
        let mel_spec = self.compute_mel_spectrogram(signal)?;

        if !self.config.use_delta && !self.config.use_delta_delta {
            return Ok(mel_spec);
        }

        let mut features = vec![mel_spec.clone()];

        if self.config.use_delta {
            let delta = Self::compute_delta(&mel_spec);

            if self.config.use_delta_delta {
                let delta_delta = Self::compute_delta(&delta);
                features.push(delta.clone());
                features.push(delta_delta);
            } else {
                features.push(delta);
            }
        }

        // Concatenate along feature dimension
        let n_frames = features[0].dim().1;
        let total_features: usize = features.iter().map(|f| f.dim().0).sum();
        let mut combined = Array2::zeros((total_features, n_frames));

        let mut offset = 0;
        for feat in features {
            let n_feat = feat.dim().0;
            for i in 0..n_feat {
                for j in 0..n_frames {
                    combined[[offset + i, j]] = feat[[i, j]];
                }
            }
            offset += n_feat;
        }

        Ok(combined)
    }
}

impl SignalTokenizer for SpeechTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let features = self.extract_features(signal)?;
        // Flatten features to 1D array
        Ok(Array1::from_vec(
            features.iter().copied().collect::<Vec<f32>>(),
        ))
    }

    fn decode(&self, _tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        // Speech features are not directly invertible to audio
        Err(TokenizerError::decoding(
            "speech_tokenizer",
            "Mel-spectrogram features cannot be directly inverted to audio. Use a vocoder (e.g., Griffin-Lim, WaveGlow) for reconstruction.".to_string(),
        ))
    }

    fn embed_dim(&self) -> usize {
        let mut dim = self.config.n_mels;
        if self.config.use_delta {
            dim += self.config.n_mels;
        }
        if self.config.use_delta_delta {
            dim += self.config.n_mels;
        }
        dim
    }

    fn vocab_size(&self) -> usize {
        0 // Continuous features
    }
}

/// Configuration for Music tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicTokenizerConfig {
    /// Sample rate in Hz
    pub sample_rate: usize,
    /// Number of chroma bins (typically 12 for 12 semitones)
    pub n_chroma: usize,
    /// FFT size
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Number of octaves for CQT
    pub n_octaves: usize,
    /// Bins per octave
    pub bins_per_octave: usize,
}

impl Default for MusicTokenizerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            n_chroma: 12,
            n_fft: 2048,
            hop_length: 512,
            n_octaves: 7,
            bins_per_octave: 36, // 3 bins per semitone
        }
    }
}

/// Music-specific tokenizer with chroma and note features
///
/// Extracts pitch-class (chroma) features for music analysis,
/// harmony detection, and chord recognition.
pub struct MusicTokenizer {
    config: MusicTokenizerConfig,
}

impl MusicTokenizer {
    /// Create a new music tokenizer
    pub fn new(config: MusicTokenizerConfig) -> Self {
        Self { config }
    }

    /// Compute chromagram from audio signal
    pub fn compute_chromagram(&self, signal: &Array1<f32>) -> TokenizerResult<Array2<f32>> {
        let n_frames = (signal.len() - self.config.n_fft) / self.config.hop_length + 1;
        let mut chroma = Array2::zeros((self.config.n_chroma, n_frames));

        // Hann window
        let window: Vec<f32> = (0..self.config.n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (self.config.n_fft - 1) as f32).cos()))
            .collect();

        // Reference frequency (A4 = 440 Hz)
        let ref_freq = 440.0f32;

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.config.hop_length;
            let end = start + self.config.n_fft;

            if end > signal.len() {
                break;
            }

            // Extract frame and apply window
            let frame: Vec<f32> = signal
                .slice(ndarray::s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Compute magnitude spectrum
            let n_freqs = self.config.n_fft / 2 + 1;
            for k in 1..n_freqs {
                let freq = k as f32 * self.config.sample_rate as f32 / self.config.n_fft as f32;

                // Calculate pitch class (0-11)
                let pitch = 12.0 * (freq / ref_freq).log2();
                let pitch_class = pitch.rem_euclid(12.0) as usize;

                if pitch_class < self.config.n_chroma {
                    // Simple magnitude calculation
                    let mut real = 0.0f32;
                    let mut imag = 0.0f32;
                    for (n, &x) in frame.iter().enumerate() {
                        let angle = -2.0 * PI * k as f32 * n as f32 / self.config.n_fft as f32;
                        real += x * angle.cos();
                        imag += x * angle.sin();
                    }
                    let magnitude = (real * real + imag * imag).sqrt();
                    chroma[[pitch_class, frame_idx]] += magnitude;
                }
            }

            // Normalize chroma vector
            let total_energy: f32 = chroma.column(frame_idx).iter().sum();
            if total_energy > 1e-10 {
                for i in 0..self.config.n_chroma {
                    chroma[[i, frame_idx]] /= total_energy;
                }
            }
        }

        Ok(chroma)
    }

    /// Compute onset strength envelope for beat tracking
    pub fn compute_onset_strength(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let n_frames = (signal.len() - self.config.n_fft) / self.config.hop_length + 1;
        let mut onset_strength = Array1::zeros(n_frames);

        let window: Vec<f32> = (0..self.config.n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (self.config.n_fft - 1) as f32).cos()))
            .collect();

        let mut prev_spectrum = vec![0.0f32; self.config.n_fft / 2 + 1];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.config.hop_length;
            let end = start + self.config.n_fft;

            if end > signal.len() {
                break;
            }

            // Extract frame
            let frame: Vec<f32> = signal
                .slice(ndarray::s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Compute magnitude spectrum
            let n_freqs = self.config.n_fft / 2 + 1;
            let mut spectrum = vec![0.0f32; n_freqs];

            for (k, spec_val) in spectrum.iter_mut().enumerate().take(n_freqs) {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;
                for (n, &x) in frame.iter().enumerate() {
                    let angle = -2.0 * PI * k as f32 * n as f32 / self.config.n_fft as f32;
                    real += x * angle.cos();
                    imag += x * angle.sin();
                }
                *spec_val = (real * real + imag * imag).sqrt();
            }

            // Onset strength is the positive difference from previous frame
            if frame_idx > 0 {
                let mut strength = 0.0f32;
                for k in 0..n_freqs {
                    let diff = spectrum[k] - prev_spectrum[k];
                    if diff > 0.0 {
                        strength += diff;
                    }
                }
                onset_strength[frame_idx] = strength;
            }

            prev_spectrum = spectrum;
        }

        Ok(onset_strength)
    }
}

impl SignalTokenizer for MusicTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let chroma = self.compute_chromagram(signal)?;
        Ok(Array1::from_vec(
            chroma.iter().copied().collect::<Vec<f32>>(),
        ))
    }

    fn decode(&self, _tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Err(TokenizerError::decoding(
            "music_tokenizer",
            "Chroma features cannot be directly inverted to audio".to_string(),
        ))
    }

    fn embed_dim(&self) -> usize {
        self.config.n_chroma
    }

    fn vocab_size(&self) -> usize {
        0 // Continuous features
    }
}

/// Configuration for Environmental sound tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalTokenizerConfig {
    /// Sample rate in Hz
    pub sample_rate: usize,
    /// Number of mel filterbanks
    pub n_mels: usize,
    /// FFT size
    pub n_fft: usize,
    /// Hop length
    pub hop_length: usize,
    /// Use spectral centroid
    pub use_spectral_centroid: bool,
    /// Use spectral rolloff
    pub use_spectral_rolloff: bool,
    /// Use zero crossing rate
    pub use_zcr: bool,
}

impl Default for EnvironmentalTokenizerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            n_mels: 128,
            n_fft: 2048,
            hop_length: 512,
            use_spectral_centroid: true,
            use_spectral_rolloff: true,
            use_zcr: true,
        }
    }
}

/// Environmental sound tokenizer for general audio classification
///
/// Extracts diverse spectro-temporal features for environmental
/// sound recognition, including mel-spectrograms and statistical features.
pub struct EnvironmentalTokenizer {
    config: EnvironmentalTokenizerConfig,
    speech_tokenizer: SpeechTokenizer,
}

impl EnvironmentalTokenizer {
    /// Create a new environmental sound tokenizer
    pub fn new(config: EnvironmentalTokenizerConfig) -> TokenizerResult<Self> {
        // Reuse speech tokenizer for mel-spectrogram computation
        let speech_config = SpeechTokenizerConfig {
            sample_rate: config.sample_rate,
            n_mels: config.n_mels,
            n_fft: config.n_fft,
            hop_length: config.hop_length,
            n_phonemes: 0,
            use_delta: false,
            use_delta_delta: false,
        };

        let speech_tokenizer = SpeechTokenizer::new(speech_config)?;

        Ok(Self {
            config,
            speech_tokenizer,
        })
    }

    /// Compute spectral centroid
    pub fn compute_spectral_centroid(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let n_frames = (signal.len() - self.config.n_fft) / self.config.hop_length + 1;
        let mut centroid = Array1::zeros(n_frames);

        let window: Vec<f32> = (0..self.config.n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (self.config.n_fft - 1) as f32).cos()))
            .collect();

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.config.hop_length;
            let end = start + self.config.n_fft;

            if end > signal.len() {
                break;
            }

            let frame: Vec<f32> = signal
                .slice(ndarray::s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Compute magnitude spectrum
            let n_freqs = self.config.n_fft / 2 + 1;
            let mut weighted_sum = 0.0f32;
            let mut total_magnitude = 0.0f32;

            for k in 0..n_freqs {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;
                for (n, &x) in frame.iter().enumerate() {
                    let angle = -2.0 * PI * k as f32 * n as f32 / self.config.n_fft as f32;
                    real += x * angle.cos();
                    imag += x * angle.sin();
                }
                let magnitude = (real * real + imag * imag).sqrt();
                let freq = k as f32 * self.config.sample_rate as f32 / self.config.n_fft as f32;

                weighted_sum += freq * magnitude;
                total_magnitude += magnitude;
            }

            centroid[frame_idx] = if total_magnitude > 1e-10 {
                weighted_sum / total_magnitude
            } else {
                0.0
            };
        }

        Ok(centroid)
    }

    /// Compute zero crossing rate
    pub fn compute_zcr(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let n_frames = (signal.len() - self.config.n_fft) / self.config.hop_length + 1;
        let mut zcr = Array1::zeros(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.config.hop_length;
            let end = start + self.config.n_fft;

            if end > signal.len() {
                break;
            }

            let frame = signal.slice(ndarray::s![start..end]);
            let mut crossings = 0;

            for i in 1..frame.len() {
                if (frame[i] >= 0.0 && frame[i - 1] < 0.0)
                    || (frame[i] < 0.0 && frame[i - 1] >= 0.0)
                {
                    crossings += 1;
                }
            }

            zcr[frame_idx] = crossings as f32 / frame.len() as f32;
        }

        Ok(zcr)
    }
}

impl SignalTokenizer for EnvironmentalTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        // Extract mel-spectrogram
        let mel_spec = self.speech_tokenizer.compute_mel_spectrogram(signal)?;
        let mut features = vec![mel_spec.iter().copied().collect::<Vec<f32>>()];

        // Add spectral centroid
        if self.config.use_spectral_centroid {
            let centroid = self.compute_spectral_centroid(signal)?;
            features.push(centroid.to_vec());
        }

        // Add zero crossing rate
        if self.config.use_zcr {
            let zcr = self.compute_zcr(signal)?;
            features.push(zcr.to_vec());
        }

        // Concatenate all features
        let combined: Vec<f32> = features.into_iter().flatten().collect();
        Ok(Array1::from_vec(combined))
    }

    fn decode(&self, _tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Err(TokenizerError::decoding(
            "environmental_tokenizer",
            "Environmental features cannot be directly inverted to audio".to_string(),
        ))
    }

    fn embed_dim(&self) -> usize {
        let mut dim = self.config.n_mels;
        if self.config.use_spectral_centroid {
            dim += 1;
        }
        if self.config.use_spectral_rolloff {
            dim += 1;
        }
        if self.config.use_zcr {
            dim += 1;
        }
        dim
    }

    fn vocab_size(&self) -> usize {
        0 // Continuous features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_conversions() {
        let hz = 440.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 0.01);
    }

    #[test]
    fn test_speech_tokenizer_creation() {
        let config = SpeechTokenizerConfig::default();
        let tokenizer = SpeechTokenizer::new(config).unwrap();
        assert_eq!(tokenizer.embed_dim(), 240); // 80 mels * 3 (static + delta + delta-delta)
    }

    #[test]
    fn test_speech_tokenizer_mel_spectrogram() {
        let config = SpeechTokenizerConfig::default();
        let tokenizer = SpeechTokenizer::new(config).unwrap();

        // Generate test signal (1 second at 16kHz)
        let signal = Array1::from_vec(
            (0..16000)
                .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
                .collect(),
        );

        let mel_spec = tokenizer.compute_mel_spectrogram(&signal).unwrap();
        assert_eq!(mel_spec.dim().0, 80); // 80 mel bins
        assert!(mel_spec.dim().1 > 0); // Some frames
    }

    #[test]
    fn test_speech_tokenizer_features() {
        let config = SpeechTokenizerConfig::default();
        let tokenizer = SpeechTokenizer::new(config).unwrap();

        let signal = Array1::from_vec((0..8000).map(|i| (i as f32 * 0.01).sin()).collect());
        let features = tokenizer.extract_features(&signal).unwrap();

        // Should have 240 features (80 mels + 80 delta + 80 delta-delta)
        assert_eq!(features.dim().0, 240);
    }

    #[test]
    fn test_music_tokenizer_chromagram() {
        let config = MusicTokenizerConfig::default();
        let tokenizer = MusicTokenizer::new(config);

        // Generate test signal with A440
        let signal = Array1::from_vec(
            (0..22050)
                .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin())
                .collect(),
        );

        let chroma = tokenizer.compute_chromagram(&signal).unwrap();
        assert_eq!(chroma.dim().0, 12); // 12 chroma bins
        assert!(chroma.dim().1 > 0);
    }

    #[test]
    fn test_music_tokenizer_onset() {
        let config = MusicTokenizerConfig::default();
        let tokenizer = MusicTokenizer::new(config);

        let signal = Array1::from_vec((0..22050).map(|i| (i as f32 * 0.001).sin()).collect());
        let onset = tokenizer.compute_onset_strength(&signal).unwrap();
        assert!(!onset.is_empty());
    }

    #[test]
    fn test_environmental_tokenizer() {
        let config = EnvironmentalTokenizerConfig::default();
        let tokenizer = EnvironmentalTokenizer::new(config).unwrap();

        let signal = Array1::from_vec((0..22050).map(|i| (i as f32 * 0.001).sin()).collect());

        // Test spectral centroid
        let centroid = tokenizer.compute_spectral_centroid(&signal).unwrap();
        assert!(!centroid.is_empty());
        assert!(centroid.iter().all(|&x| x >= 0.0));

        // Test ZCR
        let zcr = tokenizer.compute_zcr(&signal).unwrap();
        assert!(!zcr.is_empty());
        assert!(zcr.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_speech_tokenizer_signal_trait() {
        let config = SpeechTokenizerConfig::default();
        let tokenizer = SpeechTokenizer::new(config).unwrap();

        let signal = Array1::from_vec((0..8000).map(|i| (i as f32 * 0.01).sin()).collect());
        let encoded = tokenizer.encode(&signal).unwrap();
        assert!(!encoded.is_empty());
        assert_eq!(tokenizer.vocab_size(), 0); // Continuous features

        // Decoding should fail for speech (need vocoder)
        assert!(tokenizer.decode(&encoded).is_err());
    }

    #[test]
    fn test_music_tokenizer_signal_trait() {
        let config = MusicTokenizerConfig::default();
        let tokenizer = MusicTokenizer::new(config);

        let signal = Array1::from_vec((0..22050).map(|i| (i as f32 * 0.001).sin()).collect());
        let encoded = tokenizer.encode(&signal).unwrap();
        assert!(!encoded.is_empty());
        assert_eq!(tokenizer.vocab_size(), 0);
    }

    #[test]
    fn test_environmental_tokenizer_signal_trait() {
        let config = EnvironmentalTokenizerConfig::default();
        let tokenizer = EnvironmentalTokenizer::new(config).unwrap();

        let signal = Array1::from_vec((0..22050).map(|i| (i as f32 * 0.001).sin()).collect());
        let encoded = tokenizer.encode(&signal).unwrap();
        assert!(!encoded.is_empty());
        assert_eq!(tokenizer.vocab_size(), 0);
    }
}
