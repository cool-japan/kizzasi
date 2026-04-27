//! Signal processing module
//!
//! This module provides comprehensive signal processing capabilities including:
//! - Digital filters (FIR, IIR)
//! - Adaptive filters (Kalman, LMS, RLS, NLMS, Particle)
//! - Cepstral analysis (Real/Complex cepstrum, formants, pitch detection)
//! - Time-frequency analysis (Gabor, S-transform, Wigner-Ville, Choi-Williams, Reassigned)
//! - Blind source separation (FastICA, NMF, PCA, Temporal decorrelation)
//! - Multi-channel beamforming (Delay-and-Sum, MVDR, adaptive)
//! - Hilbert-Huang Transform (EMD, EEMD, instantaneous frequency)
//! - Signal quality metrics (PESQ, STOI, POLQA, MOS prediction)
//! - Wavelet transforms
//! - Resampling and sample rate conversion
//! - Spectral analysis (FFT, spectrograms, MFCCs)
//! - Signal processing utilities

pub mod adaptive;
pub mod beamforming;
pub mod cepstral;
pub mod filters;
pub mod functions;
pub mod hht;
pub mod ml;
pub mod processor;
pub mod quality;
pub mod resamplers;
pub mod separation;
pub mod signalprocessor_traits;
pub mod spectral;
pub mod timefreq;
pub mod wavelets;

// Re-export all types
pub use adaptive::{KalmanFilter, LmsFilter, NlmsFilter, ParticleFilter, RlsFilter};
pub use beamforming::{AdaptiveBeamformer, DOAEstimator, DelayAndSum, MicrophoneArray, MVDR};
pub use cepstral::{
    CepstralDistance, ComplexCepstrum, FormantTracker, QuefrencyFilter, RealCepstrum,
};
pub use filters::{Filter, FirFilter, IirFilter};
pub use hht::{
    EmdConfig, EmdResult, EmpiricalModeDecomposition, EnsembleEmd, IntrinsicModeFunction,
};
#[allow(unused_imports)]
pub use ml::{AnomalyDetector, AnomalyMethod, FeatureExtractor, MiniAutoencoder, SignalDenoiser};
pub use processor::SignalProcessor;
pub use quality::{
    MosPredictor, PesqCalculator, PolqaCalculator, QualityMetrics, SnrCalculator, StoiCalculator,
};
pub use resamplers::{
    ArbitrarySrcResampler, CubicResampler, FarrowResampler, LinearResampler, RatioModulation,
    Resampler, SincStreamingResampler, StreamingResampler, TimeVaryingResampler,
};
pub use separation::{FastICA, Nonlinearity, TemporalDecorrelation, NMF, PCA};
pub use spectral::{Spectrogram, WindowType};
pub use timefreq::{
    ChoiWilliams, GaborResult, GaborTransform, ReassignedResult, ReassignedSpectrogram, STransform,
    STransformResult, WignerVille, WignerVilleResult,
};
pub use wavelets::{DwtMultiLevel, DwtResult, WaveletAnalyzer, WaveletType};
