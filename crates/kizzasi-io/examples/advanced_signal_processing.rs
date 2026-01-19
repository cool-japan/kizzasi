//! Advanced signal processing example
//!
//! Demonstrates state-of-the-art signal processing capabilities:
//! - Adaptive filtering (Kalman, LMS, RLS)
//! - Cepstral analysis and pitch detection
//! - Advanced time-frequency analysis

use kizzasi_io::{
    CepstralDistance, ChoiWilliams, ComplexCepstrum, FormantTracker, GaborTransform, KalmanFilter,
    LmsFilter, NlmsFilter, RealCepstrum, ReassignedSpectrogram, STransform, SignalGenerator,
    SineGenerator, WignerVille, WindowType,
};
use scirs2_core::ndarray::{arr1, s, Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Signal Processing Demo ===\n");

    // Generate test signal: sine wave + noise
    let sample_rate = 16000.0;
    let duration = 1.0;
    let n_samples = (sample_rate * duration) as usize;

    let mut sine_gen = SineGenerator::new(440.0, 0.5, sample_rate);
    let signal = sine_gen.generate(n_samples);

    println!(
        "Generated signal: {} samples at {} Hz",
        n_samples, sample_rate
    );

    // 1. Kalman Filter Demo
    println!("\n--- Kalman Filter Demo ---");
    demo_kalman_filter()?;

    // 2. Adaptive LMS Filter Demo
    println!("\n--- Adaptive LMS Filter Demo ---");
    demo_lms_filter(&signal)?;

    // 3. Cepstral Analysis Demo
    println!("\n--- Cepstral Analysis Demo ---");
    demo_cepstral_analysis(&signal, sample_rate)?;

    // 4. Time-Frequency Analysis Demo
    println!("\n--- Time-Frequency Analysis Demo ---");
    demo_timefreq_analysis(&signal, sample_rate)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demo_kalman_filter() -> Result<(), Box<dyn std::error::Error>> {
    // Simple 1D position tracking example
    println!("Tracking position with noisy measurements...");

    let initial_state = arr1(&[0.0]);
    let initial_cov = Array2::from_shape_vec((1, 1), vec![1.0])?;
    let transition = Array2::from_shape_vec((1, 1), vec![1.0])?; // x(k+1) = x(k)
    let observation = Array2::from_shape_vec((1, 1), vec![1.0])?; // z(k) = x(k)
    let process_noise = Array2::from_shape_vec((1, 1), vec![0.01])?;
    let measurement_noise = Array2::from_shape_vec((1, 1), vec![0.1])?;

    let mut kf = KalmanFilter::new(
        initial_state,
        initial_cov,
        transition,
        observation,
        process_noise,
        measurement_noise,
    )?;

    // Simulate measurements with noise
    let true_position = 10.0;
    for i in 0..10 {
        let noise = (i as f32 * 0.1).sin() * 0.5; // Simulated noise
        let measurement = arr1(&[true_position + noise]);

        kf.predict();
        kf.update(&measurement)?;

        println!(
            "  Step {}: Measurement = {:.3}, Estimate = {:.3}",
            i + 1,
            measurement[0],
            kf.state()[0]
        );
    }

    println!("  Final estimate: {:.3}", kf.state()[0]);
    println!("  True value: {:.3}", true_position);

    Ok(())
}

fn demo_lms_filter(signal: &Array1<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Training adaptive LMS filter for system identification...");

    let mut lms = LmsFilter::new(8, 0.01)?;
    let mut nlms = NlmsFilter::new(8, 0.5, None)?;

    // Simulate a system to identify (simple delay + scale)
    let mut errors_lms = Vec::new();
    let mut errors_nlms = Vec::new();

    for i in 3..100.min(signal.len()) {
        let input = signal[i];
        let desired = signal[i - 3] * 0.5; // System: delay by 3, scale by 0.5

        let (_, error_lms) = lms.adapt(input, desired);
        let (_, error_nlms) = nlms.adapt(input, desired);

        errors_lms.push(error_lms.abs());
        errors_nlms.push(error_nlms.abs());
    }

    // Calculate average error over last 20 samples
    let avg_error_lms: f32 = errors_lms.iter().rev().take(20).sum::<f32>() / 20.0;
    let avg_error_nlms: f32 = errors_nlms.iter().rev().take(20).sum::<f32>() / 20.0;

    println!("  LMS final average error: {:.6}", avg_error_lms);
    println!("  NLMS final average error: {:.6}", avg_error_nlms);
    println!(
        "  NLMS converged {} faster",
        avg_error_lms / avg_error_nlms.max(1e-10)
    );

    Ok(())
}

fn demo_cepstral_analysis(
    signal: &Array1<f32>,
    sample_rate: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Performing cepstral analysis for pitch detection...");

    // Use a frame of the signal
    let frame_size = 512;
    let frame = signal.slice(s![..frame_size.min(signal.len())]).to_owned();

    // Real cepstrum for pitch detection
    let mut real_cep = RealCepstrum::new();
    let cepstrum = real_cep.compute(&frame)?;

    println!("  Computed real cepstrum: {} coefficients", cepstrum.len());

    // Detect pitch
    if let Some(period) = real_cep.detect_pitch(&cepstrum, sample_rate, 80.0, 400.0) {
        let f0 = real_cep.period_to_frequency(period, sample_rate);
        println!(
            "  Detected pitch: {:.1} Hz (period: {:.1} samples)",
            f0, period
        );
    } else {
        println!("  No clear pitch detected");
    }

    // Liftering
    let _liftered = real_cep.lifter(&cepstrum, 22.0);
    println!("  Applied liftering with coefficient 22.0");

    // Formant tracking
    let mut formant_tracker = FormantTracker::new(sample_rate, None);
    if let Ok(formants) = formant_tracker.estimate_formants(&frame, 3) {
        println!("  Estimated formants:");
        for (i, &f) in formants.iter().enumerate() {
            println!("    F{}: {:.1} Hz", i + 1, f);
        }
    }

    // Cepstral distance
    let cep1 = real_cep.compute(&frame)?;
    let shifted_frame = signal
        .slice(s![10..(frame_size + 10).min(signal.len())])
        .to_owned();
    let cep2 = real_cep.compute(&shifted_frame)?;

    let distance = CepstralDistance::compute(&cep1, &cep2)?;
    println!("  Cepstral distance between frames: {:.4}", distance);

    // Complex cepstrum
    let mut complex_cep = ComplexCepstrum::new();
    let complex_cepstrum = complex_cep.compute(&frame)?;
    println!(
        "  Complex cepstrum computed: {} coefficients",
        complex_cepstrum.len()
    );

    Ok(())
}

fn demo_timefreq_analysis(
    signal: &Array1<f32>,
    sample_rate: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Performing advanced time-frequency analysis...");

    let window_size = 256;
    let hop_size = 128;

    // Gabor Transform
    let mut gabor = GaborTransform::new(sample_rate);
    let gabor_result = gabor.compute(signal, window_size, hop_size, 32.0)?;

    println!(
        "  Gabor transform: {} frames × {} bins",
        gabor_result.num_frames, gabor_result.num_bins
    );

    // Compute average magnitude
    let avg_mag: f32 = (0..gabor_result.num_frames)
        .map(|f| {
            (0..gabor_result.num_bins)
                .map(|b| gabor_result.magnitude(f, b))
                .sum::<f32>()
                / gabor_result.num_bins as f32
        })
        .sum::<f32>()
        / gabor_result.num_frames as f32;

    println!("  Average magnitude: {:.4}", avg_mag);

    // Inverse Gabor (reconstruction)
    let reconstructed = gabor.inverse(&gabor_result)?;
    println!("  Reconstructed signal: {} samples", reconstructed.len());

    // S-Transform (on smaller subset for speed)
    let small_signal = signal.slice(s![..64]).to_owned();
    let mut stransform = STransform::new(sample_rate);
    let s_result = stransform.compute(&small_signal, 1.0)?;

    println!(
        "  S-transform computed: {}×{} time-frequency matrix",
        s_result.n, s_result.n
    );

    // Wigner-Ville Distribution (on small signal)
    let mut wv = WignerVille::new(sample_rate);
    let wv_result = wv.compute(&small_signal)?;

    println!(
        "  Wigner-Ville: {} time points × {} frequency points",
        wv_result.n_time, wv_result.n_freq
    );

    // Choi-Williams (reduced cross-terms)
    let mut cw = ChoiWilliams::new(sample_rate, 0.5);
    let cw_result = cw.compute(&small_signal)?;

    println!(
        "  Choi-Williams: {} time points × {} frequency points",
        cw_result.n_time, cw_result.n_freq
    );

    // Reassigned Spectrogram
    let mut reassigned = ReassignedSpectrogram::new(sample_rate);
    let reassigned_result = reassigned.compute(signal, window_size, hop_size, WindowType::Hann)?;

    println!(
        "  Reassigned spectrogram: {} frames × {} bins",
        reassigned_result.num_frames, reassigned_result.num_bins
    );

    // Sample some values
    if reassigned_result.num_frames > 0 && reassigned_result.num_bins > 10 {
        let mid_frame = reassigned_result.num_frames / 2;
        let sample_val = reassigned_result.get(mid_frame, 10);
        println!(
            "  Sample value at frame {}, bin 10: {:.4}",
            mid_frame, sample_val
        );
    }

    Ok(())
}
