//! Wavelet transforms and analysis
//!
//! This module provides wavelet-based signal processing capabilities including:
//! - Discrete Wavelet Transform (DWT)
//! - Multi-level wavelet decomposition
//! - Wavelet denoising
//! - Stationary Wavelet Transform (SWT)

/// Discrete Wavelet Transform (DWT) result
#[derive(Debug, Clone)]
pub struct DwtResult {
    /// Approximation coefficients (low-frequency)
    pub approximation: Vec<f32>,
    /// Detail coefficients (high-frequency)
    pub detail: Vec<f32>,
    /// Wavelet type used
    pub wavelet: WaveletType,
    /// Decomposition level
    pub level: usize,
}

impl DwtResult {
    /// Get total number of coefficients
    pub fn total_coefficients(&self) -> usize {
        self.approximation.len() + self.detail.len()
    }
}

/// Multi-level DWT result
#[derive(Debug, Clone)]
pub struct DwtMultiLevel {
    /// Approximation coefficients at the coarsest level
    pub approximation: Vec<f32>,
    /// Detail coefficients at each level (from finest to coarsest)
    pub details: Vec<Vec<f32>>,
    /// Wavelet type used
    pub wavelet: WaveletType,
    /// Number of decomposition levels
    pub levels: usize,
    /// Original signal length
    pub original_length: usize,
    /// Signal lengths at each level (for reconstruction)
    pub lengths: Vec<usize>,
}

/// Wavelet types for wavelet transforms
#[derive(Debug, Clone, Copy)]
pub enum WaveletType {
    /// Haar wavelet (simplest, good for edge detection)
    Haar,
    /// Daubechies-2 (db2) wavelet
    Daubechies2,
    /// Daubechies-4 (db4) wavelet
    Daubechies4,
    /// Daubechies-6 (db6) wavelet
    Daubechies6,
    /// Symlet-2 (sym2) wavelet
    Symlet2,
    /// Symlet-4 (sym4) wavelet
    Symlet4,
    /// Coiflet-1 (coif1) wavelet
    Coiflet1,
}

impl WaveletType {
    /// Get the low-pass decomposition filter coefficients (scaling function)
    pub fn decomposition_low(&self) -> Vec<f32> {
        match self {
            WaveletType::Haar => vec![0.707_106_77, 0.707_106_77],
            WaveletType::Daubechies2 => {
                vec![0.482_962_9, 0.836_516_3, 0.224_143_9, -0.129_409_5]
            }
            WaveletType::Daubechies4 => {
                vec![
                    0.230_377_8,
                    0.714_846_6,
                    0.630_880_8,
                    -0.027_983_77,
                    -0.187_034_8,
                    0.030_841_38,
                    0.032_883_0,
                    -0.010_597_4,
                ]
            }
            WaveletType::Daubechies6 => {
                vec![
                    0.111_540_7,
                    0.494_623_9,
                    0.751_133_9,
                    0.315_250_4,
                    -0.226_264_7,
                    -0.129_766_9,
                    0.097_501_6,
                    0.027_522_87,
                    -0.031_582_0,
                    0.000_553_84,
                    0.004_777_26,
                    -0.001_077_3,
                ]
            }
            WaveletType::Symlet2 => {
                vec![-0.129_409_5, 0.224_143_9, 0.836_516_3, 0.482_962_9]
            }
            WaveletType::Symlet4 => {
                vec![
                    -0.075_765_7,
                    -0.029_635_53,
                    0.497_618_7,
                    0.803_738_8,
                    0.297_857_8,
                    -0.099_219_5,
                    -0.012_604_0,
                    0.032_223_1,
                ]
            }
            WaveletType::Coiflet1 => {
                vec![
                    -0.015_655_73,
                    -0.072_732_6,
                    0.384_864_9,
                    0.852_572,
                    0.337_897_7,
                    -0.072_732_6,
                ]
            }
        }
    }

    /// Get the high-pass decomposition filter coefficients (wavelet function)
    pub fn decomposition_high(&self) -> Vec<f32> {
        let low = self.decomposition_low();
        let n = low.len();
        low.iter()
            .enumerate()
            .map(|(i, _)| if i % 2 == 0 { -1.0 } else { 1.0 } * low[n - 1 - i])
            .collect()
    }

    /// Get the low-pass reconstruction filter coefficients
    pub fn reconstruction_low(&self) -> Vec<f32> {
        self.decomposition_low().into_iter().rev().collect()
    }

    /// Get the high-pass reconstruction filter coefficients
    pub fn reconstruction_high(&self) -> Vec<f32> {
        self.decomposition_high().into_iter().rev().collect()
    }

    /// Get the filter length
    pub fn filter_length(&self) -> usize {
        self.decomposition_low().len()
    }
}

/// Wavelet analyzer for performing wavelet transforms
#[derive(Debug, Clone)]
pub struct WaveletAnalyzer {
    wavelet: WaveletType,
}

impl WaveletAnalyzer {
    /// Create a new wavelet analyzer
    pub fn new(wavelet: WaveletType) -> Self {
        Self { wavelet }
    }

    /// Perform single-level DWT decomposition using lifting scheme
    pub fn dwt(&self, signal: &[f32]) -> DwtResult {
        let n = signal.len();
        let out_len = n.div_ceil(2);

        let mut approx = Vec::with_capacity(out_len);
        let mut detail = Vec::with_capacity(out_len);

        match self.wavelet {
            WaveletType::Haar => {
                let sqrt2_inv = 1.0 / std::f32::consts::SQRT_2;
                for i in 0..out_len {
                    let idx0 = i * 2;
                    let idx1 = (idx0 + 1).min(n - 1);
                    let x0 = signal[idx0];
                    let x1 = signal[idx1];
                    approx.push((x0 + x1) * sqrt2_inv);
                    detail.push((x0 - x1) * sqrt2_inv);
                }
            }
            _ => {
                let low_filter = self.wavelet.decomposition_low();
                let high_filter = self.wavelet.decomposition_high();
                let f_len = low_filter.len();

                for i in 0..out_len {
                    let center = i * 2;
                    let mut sum_low = 0.0f32;
                    let mut sum_high = 0.0f32;

                    for j in 0..f_len {
                        let idx = (center + j) as isize - (f_len as isize / 2);
                        let val = self.extend_signal(signal, idx, n);
                        sum_low += val * low_filter[j];
                        sum_high += val * high_filter[j];
                    }

                    approx.push(sum_low);
                    detail.push(sum_high);
                }
            }
        }

        DwtResult {
            approximation: approx,
            detail,
            wavelet: self.wavelet,
            level: 1,
        }
    }

    /// Perform multi-level DWT decomposition
    pub fn dwt_multilevel(&self, signal: &[f32], levels: usize) -> DwtMultiLevel {
        let mut details = Vec::with_capacity(levels);
        let mut lengths = Vec::with_capacity(levels);
        let mut current = signal.to_vec();
        let original_length = signal.len();

        for _ in 0..levels {
            if current.len() < self.wavelet.filter_length() {
                break;
            }

            lengths.push(current.len());
            let result = self.dwt(&current);
            details.push(result.detail);
            current = result.approximation;
        }

        let num_levels = details.len();

        DwtMultiLevel {
            approximation: current,
            details,
            wavelet: self.wavelet,
            levels: num_levels,
            original_length,
            lengths,
        }
    }

    /// Perform inverse DWT (single level reconstruction) using lifting scheme
    pub fn idwt(&self, approx: &[f32], detail: &[f32], output_length: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; output_length];

        match self.wavelet {
            WaveletType::Haar => {
                let sqrt2_inv = 1.0 / std::f32::consts::SQRT_2;
                for (i, (&a, &d)) in approx.iter().zip(detail.iter()).enumerate() {
                    let idx0 = i * 2;
                    let idx1 = idx0 + 1;

                    if idx0 < output_length {
                        result[idx0] = (a + d) * sqrt2_inv;
                    }
                    if idx1 < output_length {
                        result[idx1] = (a - d) * sqrt2_inv;
                    }
                }
            }
            _ => {
                let low_filter = self.wavelet.reconstruction_low();
                let high_filter = self.wavelet.reconstruction_high();
                let f_len = low_filter.len();

                for (i, (&a, &d)) in approx.iter().zip(detail.iter()).enumerate() {
                    let pos = i * 2;
                    for j in 0..f_len {
                        let out_idx = pos as isize + j as isize - (f_len as isize / 2) + 1;
                        if out_idx >= 0 && (out_idx as usize) < output_length {
                            result[out_idx as usize] += a * low_filter[j] + d * high_filter[j];
                        }
                    }
                }
            }
        }

        result
    }

    /// Perform multi-level inverse DWT reconstruction
    pub fn idwt_multilevel(&self, decomp: &DwtMultiLevel) -> Vec<f32> {
        let mut current = decomp.approximation.clone();

        for (i, detail) in decomp.details.iter().enumerate().rev() {
            let output_len = if i < decomp.lengths.len() {
                decomp.lengths[i]
            } else {
                detail.len() * 2
            };
            current = self.idwt(&current, detail, output_len);
        }

        current.truncate(decomp.original_length);
        current
    }

    /// Convolve and downsample by 2 (for decomposition)
    #[allow(dead_code)]
    fn convolve_downsample(&self, signal: &[f32], filter: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let f_len = filter.len();
        let out_len = (n + f_len - 1) / 2;

        let mut result = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let center = i * 2;
            let mut sum = 0.0f32;

            for (j, &f) in filter.iter().enumerate() {
                let idx = center as isize + j as isize - (f_len as isize - 1);
                let val = self.extend_signal(signal, idx, n);
                sum += val * f;
            }

            result.push(sum);
        }

        result
    }

    /// Get signal value with symmetric extension
    fn extend_signal(&self, signal: &[f32], idx: isize, n: usize) -> f32 {
        if idx < 0 {
            signal[(-1 - idx) as usize % n]
        } else if idx >= n as isize {
            let reflected = 2 * n as isize - 2 - idx;
            if reflected >= 0 && (reflected as usize) < n {
                signal[reflected as usize]
            } else {
                signal[n - 1]
            }
        } else {
            signal[idx as usize]
        }
    }

    /// Upsample by 2 and convolve (for reconstruction)
    #[allow(dead_code)]
    fn upsample_convolve(&self, signal: &[f32], filter: &[f32], output_length: usize) -> Vec<f32> {
        let upsampled_len = signal.len() * 2;
        let mut upsampled = vec![0.0f32; upsampled_len];

        for (i, &s) in signal.iter().enumerate() {
            upsampled[i * 2] = s;
        }

        let f_len = filter.len();
        let mut result = vec![0.0; output_length];

        for (i, res) in result.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &f) in filter.iter().enumerate() {
                let idx = i as isize + j as isize - (f_len as isize - 1);
                if idx >= 0 && (idx as usize) < upsampled_len {
                    sum += upsampled[idx as usize] * f;
                }
            }
            *res = sum;
        }

        result
    }

    /// Compute stationary wavelet transform (SWT) - undecimated DWT
    /// Returns coefficients at full resolution for each level
    pub fn swt(&self, signal: &[f32], levels: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut results = Vec::with_capacity(levels);
        let mut low_filter = self.wavelet.decomposition_low();
        let mut high_filter = self.wavelet.decomposition_high();
        let mut current = signal.to_vec();

        for _ in 0..levels {
            let approx = self.convolve_full(&current, &low_filter);
            let detail = self.convolve_full(&current, &high_filter);

            results.push((approx.clone(), detail));
            current = approx;

            low_filter = self.upsample_filter(&low_filter);
            high_filter = self.upsample_filter(&high_filter);
        }

        results
    }

    /// Convolve without downsampling (for SWT)
    fn convolve_full(&self, signal: &[f32], filter: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let f_len = filter.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut sum = 0.0f32;
            for (j, &f) in filter.iter().enumerate() {
                let idx = i as isize + j as isize - (f_len as isize / 2);
                let val = if idx < 0 {
                    signal[(-idx - 1) as usize % n]
                } else if idx >= n as isize {
                    signal[(2 * n as isize - idx - 1) as usize % n]
                } else {
                    signal[idx as usize]
                };
                sum += val * f;
            }
            result.push(sum);
        }

        result
    }

    /// Upsample filter by inserting zeros
    fn upsample_filter(&self, filter: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(filter.len() * 2 - 1);
        for (i, &f) in filter.iter().enumerate() {
            result.push(f);
            if i < filter.len() - 1 {
                result.push(0.0);
            }
        }
        result
    }

    /// Compute wavelet energy at each level
    pub fn wavelet_energy(&self, decomp: &DwtMultiLevel) -> Vec<f32> {
        let mut energies = Vec::with_capacity(decomp.levels + 1);

        for detail in &decomp.details {
            let energy: f32 = detail.iter().map(|&x| x * x).sum();
            energies.push(energy);
        }

        let approx_energy: f32 = decomp.approximation.iter().map(|&x| x * x).sum();
        energies.push(approx_energy);

        let total: f32 = energies.iter().sum();
        if total > 0.0 {
            for e in &mut energies {
                *e /= total;
            }
        }

        energies
    }

    /// Denoise signal using wavelet thresholding
    pub fn denoise(&self, signal: &[f32], levels: usize, threshold: f32) -> Vec<f32> {
        let decomp = self.dwt_multilevel(signal, levels);

        let thresholded_details: Vec<Vec<f32>> = decomp
            .details
            .iter()
            .map(|detail| {
                detail
                    .iter()
                    .map(|&x| Self::soft_threshold(x, threshold))
                    .collect()
            })
            .collect();

        let thresholded = DwtMultiLevel {
            approximation: decomp.approximation,
            details: thresholded_details,
            wavelet: decomp.wavelet,
            levels: decomp.levels,
            original_length: decomp.original_length,
            lengths: decomp.lengths,
        };

        self.idwt_multilevel(&thresholded)
    }

    /// Soft thresholding function
    fn soft_threshold(x: f32, threshold: f32) -> f32 {
        if x.abs() <= threshold {
            0.0
        } else if x > 0.0 {
            x - threshold
        } else {
            x + threshold
        }
    }

    /// Estimate universal threshold (VisuShrink)
    pub fn universal_threshold(detail: &[f32]) -> f32 {
        let n = detail.len() as f32;
        let sigma = Self::mad_sigma(detail);
        sigma * (2.0 * n.ln()).sqrt()
    }

    /// Estimate noise standard deviation using MAD (Median Absolute Deviation)
    fn mad_sigma(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let mut abs_data: Vec<f32> = data.iter().map(|&x| x.abs()).collect();
        abs_data.sort_by(|a, b| a.total_cmp(b));

        let median = if abs_data.len().is_multiple_of(2) {
            (abs_data[abs_data.len() / 2 - 1] + abs_data[abs_data.len() / 2]) / 2.0
        } else {
            abs_data[abs_data.len() / 2]
        };

        median / 0.674_489_75
    }
}
