//! Type-safe wrappers for improved type-level safety
//!
//! This module provides newtype wrappers to prevent mixing up different
//! kinds of indices, dimensions, and other numeric values at the type level.

use std::fmt;

/// Newtype for codebook size to prevent confusion with other usize values
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CodebookSize(pub usize);

impl CodebookSize {
    /// Create a new codebook size
    pub fn new(size: usize) -> Self {
        Self(size)
    }

    /// Get the inner value
    pub fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for CodebookSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} codes", self.0)
    }
}

/// Newtype for embedding dimension to distinguish from signal length
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EmbedDim(pub usize);

impl EmbedDim {
    /// Create a new embedding dimension
    pub fn new(dim: usize) -> Self {
        Self(dim)
    }

    /// Get the inner value
    pub fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for EmbedDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}D", self.0)
    }
}

/// Newtype for signal length
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignalLength(pub usize);

impl SignalLength {
    /// Create a new signal length
    pub fn new(len: usize) -> Self {
        Self(len)
    }

    /// Get the inner value
    pub fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for SignalLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} samples", self.0)
    }
}

/// Newtype for batch size
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchSize(pub usize);

impl BatchSize {
    /// Create a new batch size
    pub fn new(size: usize) -> Self {
        Self(size)
    }

    /// Get the inner value
    pub fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for BatchSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "batch({})", self.0)
    }
}

/// Newtype for codebook index to prevent confusion with signal indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CodebookIndex(pub usize);

impl CodebookIndex {
    /// Create a new codebook index
    pub fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// Get the inner value
    pub fn get(self) -> usize {
        self.0
    }

    /// Check if index is valid for a given codebook size
    pub fn is_valid_for(&self, codebook_size: CodebookSize) -> bool {
        self.0 < codebook_size.get()
    }
}

impl fmt::Display for CodebookIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "code[{}]", self.0)
    }
}

/// Newtype for bit depth to ensure valid bit widths
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BitDepth(u8);

impl BitDepth {
    /// Create a new bit depth (validated to be 1-16 bits)
    pub fn new(bits: u8) -> Result<Self, String> {
        if bits == 0 || bits > 16 {
            return Err(format!("Bit depth must be 1-16, got {}", bits));
        }
        Ok(Self(bits))
    }

    /// Create an 8-bit depth (common for audio)
    pub fn bits_8() -> Self {
        Self(8)
    }

    /// Create a 16-bit depth (CD quality)
    pub fn bits_16() -> Self {
        Self(16)
    }

    /// Get the inner value
    pub fn get(self) -> u8 {
        self.0
    }

    /// Get the number of quantization levels (2^bits)
    pub fn num_levels(self) -> usize {
        1 << self.0
    }
}

impl fmt::Display for BitDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-bit", self.0)
    }
}

/// Newtype for learning rate to prevent confusion with other f32 values
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct LearningRate(pub f32);

impl LearningRate {
    /// Create a new learning rate (validated to be positive)
    pub fn new(rate: f32) -> Result<Self, String> {
        if rate <= 0.0 || !rate.is_finite() {
            return Err(format!(
                "Learning rate must be positive and finite, got {}",
                rate
            ));
        }
        Ok(Self(rate))
    }

    /// Common default learning rate (0.001)
    pub fn default_rate() -> Self {
        Self(0.001)
    }

    /// Get the inner value
    pub fn get(self) -> f32 {
        self.0
    }
}

impl fmt::Display for LearningRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lr={}", self.0)
    }
}

/// Newtype for number of training epochs
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epochs(pub usize);

impl Epochs {
    /// Create a new number of epochs
    pub fn new(epochs: usize) -> Self {
        Self(epochs)
    }

    /// Get the inner value
    pub fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for Epochs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} epochs", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_size() {
        let size = CodebookSize::new(256);
        assert_eq!(size.get(), 256);
        assert_eq!(format!("{}", size), "256 codes");
    }

    #[test]
    fn test_embed_dim() {
        let dim = EmbedDim::new(128);
        assert_eq!(dim.get(), 128);
        assert_eq!(format!("{}", dim), "128D");
    }

    #[test]
    fn test_codebook_index_validation() {
        let idx = CodebookIndex::new(10);
        let size = CodebookSize::new(256);
        assert!(idx.is_valid_for(size));

        let invalid_idx = CodebookIndex::new(300);
        assert!(!invalid_idx.is_valid_for(size));
    }

    #[test]
    fn test_bit_depth() {
        let bd8 = BitDepth::new(8).unwrap();
        assert_eq!(bd8.get(), 8);
        assert_eq!(bd8.num_levels(), 256);

        let bd16 = BitDepth::new(16).unwrap();
        assert_eq!(bd16.num_levels(), 65536);

        // Invalid bit depths
        assert!(BitDepth::new(0).is_err());
        assert!(BitDepth::new(17).is_err());
    }

    #[test]
    fn test_learning_rate() {
        let lr = LearningRate::new(0.01).unwrap();
        assert_eq!(lr.get(), 0.01);

        // Invalid learning rates
        assert!(LearningRate::new(0.0).is_err());
        assert!(LearningRate::new(-0.1).is_err());
        assert!(LearningRate::new(f32::NAN).is_err());
    }

    #[test]
    fn test_epochs() {
        let epochs = Epochs::new(100);
        assert_eq!(epochs.get(), 100);
        assert_eq!(format!("{}", epochs), "100 epochs");
    }
}
