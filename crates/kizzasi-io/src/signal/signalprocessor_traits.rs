//! # SignalProcessor - Trait Implementations
//!
//! This module contains trait implementations for `SignalProcessor`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::processor::SignalProcessor;

impl std::fmt::Debug for SignalProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignalProcessor")
            .field("buffer_size", &self.buffer_size)
            .field("sample_rate", &self.sample_rate)
            .field("fft_planner", &"<FftPlanner>")
            .finish()
    }
}
