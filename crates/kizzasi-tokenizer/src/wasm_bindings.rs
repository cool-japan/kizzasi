//! WebAssembly bindings for kizzasi-tokenizer
//!
//! Exposes core tokenizer functionality to JavaScript/TypeScript via
//! wasm-bindgen. All items in this module are gated behind the `wasm`
//! feature flag and are only meaningful when compiling for the
//! `wasm32-unknown-unknown` target.
//!
//! ## Usage from JavaScript
//!
//! ```javascript
//! import init, { WasmLinearQuantizer, WasmMuLawCodec } from 'kizzasi-tokenizer';
//!
//! await init();
//!
//! const q = new WasmLinearQuantizer(-1.0, 1.0, 8);
//! const code = q.encode(0.5);
//! const value = q.decode(code);
//!
//! const codec = new WasmMuLawCodec(8);
//! const muCode = codec.encode(0.5);
//! const muValue = codec.decode(muCode);
//! ```

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// Bring the Quantizer trait into scope so its methods are callable on LinearQuantizer
#[cfg(feature = "wasm")]
use crate::quantizer::Quantizer as _;

/// JavaScript-callable linear uniform quantizer.
///
/// Wraps [`crate::LinearQuantizer`] and exposes encode/decode operations
/// to JavaScript. Quantizes continuous floating-point values into discrete
/// integer codes using uniform step sizes.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmLinearQuantizer {
    inner: crate::LinearQuantizer,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmLinearQuantizer {
    /// Create a new linear quantizer over the range `[min, max]` with `bits` resolution.
    ///
    /// # Errors
    /// Returns a `JsValue` error string if `min >= max` or `bits` is not in `1..=16`.
    #[wasm_bindgen(constructor)]
    pub fn new(min: f32, max: f32, bits: u32) -> Result<WasmLinearQuantizer, JsValue> {
        let bits_u8 =
            u8::try_from(bits).map_err(|_| JsValue::from_str("bits must fit in a u8 (0–255)"))?;
        crate::LinearQuantizer::new(min, max, bits_u8)
            .map(|inner| WasmLinearQuantizer { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Quantize a continuous `value` and return its discrete integer code.
    pub fn encode(&self, value: f32) -> u32 {
        // quantize returns i32 clamped to [0, levels-1]; safe to cast unsigned.
        self.inner.quantize(value).max(0) as u32
    }

    /// Reconstruct the continuous value from a discrete integer `code`.
    pub fn decode(&self, code: u32) -> f32 {
        self.inner.dequantize(code as i32)
    }

    /// Return the number of quantization levels (2^bits).
    pub fn num_levels(&self) -> u32 {
        self.inner.num_levels() as u32
    }

    /// Return the quantization step size.
    pub fn step_size(&self) -> f32 {
        self.inner.step_size()
    }
}

/// JavaScript-callable μ-law (mu-law) companding codec.
///
/// Wraps [`crate::MuLawCodec`] and exposes encode/decode operations
/// to JavaScript. Uses logarithmic quantization suited for audio signals,
/// preserving dynamic range better than linear quantization for quiet sounds.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmMuLawCodec {
    inner: crate::MuLawCodec,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmMuLawCodec {
    /// Create a new μ-law codec with the given bit depth.
    ///
    /// The μ parameter is automatically set to `(2^bits) - 1` (e.g. 255 for 8-bit).
    ///
    /// # Errors
    /// Returns a `JsValue` error if `bits` cannot be represented as a `u8`,
    /// or is outside the valid range `1..=16` (a JS caller can request any
    /// `u8`, and an out-of-range bit depth would otherwise silently produce
    /// a codec with a different bit depth than requested).
    #[wasm_bindgen(constructor)]
    pub fn new(bits: u32) -> Result<WasmMuLawCodec, JsValue> {
        let bits_u8 =
            u8::try_from(bits).map_err(|_| JsValue::from_str("bits must fit in a u8 (0–255)"))?;
        crate::MuLawCodec::try_new(bits_u8)
            .map(|inner| WasmMuLawCodec { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Quantize a continuous audio `sample` in `[-1.0, 1.0]` and return its integer code.
    pub fn encode(&self, sample: f32) -> u32 {
        self.inner.quantize(sample).max(0) as u32
    }

    /// Reconstruct the continuous audio sample from a discrete integer `code`.
    pub fn decode(&self, code: u32) -> f32 {
        self.inner.dequantize(code as i32)
    }

    /// Return the μ parameter used by this codec.
    pub fn mu(&self) -> f32 {
        self.inner.mu()
    }

    /// Return the number of quantization levels (2^bits).
    pub fn num_levels(&self) -> u32 {
        // `levels()` is computed once from a validated `bits` at
        // construction time (see `MuLawCodec::try_new`); re-deriving it here
        // via a fresh `1 << bits` shift would risk overflow if `bits` were
        // ever unvalidated.
        self.inner.levels() as u32
    }
}

// No `#[cfg(test)] mod tests` here: this binding is a thin, compiler-checked
// pass-through to `MuLawCodec::try_new`/`levels()` (both already covered by
// `mulaw.rs`'s `test_mulaw_try_new_rejects_invalid_bits` and friends), and
// constructing a `wasm_bindgen`-wrapped error value natively is unsafe
// outside a real `wasm-bindgen-test` harness — `JsValue::from_str` (used in
// the `Err` path above) calls into `wasm_bindgen::__wbindgen_describe`,
// which is compiled to abort the process (`SIGABRT`, not a catchable panic)
// when there is no JS glue registered to answer it, as there isn't under
// plain `cargo test`/`cargo nextest`. This was confirmed empirically while
// preparing this fix: a native `#[test]` calling
// `WasmMuLawCodec::new(0).is_err()` aborts the whole test process rather
// than returning `Err`. Exercising this module's actual JS-facing behavior
// requires `wasm-bindgen-test` + `wasm-bindgen-test-runner` against a
// `wasm32-unknown-unknown` build, which this crate does not currently set
// up; `cargo check --all-features` still type-checks every function here on
// every run.
