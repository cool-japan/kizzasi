//! Small shared helpers for converting `ndarray` arrays to contiguous
//! slices without silently discarding data.
//!
//! `Array1::as_slice()` returns `None` when the array is not contiguous
//! (e.g. a strided owned array produced by `slice_move(s![..;2])`, which
//! keeps its stride rather than compacting). Several modules in this crate
//! previously handled that with `arr.as_slice().unwrap_or(&[])` — silently
//! substituting an **empty** slice for a same-length, merely non-contiguous
//! array. Every `ViolationComputable::check`/`violation` implementation in
//! this crate treats a dimension index at or beyond the input length as
//! vacuously satisfied / zero violation (see the doc comment on
//! [`crate::constraint::ViolationComputable`]), so that substitution turned
//! "the array happens to be strided" into "every constraint reports
//! satisfied" — an active safety constraint silently reporting satisfied.

use scirs2_core::ndarray::Array1;
use std::borrow::Cow;

/// Returns the contents of `arr` as a contiguous slice, borrowing when
/// possible and copying into an owned buffer only when `arr` is not already
/// contiguous. Never substitutes an empty slice for a non-empty array — the
/// returned `Cow` is always the same length as `arr`.
pub(crate) fn contiguous(arr: &Array1<f32>) -> Cow<'_, [f32]> {
    match arr.as_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(arr.iter().copied().collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::s;

    #[test]
    fn test_contiguous_borrows_when_already_contiguous() {
        let arr = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let out = contiguous(&arr);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(&*out, &[1.0, 2.0, 3.0]);
    }

    /// Regression (finding 138): a strided owned array must round-trip its
    /// full contents, not collapse to an empty slice.
    #[test]
    fn test_contiguous_copies_strided_array_without_losing_data() {
        let source = Array1::from_vec(vec![0.0f32, 10.0, 1.0, 11.0, 2.0, 12.0]);
        let strided = source.slice_move(s![..;2]);
        assert_eq!(strided.as_slice(), None, "fixture must be non-contiguous");

        let out = contiguous(&strided);
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(&*out, &[0.0, 1.0, 2.0]);
        assert_eq!(out.len(), strided.len());
    }
}
