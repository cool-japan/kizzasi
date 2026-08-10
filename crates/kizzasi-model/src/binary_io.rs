//! Binary read primitives shared by GGUF and other binary loaders.
//!
//! All helpers take a byte slice and a mutable cursor `pos: &mut usize`,
//! advancing the cursor by the number of bytes consumed and returning the
//! decoded value. On buffer underflow they return [`ModelError::LoadError`]
//! (via [`ModelError::simple_load_error`]).
//!
//! Values are little-endian, matching the GGUF on-disk format.

use crate::error::{ModelError, ModelResult};

/// Read a single little-endian `u8` from `buf` at `*pos`, advancing `pos` by 1.
pub(crate) fn read_u8(buf: &[u8], pos: &mut usize) -> ModelResult<u8> {
    if *pos >= buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading u8 at position {}",
            pos
        )));
    }
    let v = buf[*pos];
    *pos += 1;
    Ok(v)
}

/// Read a little-endian `u16` from `buf` at `*pos`, advancing `pos` by 2.
pub(crate) fn read_u16_le(buf: &[u8], pos: &mut usize) -> ModelResult<u16> {
    let end = *pos + 2;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading u16 at position {}",
            pos
        )));
    }
    let v = u16::from_le_bytes([buf[*pos], buf[*pos + 1]]);
    *pos = end;
    Ok(v)
}

/// Read a little-endian `u32` from `buf` at `*pos`, advancing `pos` by 4.
pub(crate) fn read_u32_le(buf: &[u8], pos: &mut usize) -> ModelResult<u32> {
    let end = *pos + 4;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading u32 at position {}",
            pos
        )));
    }
    let v = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
    *pos = end;
    Ok(v)
}

/// Read a little-endian `u64` from `buf` at `*pos`, advancing `pos` by 8.
pub(crate) fn read_u64_le(buf: &[u8], pos: &mut usize) -> ModelResult<u64> {
    let end = *pos + 8;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading u64 at position {}",
            pos
        )));
    }
    let v = u64::from_le_bytes([
        buf[*pos],
        buf[*pos + 1],
        buf[*pos + 2],
        buf[*pos + 3],
        buf[*pos + 4],
        buf[*pos + 5],
        buf[*pos + 6],
        buf[*pos + 7],
    ]);
    *pos = end;
    Ok(v)
}

/// Read a single signed byte from `buf`.
pub(crate) fn read_i8(buf: &[u8], pos: &mut usize) -> ModelResult<i8> {
    read_u8(buf, pos).map(|v| v as i8)
}

/// Read a little-endian `i16` from `buf`.
pub(crate) fn read_i16_le(buf: &[u8], pos: &mut usize) -> ModelResult<i16> {
    read_u16_le(buf, pos).map(|v| v as i16)
}

/// Read a little-endian `i32` from `buf`.
pub(crate) fn read_i32_le(buf: &[u8], pos: &mut usize) -> ModelResult<i32> {
    read_u32_le(buf, pos).map(|v| v as i32)
}

/// Read a little-endian `i64` from `buf`.
pub(crate) fn read_i64_le(buf: &[u8], pos: &mut usize) -> ModelResult<i64> {
    read_u64_le(buf, pos).map(|v| v as i64)
}

/// Read a little-endian IEEE-754 `f32` from `buf`.
pub(crate) fn read_f32_le(buf: &[u8], pos: &mut usize) -> ModelResult<f32> {
    read_u32_le(buf, pos).map(f32::from_bits)
}

/// Read a little-endian IEEE-754 `f64` from `buf`.
pub(crate) fn read_f64_le(buf: &[u8], pos: &mut usize) -> ModelResult<f64> {
    read_u64_le(buf, pos).map(f64::from_bits)
}

/// Read a single boolean byte from `buf`. Any non-zero value is `true`.
pub(crate) fn read_bool(buf: &[u8], pos: &mut usize) -> ModelResult<bool> {
    read_u8(buf, pos).map(|v| v != 0)
}

/// Read a GGUF string using v2+ encoding (u64 length prefix).
pub(crate) fn read_string_v2(buf: &[u8], pos: &mut usize) -> ModelResult<String> {
    let len = read_u64_le(buf, pos)? as usize;
    let end = *pos + len;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading string of length {} at position {}",
            len, pos
        )));
    }
    let s = std::str::from_utf8(&buf[*pos..end]).map_err(|e| {
        ModelError::simple_load_error(format!("Invalid UTF-8 in GGUF string: {}", e))
    })?;
    let owned = s.to_owned();
    *pos = end;
    Ok(owned)
}

/// Read a GGUF string using v1 encoding (u32 length prefix).
pub(crate) fn read_string_v1(buf: &[u8], pos: &mut usize) -> ModelResult<String> {
    let len = read_u32_le(buf, pos)? as usize;
    let end = *pos + len;
    if end > buf.len() {
        return Err(ModelError::simple_load_error(format!(
            "Buffer underflow reading v1 string of length {} at position {}",
            len, pos
        )));
    }
    let s = std::str::from_utf8(&buf[*pos..end]).map_err(|e| {
        ModelError::simple_load_error(format!("Invalid UTF-8 in GGUF v1 string: {}", e))
    })?;
    let owned = s.to_owned();
    *pos = end;
    Ok(owned)
}

/// Advance `*offset` to the next `alignment`-byte aligned boundary.
///
/// `alignment` must be a power of two; the function uses the standard
/// `(offset + alignment - 1) & !(alignment - 1)` round-up trick. Passing
/// `alignment = 0` is a programmer error and will panic in debug builds.
pub(crate) fn align_offset(offset: &mut usize, alignment: usize) {
    debug_assert!(alignment > 0, "alignment must be non-zero");
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two, got {}",
        alignment
    );
    let mask = alignment - 1;
    *offset = (*offset + mask) & !mask;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_u32_le_roundtrip() {
        // Encode a known value and decode it back through `read_u32_le`,
        // verifying the cursor advances exactly four bytes.
        let value: u32 = 0xDEAD_BEEF;
        let bytes = value.to_le_bytes();
        let mut pos = 0usize;
        let got = read_u32_le(&bytes, &mut pos).expect("read should succeed");
        assert_eq!(got, value);
        assert_eq!(pos, 4);
    }

    #[test]
    fn test_align_offset_various() {
        // 32-byte alignment (GGUF data section uses 32-byte padding).
        let mut o = 0usize;
        align_offset(&mut o, 32);
        assert_eq!(o, 0, "already aligned");

        let mut o = 1usize;
        align_offset(&mut o, 32);
        assert_eq!(o, 32, "round up from 1 → 32");

        let mut o = 31usize;
        align_offset(&mut o, 32);
        assert_eq!(o, 32, "round up from 31 → 32");

        let mut o = 33usize;
        align_offset(&mut o, 32);
        assert_eq!(o, 64, "round up from 33 → 64");

        // 8-byte alignment.
        let mut o = 9usize;
        align_offset(&mut o, 8);
        assert_eq!(o, 16, "round up from 9 → 16 with 8-byte alignment");

        // 1-byte alignment is a no-op for every input.
        let mut o = 7usize;
        align_offset(&mut o, 1);
        assert_eq!(o, 7);
    }

    #[test]
    fn test_read_string_v2_small() {
        // Build a v2 length-prefixed string ("hi") and round-trip it.
        let s = "hi";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());

        let mut pos = 0usize;
        let got = read_string_v2(&buf, &mut pos).expect("read should succeed");
        assert_eq!(got, s);
        // 8 bytes of length prefix + 2 bytes of content.
        assert_eq!(pos, 8 + s.len());
    }

    #[test]
    fn test_read_underflow_returns_error() {
        // Three-byte buffer cannot hold a u32; reading must error.
        let buf = [1u8, 2, 3];
        let mut pos = 0usize;
        assert!(read_u32_le(&buf, &mut pos).is_err());
    }
}
