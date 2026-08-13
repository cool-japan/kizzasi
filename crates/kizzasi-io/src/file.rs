//! File stream support
//!
//! Provides file-based data streams for WAV audio, CSV time series, and HDF5 datasets.
//!
//! ## Features
//! - WAV audio file reading/writing
//! - CSV time series reading/writing
//! - HDF5 dataset reading/writing
//! - Async I/O support
//! - Streaming large WAV files: `WavReader` decodes on demand via
//!   `read_chunk`/`read_all` instead of loading the whole file into memory
//!   up front. CSV (`CsvReader::read_array`) and HDF5
//!   (`Hdf5Reader::read_dataset_1d`/`read_dataset_2d`) currently read their
//!   entire dataset eagerly.
//!
//! ## Example
//! ```rust,no_run
//! use kizzasi_io::{WavReader, WavWriter, WavSpec};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Read WAV file
//!     let mut reader = WavReader::open("input.wav").await?;
//!     let samples = reader.read_all().await?;
//!
//!     // Write WAV file
//!     let spec = WavSpec {
//!         sample_rate: 44100,
//!         channels: 2,
//!         bits_per_sample: 16,
//!         ..Default::default()
//!     };
//!     let mut writer = WavWriter::create("output.wav", spec).await?;
//!     writer.write_samples(&samples).await?;
//!
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info};

// ============================================================================
// WAV Audio Files
// ============================================================================

/// WAV sample storage format: linear PCM integers, or IEEE-754 floating
/// point.
///
/// `WavReader` always reflects whichever format the file on disk actually
/// uses (via `spec().sample_format`, read from the file's header) regardless
/// of this type's default; `WavWriter` uses it to decide whether to write
/// scaled integers or raw `f32` values. A previous version had no such
/// field and hard-coded `hound::SampleFormat::Int` in `WavWriter::create`,
/// making it impossible to write a 32-bit float WAV file through this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SampleFormat {
    /// Linear PCM integers (8/16/24/32-bit), scaled from/to the `[-1.0,
    /// 1.0]` range used by this crate's `f32` sample buffers.
    #[default]
    Int,
    /// IEEE-754 32-bit floating point samples (`bits_per_sample` must be
    /// 32), written/read without integer scaling.
    Float,
}

impl From<SampleFormat> for hound::SampleFormat {
    fn from(value: SampleFormat) -> Self {
        match value {
            SampleFormat::Int => hound::SampleFormat::Int,
            SampleFormat::Float => hound::SampleFormat::Float,
        }
    }
}

impl From<hound::SampleFormat> for SampleFormat {
    fn from(value: hound::SampleFormat) -> Self {
        match value {
            hound::SampleFormat::Int => SampleFormat::Int,
            hound::SampleFormat::Float => SampleFormat::Float,
        }
    }
}

/// WAV file specification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WavSpec {
    /// Sample rate (Hz)
    pub sample_rate: u32,

    /// Number of channels
    pub channels: u16,

    /// Bits per sample (8, 16, 24, 32)
    pub bits_per_sample: u16,

    /// Sample storage format. Defaults to `Int` for backward compatibility;
    /// set to `Float` (with `bits_per_sample: 32`) to write/read 32-bit
    /// floating point WAV data.
    #[serde(default)]
    pub sample_format: SampleFormat,
}

impl Default for WavSpec {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 1,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        }
    }
}

/// Full-scale PCM integer magnitude for a given bit depth, used
/// symmetrically by both `WavReader` and `WavWriter` so encode/decode
/// cannot drift out of sync.
///
/// A previous version scaled by `2^(bits-1)` in both directions: for
/// `bits_per_sample = 16` a full-scale `1.0` sample produced `32768`,
/// outside the valid `i16` range `[-32768, 32767]` -- one past the
/// positive limit. `2^(bits-1) - 1` keeps `1.0` in range for every
/// supported depth. `bits_per_sample == 0` (which made `bits - 1` underflow
/// the `u16` and panic) is rejected here instead.
fn pcm_scale(bits_per_sample: u16) -> IoResult<f32> {
    match bits_per_sample {
        8 | 16 | 24 | 32 => Ok(((1i64 << (bits_per_sample - 1)) - 1) as f32),
        other => Err(IoError::ConfigError(format!(
            "Unsupported WAV bits_per_sample: {other} (expected 8, 16, 24, or 32)"
        ))),
    }
}

/// WAV file reader
///
/// Streams samples on demand from disk via `hound`'s borrowing sample
/// iterator (`hound::WavReader::samples`) instead of decoding the whole
/// file into memory in `open()`: a previous version eagerly collected
/// every sample into a `Vec<f32>` up front, so a multi-gigabyte WAV file
/// was fully resident despite the module doc claiming "streaming large
/// files".
pub struct WavReader {
    spec: WavSpec,
    sample_format: hound::SampleFormat,
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
    /// Interleaved values decoded so far, tracked independently of hound's
    /// own (private) internal position so `position()` keeps its original
    /// "values consumed" meaning.
    position: usize,
}

impl WavReader {
    /// Open WAV file for reading
    pub async fn open<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let path_ref = path.as_ref();
        let reader = hound::WavReader::open(path_ref)
            .map_err(|e| IoError::ReadFailed(format!("Failed to open WAV file: {}", e)))?;

        let spec = reader.spec();
        let wav_spec = WavSpec {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            bits_per_sample: spec.bits_per_sample,
            sample_format: spec.sample_format.into(),
        };

        info!(
            "WAV file opened: {} samples, {}Hz, {} channels",
            reader.len(),
            wav_spec.sample_rate,
            wav_spec.channels
        );

        Ok(Self {
            spec: wav_spec,
            sample_format: spec.sample_format,
            reader,
            position: 0,
        })
    }

    /// Get WAV specification
    pub fn spec(&self) -> WavSpec {
        self.spec
    }

    /// Get total number of samples (interleaved value count)
    pub fn len(&self) -> usize {
        self.reader.len() as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.reader.len() == 0
    }

    /// Decode up to `max_values` more interleaved values from wherever the
    /// underlying reader currently is. Returns fewer than `max_values` (or
    /// an empty `Vec`) once the end of the file is reached.
    fn decode_next(&mut self, max_values: usize) -> IoResult<Vec<f32>> {
        if max_values == 0 {
            return Ok(Vec::new());
        }
        let decoded = match self.sample_format {
            hound::SampleFormat::Float => self
                .reader
                .samples::<f32>()
                .take(max_values)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| IoError::ReadFailed(format!("Failed to read WAV samples: {}", e)))?,
            hound::SampleFormat::Int => {
                let scale = pcm_scale(self.spec.bits_per_sample)?;
                let ints: Vec<i32> = self
                    .reader
                    .samples::<i32>()
                    .take(max_values)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| {
                        IoError::ReadFailed(format!("Failed to read WAV samples: {}", e))
                    })?;
                ints.into_iter().map(|s| s as f32 / scale).collect()
            }
        };
        self.position += decoded.len();
        Ok(decoded)
    }

    /// Read all remaining samples
    ///
    /// Decodes in bounded-size chunks internally rather than one giant
    /// `hound` call, but the result is still fully materialized as the
    /// returned `Array1`; use `read_chunk` directly for a caller-controlled
    /// memory ceiling.
    pub async fn read_all(&mut self) -> IoResult<Array1<f32>> {
        const CHUNK: usize = 65536;
        let mut all = Vec::with_capacity(self.reader.len() as usize);
        loop {
            let chunk = self.decode_next(CHUNK)?;
            if chunk.is_empty() {
                break;
            }
            all.extend(chunk);
        }
        Ok(Array1::from_vec(all))
    }

    /// Read all remaining samples as a multi-channel array
    pub async fn read_channels(&mut self) -> IoResult<Array2<f32>> {
        let channels = self.spec.channels as usize;
        if channels == 0 {
            return Err(IoError::ReadFailed(
                "WAV file declares zero channels".to_string(),
            ));
        }

        let flat = self.read_all().await?;
        let num_frames = flat.len() / channels;
        let mut data = Array2::zeros((num_frames, channels));

        for (idx, &sample) in flat.iter().enumerate() {
            let frame = idx / channels;
            if frame >= num_frames {
                // Trailing partial frame (flat.len() not a multiple of
                // channels); nothing further to place.
                break;
            }
            data[[frame, idx % channels]] = sample;
        }

        Ok(data)
    }

    /// Read next chunk of samples, decoding on demand. Returns `None` once
    /// the end of the file has been reached.
    pub async fn read_chunk(&mut self, chunk_size: usize) -> IoResult<Option<Array1<f32>>> {
        let chunk = self.decode_next(chunk_size)?;
        if chunk.is_empty() {
            return Ok(None);
        }

        debug!("WAV read chunk: {} samples", chunk.len());
        Ok(Some(Array1::from_vec(chunk)))
    }

    /// Reset reading position to the start of the audio data
    pub fn reset(&mut self) -> IoResult<()> {
        self.reader
            .seek(0)
            .map_err(|e| IoError::ReadFailed(format!("Failed to seek WAV file: {}", e)))?;
        self.position = 0;
        Ok(())
    }

    /// Get current position (interleaved values already consumed)
    pub fn position(&self) -> usize {
        self.position
    }
}

/// WAV file writer
pub struct WavWriter {
    spec: WavSpec,
    writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
}

impl WavWriter {
    /// Create WAV file for writing
    pub async fn create<P: AsRef<Path>>(path: P, spec: WavSpec) -> IoResult<Self> {
        match spec.sample_format {
            SampleFormat::Int => {
                // Validate eagerly (also rejects bits_per_sample == 0, which
                // would otherwise underflow `bits_per_sample - 1` on the
                // first write and panic).
                pcm_scale(spec.bits_per_sample)?;
            }
            SampleFormat::Float => {
                // hound only supports 32-bit IEEE float samples. Reject a
                // mismatched bits_per_sample up front instead of deferring
                // to hound's generic `Error::Unsupported` on the first
                // `write_sample` call.
                if spec.bits_per_sample != 32 {
                    return Err(IoError::ConfigError(format!(
                        "WAV Float sample format requires bits_per_sample = 32, got {}",
                        spec.bits_per_sample
                    )));
                }
            }
        }

        let hound_spec = hound::WavSpec {
            channels: spec.channels,
            sample_rate: spec.sample_rate,
            bits_per_sample: spec.bits_per_sample,
            sample_format: spec.sample_format.into(),
        };

        let writer = hound::WavWriter::create(path, hound_spec)
            .map_err(|e| IoError::WriteFailed(format!("Failed to create WAV file: {}", e)))?;

        info!(
            "WAV file created: {}Hz, {} channels, {} bits, format={:?}",
            spec.sample_rate, spec.channels, spec.bits_per_sample, spec.sample_format
        );

        Ok(Self { spec, writer })
    }

    /// Write samples (f32 values in [-1.0, 1.0])
    pub async fn write_samples(&mut self, samples: &Array1<f32>) -> IoResult<()> {
        match self.spec.sample_format {
            SampleFormat::Int => {
                let scale = pcm_scale(self.spec.bits_per_sample)?;
                for &sample in samples.iter() {
                    // `2^(bits-1) - 1` (not `2^(bits-1)`) keeps a full-scale
                    // 1.0 sample in range; `.round()` avoids the
                    // truncation-toward-zero bias of a plain `as i32` cast.
                    let int_sample = (sample.clamp(-1.0, 1.0) * scale).round() as i32;
                    self.writer.write_sample(int_sample).map_err(|e| {
                        IoError::WriteFailed(format!("Failed to write WAV sample: {}", e))
                    })?;
                }
            }
            SampleFormat::Float => {
                // Float PCM stores the f32 value as-is; no integer scaling.
                for &sample in samples.iter() {
                    self.writer.write_sample(sample).map_err(|e| {
                        IoError::WriteFailed(format!("Failed to write WAV sample: {}", e))
                    })?;
                }
            }
        }

        debug!("WAV wrote {} samples", samples.len());
        Ok(())
    }

    /// Write multi-channel samples
    pub async fn write_channels(&mut self, samples: &Array2<f32>) -> IoResult<()> {
        match self.spec.sample_format {
            SampleFormat::Int => {
                let scale = pcm_scale(self.spec.bits_per_sample)?;
                for row in samples.outer_iter() {
                    for &sample in row.iter() {
                        let int_sample = (sample.clamp(-1.0, 1.0) * scale).round() as i32;
                        self.writer.write_sample(int_sample).map_err(|e| {
                            IoError::WriteFailed(format!("Failed to write WAV sample: {}", e))
                        })?;
                    }
                }
            }
            SampleFormat::Float => {
                for row in samples.outer_iter() {
                    for &sample in row.iter() {
                        self.writer.write_sample(sample).map_err(|e| {
                            IoError::WriteFailed(format!("Failed to write WAV sample: {}", e))
                        })?;
                    }
                }
            }
        }

        debug!("WAV wrote {} frames", samples.nrows());
        Ok(())
    }

    /// Finalize and close the file
    pub async fn finalize(self) -> IoResult<()> {
        self.writer
            .finalize()
            .map_err(|e| IoError::WriteFailed(format!("Failed to finalize WAV file: {}", e)))?;

        info!("WAV file finalized");
        Ok(())
    }
}

// ============================================================================
// CSV Time Series Files
// ============================================================================

/// CSV stream reader
pub struct CsvReader {
    path: String,
    delimiter: u8,
    has_header: bool,
}

impl CsvReader {
    /// Open CSV file for reading
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            delimiter: b',',
            has_header: true,
        }
    }

    /// Set delimiter
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set whether file has header
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Read all data as 2D array
    pub async fn read_array(&self) -> IoResult<Array2<f32>> {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_header)
            .from_path(&self.path)
            .map_err(|e| IoError::ReadFailed(format!("Failed to open CSV file: {}", e)))?;

        let mut rows: Vec<Vec<f32>> = Vec::new();

        for result in reader.records() {
            let record = result
                .map_err(|e| IoError::ReadFailed(format!("Failed to read CSV record: {}", e)))?;

            let row: Vec<f32> = record
                .iter()
                .map(|s| {
                    s.parse::<f32>()
                        .map_err(|e| IoError::ParseError(format!("Failed to parse float: {}", e)))
                })
                .collect::<IoResult<Vec<_>>>()?;

            rows.push(row);
        }

        if rows.is_empty() {
            return Err(IoError::ReadFailed("CSV file is empty".into()));
        }

        let num_cols = rows[0].len();
        let num_rows = rows.len();

        // A previous version derived num_cols solely from the first row
        // and then wrote every parsed cell unconditionally: a later row
        // with MORE columns panicked with an out-of-bounds ndarray index,
        // and a row with FEWER columns was silently zero-filled with no
        // warning. Validate every row's width up front instead.
        for (i, row) in rows.iter().enumerate() {
            if row.len() != num_cols {
                return Err(IoError::ParseError(format!(
                    "CSV row {} has {} column(s), expected {} (from the first row)",
                    i,
                    row.len(),
                    num_cols
                )));
            }
        }

        let mut data = Array2::zeros((num_rows, num_cols));
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[[i, j]] = val;
            }
        }

        info!("CSV file read: {} rows, {} columns", num_rows, num_cols);
        Ok(data)
    }

    /// Read single column
    pub async fn read_column(&self, column_index: usize) -> IoResult<Array1<f32>> {
        let array = self.read_array().await?;

        if column_index >= array.ncols() {
            return Err(IoError::ReadFailed(format!(
                "Column index {} out of bounds",
                column_index
            )));
        }

        Ok(array.column(column_index).to_owned())
    }
}

/// CSV stream writer
pub struct CsvWriter {
    writer: csv::Writer<std::fs::File>,
}

impl CsvWriter {
    /// Create CSV file for writing
    pub async fn create<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let writer = csv::Writer::from_path(path)
            .map_err(|e| IoError::WriteFailed(format!("Failed to create CSV file: {}", e)))?;

        Ok(Self { writer })
    }

    /// Write header row
    pub async fn write_header(&mut self, headers: &[&str]) -> IoResult<()> {
        self.writer
            .write_record(headers)
            .map_err(|e| IoError::WriteFailed(format!("Failed to write CSV header: {}", e)))?;

        Ok(())
    }

    /// Write a row
    pub async fn write_row(&mut self, row: &[f32]) -> IoResult<()> {
        let strings: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        self.writer
            .write_record(&strings)
            .map_err(|e| IoError::WriteFailed(format!("Failed to write CSV row: {}", e)))?;

        Ok(())
    }

    /// Write 2D array
    pub async fn write_array(&mut self, array: &Array2<f32>) -> IoResult<()> {
        for row in array.outer_iter() {
            let row_data: Vec<f32> = row.to_vec();
            self.write_row(&row_data).await?;
        }

        info!("CSV array written: {} rows", array.nrows());
        Ok(())
    }

    /// Flush and finalize
    pub async fn finalize(mut self) -> IoResult<()> {
        self.writer
            .flush()
            .map_err(|e| IoError::WriteFailed(format!("Failed to flush CSV file: {}", e)))?;

        Ok(())
    }
}

// ============================================================================
// HDF5 Dataset Files
//
// Behind the `hdf5` feature, not `file`, for historical reasons: this used to
// bind the libhdf5 C library through the `hdf5` crate. Since the migration to
// OxiH5 the whole section is pure Rust (no libhdf5 install needed), and the
// separate feature only marks HDF5 support as optional.
// ============================================================================

/// Decode a dataset into `f32` samples regardless of its stored dtype.
///
/// libhdf5 converted any numeric on-disk dtype to the requested `f32` during
/// the read call, and callers of [`Hdf5Reader`] rely on that (h5py writes
/// float64 by default). OxiH5's accessors are dtype-strict, so replicate the
/// tolerant behaviour here with an explicit conversion chain.
#[cfg(feature = "hdf5")]
fn dataset_to_f32(dataset: &oxih5::Dataset) -> IoResult<Vec<f32>> {
    if let Ok(v) = dataset.as_f32() {
        return Ok(v);
    }
    if let Ok(v) = dataset.as_f64() {
        return Ok(v.into_iter().map(|x| x as f32).collect());
    }
    // `as_f16` already widens half-precision floats to f32.
    if let Ok(v) = dataset.as_f16() {
        return Ok(v);
    }
    if let Ok(v) = dataset.as_i8() {
        return Ok(v.into_iter().map(f32::from).collect());
    }
    if let Ok(v) = dataset.as_i16() {
        return Ok(v.into_iter().map(f32::from).collect());
    }
    if let Ok(v) = dataset.as_i32() {
        return Ok(v.into_iter().map(|x| x as f32).collect());
    }
    if let Ok(v) = dataset.as_i64() {
        return Ok(v.into_iter().map(|x| x as f32).collect());
    }
    if let Ok(v) = dataset.as_u8() {
        return Ok(v.into_iter().map(f32::from).collect());
    }
    if let Ok(v) = dataset.as_u16() {
        return Ok(v.into_iter().map(f32::from).collect());
    }
    if let Ok(v) = dataset.as_u32() {
        return Ok(v.into_iter().map(|x| x as f32).collect());
    }
    if let Ok(v) = dataset.as_u64() {
        return Ok(v.into_iter().map(|x| x as f32).collect());
    }
    Err(IoError::ReadFailed(format!(
        "Unsupported HDF5 dataset dtype for f32 conversion: {:?}",
        dataset.dtype
    )))
}

/// HDF5 file reader (pure Rust, via OxiH5)
#[cfg(feature = "hdf5")]
pub struct Hdf5Reader {
    file: oxih5::File,
}

#[cfg(feature = "hdf5")]
impl Hdf5Reader {
    /// Open HDF5 file for reading
    pub async fn open<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let file = oxih5::open(path)
            .map_err(|e| IoError::ReadFailed(format!("Failed to open HDF5 file: {}", e)))?;

        info!("HDF5 file opened");
        Ok(Self { file })
    }

    /// Read 1D dataset
    pub async fn read_dataset_1d(&self, name: &str) -> IoResult<Array1<f32>> {
        let dataset = self
            .file
            .dataset(name)
            .map_err(|e| IoError::ReadFailed(format!("Failed to open dataset: {}", e)))?;

        // The libhdf5-based predecessor's `read_1d()` rejected datasets of any
        // other rank; keep that contract.
        if dataset.shape.len() != 1 {
            return Err(IoError::ReadFailed(format!(
                "Dataset is not 1D: {:?}",
                dataset.shape
            )));
        }

        let data = dataset_to_f32(&dataset)?;

        debug!("HDF5 dataset '{}' read: {} elements", name, data.len());
        Ok(Array1::from_vec(data))
    }

    /// Read 2D dataset
    pub async fn read_dataset_2d(&self, name: &str) -> IoResult<Array2<f32>> {
        let dataset = self
            .file
            .dataset(name)
            .map_err(|e| IoError::ReadFailed(format!("Failed to open dataset: {}", e)))?;

        if dataset.shape.len() != 2 {
            return Err(IoError::ReadFailed(format!(
                "Dataset is not 2D: {:?}",
                dataset.shape
            )));
        }

        let nrows = dataset.shape[0];
        let ncols = dataset.shape[1];

        // HDF5 stores row-major (C order), which is also ndarray's default
        // layout, so the flat buffer maps straight into an Array2.
        let raw_data = dataset_to_f32(&dataset)?;
        let data = Array2::from_shape_vec((nrows, ncols), raw_data)
            .map_err(|e| IoError::ReadFailed(format!("Dataset shape/content mismatch: {}", e)))?;

        debug!("HDF5 dataset '{}' read: {:?} shape", name, (nrows, ncols));
        Ok(data)
    }

    /// List all datasets
    pub async fn list_datasets(&self) -> IoResult<Vec<String>> {
        self.file
            .dataset_names()
            .map_err(|e| IoError::ReadFailed(format!("Failed to list HDF5 members: {}", e)))
    }
}

/// HDF5 file writer (pure Rust, via OxiH5)
///
/// OxiH5's [`oxih5::FileWriter`] is a builder: datasets are staged in memory
/// by the `write_*` calls and the file itself is materialized once by
/// [`Hdf5Writer::finalize`]. (The libhdf5-based predecessor wrote through to
/// disk incrementally.) Dropping the writer without calling `finalize`
/// therefore creates no file at all -- it does not leave a partial one.
#[cfg(feature = "hdf5")]
pub struct Hdf5Writer {
    writer: oxih5::FileWriter,
    path: std::path::PathBuf,
}

#[cfg(feature = "hdf5")]
impl Hdf5Writer {
    /// Create HDF5 file for writing
    ///
    /// The path is validated and written when [`Hdf5Writer::finalize`] runs;
    /// see the struct-level note on builder semantics.
    pub async fn create<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        info!("HDF5 file created");
        Ok(Self {
            writer: oxih5::FileWriter::new(),
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Write 1D dataset
    pub async fn write_dataset_1d(&mut self, name: &str, data: &Array1<f32>) -> IoResult<()> {
        // `as_slice()` returns `None` for a non-contiguous view (e.g. one
        // produced by slicing with a step, or a transposed view); fall
        // back to an owned, contiguous copy instead of panicking via
        // `.expect(...)`. `write_dataset_2d` already does this
        // unconditionally by flattening through `.iter()`.
        let owned;
        let flat: &[f32] = match data.as_slice() {
            Some(slice) => slice,
            None => {
                owned = data.iter().copied().collect::<Vec<f32>>();
                &owned
            }
        };

        self.writer
            .write_dataset_f32(name, flat, &[data.len()])
            .map_err(|e| IoError::WriteFailed(format!("Failed to write dataset: {}", e)))?;

        debug!("HDF5 dataset '{}' written: {} elements", name, data.len());
        Ok(())
    }

    /// Write 2D dataset
    pub async fn write_dataset_2d(&mut self, name: &str, data: &Array2<f32>) -> IoResult<()> {
        let shape = (data.nrows(), data.ncols());

        // Flatten 2D array to 1D for writing (row-major, matching HDF5's
        // C-order layout).
        let flat_data: Vec<f32> = data.iter().cloned().collect();
        self.writer
            .write_dataset_f32(name, &flat_data, &[shape.0, shape.1])
            .map_err(|e| IoError::WriteFailed(format!("Failed to write dataset: {}", e)))?;

        debug!("HDF5 dataset '{}' written: {:?} shape", name, shape);
        Ok(())
    }

    /// Build the HDF5 file on disk and close the writer.
    ///
    /// This is where the staged datasets actually reach the filesystem;
    /// I/O errors (unwritable path, missing parent directory, full disk)
    /// surface here rather than in `create`.
    pub async fn finalize(mut self) -> IoResult<()> {
        self.writer
            .build(&self.path)
            .map_err(|e| IoError::WriteFailed(format!("Failed to write HDF5 file: {}", e)))?;
        info!("HDF5 file finalized");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_wav_spec_default() {
        let spec = WavSpec::default();
        assert_eq!(spec.sample_rate, 44100);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
    }

    #[tokio::test]
    async fn test_csv_round_trip() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_csv.csv");

        // Write
        let mut writer = CsvWriter::create(&path).await.unwrap();
        writer.write_header(&["col1", "col2"]).await.unwrap();
        writer.write_row(&[1.0, 2.0]).await.unwrap();
        writer.write_row(&[3.0, 4.0]).await.unwrap();
        writer.finalize().await.unwrap();

        // Read
        let reader = CsvReader::new(&path);
        let data = reader.read_array().await.unwrap();

        assert_eq!(data.nrows(), 2);
        assert_eq!(data.ncols(), 2);
        assert_eq!(data[[0, 0]], 1.0);
        assert_eq!(data[[1, 1]], 4.0);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    fn unique_temp_path(name: &str) -> std::path::PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        env::temp_dir().join(format!("kizzasi_{}_{}_{}", name, std::process::id(), id))
    }

    // === Regression tests: WAV integer scaling off-by-one (medium, id=48) ===

    #[test]
    fn test_pcm_scale_keeps_full_scale_in_range() {
        // 2^(bits-1) - 1 (not 2^(bits-1)): a full-scale 1.0 sample must not
        // overflow the signed integer range for that bit depth.
        let scale16 = pcm_scale(16).unwrap();
        assert_eq!(scale16, 32767.0);
        assert!((1.0f32 * scale16) as i32 <= i16::MAX as i32);

        let scale8 = pcm_scale(8).unwrap();
        assert_eq!(scale8, 127.0);
    }

    #[test]
    fn test_pcm_scale_rejects_zero_and_unsupported_depths() {
        assert!(pcm_scale(0).is_err(), "bits=0 must not underflow bits-1");
        assert!(pcm_scale(12).is_err(), "12 is not a supported bit depth");
        assert!(pcm_scale(16).is_ok());
    }

    #[tokio::test]
    async fn test_wav_round_trip_full_scale() {
        let path = unique_temp_path("wav_full_scale.wav");

        let spec = WavSpec {
            sample_rate: 44100,
            channels: 1,
            bits_per_sample: 16,
            ..Default::default()
        };
        let mut writer = WavWriter::create(&path, spec).await.unwrap();
        // Exercise the exact full-scale boundary values that used to
        // overflow i16 on the positive side.
        let samples = Array1::from_vec(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        writer.write_samples(&samples).await.unwrap();
        writer.finalize().await.unwrap();

        let mut reader = WavReader::open(&path).await.unwrap();
        let read_back = reader.read_all().await.unwrap();

        assert_eq!(read_back.len(), samples.len());
        for (original, roundtripped) in samples.iter().zip(read_back.iter()) {
            // 16-bit quantization error tolerance.
            assert!(
                (original - roundtripped).abs() < 1e-3,
                "original={original}, roundtripped={roundtripped}"
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_wav_create_rejects_zero_bits_per_sample() {
        let path = unique_temp_path("wav_zero_bits.wav");
        let spec = WavSpec {
            sample_rate: 44100,
            channels: 1,
            bits_per_sample: 0,
            ..Default::default()
        };
        // bits_per_sample - 1 used to underflow the u16 and panic on the
        // first write; it must now be rejected up front.
        let result = WavWriter::create(&path, spec).await;
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_wav_streaming_read_chunk_matches_read_all() {
        let path = unique_temp_path("wav_streaming.wav");

        let spec = WavSpec {
            sample_rate: 8000,
            channels: 1,
            bits_per_sample: 16,
            ..Default::default()
        };
        let mut writer = WavWriter::create(&path, spec).await.unwrap();
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) * 2.0 - 1.0).collect();
        writer
            .write_samples(&Array1::from_vec(samples.clone()))
            .await
            .unwrap();
        writer.finalize().await.unwrap();

        // Read via bounded chunks (the streaming primitive) and via
        // read_all(); both must agree on total sample count and values.
        let mut chunked_reader = WavReader::open(&path).await.unwrap();
        let mut chunked = Vec::new();
        while let Some(chunk) = chunked_reader.read_chunk(37).await.unwrap() {
            chunked.extend(chunk.iter().copied());
        }
        assert_eq!(chunked.len(), samples.len());
        assert_eq!(chunked_reader.position(), samples.len());

        let mut whole_reader = WavReader::open(&path).await.unwrap();
        let whole = whole_reader.read_all().await.unwrap();
        assert_eq!(whole.len(), samples.len());

        for (a, b) in chunked.iter().zip(whole.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_wav_len_does_not_require_reading_samples() {
        let path = unique_temp_path("wav_len.wav");
        let spec = WavSpec {
            sample_rate: 8000,
            channels: 1,
            bits_per_sample: 16,
            ..Default::default()
        };
        let mut writer = WavWriter::create(&path, spec).await.unwrap();
        writer
            .write_samples(&Array1::from_vec(vec![0.1; 500]))
            .await
            .unwrap();
        writer.finalize().await.unwrap();

        let reader = WavReader::open(&path).await.unwrap();
        // len()/is_empty() must reflect the header-declared length without
        // decoding any sample data (no read_chunk/read_all call above).
        assert_eq!(reader.len(), 500);
        assert!(!reader.is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_wav_reset_rereads_from_start() {
        let path = unique_temp_path("wav_reset.wav");
        let spec = WavSpec {
            sample_rate: 8000,
            channels: 1,
            bits_per_sample: 16,
            ..Default::default()
        };
        let mut writer = WavWriter::create(&path, spec).await.unwrap();
        writer
            .write_samples(&Array1::from_vec(vec![0.25, 0.5, 0.75]))
            .await
            .unwrap();
        writer.finalize().await.unwrap();

        let mut reader = WavReader::open(&path).await.unwrap();
        let first_pass = reader.read_all().await.unwrap();
        assert_eq!(first_pass.len(), 3);
        assert_eq!(reader.position(), 3);

        reader.reset().unwrap();
        assert_eq!(reader.position(), 0);
        let second_pass = reader.read_all().await.unwrap();
        assert_eq!(second_pass.len(), 3);
        for (a, b) in first_pass.iter().zip(second_pass.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(&path).ok();
    }

    // === Regression tests: WavSpec::sample_format / Float WAV support (medium, id=48) ===
    //
    // A previous version had no `sample_format` field at all and
    // hard-coded `hound::SampleFormat::Int` in `WavWriter::create`, making
    // it impossible to write a 32-bit float WAV file through this API even
    // though `WavReader` already handled reading one.

    #[tokio::test]
    async fn test_wav_write_and_read_float_format_round_trips_without_quantization() {
        let path = unique_temp_path("wav_float.wav");

        let spec = WavSpec {
            sample_rate: 48000,
            channels: 1,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(&path, spec).await.unwrap();
        let samples = Array1::from_vec(vec![-1.0, -0.333_333, 0.0, 0.5, 1.0, 0.123_456_7]);
        writer.write_samples(&samples).await.unwrap();
        writer.finalize().await.unwrap();

        let mut reader = WavReader::open(&path).await.unwrap();
        assert_eq!(
            reader.spec().sample_format,
            SampleFormat::Float,
            "the format actually written to the file header must round-trip through spec()"
        );
        let read_back = reader.read_all().await.unwrap();

        assert_eq!(read_back.len(), samples.len());
        for (original, roundtripped) in samples.iter().zip(read_back.iter()) {
            // Float PCM stores the f32 bit pattern directly (no integer
            // quantization), so this should be very close to exact.
            assert!(
                (original - roundtripped).abs() < 1e-6,
                "original={original}, roundtripped={roundtripped}"
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_wav_create_rejects_float_format_with_non_32_bits() {
        let path = unique_temp_path("wav_float_bad_bits.wav");
        let spec = WavSpec {
            sample_rate: 44100,
            channels: 1,
            bits_per_sample: 16,
            sample_format: SampleFormat::Float,
        };
        let result = WavWriter::create(&path, spec).await;
        assert!(
            result.is_err(),
            "Float format requires bits_per_sample = 32"
        );
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_wav_spec_default_is_int_format() {
        // Backward compatibility: existing callers that only set
        // sample_rate/channels/bits_per_sample via `..Default::default()`
        // must keep getting Int-format PCM output, unchanged.
        assert_eq!(WavSpec::default().sample_format, SampleFormat::Int);
    }

    // === Regression test: ragged CSV rows (medium, id=48) ===

    #[tokio::test]
    async fn test_csv_read_array_rejects_ragged_rows() {
        let path = unique_temp_path("csv_ragged.csv");

        // Write a CSV with a header + a short row directly, bypassing
        // CsvWriter (which always writes consistent row widths) to
        // reproduce a malformed/hand-edited file.
        tokio::fs::write(&path, "col1,col2,col3\n1.0,2.0,3.0\n4.0,5.0\n")
            .await
            .unwrap();

        let reader = CsvReader::new(&path);
        let result = reader.read_array().await;
        assert!(
            result.is_err(),
            "a row with fewer columns than the first row must error, not silently zero-fill"
        );

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_csv_read_array_rejects_row_with_extra_columns() {
        let path = unique_temp_path("csv_extra_cols.csv");
        tokio::fs::write(&path, "col1,col2\n1.0,2.0\n3.0,4.0,5.0\n")
            .await
            .unwrap();

        let reader = CsvReader::new(&path);
        let result = reader.read_array().await;
        assert!(
            result.is_err(),
            "a row with more columns than the first row must error, not panic on out-of-bounds"
        );

        std::fs::remove_file(&path).ok();
    }

    // === Regression test: HDF5 non-contiguous array write (medium, id=48) ===

    #[cfg(feature = "hdf5")]
    #[tokio::test]
    async fn test_hdf5_write_dataset_1d_non_contiguous_does_not_panic() {
        let path = unique_temp_path("hdf5_noncontig.h5");
        let mut writer = Hdf5Writer::create(&path).await.unwrap();

        // `invert_axis` flips the stride sign in place (no reallocation),
        // producing an owned array that is no longer in standard
        // (contiguous) layout. A previous version unconditionally called
        // `.expect("Array must have contiguous layout")` on `as_slice()`
        // here and would panic.
        let mut data = Array1::from_vec((0..10).map(|i| i as f32).collect());
        data.invert_axis(scirs2_core::ndarray::Axis(0));
        assert!(
            data.as_slice().is_none(),
            "test fixture must be non-contiguous to actually exercise the as_slice() fallback"
        );

        let result = writer.write_dataset_1d("data", &data).await;
        assert!(result.is_ok(), "{result:?}");

        std::fs::remove_file(&path).ok();
    }

    // === Regression tests: OxiH5 migration (write -> finalize -> read) ======

    #[cfg(feature = "hdf5")]
    #[tokio::test]
    async fn test_hdf5_roundtrip_1d_and_2d() {
        let path = unique_temp_path("hdf5_roundtrip.h5");

        let v1 = Array1::from_vec(vec![1.0f32, -2.5, 3.25, 0.0]);
        let v2 = Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut writer = Hdf5Writer::create(&path).await.unwrap();
        writer.write_dataset_1d("vector", &v1).await.unwrap();
        writer.write_dataset_2d("matrix", &v2).await.unwrap();
        writer.finalize().await.unwrap();

        let reader = Hdf5Reader::open(&path).await.unwrap();
        let mut names = reader.list_datasets().await.unwrap();
        names.sort();
        assert_eq!(names, vec!["matrix".to_string(), "vector".to_string()]);

        let r1 = reader.read_dataset_1d("vector").await.unwrap();
        assert_eq!(r1, v1);
        let r2 = reader.read_dataset_2d("matrix").await.unwrap();
        assert_eq!(r2, v2);

        // Rank mismatches must error, matching the libhdf5-era contract.
        assert!(reader.read_dataset_1d("matrix").await.is_err());
        assert!(reader.read_dataset_2d("vector").await.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "hdf5")]
    #[tokio::test]
    async fn test_hdf5_read_converts_f64_dataset_to_f32() {
        // h5py writes float64 by default, and the libhdf5-based reader
        // converted on read; `dataset_to_f32` must preserve that tolerance.
        let path = unique_temp_path("hdf5_f64.h5");

        let mut writer = oxih5::FileWriter::new();
        writer
            .write_dataset_f64("wide", &[1.5f64, -2.0, 1.0e10], &[3])
            .unwrap();
        writer.build(&path).unwrap();

        let reader = Hdf5Reader::open(&path).await.unwrap();
        let data = reader.read_dataset_1d("wide").await.unwrap();
        assert_eq!(data.to_vec(), vec![1.5f32, -2.0, 1.0e10]);

        std::fs::remove_file(&path).ok();
    }
}
