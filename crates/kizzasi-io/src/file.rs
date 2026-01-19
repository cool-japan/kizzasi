//! File stream support
//!
//! Provides file-based data streams for WAV audio, CSV time series, and HDF5 datasets.
//!
//! ## Features
//! - WAV audio file reading/writing
//! - CSV time series reading/writing
//! - HDF5 dataset reading/writing
//! - Async I/O support
//! - Streaming large files
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

/// WAV file specification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WavSpec {
    /// Sample rate (Hz)
    pub sample_rate: u32,

    /// Number of channels
    pub channels: u16,

    /// Bits per sample (8, 16, 24, 32)
    pub bits_per_sample: u16,
}

impl Default for WavSpec {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 1,
            bits_per_sample: 16,
        }
    }
}

/// WAV file reader
pub struct WavReader {
    spec: WavSpec,
    samples: Vec<f32>,
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
        };

        // Read all samples and convert to f32
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| IoError::ReadFailed(format!("Failed to read WAV samples: {}", e)))?,
            hound::SampleFormat::Int => {
                let int_samples = reader
                    .into_samples::<i32>()
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| {
                        IoError::ReadFailed(format!("Failed to read WAV samples: {}", e))
                    })?;

                // Normalize to [-1.0, 1.0]
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                int_samples
                    .into_iter()
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
        };

        info!(
            "WAV file opened: {} samples, {}Hz, {} channels",
            samples.len(),
            wav_spec.sample_rate,
            wav_spec.channels
        );

        Ok(Self {
            spec: wav_spec,
            samples,
            position: 0,
        })
    }

    /// Get WAV specification
    pub fn spec(&self) -> WavSpec {
        self.spec
    }

    /// Get total number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Read all samples
    pub async fn read_all(&self) -> IoResult<Array1<f32>> {
        Ok(Array1::from_vec(self.samples.clone()))
    }

    /// Read samples as multi-channel array
    pub async fn read_channels(&self) -> IoResult<Array2<f32>> {
        let num_frames = self.samples.len() / self.spec.channels as usize;
        let mut data = Array2::zeros((num_frames, self.spec.channels as usize));

        for (i, chunk) in self.samples.chunks(self.spec.channels as usize).enumerate() {
            for (ch, &sample) in chunk.iter().enumerate() {
                data[[i, ch]] = sample;
            }
        }

        Ok(data)
    }

    /// Read next chunk of samples
    pub async fn read_chunk(&mut self, chunk_size: usize) -> IoResult<Option<Array1<f32>>> {
        if self.position >= self.samples.len() {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(self.samples.len());
        let chunk = self.samples[self.position..end].to_vec();
        self.position = end;

        debug!("WAV read chunk: {} samples", chunk.len());
        Ok(Some(Array1::from_vec(chunk)))
    }

    /// Reset reading position
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get current position
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
        let hound_spec = hound::WavSpec {
            channels: spec.channels,
            sample_rate: spec.sample_rate,
            bits_per_sample: spec.bits_per_sample,
            sample_format: hound::SampleFormat::Int,
        };

        let writer = hound::WavWriter::create(path, hound_spec)
            .map_err(|e| IoError::WriteFailed(format!("Failed to create WAV file: {}", e)))?;

        info!(
            "WAV file created: {}Hz, {} channels, {} bits",
            spec.sample_rate, spec.channels, spec.bits_per_sample
        );

        Ok(Self { spec, writer })
    }

    /// Write samples (f32 values in [-1.0, 1.0])
    pub async fn write_samples(&mut self, samples: &Array1<f32>) -> IoResult<()> {
        let max_val = (1i64 << (self.spec.bits_per_sample - 1)) as f32;

        for &sample in samples.iter() {
            let int_sample = (sample.clamp(-1.0, 1.0) * max_val) as i32;
            self.writer
                .write_sample(int_sample)
                .map_err(|e| IoError::WriteFailed(format!("Failed to write WAV sample: {}", e)))?;
        }

        debug!("WAV wrote {} samples", samples.len());
        Ok(())
    }

    /// Write multi-channel samples
    pub async fn write_channels(&mut self, samples: &Array2<f32>) -> IoResult<()> {
        let max_val = (1i64 << (self.spec.bits_per_sample - 1)) as f32;

        for row in samples.outer_iter() {
            for &sample in row.iter() {
                let int_sample = (sample.clamp(-1.0, 1.0) * max_val) as i32;
                self.writer.write_sample(int_sample).map_err(|e| {
                    IoError::WriteFailed(format!("Failed to write WAV sample: {}", e))
                })?;
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
// ============================================================================

/// HDF5 file reader
pub struct Hdf5Reader {
    file: hdf5::File,
}

impl Hdf5Reader {
    /// Open HDF5 file for reading
    pub async fn open<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let file = hdf5::File::open(path)
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

        let data: Vec<f32> = dataset
            .read_1d()
            .map_err(|e| IoError::ReadFailed(format!("Failed to read dataset: {}", e)))?
            .to_vec();

        debug!("HDF5 dataset '{}' read: {} elements", name, data.len());
        Ok(Array1::from_vec(data))
    }

    /// Read 2D dataset
    pub async fn read_dataset_2d(&self, name: &str) -> IoResult<Array2<f32>> {
        let dataset = self
            .file
            .dataset(name)
            .map_err(|e| IoError::ReadFailed(format!("Failed to open dataset: {}", e)))?;

        // Read raw data
        let raw_data: Vec<f32> = dataset
            .read_raw()
            .map_err(|e| IoError::ReadFailed(format!("Failed to read dataset: {}", e)))?;

        let shape = dataset.shape();
        if shape.len() != 2 {
            return Err(IoError::ReadFailed(format!(
                "Dataset is not 2D: {:?}",
                shape
            )));
        }

        let nrows = shape[0];
        let ncols = shape[1];

        // Reshape into Array2
        let mut data = Array2::zeros((nrows, ncols));
        for i in 0..nrows {
            for j in 0..ncols {
                data[[i, j]] = raw_data[i * ncols + j];
            }
        }

        debug!("HDF5 dataset '{}' read: {:?} shape", name, (nrows, ncols));
        Ok(data)
    }

    /// List all datasets
    pub async fn list_datasets(&self) -> IoResult<Vec<String>> {
        let mut datasets = Vec::new();

        for name in self
            .file
            .member_names()
            .map_err(|e| IoError::ReadFailed(format!("Failed to list HDF5 members: {}", e)))?
        {
            if self.file.dataset(&name).is_ok() {
                datasets.push(name);
            }
        }

        Ok(datasets)
    }
}

/// HDF5 file writer
pub struct Hdf5Writer {
    file: hdf5::File,
}

impl Hdf5Writer {
    /// Create HDF5 file for writing
    pub async fn create<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let file = hdf5::File::create(path)
            .map_err(|e| IoError::WriteFailed(format!("Failed to create HDF5 file: {}", e)))?;

        info!("HDF5 file created");
        Ok(Self { file })
    }

    /// Write 1D dataset
    pub async fn write_dataset_1d(&self, name: &str, data: &Array1<f32>) -> IoResult<()> {
        let dataset = self
            .file
            .new_dataset::<f32>()
            .shape(data.len())
            .create(name)
            .map_err(|e| IoError::WriteFailed(format!("Failed to create dataset: {}", e)))?;

        dataset
            .write_raw(data.as_slice().expect("Array must have contiguous layout"))
            .map_err(|e| IoError::WriteFailed(format!("Failed to write dataset: {}", e)))?;

        debug!("HDF5 dataset '{}' written: {} elements", name, data.len());
        Ok(())
    }

    /// Write 2D dataset
    pub async fn write_dataset_2d(&self, name: &str, data: &Array2<f32>) -> IoResult<()> {
        let shape = (data.nrows(), data.ncols());
        let dataset = self
            .file
            .new_dataset::<f32>()
            .shape(shape)
            .create(name)
            .map_err(|e| IoError::WriteFailed(format!("Failed to create dataset: {}", e)))?;

        // Flatten 2D array to 1D for writing
        let flat_data: Vec<f32> = data.iter().cloned().collect();
        dataset
            .write_raw(&flat_data)
            .map_err(|e| IoError::WriteFailed(format!("Failed to write dataset: {}", e)))?;

        debug!("HDF5 dataset '{}' written: {:?} shape", name, shape);
        Ok(())
    }

    /// Flush and close
    pub async fn finalize(self) -> IoResult<()> {
        drop(self.file);
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
}
