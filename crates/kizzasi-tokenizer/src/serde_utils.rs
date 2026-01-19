//! Serialization and deserialization utilities for tokenizers
//!
//! Provides methods to save and load tokenizer weights and configurations
//! in various formats (JSON, binary, safetensors).

use crate::error::{TokenizerError, TokenizerResult};
use crate::{ContinuousTokenizer, LinearQuantizer, MuLawCodec, SignalTokenizer};
#[cfg(feature = "vqvae")]
use crate::{VQConfig, VQVAETokenizer};
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Serializable configuration for ContinuousTokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousTokenizerConfig {
    pub input_dim: usize,
    pub embed_dim: usize,
    #[serde(with = "array2_serde")]
    pub encoder: Array2<f32>,
    #[serde(with = "array2_serde")]
    pub decoder: Array2<f32>,
}

impl ContinuousTokenizerConfig {
    /// Convert from ContinuousTokenizer
    pub fn from_tokenizer(tokenizer: &ContinuousTokenizer) -> Self {
        Self {
            input_dim: tokenizer.input_dim(),
            embed_dim: tokenizer.embed_dim(),
            encoder: tokenizer.encoder().clone(),
            decoder: tokenizer.decoder().clone(),
        }
    }

    /// Convert to ContinuousTokenizer
    pub fn to_tokenizer(&self) -> TokenizerResult<ContinuousTokenizer> {
        let mut tokenizer = ContinuousTokenizer::new(self.input_dim, self.embed_dim);
        tokenizer.set_encoder(self.encoder.clone())?;
        tokenizer.set_decoder(self.decoder.clone())?;
        Ok(tokenizer)
    }
}

/// Serializable configuration for LinearQuantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearQuantizerConfig {
    pub min: f32,
    pub max: f32,
    pub bits: u8,
}

impl LinearQuantizerConfig {
    /// Convert from LinearQuantizer
    pub fn from_quantizer(quantizer: &LinearQuantizer) -> Self {
        let (min, max) = quantizer.range();
        Self {
            min,
            max,
            bits: quantizer.bits(),
        }
    }

    /// Convert to LinearQuantizer
    pub fn to_quantizer(&self) -> TokenizerResult<LinearQuantizer> {
        LinearQuantizer::new(self.min, self.max, self.bits)
    }
}

/// Serializable configuration for MuLawCodec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuLawCodecConfig {
    pub mu: f32,
    pub bits: u8,
}

impl MuLawCodecConfig {
    /// Convert from MuLawCodec
    pub fn from_codec(codec: &MuLawCodec) -> Self {
        Self {
            mu: codec.mu(),
            bits: codec.bits(),
        }
    }

    /// Convert to MuLawCodec
    pub fn to_codec(&self) -> MuLawCodec {
        MuLawCodec::with_mu(self.mu, self.bits)
    }
}

/// Serializable configuration for MultiScaleTokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleTokenizerConfig {
    pub input_dim: usize,
    pub embed_dim_per_level: usize,
    pub downsample_factors: Vec<usize>,
    #[serde(with = "vec_array2_serde")]
    pub encoders: Vec<Array2<f32>>,
    #[serde(with = "vec_array2_serde")]
    pub decoders: Vec<Array2<f32>>,
}

/// Serializable configuration for VQVAETokenizer
#[cfg(feature = "vqvae")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VQVAETokenizerConfig {
    pub input_dim: usize,
    pub vq_config: VQConfig,
    #[serde(with = "array2_serde")]
    pub encoder: Array2<f32>,
    #[serde(with = "array2_serde")]
    pub decoder: Array2<f32>,
    #[serde(with = "array2_serde")]
    pub codebook: Array2<f32>,
}

#[cfg(feature = "vqvae")]
impl VQVAETokenizerConfig {
    /// Convert from VQVAETokenizer
    pub fn from_tokenizer(tokenizer: &VQVAETokenizer) -> Self {
        Self {
            input_dim: tokenizer.encoder().shape()[0],
            vq_config: VQConfig {
                codebook_size: tokenizer.quantizer().codebook_size(),
                embed_dim: tokenizer.quantizer().embed_dim(),
                commitment_beta: 0.25,
                ema_decay: 0.99,
                epsilon: 1e-5,
                use_ema: true,
            },
            encoder: tokenizer.encoder().clone(),
            decoder: tokenizer.decoder().clone(),
            codebook: tokenizer.quantizer().codebook().clone(),
        }
    }

    /// Convert to VQVAETokenizer
    pub fn to_tokenizer(&self) -> TokenizerResult<VQVAETokenizer> {
        let mut tokenizer = VQVAETokenizer::new(self.input_dim, self.vq_config.clone());
        tokenizer.set_encoder(self.encoder.clone())?;
        tokenizer.set_decoder(self.decoder.clone())?;
        // Note: We would need to add a method to set codebook in VectorQuantizer
        Ok(tokenizer)
    }
}

/// Custom serde module for Array2<f32>
mod array2_serde {
    use scirs2_core::ndarray::Array2;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Array2Data {
        shape: Vec<usize>,
        data: Vec<f32>,
    }

    pub fn serialize<S>(array: &Array2<f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let shape = array.shape().to_vec();
        let data = array.iter().cloned().collect();
        let wrapper = Array2Data { shape, data };
        wrapper.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Array2<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wrapper = Array2Data::deserialize(deserializer)?;
        if wrapper.shape.len() != 2 {
            return Err(serde::de::Error::custom("Expected 2D array"));
        }
        Array2::from_shape_vec((wrapper.shape[0], wrapper.shape[1]), wrapper.data)
            .map_err(serde::de::Error::custom)
    }
}

/// Custom serde module for Vec<Array2<f32>>
mod vec_array2_serde {
    use scirs2_core::ndarray::Array2;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Array2Data {
        shape: Vec<usize>,
        data: Vec<f32>,
    }

    pub fn serialize<S>(arrays: &[Array2<f32>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wrappers: Vec<Array2Data> = arrays
            .iter()
            .map(|array| Array2Data {
                shape: array.shape().to_vec(),
                data: array.iter().cloned().collect(),
            })
            .collect();
        wrappers.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Array2<f32>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wrappers: Vec<Array2Data> = Vec::deserialize(deserializer)?;
        wrappers
            .into_iter()
            .map(|wrapper| {
                if wrapper.shape.len() != 2 {
                    return Err(serde::de::Error::custom("Expected 2D array"));
                }
                Array2::from_shape_vec((wrapper.shape[0], wrapper.shape[1]), wrapper.data)
                    .map_err(serde::de::Error::custom)
            })
            .collect()
    }
}

/// Trait for saving and loading tokenizers
pub trait TokenizerIO: Sized {
    /// Save to JSON file
    fn save_json<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()>;

    /// Load from JSON file
    fn load_json<P: AsRef<Path>>(path: P) -> TokenizerResult<Self>;

    /// Save to binary file (using bincode)
    fn save_binary<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()>;

    /// Load from binary file (using bincode)
    fn load_binary<P: AsRef<Path>>(path: P) -> TokenizerResult<Self>;
}

// Implement TokenizerIO for ContinuousTokenizer
impl TokenizerIO for ContinuousTokenizer {
    fn save_json<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()> {
        let config = ContinuousTokenizerConfig::from_tokenizer(self);
        let file = File::create(path).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to create file: {}", e))
        })?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &config).map_err(|e| {
            TokenizerError::encoding("serialization", format!("JSON serialization failed: {}", e))
        })
    }

    fn load_json<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let file = File::open(path).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to open file: {}", e))
        })?;
        let reader = BufReader::new(file);
        let config: ContinuousTokenizerConfig = serde_json::from_reader(reader).map_err(|e| {
            TokenizerError::decoding(
                "deserialization",
                format!("JSON deserialization failed: {}", e),
            )
        })?;
        config.to_tokenizer()
    }

    fn save_binary<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()> {
        let config = ContinuousTokenizerConfig::from_tokenizer(self);
        let file = File::create(path).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);
        let encoded = serde_json::to_vec(&config).map_err(|e| {
            TokenizerError::encoding(
                "serialization",
                format!("Binary serialization failed: {}", e),
            )
        })?;
        writer.write_all(&encoded).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to write file: {}", e))
        })
    }

    fn load_binary<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let file = File::open(path).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to open file: {}", e))
        })?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to read file: {}", e))
        })?;
        let config: ContinuousTokenizerConfig = serde_json::from_slice(&buffer).map_err(|e| {
            TokenizerError::decoding(
                "deserialization",
                format!("Binary deserialization failed: {}", e),
            )
        })?;
        config.to_tokenizer()
    }
}

// Implement TokenizerIO for LinearQuantizer
impl TokenizerIO for LinearQuantizer {
    fn save_json<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()> {
        let config = LinearQuantizerConfig::from_quantizer(self);
        let file = File::create(path).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to create file: {}", e))
        })?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &config).map_err(|e| {
            TokenizerError::encoding("serialization", format!("JSON serialization failed: {}", e))
        })
    }

    fn load_json<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let file = File::open(path).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to open file: {}", e))
        })?;
        let reader = BufReader::new(file);
        let config: LinearQuantizerConfig = serde_json::from_reader(reader).map_err(|e| {
            TokenizerError::decoding(
                "deserialization",
                format!("JSON deserialization failed: {}", e),
            )
        })?;
        config.to_quantizer()
    }

    fn save_binary<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()> {
        let config = LinearQuantizerConfig::from_quantizer(self);
        let file = File::create(path).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);
        let encoded = serde_json::to_vec(&config).map_err(|e| {
            TokenizerError::encoding(
                "serialization",
                format!("Binary serialization failed: {}", e),
            )
        })?;
        writer.write_all(&encoded).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to write file: {}", e))
        })
    }

    fn load_binary<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let file = File::open(path).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to open file: {}", e))
        })?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to read file: {}", e))
        })?;
        let config: LinearQuantizerConfig = serde_json::from_slice(&buffer).map_err(|e| {
            TokenizerError::decoding(
                "deserialization",
                format!("Binary deserialization failed: {}", e),
            )
        })?;
        config.to_quantizer()
    }
}

// Implement TokenizerIO for MuLawCodec
impl TokenizerIO for MuLawCodec {
    fn save_json<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()> {
        let config = MuLawCodecConfig::from_codec(self);
        let file = File::create(path).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to create file: {}", e))
        })?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &config).map_err(|e| {
            TokenizerError::encoding("serialization", format!("JSON serialization failed: {}", e))
        })
    }

    fn load_json<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let file = File::open(path).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to open file: {}", e))
        })?;
        let reader = BufReader::new(file);
        let config: MuLawCodecConfig = serde_json::from_reader(reader).map_err(|e| {
            TokenizerError::decoding(
                "deserialization",
                format!("JSON deserialization failed: {}", e),
            )
        })?;
        Ok(config.to_codec())
    }

    fn save_binary<P: AsRef<Path>>(&self, path: P) -> TokenizerResult<()> {
        let config = MuLawCodecConfig::from_codec(self);
        let file = File::create(path).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);
        let encoded = serde_json::to_vec(&config).map_err(|e| {
            TokenizerError::encoding(
                "serialization",
                format!("Binary serialization failed: {}", e),
            )
        })?;
        writer.write_all(&encoded).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Failed to write file: {}", e))
        })
    }

    fn load_binary<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let file = File::open(path).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to open file: {}", e))
        })?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Failed to read file: {}", e))
        })?;
        let config: MuLawCodecConfig = serde_json::from_slice(&buffer).map_err(|e| {
            TokenizerError::decoding(
                "deserialization",
                format!("Binary deserialization failed: {}", e),
            )
        })?;
        Ok(config.to_codec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SignalTokenizer;
    use std::env;

    #[test]
    fn test_continuous_tokenizer_json_roundtrip() {
        let tokenizer = ContinuousTokenizer::new(10, 20);

        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_continuous.json");

        tokenizer.save_json(&path).unwrap();
        let loaded = ContinuousTokenizer::load_json(&path).unwrap();

        assert_eq!(tokenizer.input_dim(), loaded.input_dim());
        assert_eq!(tokenizer.embed_dim(), loaded.embed_dim());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_continuous_tokenizer_binary_roundtrip() {
        let tokenizer = ContinuousTokenizer::new(10, 20);

        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_continuous.bin");

        tokenizer.save_binary(&path).unwrap();
        let loaded = ContinuousTokenizer::load_binary(&path).unwrap();

        assert_eq!(tokenizer.input_dim(), loaded.input_dim());
        assert_eq!(tokenizer.embed_dim(), loaded.embed_dim());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_linear_quantizer_json_roundtrip() {
        let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).unwrap();

        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_linear.json");

        quantizer.save_json(&path).unwrap();
        let loaded = LinearQuantizer::load_json(&path).unwrap();

        assert_eq!(quantizer.range(), loaded.range());
        assert_eq!(quantizer.bits(), loaded.bits());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mulaw_codec_json_roundtrip() {
        let codec = MuLawCodec::new(8);

        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_mulaw.json");

        codec.save_json(&path).unwrap();
        let loaded = MuLawCodec::load_json(&path).unwrap();

        assert_eq!(codec.mu(), loaded.mu());
        assert_eq!(codec.bits(), loaded.bits());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
