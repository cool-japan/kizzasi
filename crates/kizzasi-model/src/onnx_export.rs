//! ONNX weight export for kizzasi-model
//!
//! Serializes kizzasi model **weights** to ONNX protobuf tensors using manual
//! byte encoding (Pure Rust approach — no prost, no build.rs). This module
//! provides two levels of support:
//!
//! - [`export_weights_to_onnx`]: a flat weight dump — every tensor in the
//!   input map becomes a graph *initializer*, with no `nodes`/`inputs`/
//!   `outputs`. This is a portable container for weights, not a runnable
//!   ONNX graph.
//! - [`export_linear_layer`]: builds one real, runnable ONNX graph (a
//!   `MatMul` node, optionally followed by an `Add` for the bias) for a
//!   single linear layer.
//!
//! No model type in this crate (Mamba, RWKV, S4, Transformer, …) has a
//! `to_onnx`/`export` method that walks its full architecture and emits the
//! corresponding sequence of ONNX ops — several of these SSM recurrences
//! have no direct ONNX operator equivalent, so a faithful per-architecture
//! graph exporter is a separate, much larger undertaking than tensor
//! serialization. Callers who need a full computation graph must currently
//! compose one from [`OnnxGraph`]/[`OnnxTensor`]/[`OnnxValueInfo`] (or
//! [`export_linear_layer`] repeated per layer) themselves.
//!
//! ## ONNX Protobuf Reference
//! <https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3>
//!
//! Wire types used:
//! - 0: varint (int32, int64, bool, enum)
//! - 2: length-delimited (string, bytes, embedded messages, repeated fields)
//! - 5: 32-bit (float)

use crate::{ModelError, ModelResult};
use std::collections::HashMap;

// ============================================================
// Minimal protobuf encoder
// ============================================================

/// Minimal protobuf wire-format encoder.
///
/// All functions produce raw bytes that can be concatenated to form
/// a valid protobuf message. This module intentionally exposes its
/// functions as `pub` so the tests can unit-test individual primitives.
pub mod proto {
    /// Encode an unsigned 64-bit integer as a varint.
    pub fn encode_varint(value: u64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(10);
        let mut v = value;
        loop {
            if v < 0x80 {
                buf.push(v as u8);
                break;
            }
            buf.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }
        buf
    }

    /// Build a field tag byte sequence: `(field << 3) | wire_type`.
    pub fn field_tag(field: u32, wire_type: u32) -> Vec<u8> {
        encode_varint(((field as u64) << 3) | wire_type as u64)
    }

    /// Encode a `string` field (wire type 2).
    pub fn encode_string(field: u32, s: &str) -> Vec<u8> {
        let bytes = s.as_bytes();
        let mut out = field_tag(field, 2);
        out.extend(encode_varint(bytes.len() as u64));
        out.extend_from_slice(bytes);
        out
    }

    /// Encode a `bytes` field (wire type 2).
    pub fn encode_bytes(field: u32, b: &[u8]) -> Vec<u8> {
        let mut out = field_tag(field, 2);
        out.extend(encode_varint(b.len() as u64));
        out.extend_from_slice(b);
        out
    }

    /// Encode an `int32` field (wire type 0).
    ///
    /// Returns empty bytes when the value is zero (default protobuf behaviour).
    pub fn encode_i32(field: u32, v: i32) -> Vec<u8> {
        if v == 0 {
            return vec![];
        }
        let mut out = field_tag(field, 0);
        out.extend(encode_varint(v as i64 as u64));
        out
    }

    /// Encode an `int64` field (wire type 0).
    ///
    /// Returns empty bytes when the value is zero (default protobuf behaviour).
    pub fn encode_i64(field: u32, v: i64) -> Vec<u8> {
        if v == 0 {
            return vec![];
        }
        let mut out = field_tag(field, 0);
        out.extend(encode_varint(v as u64));
        out
    }

    /// Encode a `float` field (wire type 5 = 32-bit little-endian).
    pub fn encode_f32(field: u32, v: f32) -> Vec<u8> {
        let mut out = field_tag(field, 5);
        out.extend_from_slice(&v.to_le_bytes());
        out
    }

    /// Encode an embedded sub-message (wire type 2).
    ///
    /// Returns empty bytes when `msg` is empty.
    pub fn encode_submessage(field: u32, msg: &[u8]) -> Vec<u8> {
        if msg.is_empty() {
            return vec![];
        }
        let mut out = field_tag(field, 2);
        out.extend(encode_varint(msg.len() as u64));
        out.extend_from_slice(msg);
        out
    }

    /// Encode a `[]f32` slice as `raw_data` bytes (field, wire type 2).
    ///
    /// Uses the efficient `raw_data` approach (field 9 in `TensorProto`)
    /// rather than per-element `float_data`.
    pub fn encode_float_slice_as_raw(field: u32, floats: &[f32]) -> Vec<u8> {
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        encode_bytes(field, &bytes)
    }

    /// Encode repeated `int64` values as a packed field (wire type 2).
    ///
    /// Returns empty bytes when `vals` is empty.
    pub fn encode_packed_i64(field: u32, vals: &[i64]) -> Vec<u8> {
        if vals.is_empty() {
            return vec![];
        }
        let mut inner = Vec::new();
        for &v in vals {
            inner.extend(encode_varint(v as u64));
        }
        encode_bytes(field, &inner)
    }

    /// Encode repeated `float` values as a packed field (wire type 2).
    ///
    /// Returns empty bytes when `vals` is empty.
    pub fn encode_packed_f32(field: u32, vals: &[f32]) -> Vec<u8> {
        if vals.is_empty() {
            return vec![];
        }
        // packed floats: concatenate 4-byte LE representations
        let mut inner = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            inner.extend_from_slice(&v.to_le_bytes());
        }
        encode_bytes(field, &inner)
    }
}

// ============================================================
// ONNX data type constants (TensorProto.DataType)
// ============================================================

/// ONNX data type constants (from `onnx.proto` `TensorProto.DataType`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(i32)]
pub enum OnnxDataType {
    /// 32-bit IEEE floating point
    Float = 1,
    /// 8-bit signed integer
    Int8 = 3,
    /// 32-bit signed integer
    Int32 = 6,
    /// 64-bit signed integer
    Int64 = 7,
    /// 16-bit IEEE floating point (half)
    Float16 = 10,
    /// 64-bit IEEE floating point (double)
    Double = 11,
}

impl OnnxDataType {
    /// Return the protobuf enum integer value.
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

// ============================================================
// ONNX graph element types
// ============================================================

/// A single ONNX tensor initializer (weight blob).
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Tensor name (used to reference this weight in the graph).
    pub name: String,
    /// Shape dimensions.
    pub dims: Vec<i64>,
    /// Element data type.
    pub data_type: OnnxDataType,
    /// Float values (only populated when `data_type == Float`).
    pub float_data: Vec<f32>,
}

impl OnnxTensor {
    /// Serialize to `TensorProto` bytes.
    ///
    /// Field numbers (onnx.proto3 `TensorProto`):
    /// - dims:      1 (int64 repeated, packed)
    /// - data_type: 2 (int32)
    /// - name:      8 (string)
    /// - raw_data:  9 (bytes)
    ///
    /// # Errors
    /// Returns [`ModelError::InvalidConfig`] when `data_type` is not `Float`
    /// but `float_data` is populated. `OnnxTensor` has only one data payload
    /// field (`float_data: Vec<f32>`, 4 bytes/element); writing it out as
    /// `raw_data` while the header declares a different `data_type` (e.g.
    /// `Float16`, 2 bytes/element, or `Int8`, 1 byte/element) would produce a
    /// `TensorProto` whose byte length contradicts its own dtype tag —
    /// silent corruption for any reader that trusts the header.
    fn to_proto_bytes(&self) -> ModelResult<Vec<u8>> {
        if self.data_type != OnnxDataType::Float && !self.float_data.is_empty() {
            return Err(ModelError::invalid_config(format!(
                "OnnxTensor '{}': data_type is {:?} but float_data has {} values \
                 populated — only Float tensors can be serialized from float_data \
                 (raw_data is always written as an f32 little-endian blob)",
                self.name,
                self.data_type,
                self.float_data.len()
            )));
        }
        let mut buf = Vec::new();
        // field 1: dims (packed int64)
        buf.extend(proto::encode_packed_i64(1, &self.dims));
        // field 2: data_type
        buf.extend(proto::encode_i32(2, self.data_type.as_i32()));
        // field 8: name
        buf.extend(proto::encode_string(8, &self.name));
        // field 9: raw_data (f32 little-endian blob)
        if !self.float_data.is_empty() {
            buf.extend(proto::encode_float_slice_as_raw(9, &self.float_data));
        }
        Ok(buf)
    }
}

// ============================================================
// ONNX value info
// ============================================================

/// ONNX value info (input/output declaration).
#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    /// Tensor name.
    pub name: String,
    /// Element data type.
    pub data_type: OnnxDataType,
    /// Shape dimensions. `None` means a dynamic (symbolic) dimension.
    pub shape: Vec<Option<i64>>,
}

impl OnnxValueInfo {
    /// Serialize to `ValueInfoProto` bytes.
    ///
    /// Field numbers:
    /// - name: 1 (string)
    /// - type: 2 (TypeProto)
    ///
    /// `TypeProto`:
    /// - tensor_type: 1 (TypeProto.Tensor)
    ///
    /// `TypeProto.Tensor`:
    /// - elem_type: 1 (int32)
    /// - shape:     2 (TensorShapeProto)
    ///
    /// `TensorShapeProto`:
    /// - dim: 1 (Dimension repeated)
    ///
    /// `TensorShapeProto.Dimension`:
    /// - dim_value: 1 (int64)  — static
    /// - dim_param: 2 (string) — dynamic (symbolic)
    fn to_proto_bytes(&self) -> Vec<u8> {
        // Build TensorShapeProto
        let mut shape_buf = Vec::new();
        for dim_opt in &self.shape {
            let dim_bytes = match dim_opt {
                Some(v) => {
                    // dim_value: field 1 int64. Emitted unconditionally, even
                    // when `*v == 0` — this is a `TensorShapeProto.Dimension`
                    // oneof entry, where `proto::encode_i64`'s usual
                    // default-value elision (correct for a plain optional
                    // scalar field) would instead drop the *entire*
                    // `Dimension` submessage below, silently shrinking the
                    // declared tensor rank instead of representing a
                    // genuine 0-sized dimension.
                    let mut b = proto::field_tag(1, 0);
                    b.extend(proto::encode_varint(*v as u64));
                    b
                }
                None => {
                    // dim_param: field 2 string (use "?" as placeholder)
                    proto::encode_string(2, "?")
                }
            };
            shape_buf.extend(proto::encode_submessage(1, &dim_bytes));
        }

        // Build TypeProto.Tensor
        let mut tensor_type_buf = Vec::new();
        // elem_type: field 1 int32
        tensor_type_buf.extend(proto::encode_i32(1, self.data_type.as_i32()));
        // shape: field 2 TensorShapeProto
        if !shape_buf.is_empty() {
            tensor_type_buf.extend(proto::encode_submessage(2, &shape_buf));
        }

        // Build TypeProto (tensor_type is field 1)
        let type_proto_buf = proto::encode_submessage(1, &tensor_type_buf);

        // Build ValueInfoProto
        let mut buf = Vec::new();
        buf.extend(proto::encode_string(1, &self.name));
        buf.extend(proto::encode_submessage(2, &type_proto_buf));
        buf
    }
}

// ============================================================
// ONNX attribute
// ============================================================

/// Attribute type enum values (AttributeProto.AttributeType).
#[repr(i32)]
enum AttributeType {
    Int = 1,
    Float = 4,
    String = 3,
    Ints = 7,
    Floats = 6,
}

/// An ONNX operator attribute.
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    /// Integer attribute.
    Int(String, i64),
    /// Float attribute.
    Float(String, f32),
    /// String/bytes attribute.
    String(String, Vec<u8>),
    /// Repeated integer attribute.
    Ints(String, Vec<i64>),
    /// Repeated float attribute.
    Floats(String, Vec<f32>),
}

impl OnnxAttribute {
    /// Serialize to `AttributeProto` bytes.
    ///
    /// Field numbers:
    /// - name:   1 (string)
    /// - i:      3 (int64)
    /// - f:      4 (float)
    /// - s:      8 (bytes)
    /// - floats: 6 (repeated float, packed)
    /// - ints:   7 (repeated int64, packed)
    /// - type:  20 (int32, AttributeType enum)
    fn to_proto_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            OnnxAttribute::Int(name, v) => {
                buf.extend(proto::encode_string(1, name));
                buf.extend(proto::encode_i64(3, *v));
                buf.extend(proto::encode_i32(20, AttributeType::Int as i32));
            }
            OnnxAttribute::Float(name, v) => {
                buf.extend(proto::encode_string(1, name));
                buf.extend(proto::encode_f32(4, *v));
                buf.extend(proto::encode_i32(20, AttributeType::Float as i32));
            }
            OnnxAttribute::String(name, v) => {
                buf.extend(proto::encode_string(1, name));
                buf.extend(proto::encode_bytes(8, v));
                buf.extend(proto::encode_i32(20, AttributeType::String as i32));
            }
            OnnxAttribute::Ints(name, vals) => {
                buf.extend(proto::encode_string(1, name));
                buf.extend(proto::encode_packed_i64(7, vals));
                buf.extend(proto::encode_i32(20, AttributeType::Ints as i32));
            }
            OnnxAttribute::Floats(name, vals) => {
                buf.extend(proto::encode_string(1, name));
                buf.extend(proto::encode_packed_f32(6, vals));
                buf.extend(proto::encode_i32(20, AttributeType::Floats as i32));
            }
        }
        buf
    }
}

// ============================================================
// ONNX node (operator)
// ============================================================

/// An ONNX computation node.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Operator type (e.g. `"MatMul"`, `"Add"`).
    pub op_type: String,
    /// Node name (for debugging).
    pub name: String,
    /// Names of input tensors/values.
    pub inputs: Vec<String>,
    /// Names of output tensors/values.
    pub outputs: Vec<String>,
    /// Operator attributes.
    pub attributes: Vec<OnnxAttribute>,
}

impl OnnxNode {
    /// Serialize to `NodeProto` bytes.
    ///
    /// Field numbers:
    /// - input:     1 (string repeated)
    /// - output:    2 (string repeated)
    /// - name:      3 (string)
    /// - op_type:   4 (string)
    /// - attribute: 5 (AttributeProto repeated)
    fn to_proto_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for inp in &self.inputs {
            buf.extend(proto::encode_string(1, inp));
        }
        for out in &self.outputs {
            buf.extend(proto::encode_string(2, out));
        }
        buf.extend(proto::encode_string(3, &self.name));
        buf.extend(proto::encode_string(4, &self.op_type));
        for attr in &self.attributes {
            let attr_bytes = attr.to_proto_bytes();
            buf.extend(proto::encode_submessage(5, &attr_bytes));
        }
        buf
    }
}

// ============================================================
// ONNX graph
// ============================================================

/// A complete ONNX computational graph.
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// Graph name.
    pub name: String,
    /// Computation nodes in topological order.
    pub nodes: Vec<OnnxNode>,
    /// Graph-level inputs (dynamic activations, not weights).
    pub inputs: Vec<OnnxValueInfo>,
    /// Graph-level outputs.
    pub outputs: Vec<OnnxValueInfo>,
    /// Weight tensors (initializers).
    pub initializers: Vec<OnnxTensor>,
}

impl OnnxGraph {
    /// Serialize to `GraphProto` bytes.
    ///
    /// Field numbers:
    /// - node:        1 (NodeProto repeated)
    /// - name:        2 (string)
    /// - initializer: 6 (TensorProto repeated)
    /// - input:      11 (ValueInfoProto repeated)
    /// - output:     12 (ValueInfoProto repeated)
    ///
    /// # Errors
    /// Propagates [`OnnxTensor::to_proto_bytes`]'s dtype/payload validation
    /// error for any mismatched initializer.
    fn to_proto_bytes(&self) -> ModelResult<Vec<u8>> {
        let mut buf = Vec::new();
        // field 1: nodes
        for node in &self.nodes {
            let node_bytes = node.to_proto_bytes();
            buf.extend(proto::encode_submessage(1, &node_bytes));
        }
        // field 2: name
        buf.extend(proto::encode_string(2, &self.name));
        // field 6: initializers (weight tensors)
        for init in &self.initializers {
            let init_bytes = init.to_proto_bytes()?;
            buf.extend(proto::encode_submessage(6, &init_bytes));
        }
        // field 11: inputs
        for inp in &self.inputs {
            let inp_bytes = inp.to_proto_bytes();
            buf.extend(proto::encode_submessage(11, &inp_bytes));
        }
        // field 12: outputs
        for out in &self.outputs {
            let out_bytes = out.to_proto_bytes();
            buf.extend(proto::encode_submessage(12, &out_bytes));
        }
        Ok(buf)
    }
}

// ============================================================
// ONNX model
// ============================================================

/// A complete ONNX model ready for serialization.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// ONNX IR version (typically 8).
    pub ir_version: i64,
    /// Default opset version (typically 17).
    pub opset_version: i64,
    /// Operator domain (empty string for the default ONNX domain).
    pub domain: String,
    /// The computation graph.
    pub graph: OnnxGraph,
    /// Human-readable description.
    pub doc_string: String,
}

impl OnnxModel {
    /// Create a new model with sane ONNX defaults.
    pub fn new(graph: OnnxGraph) -> Self {
        Self {
            ir_version: 8,
            opset_version: 17,
            domain: String::new(),
            graph,
            doc_string: "Generated by Kizzasi".to_string(),
        }
    }

    /// Serialize to ONNX protobuf bytes.
    ///
    /// `ModelProto` field numbers:
    /// - ir_version:    1 (int64)
    /// - opset_import:  8 (OperatorSetIdProto repeated)
    /// - graph:         7 (GraphProto)
    /// - doc_string:   12 (string)
    ///
    /// `OperatorSetIdProto`:
    /// - domain:  1 (string)
    /// - version: 2 (int64)
    pub fn to_bytes(&self) -> ModelResult<Vec<u8>> {
        let mut buf = Vec::new();

        // field 1: ir_version
        buf.extend(proto::encode_i64(1, self.ir_version));

        // field 7: graph (GraphProto)
        let graph_bytes = self.graph.to_proto_bytes()?;
        buf.extend(proto::encode_submessage(7, &graph_bytes));

        // field 8: opset_import (OperatorSetIdProto)
        //   domain:  field 1 (string)
        //   version: field 2 (int64)
        let mut opset_buf = Vec::new();
        opset_buf.extend(proto::encode_string(1, &self.domain));
        opset_buf.extend(proto::encode_i64(2, self.opset_version));
        buf.extend(proto::encode_submessage(8, &opset_buf));

        // field 12: doc_string
        if !self.doc_string.is_empty() {
            buf.extend(proto::encode_string(12, &self.doc_string));
        }

        Ok(buf)
    }

    /// Write the serialized model to a file at `path`.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> ModelResult<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(ModelError::IoError)
    }
}

// ============================================================
// High-level export helpers
// ============================================================

/// Build a weight-only ONNX model from a flat weight map.
///
/// This creates a graph with no computation nodes — only initializer
/// tensors. This is useful for checkpoint export and weight inspection
/// by tools such as Netron.
///
/// # Arguments
/// * `weights` — map from tensor name to flat `f32` data.
/// * `shapes`  — map from tensor name to `[d0, d1, …]` shape (same keys).
/// * `model_name` — used as the graph name.
///
/// # Errors
/// Returns [`ModelError::InvalidConfig`] if a name appears in `weights`
/// but not in `shapes`, or if the shape volume does not match the data length.
pub fn export_weights_to_onnx(
    weights: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    model_name: &str,
) -> ModelResult<OnnxModel> {
    let mut initializers = Vec::with_capacity(weights.len());

    // Iterate in a deterministic order for reproducibility.
    let mut names: Vec<&String> = weights.keys().collect();
    names.sort();

    for name in names {
        let data = &weights[name];
        let shape = shapes.get(name).ok_or_else(|| {
            ModelError::invalid_config(format!(
                "export_weights_to_onnx: shape missing for tensor '{name}'"
            ))
        })?;

        // Validate shape volume vs data length.
        let volume: usize = shape.iter().product();
        if volume != data.len() {
            return Err(ModelError::InvalidConfig {
                message: format!(
                    "export_weights_to_onnx: tensor '{name}' shape {:?} has volume {volume} \
                     but data length is {}",
                    shape,
                    data.len()
                ),
            });
        }

        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        initializers.push(OnnxTensor {
            name: name.clone(),
            dims,
            data_type: OnnxDataType::Float,
            float_data: data.clone(),
        });
    }

    let graph = OnnxGraph {
        name: model_name.to_string(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        initializers,
    };

    Ok(OnnxModel::new(graph))
}

/// Export a single fully-connected (linear) layer as an ONNX sub-graph.
///
/// Produces:
/// - One `MatMul` node: `input × weight^T → matmul_out`
/// - Optionally one `Add` node: `matmul_out + bias → output`
///
/// The weight tensor is stored transposed relative to PyTorch convention
/// so that standard `MatMul` (`[batch, in_features] × [in_features, out_features]`)
/// produces the correct result.
///
/// # Arguments
/// * `weight`       — flat row-major weight data, shape `[out, in]`.
/// * `weight_shape` — `[out_features, in_features]` (2 elements).
/// * `bias`         — optional bias of length `out_features`.
/// * `input_name`   — name of the graph-level input value.
/// * `output_name`  — name of the graph-level output value.
/// * `layer_name`   — prefix used to name internal tensors and nodes.
///
/// # Errors
/// Returns [`ModelError::InvalidConfig`] if `weight_shape` does not have
/// exactly 2 elements or if the bias length is inconsistent.
pub fn export_linear_layer(
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    input_name: &str,
    output_name: &str,
    layer_name: &str,
) -> ModelResult<OnnxGraph> {
    if weight_shape.len() != 2 {
        return Err(ModelError::invalid_config(format!(
            "export_linear_layer: weight_shape must have exactly 2 elements, \
             got {}",
            weight_shape.len()
        )));
    }

    let out_features = weight_shape[0];
    let in_features = weight_shape[1];
    let expected_len = out_features * in_features;

    if weight.len() != expected_len {
        return Err(ModelError::invalid_config(format!(
            "export_linear_layer: weight length {} does not match shape {:?} (volume {})",
            weight.len(),
            weight_shape,
            expected_len
        )));
    }

    // Transpose weight from [out, in] to [in, out] for ONNX MatMul.
    // ONNX MatMul: [batch, in] × [in, out] → [batch, out]
    let mut weight_transposed = vec![0.0f32; expected_len];
    for o in 0..out_features {
        for i in 0..in_features {
            weight_transposed[i * out_features + o] = weight[o * in_features + i];
        }
    }

    let weight_name = format!("{layer_name}.weight");
    let matmul_out_name = format!("{layer_name}.matmul_out");

    let weight_tensor = OnnxTensor {
        name: weight_name.clone(),
        dims: vec![in_features as i64, out_features as i64],
        data_type: OnnxDataType::Float,
        float_data: weight_transposed,
    };

    let mut nodes = Vec::new();
    let mut initializers = vec![weight_tensor];

    // Decide whether the MatMul feeds directly into the output or into Add.
    let matmul_output = if bias.is_some() {
        matmul_out_name.clone()
    } else {
        output_name.to_string()
    };

    let matmul_node = OnnxNode {
        op_type: "MatMul".to_string(),
        name: format!("{layer_name}/MatMul"),
        inputs: vec![input_name.to_string(), weight_name],
        outputs: vec![matmul_output],
        attributes: vec![],
    };
    nodes.push(matmul_node);

    if let Some(bias_data) = bias {
        if bias_data.len() != out_features {
            return Err(ModelError::invalid_config(format!(
                "export_linear_layer: bias length {} does not match out_features {}",
                bias_data.len(),
                out_features
            )));
        }

        let bias_name = format!("{layer_name}.bias");
        let bias_tensor = OnnxTensor {
            name: bias_name.clone(),
            dims: vec![out_features as i64],
            data_type: OnnxDataType::Float,
            float_data: bias_data.to_vec(),
        };
        initializers.push(bias_tensor);

        let add_node = OnnxNode {
            op_type: "Add".to_string(),
            name: format!("{layer_name}/Add"),
            inputs: vec![matmul_out_name, bias_name],
            outputs: vec![output_name.to_string()],
            attributes: vec![],
        };
        nodes.push(add_node);
    }

    // Build graph-level I/O declarations.
    let graph_input = OnnxValueInfo {
        name: input_name.to_string(),
        data_type: OnnxDataType::Float,
        // batch dimension is dynamic
        shape: vec![None, Some(in_features as i64)],
    };
    let graph_output = OnnxValueInfo {
        name: output_name.to_string(),
        data_type: OnnxDataType::Float,
        shape: vec![None, Some(out_features as i64)],
    };

    Ok(OnnxGraph {
        name: layer_name.to_string(),
        nodes,
        inputs: vec![graph_input],
        outputs: vec![graph_output],
        initializers,
    })
}

// ============================================================
// Additional builder helpers
// ============================================================

/// Builder for constructing `OnnxGraph`s incrementally.
///
/// Useful when programmatically assembling multi-layer graphs.
#[derive(Debug, Default)]
pub struct OnnxGraphBuilder {
    name: String,
    nodes: Vec<OnnxNode>,
    inputs: Vec<OnnxValueInfo>,
    outputs: Vec<OnnxValueInfo>,
    initializers: Vec<OnnxTensor>,
}

impl OnnxGraphBuilder {
    /// Create a new builder with the given graph name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Add a computation node.
    pub fn add_node(mut self, node: OnnxNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Declare a graph-level input.
    pub fn add_input(mut self, vi: OnnxValueInfo) -> Self {
        self.inputs.push(vi);
        self
    }

    /// Declare a graph-level output.
    pub fn add_output(mut self, vi: OnnxValueInfo) -> Self {
        self.outputs.push(vi);
        self
    }

    /// Add a weight initializer tensor.
    pub fn add_initializer(mut self, tensor: OnnxTensor) -> Self {
        self.initializers.push(tensor);
        self
    }

    /// Consume the builder and produce an [`OnnxGraph`].
    pub fn build(self) -> OnnxGraph {
        OnnxGraph {
            name: self.name,
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
            initializers: self.initializers,
        }
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_proto_varint_encoding() {
        use super::proto::encode_varint;
        assert_eq!(encode_varint(0), vec![0]);
        assert_eq!(encode_varint(1), vec![1]);
        assert_eq!(encode_varint(127), vec![127]);
        assert_eq!(encode_varint(128), vec![0x80, 0x01]);
        assert_eq!(encode_varint(300), vec![0xAC, 0x02]);
    }

    #[test]
    fn test_onnx_model_to_bytes_nonempty() {
        let graph = OnnxGraph {
            name: "test".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![OnnxTensor {
                name: "w".to_string(),
                dims: vec![2, 3],
                data_type: OnnxDataType::Float,
                float_data: vec![1.0f32; 6],
            }],
        };
        let model = OnnxModel::new(graph);
        let bytes = model.to_bytes().expect("serialization must succeed");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_value_info_zero_dim_is_not_dropped() {
        // A dimension of exactly 0 must still be represented as a
        // `TensorShapeProto.Dimension` submessage, not silently omitted
        // (which would shrink the declared tensor rank).
        let with_zero_dim = OnnxValueInfo {
            name: "x".to_string(),
            data_type: OnnxDataType::Float,
            shape: vec![Some(0), Some(4)],
        };
        let without_dims = OnnxValueInfo {
            name: "x".to_string(),
            data_type: OnnxDataType::Float,
            shape: vec![],
        };
        let with_bytes = with_zero_dim.to_proto_bytes();
        let without_bytes = without_dims.to_proto_bytes();
        assert!(
            with_bytes.len() > without_bytes.len(),
            "a shape containing a 0-sized dimension must serialize to more \
             bytes than no shape at all -- the 0 dimension must not be \
             silently dropped"
        );
    }

    #[test]
    fn test_tensor_dtype_mismatch_rejected() {
        let bad_tensor = OnnxTensor {
            name: "bad".to_string(),
            dims: vec![2],
            data_type: OnnxDataType::Float16,
            float_data: vec![1.0, 2.0],
        };
        let graph = OnnxGraph {
            name: "test".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![bad_tensor],
        };
        let model = OnnxModel::new(graph);
        assert!(
            model.to_bytes().is_err(),
            "a Float16-tagged tensor with float_data populated (always \
             encoded as f32 raw bytes) must be rejected, not silently \
             serialized with a contradicting dtype tag"
        );
    }

    #[test]
    fn test_tensor_dtype_mismatch_allows_empty_payload() {
        // A non-Float dtype with EMPTY float_data is fine (e.g. a
        // shape-only placeholder) -- only a *populated* mismatched payload
        // is an error.
        let placeholder = OnnxTensor {
            name: "placeholder".to_string(),
            dims: vec![2],
            data_type: OnnxDataType::Int8,
            float_data: vec![],
        };
        let graph = OnnxGraph {
            name: "test".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![placeholder],
        };
        let model = OnnxModel::new(graph);
        assert!(model.to_bytes().is_ok());
    }

    #[test]
    fn test_onnx_save_and_file_size() {
        let graph = OnnxGraph {
            name: "weight_export".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![OnnxTensor {
                name: "embed.weight".to_string(),
                dims: vec![4, 8],
                data_type: OnnxDataType::Float,
                float_data: (0..32).map(|i| i as f32 * 0.01).collect(),
            }],
        };
        let model = OnnxModel::new(graph);
        let path = std::env::temp_dir().join("test_kizzasi_export.onnx");
        model.save(&path).expect("save must succeed");
        let metadata = std::fs::metadata(&path).expect("file must exist after save");
        assert!(metadata.len() > 10, "exported file must be non-trivial");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_weights_to_onnx() {
        let mut weights = HashMap::new();
        weights.insert("layer.weight".to_string(), vec![1.0f32; 12]);
        let mut shapes = HashMap::new();
        shapes.insert("layer.weight".to_string(), vec![3, 4]);
        let model =
            export_weights_to_onnx(&weights, &shapes, "test_model").expect("export must succeed");
        assert_eq!(model.graph.initializers.len(), 1);
        assert_eq!(model.graph.initializers[0].dims, vec![3i64, 4]);
    }

    #[test]
    fn test_export_linear_layer() {
        let weight = vec![1.0f32; 6]; // [2, 3]
        let bias = vec![0.1f32; 2];
        let graph =
            export_linear_layer(&weight, &[2, 3], Some(&bias), "input", "output", "linear0")
                .expect("export_linear_layer must succeed");
        // Should have MatMul and Add nodes
        assert!(!graph.nodes.is_empty(), "must have at least one node");
        assert!(
            !graph.initializers.is_empty(),
            "must have at least one initializer"
        );
        // With bias we expect exactly two nodes: MatMul and Add
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].op_type, "MatMul");
        assert_eq!(graph.nodes[1].op_type, "Add");
    }

    #[test]
    fn test_export_linear_layer_no_bias() {
        let weight = vec![0.5f32; 12]; // [3, 4]
        let graph = export_linear_layer(&weight, &[3, 4], None, "x", "y", "fc")
            .expect("export must succeed");
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "MatMul");
        assert_eq!(graph.initializers.len(), 1);
    }

    #[test]
    fn test_onnx_tensor_raw_data() {
        // Verify that raw_data encoding stores f32 as little-endian bytes.
        let floats = vec![1.0f32, 2.0, 3.0];
        let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(raw.len(), 12);
        let recovered: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(recovered, floats);
    }

    #[test]
    fn test_export_weights_shape_mismatch() {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), vec![1.0f32; 6]);
        let mut shapes = HashMap::new();
        // Wrong volume: 3×3 = 9 ≠ 6
        shapes.insert("w".to_string(), vec![3, 3]);
        let result = export_weights_to_onnx(&weights, &shapes, "bad");
        assert!(result.is_err(), "must fail on volume mismatch");
    }

    #[test]
    fn test_export_weights_missing_shape() {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), vec![1.0f32; 6]);
        let shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let result = export_weights_to_onnx(&weights, &shapes, "bad");
        assert!(result.is_err(), "must fail when shape is absent");
    }

    #[test]
    fn test_onnx_graph_builder() {
        let graph = OnnxGraphBuilder::new("built_graph")
            .add_initializer(OnnxTensor {
                name: "param".to_string(),
                dims: vec![4, 4],
                data_type: OnnxDataType::Float,
                float_data: vec![0.0f32; 16],
            })
            .add_input(OnnxValueInfo {
                name: "x".to_string(),
                data_type: OnnxDataType::Float,
                shape: vec![None, Some(4)],
            })
            .add_output(OnnxValueInfo {
                name: "y".to_string(),
                data_type: OnnxDataType::Float,
                shape: vec![None, Some(4)],
            })
            .build();
        assert_eq!(graph.name, "built_graph");
        assert_eq!(graph.initializers.len(), 1);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_attribute_int_encoding() {
        let attr = OnnxAttribute::Int("axis".to_string(), 1);
        let bytes = attr.to_proto_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_attribute_floats_encoding() {
        let attr = OnnxAttribute::Floats("scales".to_string(), vec![1.0, 2.0, 3.0]);
        let bytes = attr.to_proto_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_multi_tensor_export_round_trip() {
        // Create a model with multiple initializers and write/check it.
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        for layer in 0..4 {
            let key = format!("layer{layer}.weight");
            weights.insert(key.clone(), vec![0.1f32; 8]);
            shapes.insert(key, vec![2, 4]);
            let bkey = format!("layer{layer}.bias");
            weights.insert(bkey.clone(), vec![0.0f32; 2]);
            shapes.insert(bkey, vec![2]);
        }

        let model =
            export_weights_to_onnx(&weights, &shapes, "multi_layer").expect("export must succeed");
        assert_eq!(model.graph.initializers.len(), 8);

        let bytes = model.to_bytes().expect("to_bytes must succeed");
        assert!(!bytes.is_empty());

        // Write and read back.
        let path = std::env::temp_dir().join("test_kizzasi_multi_layer.onnx");
        model.save(&path).expect("save must succeed");
        let on_disk = std::fs::read(&path).expect("must read back file");
        assert_eq!(on_disk, bytes);
        let _ = std::fs::remove_file(&path);
    }
}
