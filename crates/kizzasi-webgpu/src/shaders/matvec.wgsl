// Matrix-vector multiplication kernel.
//
// Computes: output[row] = Σ_col matrix[row * cols + col] * vector[col]
//
// Layout:
//   matrix is row-major with shape [rows, cols]
//   vector has length cols
//   output has length rows
//
// Each thread processes one output row. Dispatch with
//   workgroups = (rows + 63) / 64

struct MatvecParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform>             params: MatvecParams;
@group(0) @binding(1) var<storage, read>       matrix: array<f32>;
@group(0) @binding(2) var<storage, read>       vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row: u32 = gid.x;
    if row >= params.rows {
        return;
    }

    var acc: f32 = 0.0;
    for (var col: u32 = 0u; col < params.cols; col = col + 1u) {
        acc = acc + matrix[row * params.cols + col] * vector[col];
    }
    output[row] = acc;
}
