// RMS Norm kernel (single-pass, single work-group, n ≤ 256).
//
// Computes: output[i] = (input[i] / rms(input)) * weight[i]
// where rms(x) = sqrt(mean(x²) + eps)
//
// Bind group layout (4 bindings):
//   binding 0: uniform { n: u32, eps: f32 }
//   binding 1: input  (storage, read)
//   binding 2: weight (storage, read)
//   binding 3: output (storage, read_write)
//
// Dispatch with exactly 1 workgroup. The caller MUST check n ≤ 256.

struct RmsNormParams {
    n:   u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform>             params: RmsNormParams;
@group(0) @binding(1) var<storage, read>       input:  array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Shared memory for parallel tree reduction of x².
var<workgroup> partial_sq: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i:       u32 = gid.x;
    let local_i: u32 = lid.x;

    // Load x² or 0.0 for out-of-bounds threads.
    if i < params.n {
        let x: f32 = input[i];
        partial_sq[local_i] = x * x;
    } else {
        partial_sq[local_i] = 0.0;
    }
    workgroupBarrier();

    // Parallel tree reduction to sum all x².
    var stride: u32 = 128u;
    loop {
        if stride == 0u {
            break;
        }
        if local_i < stride {
            partial_sq[local_i] = partial_sq[local_i] + partial_sq[local_i + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Normalize using the computed RMS.
    if i < params.n {
        let mean_sq: f32 = partial_sq[0] / f32(params.n);
        let rms: f32     = sqrt(mean_sq + params.eps);
        output[i] = (input[i] / rms) * weight[i];
    }
}
