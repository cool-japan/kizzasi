// SiLU (Swish) activation kernel.
//
// Computes: output[i] = input[i] / (1 + exp(-input[i]))
//
// Bind group layout (3 bindings — no weight):
//   binding 0: uniform { n: u32 }
//   binding 1: input  (storage, read)
//   binding 2: output (storage, read_write)
//
// Dispatch with workgroups = ceil(n / 256).

struct SiluParams {
    n: u32,
}

@group(0) @binding(0) var<uniform>             params: SiluParams;
@group(0) @binding(1) var<storage, read>       input:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn silu_scalar(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if i >= params.n {
        return;
    }
    output[i] = silu_scalar(input[i]);
}
