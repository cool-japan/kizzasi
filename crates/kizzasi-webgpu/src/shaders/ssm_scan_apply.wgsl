// SSM scan — block-prefix application pass.
//
// Second half of the multi-block scan.  `data` holds the block-local inclusive
// scan produced by `ssm_scan_block.wgsl`; `prefixes` holds the *scanned* block
// aggregates, i.e. prefixes[b] is the aggregate of blocks 0..=b.
//
// For every element of block b > 0 this applies the running prefix of all
// preceding blocks:
//   data[i] = prefixes[b - 1] ⊗ data[i]
//           = (la·pa, la·pbu + lbu)
// where (pa, pbu) = prefixes[b - 1] and (la, lbu) = data[i].
//
// Block 0 already carries the correct values and is skipped.
//
// Dispatch with the same work-group size as the block kernel so that
// `workgroup_id.x` is the block index.

struct Params {
    n: u32,
}

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read>       prefixes: array<f32>;
@group(0) @binding(2) var<storage, read_write> data:     array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let i: u32 = gid.x;
    let b: u32 = wid.x;

    if b == 0u || i >= params.n {
        return;
    }

    let pa:  f32 = prefixes[2u * (b - 1u)];
    let pbu: f32 = prefixes[2u * (b - 1u) + 1u];
    let la:  f32 = data[2u * i];
    let lbu: f32 = data[2u * i + 1u];

    data[2u * i]      = la * pa;
    data[2u * i + 1u] = la * pbu + lbu;
}
