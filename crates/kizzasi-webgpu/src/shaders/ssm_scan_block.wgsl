// Blelloch work-efficient parallel SSM scan — per-block pass.
//
// Implements an inclusive prefix scan over SSM elements (a, bu) with the
// associative operator:
//   (a1, bu1) ⊗ (a2, bu2) = (a2·a1, a2·bu1 + bu2)
// Identity element: (1.0, 0.0)
//
// Input/output layout: flat f32 buffer interleaved as
//   [a_0, bu_0, a_1, bu_1, ..., a_{n-1}, bu_{n-1}]
//
// Each work-group scans its own block of 256 elements and additionally writes
// the block aggregate (the inclusive value of the block's last lane) to
// `block_sums[workgroup_id]`.  Lanes past `n` load the identity element, which
// is a no-op on the right of the operator, so the aggregate is exact for a
// partially filled trailing block.
//
// A driver scans `block_sums` with the same kernel (recursively when there is
// more than one block of aggregates) and then applies the resulting prefixes
// with `ssm_scan_apply.wgsl`, so sequences of any length are handled exactly.

struct Params {
    n: u32,
}

@group(0) @binding(0) var<uniform>             params:     Params;
@group(0) @binding(1) var<storage, read>       input_buf:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> sh_a:  array<f32, 256>;
var<workgroup> sh_bu: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let i: u32       = gid.x;
    let local_i: u32 = lid.x;

    // --- Load into shared memory (pad with identity element beyond seq_len) ---
    var inp_a:  f32 = 1.0;
    var inp_bu: f32 = 0.0;
    if i < params.n {
        inp_a  = input_buf[2u * i];
        inp_bu = input_buf[2u * i + 1u];
    }
    sh_a[local_i]  = inp_a;
    sh_bu[local_i] = inp_bu;
    workgroupBarrier();

    // --- Up-sweep (reduce phase) ---
    // Each thread with index (stride*2 - 1, 3*stride*2 - 1, ...) combines
    // its element with the one stride positions to the left.
    var stride: u32 = 1u;
    loop {
        if stride >= WORKGROUP_SIZE { break; }
        if local_i >= stride && (local_i + 1u) % (2u * stride) == 0u {
            let a2: f32  = sh_a[local_i];
            let b2: f32  = sh_bu[local_i];
            let a1: f32  = sh_a[local_i - stride];
            let b1: f32  = sh_bu[local_i - stride];
            // Combine: (a1, b1) ⊗ right = (a2·a1, a2·b1 + b2)
            sh_a[local_i]  = a2 * a1;
            sh_bu[local_i] = a2 * b1 + b2;
        }
        workgroupBarrier();
        stride = stride << 1u;
    }

    // --- Clear last element to identity to start exclusive scan ---
    if local_i == WORKGROUP_SIZE - 1u {
        sh_a[local_i]  = 1.0;
        sh_bu[local_i] = 0.0;
    }
    workgroupBarrier();

    // --- Down-sweep (distribute phase) ---
    // Propagate the identity down to produce an exclusive prefix scan.
    stride = WORKGROUP_SIZE >> 1u;
    loop {
        if stride == 0u { break; }
        if local_i >= stride && (local_i + 1u) % (2u * stride) == 0u {
            let a_right: f32  = sh_a[local_i];
            let bu_right: f32 = sh_bu[local_i];
            let a_left: f32   = sh_a[local_i - stride];
            let bu_left: f32  = sh_bu[local_i - stride];
            // Left child gets the current (parent) exclusive prefix.
            sh_a[local_i - stride]  = a_right;
            sh_bu[local_i - stride] = bu_right;
            // Right child = left_child_exclusive ⊗ left_subtree_reduction
            // = (a_right, bu_right) ⊗ (a_left, bu_left)
            // = (a_left·a_right, a_left·bu_right + bu_left)
            sh_a[local_i]  = a_left * a_right;
            sh_bu[local_i] = a_left * bu_right + bu_left;
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // --- Convert exclusive → inclusive: incl[i] = excl[i] ⊗ input[i] ---
    // The exclusive scan gives the product of all elements *before* i within
    // this block.  Combining with input[i] yields the block-local inclusive
    // prefix at i.
    let excl_a: f32  = sh_a[local_i];
    let excl_bu: f32 = sh_bu[local_i];
    let incl_a: f32  = inp_a * excl_a;
    let incl_bu: f32 = inp_a * excl_bu + inp_bu;

    if i < params.n {
        output_buf[2u * i]      = incl_a;
        output_buf[2u * i + 1u] = incl_bu;
    }

    // The last lane holds the aggregate of the whole block (padding lanes are
    // identity elements and therefore do not perturb it).
    if local_i == WORKGROUP_SIZE - 1u {
        block_sums[2u * wid.x]      = incl_a;
        block_sums[2u * wid.x + 1u] = incl_bu;
    }
}
