//! SSM-specific backward pass types and free functions.
//!
//! This module is a sub-module of [`crate::backprop`] and contains the
//! higher-level SSM backward infrastructure that operates on flat `Vec<f32>`
//! buffers (rather than `ndarray` arrays used by the lower-level tape):
//!
//! - [`SsmForwardCache`]: cached forward activations for an SSM sequence.
//! - [`SsmGradientsVec`]: gradient output type with flat `Vec<f32>` fields.
//! - [`ssm_backward`]: the core backward pass function.
//! - [`associative_scan_backward`]: backward through a parallel prefix scan.
//! - [`GradientCheckpointedSSM`]: memory-efficient checkpointed backward.

use crate::error::{ModelError, ModelResult};

// ---------------------------------------------------------------------------
// SsmForwardCache — cached activations for the SSM backward pass
// ---------------------------------------------------------------------------

/// Cached forward-pass activations needed for an SSM backward pass.
///
/// All matrices are stored in row-major (C-contiguous) flat `Vec<f32>`.
///
/// Dimension conventions:
/// - `hidden_states`: `seq_len × d_state` (includes the initial state at index 0,
///   so total length is `(seq_len + 1) × d_state`).
/// - `inputs`: `seq_len × d_model`.
/// - `outputs`: `seq_len × d_model`.
/// - `a_bar`: `d_state` (diagonal, shared across time steps).
/// - `b_bar`: `d_state × d_model` (row-major).
/// - `c_mat`: `d_model × d_state` (row-major).
#[derive(Debug, Clone)]
pub struct SsmForwardCache {
    /// Hidden states at each step: `(seq_len + 1) × d_state`.
    /// `hidden_states[t * d_state .. (t+1) * d_state]` is h_t.
    pub hidden_states: Vec<f32>,
    /// Input sequence: `seq_len × d_model`.
    pub inputs: Vec<f32>,
    /// Output sequence: `seq_len × d_model`.
    pub outputs: Vec<f32>,
    /// Discretised A (diagonal): length `d_state`.
    pub a_bar: Vec<f32>,
    /// Discretised B: `d_state × d_model` (row-major).
    pub b_bar: Vec<f32>,
    /// C output projection: `d_model × d_state` (row-major).
    pub c_mat: Vec<f32>,
    /// Sequence length.
    pub seq_len: usize,
    /// Model / input dimension.
    pub d_model: usize,
    /// State dimension.
    pub d_state: usize,
}

impl SsmForwardCache {
    /// Construct a cache from raw component slices.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::dimension_mismatch`] if any slice has the wrong length.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_states: Vec<f32>,
        inputs: Vec<f32>,
        outputs: Vec<f32>,
        a_bar: Vec<f32>,
        b_bar: Vec<f32>,
        c_mat: Vec<f32>,
        seq_len: usize,
        d_model: usize,
        d_state: usize,
    ) -> ModelResult<Self> {
        let expected_h = (seq_len + 1) * d_state;
        if hidden_states.len() != expected_h {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache hidden_states",
                expected_h,
                hidden_states.len(),
            ));
        }
        let expected_in = seq_len * d_model;
        if inputs.len() != expected_in {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache inputs",
                expected_in,
                inputs.len(),
            ));
        }
        if outputs.len() != expected_in {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache outputs",
                expected_in,
                outputs.len(),
            ));
        }
        if a_bar.len() != d_state {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache a_bar",
                d_state,
                a_bar.len(),
            ));
        }
        let expected_b = d_state * d_model;
        if b_bar.len() != expected_b {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache b_bar",
                expected_b,
                b_bar.len(),
            ));
        }
        let expected_c = d_model * d_state;
        if c_mat.len() != expected_c {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache c_mat",
                expected_c,
                c_mat.len(),
            ));
        }
        Ok(Self {
            hidden_states,
            inputs,
            outputs,
            a_bar,
            b_bar,
            c_mat,
            seq_len,
            d_model,
            d_state,
        })
    }

    /// Run the SSM forward pass, filling `self` with activations.
    ///
    /// Recurrence: `h_t = A_bar * h_{t-1} + B_bar * x_t`, `y_t = C * h_t`
    ///
    /// - `A_bar` is diagonal (length `d_state`).
    /// - `B_bar` is `d_state × d_model` row-major.
    /// - `C` is `d_model × d_state` row-major.
    /// - `inputs` is `seq_len × d_model` row-major.
    ///
    /// Returns a new cache populated with the forward activations.
    pub fn run_forward(
        inputs: &[f32],
        a_bar: &[f32],
        b_bar: &[f32],
        c_mat: &[f32],
        seq_len: usize,
        d_model: usize,
        d_state: usize,
    ) -> ModelResult<Self> {
        if inputs.len() != seq_len * d_model {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache::run_forward inputs",
                seq_len * d_model,
                inputs.len(),
            ));
        }
        if a_bar.len() != d_state {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache::run_forward a_bar",
                d_state,
                a_bar.len(),
            ));
        }
        if b_bar.len() != d_state * d_model {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache::run_forward b_bar",
                d_state * d_model,
                b_bar.len(),
            ));
        }
        if c_mat.len() != d_model * d_state {
            return Err(ModelError::dimension_mismatch(
                "SsmForwardCache::run_forward c_mat",
                d_model * d_state,
                c_mat.len(),
            ));
        }

        let mut hidden_states = vec![0.0_f32; (seq_len + 1) * d_state];
        let mut outputs_buf = vec![0.0_f32; seq_len * d_model];

        // Initial state is zeros (already set by vec! initialiser).
        for t in 0..seq_len {
            let h_prev_start = t * d_state;
            let h_curr_start = (t + 1) * d_state;
            let x_start = t * d_model;

            // h_t[s] = a_bar[s] * h_{t-1}[s] + sum_d(b_bar[s, d] * x[t, d])
            for s in 0..d_state {
                let mut bx = 0.0_f32;
                for d in 0..d_model {
                    bx += b_bar[s * d_model + d] * inputs[x_start + d];
                }
                hidden_states[h_curr_start + s] = a_bar[s] * hidden_states[h_prev_start + s] + bx;
            }

            // y_t[d] = sum_s(c_mat[d, s] * h_t[s])
            let y_start = t * d_model;
            for d in 0..d_model {
                let mut y_val = 0.0_f32;
                for s in 0..d_state {
                    y_val += c_mat[d * d_state + s] * hidden_states[h_curr_start + s];
                }
                outputs_buf[y_start + d] = y_val;
            }
        }

        Self::new(
            hidden_states,
            inputs.to_vec(),
            outputs_buf,
            a_bar.to_vec(),
            b_bar.to_vec(),
            c_mat.to_vec(),
            seq_len,
            d_model,
            d_state,
        )
    }
}

// ---------------------------------------------------------------------------
// Gradients for SSM free-function interface
// ---------------------------------------------------------------------------

/// Gradients with respect to all SSM parameters, produced by [`ssm_backward`].
#[derive(Debug, Clone, Default)]
pub struct SsmGradientsVec {
    /// `∂L/∂A_bar`: length `d_state` (diagonal).
    pub grad_a_bar: Vec<f32>,
    /// `∂L/∂B_bar`: `d_state × d_model` row-major.
    pub grad_b_bar: Vec<f32>,
    /// `∂L/∂C`: `d_model × d_state` row-major.
    pub grad_c: Vec<f32>,
    /// `∂L/∂x`: `seq_len × d_model` row-major.
    pub grad_input: Vec<f32>,
    /// `∂L/∂h_0`: length `d_state` (gradient w.r.t. initial hidden state).
    pub grad_hidden_init: Vec<f32>,
}

/// Compute the backward pass through an SSM recurrence.
///
/// Recurrence forward: `h_t = A_bar * h_{t-1} + B_bar * x_t`, `y_t = C * h_t`
///
/// Backward equations:
/// ```text
/// ∂L/∂h_t   = C^T · ∂L/∂y_t + A_bar^T · ∂L/∂h_{t+1}
/// ∂L/∂x_t   = B_bar^T · ∂L/∂h_t
/// ∂L/∂h_{t-1} = A_bar^T · ∂L/∂h_t          (propagated as dh_next)
/// ∂L/∂A_bar   += ∂L/∂h_t ⊙ h_{t-1}
/// ∂L/∂B_bar   += ∂L/∂h_t ⊗ x_t
/// ∂L/∂C       += ∂L/∂y_t ⊗ h_t
/// ```
///
/// # Parameters
///
/// - `cache`: forward activations cached during the forward pass.
/// - `grad_output`: upstream gradient `∂L/∂y`, shape `seq_len × d_model` (row-major flat).
///
/// # Errors
///
/// Returns [`ModelError`] on dimension mismatch or non-finite values.
pub fn ssm_backward(cache: &SsmForwardCache, grad_output: &[f32]) -> ModelResult<SsmGradientsVec> {
    let seq = cache.seq_len;
    let dm = cache.d_model;
    let ds = cache.d_state;

    if grad_output.len() != seq * dm {
        return Err(ModelError::dimension_mismatch(
            "ssm_backward grad_output",
            seq * dm,
            grad_output.len(),
        ));
    }

    // Check finite inputs.
    for &v in grad_output.iter() {
        if !v.is_finite() {
            return Err(ModelError::numerical_instability(
                "ssm_backward",
                "non-finite value in grad_output",
            ));
        }
    }
    for &v in cache.hidden_states.iter() {
        if !v.is_finite() {
            return Err(ModelError::numerical_instability(
                "ssm_backward",
                "non-finite value in hidden_states",
            ));
        }
    }

    let mut grad_a_bar = vec![0.0_f32; ds];
    let mut grad_b_bar = vec![0.0_f32; ds * dm];
    let mut grad_c = vec![0.0_f32; dm * ds];
    let mut grad_input = vec![0.0_f32; seq * dm];
    let mut grad_hidden_init = vec![0.0_f32; ds];

    // dh flowing back from step t+1.
    let mut dh_next = vec![0.0_f32; ds];

    for t in (0..seq).rev() {
        let x_start = t * dm;
        let h_prev_start = t * ds; // h_{t-1}
        let h_curr_start = (t + 1) * ds; // h_t
        let y_start = t * dm;

        // ∂L/∂C += ∂L/∂y_t ⊗ h_t   (d_model × d_state outer product)
        for d in 0..dm {
            let dy_td = grad_output[y_start + d];
            for s in 0..ds {
                grad_c[d * ds + s] += dy_td * cache.hidden_states[h_curr_start + s];
            }
        }

        // ∂L/∂h_t = C^T · ∂L/∂y_t + A_bar^T · dh_{t+1}
        // C is (d_model × d_state): C^T has shape (d_state × d_model)
        // So (C^T · dy_t)[s] = sum_d C[d, s] * dy_t[d]
        let mut dh_t = vec![0.0_f32; ds];
        for s in 0..ds {
            let mut ct_dy = 0.0_f32;
            for d in 0..dm {
                ct_dy += cache.c_mat[d * ds + s] * grad_output[y_start + d];
            }
            // A_bar is diagonal: A_bar^T = A_bar (diagonal)
            dh_t[s] = ct_dy + cache.a_bar[s] * dh_next[s];
        }

        // ∂L/∂A_bar[s] += dh_t[s] * h_{t-1}[s]
        for s in 0..ds {
            grad_a_bar[s] += dh_t[s] * cache.hidden_states[h_prev_start + s];
        }

        // ∂L/∂B_bar[s, d] += dh_t[s] * x_t[d]
        for s in 0..ds {
            for d in 0..dm {
                grad_b_bar[s * dm + d] += dh_t[s] * cache.inputs[x_start + d];
            }
        }

        // ∂L/∂x_t[d] = sum_s B_bar[s, d] * dh_t[s]
        for d in 0..dm {
            let mut bx = 0.0_f32;
            #[allow(clippy::needless_range_loop)]
            for s in 0..ds {
                bx += cache.b_bar[s * dm + d] * dh_t[s];
            }
            grad_input[x_start + d] = bx;
        }

        dh_next = dh_t;
    }

    // Gradient w.r.t. initial hidden state (h_0).
    for s in 0..ds {
        grad_hidden_init[s] = cache.a_bar[s] * dh_next[s];
    }

    Ok(SsmGradientsVec {
        grad_a_bar,
        grad_b_bar,
        grad_c,
        grad_input,
        grad_hidden_init,
    })
}

// ---------------------------------------------------------------------------
// Associative scan backward
// ---------------------------------------------------------------------------

/// Backward pass through an associative (parallel prefix) scan.
///
/// Given the forward scan:
/// ```text
/// out[0] = x[0]
/// out[t] = a[t] * out[t-1] + x[t]   for t >= 1
/// ```
/// where `a` and `x` are `seq_len × d_state` flat arrays (row-major).
///
/// The reverse scan accumulates upstream gradients:
/// ```text
/// grad_x[t] = grad_out[t] + a[t+1] * grad_x[t+1]  (reverse)
/// ```
///
/// # Parameters
///
/// - `scan_outputs`: forward scan outputs, `seq_len × d_state` flat.
/// - `a_seq`: per-step diagonal A values, `seq_len × d_state` flat.
/// - `grad_output`: upstream gradient, `seq_len × d_state` flat.
/// - `d_state`: state dimension.
///
/// # Returns
///
/// `grad_input`: `seq_len × d_state` flat, gradient w.r.t. the scan input `x`.
///
/// # Errors
///
/// Returns [`ModelError`] on dimension mismatch or non-finite values.
pub fn associative_scan_backward(
    scan_outputs: &[f32],
    a_seq: &[f32],
    grad_output: &[f32],
    d_state: usize,
) -> ModelResult<Vec<f32>> {
    if d_state == 0 {
        return Err(ModelError::invalid_config(
            "associative_scan_backward: d_state must be > 0",
        ));
    }
    let total = scan_outputs.len();
    if !total.is_multiple_of(d_state) {
        return Err(ModelError::dimension_mismatch(
            "associative_scan_backward scan_outputs not divisible by d_state",
            0,
            total,
        ));
    }
    let seq_len = total / d_state;

    if a_seq.len() != seq_len * d_state {
        return Err(ModelError::dimension_mismatch(
            "associative_scan_backward a_seq",
            seq_len * d_state,
            a_seq.len(),
        ));
    }
    if grad_output.len() != seq_len * d_state {
        return Err(ModelError::dimension_mismatch(
            "associative_scan_backward grad_output",
            seq_len * d_state,
            grad_output.len(),
        ));
    }

    for &v in scan_outputs
        .iter()
        .chain(a_seq.iter())
        .chain(grad_output.iter())
    {
        if !v.is_finite() {
            return Err(ModelError::numerical_instability(
                "associative_scan_backward",
                "non-finite input value",
            ));
        }
    }

    let mut grad_input = vec![0.0_f32; seq_len * d_state];

    // Reverse scan: accumulate dL/dx[t].
    // dg[t] = dL/d out[t] + a[t+1] * dg[t+1]
    // dL/dx[t] = dg[t]
    let mut dg = vec![0.0_f32; d_state];

    for t in (0..seq_len).rev() {
        let base = t * d_state;
        #[allow(clippy::manual_memcpy)]
        for s in 0..d_state {
            // Accumulate upstream gradient then store; not a plain copy.
            dg[s] += grad_output[base + s];
            grad_input[base + s] = dg[s];
        }
        // Propagate: dg for the previous step = a[t] * dg_current
        if t > 0 {
            for s in 0..d_state {
                dg[s] = a_seq[base + s] * grad_input[base + s];
            }
        }
    }

    Ok(grad_input)
}

// ---------------------------------------------------------------------------
// GradientCheckpointedSSM
// ---------------------------------------------------------------------------

/// Gradient checkpointing for SSM sequences.
///
/// Instead of caching all hidden states (O(seq_len × d_state) memory), this
/// divides the sequence into `segments` equal-sized chunks.  Only the
/// **boundary** hidden states are stored; each segment is recomputed from its
/// boundary during the backward pass.
///
/// This trades compute for memory: backward pass runs the forward pass once
/// more per segment (2× compute, O(segments × d_state) memory).
pub struct GradientCheckpointedSSM {
    /// Number of checkpointed segments.
    pub segments: usize,
    /// Model / input dimension.
    pub d_model: usize,
    /// State dimension.
    pub d_state: usize,
}

impl GradientCheckpointedSSM {
    /// Create a new checkpointed SSM.
    pub fn new(d_model: usize, d_state: usize, segments: usize) -> Self {
        Self {
            segments,
            d_model,
            d_state,
        }
    }

    /// Forward pass that caches only segment-boundary hidden states.
    ///
    /// Returns `(outputs, checkpoint_states)`:
    /// - `outputs`: `seq_len × d_model` flat.
    /// - `checkpoint_states`: `(segments + 1) × d_state` flat — the hidden
    ///   state **at the start** of each segment plus the final state.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] on dimension mismatch.
    pub fn forward_with_checkpoints(
        &self,
        inputs: &[f32],
        a_bar: &[f32],
        b_bar: &[f32],
        c_mat: &[f32],
    ) -> ModelResult<(Vec<f32>, Vec<f32>)> {
        let dm = self.d_model;
        let ds = self.d_state;
        let segs = self.segments.max(1);

        let seq_len = inputs.len().checked_div(dm).ok_or_else(|| {
            ModelError::invalid_config("GradientCheckpointedSSM: d_model is zero")
        })?;
        if inputs.len() != seq_len * dm {
            return Err(ModelError::dimension_mismatch(
                "GradientCheckpointedSSM inputs",
                seq_len * dm,
                inputs.len(),
            ));
        }
        if a_bar.len() != ds {
            return Err(ModelError::dimension_mismatch(
                "GradientCheckpointedSSM a_bar",
                ds,
                a_bar.len(),
            ));
        }
        if b_bar.len() != ds * dm {
            return Err(ModelError::dimension_mismatch(
                "GradientCheckpointedSSM b_bar",
                ds * dm,
                b_bar.len(),
            ));
        }
        if c_mat.len() != dm * ds {
            return Err(ModelError::dimension_mismatch(
                "GradientCheckpointedSSM c_mat",
                dm * ds,
                c_mat.len(),
            ));
        }

        // segment_size rounds up so all steps are covered.
        let seg_size = seq_len.div_ceil(segs);

        let mut outputs = vec![0.0_f32; seq_len * dm];
        // (segs + 1) checkpoint states: state at start of each segment + final.
        let mut checkpoint_states = vec![0.0_f32; (segs + 1) * ds];
        // h starts as zeros (initial state already written at index 0).

        let mut h = vec![0.0_f32; ds];

        for seg in 0..segs {
            // Save state at start of this segment.
            let cp_start = seg * ds;
            checkpoint_states[cp_start..cp_start + ds].copy_from_slice(&h);

            let t_start = seg * seg_size;
            let t_end = ((seg + 1) * seg_size).min(seq_len);

            for t in t_start..t_end {
                let x_start = t * dm;
                // h[s] = a_bar[s]*h[s] + sum_d b_bar[s,d]*x[t,d]
                for s in 0..ds {
                    let mut bx = 0.0_f32;
                    for d in 0..dm {
                        bx += b_bar[s * dm + d] * inputs[x_start + d];
                    }
                    h[s] = a_bar[s] * h[s] + bx;
                }
                // y[t,d] = sum_s c_mat[d,s]*h[s]
                let y_start = t * dm;
                for d in 0..dm {
                    let mut yv = 0.0_f32;
                    for s in 0..ds {
                        yv += c_mat[d * ds + s] * h[s];
                    }
                    outputs[y_start + d] = yv;
                }
            }
        }

        // Save final state.
        let cp_final = segs * ds;
        checkpoint_states[cp_final..cp_final + ds].copy_from_slice(&h);

        Ok((outputs, checkpoint_states))
    }

    /// Backward pass using checkpointed boundary states.
    ///
    /// Each segment is recomputed from its boundary state, then the backward
    /// pass is run through that segment.
    ///
    /// # Parameters
    ///
    /// - `inputs`: `seq_len × d_model` flat.
    /// - `checkpoint_states`: `(segments + 1) × d_state` flat (from forward pass).
    /// - `grad_output`: upstream gradient `seq_len × d_model` flat.
    /// - `a_bar`, `b_bar`, `c_mat`: SSM parameters (same as forward).
    ///
    /// # Returns
    ///
    /// Accumulated [`SsmGradientsVec`].
    pub fn backward_with_checkpoints(
        &self,
        inputs: &[f32],
        checkpoint_states: &[f32],
        grad_output: &[f32],
        a_bar: &[f32],
        b_bar: &[f32],
        c_mat: &[f32],
    ) -> ModelResult<SsmGradientsVec> {
        let dm = self.d_model;
        let ds = self.d_state;
        let segs = self.segments.max(1);

        let seq_len = inputs.len().checked_div(dm).ok_or_else(|| {
            ModelError::invalid_config("GradientCheckpointedSSM::backward: d_model is zero")
        })?;

        let seg_size = seq_len.div_ceil(segs);

        let mut acc = SsmGradientsVec {
            grad_a_bar: vec![0.0_f32; ds],
            grad_b_bar: vec![0.0_f32; ds * dm],
            grad_c: vec![0.0_f32; dm * ds],
            grad_input: vec![0.0_f32; seq_len * dm],
            grad_hidden_init: vec![0.0_f32; ds],
        };

        // dh flowing back from the end of the sequence.
        let mut dh_carry = vec![0.0_f32; ds];

        for seg in (0..segs).rev() {
            let t_start = seg * seg_size;
            let t_end = ((seg + 1) * seg_size).min(seq_len);
            let seg_len = t_end - t_start;
            if seg_len == 0 {
                continue;
            }

            // Recompute this segment's hidden states from checkpoint.
            let cp_start = seg * ds;
            let init_h = &checkpoint_states[cp_start..cp_start + ds];
            let mut seg_hidden = vec![0.0_f32; (seg_len + 1) * ds];
            seg_hidden[0..ds].copy_from_slice(init_h);

            for (i, t) in (t_start..t_end).enumerate() {
                let h_prev = i * ds;
                let h_curr = (i + 1) * ds;
                let x_start = t * dm;
                for s in 0..ds {
                    let mut bx = 0.0_f32;
                    for d in 0..dm {
                        bx += b_bar[s * dm + d] * inputs[x_start + d];
                    }
                    seg_hidden[h_curr + s] = a_bar[s] * seg_hidden[h_prev + s] + bx;
                }
            }

            // Build a mini-cache and call ssm_backward.
            let seg_inputs = inputs[t_start * dm..t_end * dm].to_vec();
            let seg_grad_out = grad_output[t_start * dm..t_end * dm].to_vec();

            let seg_cache = SsmForwardCache::new(
                seg_hidden,
                seg_inputs,
                vec![0.0_f32; seg_len * dm], // outputs not needed for backward
                a_bar.to_vec(),
                b_bar.to_vec(),
                c_mat.to_vec(),
                seg_len,
                dm,
                ds,
            )?;

            let mut seg_grads = ssm_backward(&seg_cache, &seg_grad_out)?;

            // Add dh_carry contribution from later segments.
            #[allow(clippy::needless_range_loop)]
            for s in 0..ds {
                seg_grads.grad_hidden_init[s] += dh_carry[s];
                acc.grad_a_bar[s] += seg_grads.grad_a_bar[s];
            }
            for i in 0..ds * dm {
                acc.grad_b_bar[i] += seg_grads.grad_b_bar[i];
            }
            for i in 0..dm * ds {
                acc.grad_c[i] += seg_grads.grad_c[i];
            }
            let in_start = t_start * dm;
            for i in 0..seg_len * dm {
                acc.grad_input[in_start + i] += seg_grads.grad_input[i];
            }

            // The carry for the next (earlier) segment is the hidden-init gradient.
            dh_carry.copy_from_slice(&seg_grads.grad_hidden_init);
        }

        // Final carry is the gradient w.r.t. the very first hidden state.
        acc.grad_hidden_init.copy_from_slice(&dh_carry);

        Ok(acc)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: scalar loss = sum of all outputs.
    fn ssm_scalar_loss(
        inputs: &[f32],
        a_bar: &[f32],
        b_bar: &[f32],
        c_mat: &[f32],
        seq_len: usize,
        d_model: usize,
        d_state: usize,
    ) -> f32 {
        let cache =
            SsmForwardCache::run_forward(inputs, a_bar, b_bar, c_mat, seq_len, d_model, d_state)
                .expect("forward ok");
        cache.outputs.iter().sum::<f32>()
    }

    // -----------------------------------------------------------------------
    // 1. ssm_backward — correct output tensor shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssm_backward_shapes() {
        let d_model = 4;
        let d_state = 4;
        let seq_len = 3;

        let a_bar = vec![0.9_f32; d_state];
        let b_bar = vec![0.1_f32; d_state * d_model];
        let c_mat = vec![1.0_f32; d_model * d_state];
        let inputs = vec![0.5_f32; seq_len * d_model];

        let cache = SsmForwardCache::run_forward(
            &inputs, &a_bar, &b_bar, &c_mat, seq_len, d_model, d_state,
        )
        .expect("run_forward ok");

        let grad_output = vec![1.0_f32; seq_len * d_model];
        let grads = ssm_backward(&cache, &grad_output).expect("ssm_backward ok");

        assert_eq!(grads.grad_a_bar.len(), d_state, "grad_a_bar length");
        assert_eq!(
            grads.grad_b_bar.len(),
            d_state * d_model,
            "grad_b_bar length"
        );
        assert_eq!(grads.grad_c.len(), d_model * d_state, "grad_c length");
        assert_eq!(
            grads.grad_input.len(),
            seq_len * d_model,
            "grad_input length"
        );
        assert_eq!(
            grads.grad_hidden_init.len(),
            d_state,
            "grad_hidden_init length"
        );
    }

    // -----------------------------------------------------------------------
    // 2. ssm_backward — numerical finite-difference gradient check
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssm_backward_finite_diff() {
        let d_model = 4;
        let d_state = 4;
        let seq_len = 3;
        let eps = 1e-4_f32;
        let tol = 1e-2_f32;

        let a_bar = vec![0.8_f32, 0.7, 0.9, 0.6];
        let b_bar = vec![0.1_f32; d_state * d_model];
        let c_mat = vec![0.5_f32; d_model * d_state];
        let inputs = vec![
            1.0_f32, -0.5, 0.3, 0.7, 0.2, -0.1, 0.4, 0.8, 0.0, 0.6, -0.3, 0.1,
        ];

        // Analytic gradient.
        let cache = SsmForwardCache::run_forward(
            &inputs, &a_bar, &b_bar, &c_mat, seq_len, d_model, d_state,
        )
        .expect("forward ok");
        let grad_output = vec![1.0_f32; seq_len * d_model];
        let grads = ssm_backward(&cache, &grad_output).expect("backward ok");

        // Finite-difference gradient w.r.t. inputs.
        for i in 0..inputs.len() {
            let mut inp_p = inputs.clone();
            let mut inp_m = inputs.clone();
            inp_p[i] += eps;
            inp_m[i] -= eps;
            let fp = ssm_scalar_loss(&inp_p, &a_bar, &b_bar, &c_mat, seq_len, d_model, d_state);
            let fm = ssm_scalar_loss(&inp_m, &a_bar, &b_bar, &c_mat, seq_len, d_model, d_state);
            let fd = (fp - fm) / (2.0 * eps);
            let analytic = grads.grad_input[i];
            assert!(
                (analytic - fd).abs() < tol,
                "grad_input[{i}]: analytic={analytic:.6} fd={fd:.6} diff={:.6}",
                (analytic - fd).abs()
            );
        }

        // Finite-difference gradient w.r.t. a_bar.
        for i in 0..a_bar.len() {
            let mut ap = a_bar.clone();
            let mut am = a_bar.clone();
            ap[i] += eps;
            am[i] -= eps;
            let fp = ssm_scalar_loss(&inputs, &ap, &b_bar, &c_mat, seq_len, d_model, d_state);
            let fm = ssm_scalar_loss(&inputs, &am, &b_bar, &c_mat, seq_len, d_model, d_state);
            let fd = (fp - fm) / (2.0 * eps);
            let analytic = grads.grad_a_bar[i];
            assert!(
                (analytic - fd).abs() < tol,
                "grad_a_bar[{i}]: analytic={analytic:.6} fd={fd:.6} diff={:.6}",
                (analytic - fd).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. GradientCheckpointedSSM — checkpointed forward matches full forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_checkpointing_forward_matches() {
        let d_model = 3;
        let d_state = 4;
        let seq_len = 9;
        let segments = 3;

        let a_bar = vec![0.9_f32, 0.8, 0.95, 0.7];
        let b_bar = vec![0.1_f32; d_state * d_model];
        let c_mat = vec![0.5_f32; d_model * d_state];
        let inputs: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.1) - 0.5)
            .collect();

        // Full forward pass via SsmForwardCache.
        let full_cache = SsmForwardCache::run_forward(
            &inputs, &a_bar, &b_bar, &c_mat, seq_len, d_model, d_state,
        )
        .expect("full forward ok");

        // Checkpointed forward.
        let ckpt = GradientCheckpointedSSM::new(d_model, d_state, segments);
        let (ckpt_outputs, _ckpt_states) = ckpt
            .forward_with_checkpoints(&inputs, &a_bar, &b_bar, &c_mat)
            .expect("checkpointed forward ok");

        assert_eq!(
            ckpt_outputs.len(),
            full_cache.outputs.len(),
            "output length mismatch"
        );
        for (i, (&full, &ckpt_v)) in full_cache
            .outputs
            .iter()
            .zip(ckpt_outputs.iter())
            .enumerate()
        {
            assert!(
                (full - ckpt_v).abs() < 1e-5,
                "outputs[{i}] mismatch: full={full:.8} ckpt={ckpt_v:.8}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. associative_scan_backward — consistency check vs finite differences
    // -----------------------------------------------------------------------

    #[test]
    fn test_associative_scan_backward() {
        let d_state = 2usize;
        let seq_len = 4usize;

        let a_seq = vec![
            0.9_f32, 0.8, // t=0
            0.7_f32, 0.6, // t=1
            0.95_f32, 0.85, // t=2
            0.5_f32, 0.4, // t=3
        ];
        let x = vec![1.0_f32, 0.5, -0.5_f32, 0.3, 0.2_f32, -0.1, 0.4_f32, 0.8];

        // Compute forward scan.
        let mut scan_out = vec![0.0_f32; seq_len * d_state];
        let mut prev = vec![0.0_f32; d_state];
        for t in 0..seq_len {
            let base = t * d_state;
            for s in 0..d_state {
                scan_out[base + s] = a_seq[base + s] * prev[s] + x[base + s];
            }
            prev = scan_out[base..base + d_state].to_vec();
        }

        let grad_output = vec![1.0_f32; seq_len * d_state];

        let grad_input = associative_scan_backward(&scan_out, &a_seq, &grad_output, d_state)
            .expect("associative_scan_backward ok");

        assert_eq!(grad_input.len(), seq_len * d_state, "grad_input length");

        // Numerical finite-difference check.
        let eps = 1e-4_f32;
        for perturb_t in 0..seq_len {
            for perturb_s in 0..d_state {
                let idx = perturb_t * d_state + perturb_s;

                let run_scan = |xv: &[f32]| -> f32 {
                    let mut out = vec![0.0_f32; seq_len * d_state];
                    let mut pv = vec![0.0_f32; d_state];
                    for t in 0..seq_len {
                        let b = t * d_state;
                        for s in 0..d_state {
                            out[b + s] = a_seq[b + s] * pv[s] + xv[b + s];
                        }
                        pv = out[b..b + d_state].to_vec();
                    }
                    out.iter().sum()
                };

                let mut xp = x.clone();
                xp[idx] += eps;
                let mut xm = x.clone();
                xm[idx] -= eps;
                let fd = (run_scan(&xp) - run_scan(&xm)) / (2.0 * eps);
                let analytic = grad_input[idx];
                assert!(
                    (analytic - fd).abs() < 1e-2,
                    "scan_backward grad[t={perturb_t},s={perturb_s}]: \
                     analytic={analytic:.6} fd={fd:.6}"
                );
            }
        }
    }
}
