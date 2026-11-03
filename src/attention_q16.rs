//! The encoder's attention core on int16 values (`--precision int16`), flash-style as `encoder::attend`, with both
//! products on `q16`'s kernel:
//!
//! * `q k^T`: q rounded to 13 bits per row (once per task), each key to 12 bits with its own scale;
//! * `p v`: the probabilities (at most 1 after the running max is subtracted) to 15 bits on one fixed scale, each
//!   value dimension to 10 bits with a scale per head and dimension; the int32 sums go back to float32 every
//!   `FLUSH_KEYS` keys (`64 * 32767 * 1023 < 2^31`).  The softmax's sum is taken over the rounded probabilities, so
//!   numerator and denominator round alike.

use rayon::prelude::*;

use crate::encoder::{HEAD_DIM, HEADS, WIDTH, exp_shifted, row_max};
use crate::gemm::{MR, NR};
use crate::q16;

const KEY_MAX: f32 = 2047.0;
const VALUE_MAX: f32 = 1023.0;
const PROB_MAX: f32 = 32767.0;
const FLUSH_KEYS: usize = 64;
const _: () = assert!((FLUSH_KEYS as u64) * (PROB_MAX as u64) * (VALUE_MAX as u64) < (1u64 << 31));
const KEY_PAIRS: usize = HEAD_DIM / 2;
const PANEL: usize = KEY_PAIRS * NR * 2; // int16 per panel of 16 keys
const DIM_PANELS: usize = HEAD_DIM / NR;

/// The keys and values of one window as int16.
pub struct Quantized {
    keys: Vec<i16>,    // [HEADS][padded / 16][32 pairs][16 keys][2]
    kscale: Vec<f32>,  // [HEADS][padded]
    values: Vec<i16>,  // [HEADS][4 dimension panels][vkeys / 2 pairs][16 dimensions][2]
    vscale: Vec<f32>,  // [HEADS][HEAD_DIM]
    padded: usize,     // keys rounded to a panel of 16
    vkeys: usize,      // keys rounded to `FLUSH_KEYS`
}

impl Quantized {
    pub fn new(padded: usize) -> Self {
        let vkeys = padded.div_ceil(FLUSH_KEYS) * FLUSH_KEYS;
        Self {
            keys: vec![0; HEADS * padded / NR * PANEL],
            kscale: vec![0.0; HEADS * padded],
            values: vec![0; HEADS * HEAD_DIM * vkeys],
            vscale: vec![0.0; HEADS * HEAD_DIM],
            padded,
            vkeys,
        }
    }

    /// From `encoder`'s float32 layouts: keys `[HEADS][padded / 16][HEAD_DIM][16]`, values `[HEADS][padded][HEAD_DIM]`
    /// (their first `frames` rows).
    pub fn fill(&mut self, keys: &[f32], values: &[f32], frames: usize) {
        let (padded, vkeys) = (self.padded, self.vkeys);
        let panels = padded / NR;
        self.keys.par_chunks_mut(PANEL).zip(self.kscale.par_chunks_mut(NR)).enumerate().for_each(|(i, (q, scale))| {
            let src = &keys[i * HEAD_DIM * NR..(i + 1) * HEAD_DIM * NR]; // [HEAD_DIM][16]
            for (j, s) in scale.iter_mut().enumerate() {
                let max = (0..HEAD_DIM).map(|d| src[d * NR + j].abs()).fold(0.0f32, f32::max);
                *s = max / KEY_MAX;
                let inv = if max > 0.0 { KEY_MAX / max } else { 0.0 };
                for d in 0..HEAD_DIM {
                    q[((d / 2) * NR + j) * 2 + d % 2] = (src[d * NR + j] * inv).round_ties_even() as i16;
                }
            }
        });
        debug_assert_eq!(self.keys.len() / PANEL, HEADS * panels);
        let per_head = HEAD_DIM * vkeys;
        self.values.par_chunks_mut(per_head).zip(self.vscale.par_chunks_mut(HEAD_DIM)).enumerate().for_each(
            |(head, (q, scale))| {
                let src = &values[head * padded * HEAD_DIM..][..frames * HEAD_DIM];
                let mut max = [0.0f32; HEAD_DIM];
                for row in src.chunks_exact(HEAD_DIM) {
                    max.iter_mut().zip(row).for_each(|(m, v)| *m = m.max(v.abs()));
                }
                let mut inv = [0.0f32; HEAD_DIM];
                for d in 0..HEAD_DIM {
                    scale[d] = max[d] / VALUE_MAX;
                    inv[d] = if max[d] > 0.0 { VALUE_MAX / max[d] } else { 0.0 };
                }
                q.fill(0);
                for (t, row) in src.chunks_exact(HEAD_DIM).enumerate() {
                    for (d, (&v, g)) in row.iter().zip(&inv).enumerate() {
                        let (dp, j) = (d / NR, d % NR);
                        q[((dp * (vkeys / 2) + t / 2) * NR + j) * 2 + t % 2] = (v * g).round_ties_even() as i16;
                    }
                }
            },
        );
    }
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

/// `encoder::attend` on the int16 keys and values: `qo` holds q `[frames + MR, WIDTH]` and gets the output.
pub fn attend(qo: &mut [f32], quant: &Quantized, frames: usize) {
    const QB: usize = 16 * MR;
    const KB: usize = 256; // a multiple of FLUSH_KEYS
    const _: () = assert!(KB.is_multiple_of(FLUSH_KEYS));
    assert!(qo.len() >= (frames + MR) * WIDTH);
    let (padded, vkeys) = (quant.padded, quant.vkeys);
    let blocks = frames.div_ceil(QB);
    let out_ptr = SendPtr(qo.as_mut_ptr());
    let prob_scale = [1.0 / PROB_MAX; MR];
    (0..HEADS * blocks).into_par_iter().for_each(|task| {
        let (head, qb) = (task / blocks, task % blocks);
        let q0 = qb * QB;
        let rows = QB.min(frames - q0);
        let row_panels = rows.div_ceil(MR);
        let keys = &quant.keys[head * (padded / NR) * PANEL..(head + 1) * (padded / NR) * PANEL];
        let kscale = &quant.kscale[head * padded..(head + 1) * padded];
        let values = &quant.values[head * HEAD_DIM * vkeys..(head + 1) * HEAD_DIM * vkeys];
        let vscale = &quant.vscale[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        // the task's q rows as int16 pairs, `[row panel][32 pairs][MR]`, read before any output is written
        let mut qpack = vec![0i32; row_panels * KEY_PAIRS * MR];
        let mut qscale = vec![0.0f32; row_panels * MR];
        for (r, s) in qscale.iter_mut().take(rows).enumerate() {
            let q = unsafe { std::slice::from_raw_parts(out_ptr.get().add((q0 + r) * WIDTH + head * HEAD_DIM), HEAD_DIM) };
            let rp = r / MR;
            *s = q16::quantize_row(q, None, &mut qpack[rp * KEY_PAIRS * MR..(rp + 1) * KEY_PAIRS * MR], r % MR);
        }
        let mut scores = vec![0.0f32; row_panels * MR * KB];
        let mut ppack = vec![0i32; row_panels * (KB / 2) * MR];
        let mut acc = vec![0.0f32; row_panels * MR * HEAD_DIM];
        let mut max = vec![f32::NEG_INFINITY; QB];
        let mut sum = vec![0.0f32; QB];
        for k0 in (0..frames).step_by(KB) {
            let kb = KB.min(frames - k0);
            let pairs = kb.div_ceil(FLUSH_KEYS) * FLUSH_KEYS / 2;
            // scores[r, j] = q_r . k_(k0 + j)
            for rp in 0..row_panels {
                let a = &qpack[rp * KEY_PAIRS * MR..(rp + 1) * KEY_PAIRS * MR];
                for cp in 0..kb.div_ceil(NR) {
                    let p = k0 / NR + cp;
                    q16::kernel(
                        KEY_PAIRS,
                        a,
                        &keys[p * PANEL..(p + 1) * PANEL],
                        &qscale[rp * MR..(rp + 1) * MR],
                        &kscale[p * NR..(p + 1) * NR],
                        &mut scores[rp * MR * KB + cp * NR..],
                        KB,
                        false,
                    );
                }
            }
            // the running max and sum; the probabilities as int16 pairs `[row panel][pairs][MR]`
            for r in 0..row_panels * MR {
                let (rp, ri) = (r / MR, r % MR);
                let dst = &mut ppack[rp * (KB / 2) * MR..];
                if r >= rows {
                    (0..pairs).for_each(|p| dst[p * MR + ri] = 0);
                    continue;
                }
                let row = &mut scores[r * KB..r * KB + kb];
                let new_max = max[r].max(row_max(row));
                let correction = (max[r] - new_max).exp();
                max[r] = new_max;
                exp_shifted(row, new_max);
                let total = pack_probs(row, pairs, dst, ri);
                sum[r] = sum[r] * correction + total as f32 / PROB_MAX;
                if correction != 1.0 {
                    for v in &mut acc[r * HEAD_DIM..(r + 1) * HEAD_DIM] {
                        *v *= correction;
                    }
                }
            }
            // acc += p v, back to float32 every FLUSH_KEYS keys
            for rp in 0..row_panels {
                for dp in 0..DIM_PANELS {
                    for f in 0..pairs / (FLUSH_KEYS / 2) {
                        let p0 = f * FLUSH_KEYS / 2;
                        let b0 = (dp * (vkeys / 2) + k0 / 2 + p0) * NR * 2;
                        q16::kernel(
                            FLUSH_KEYS / 2,
                            &ppack[(rp * (KB / 2) + p0) * MR..],
                            &values[b0..b0 + FLUSH_KEYS * NR],
                            &prob_scale,
                            &vscale[dp * NR..(dp + 1) * NR],
                            &mut acc[rp * MR * HEAD_DIM + dp * NR..],
                            HEAD_DIM,
                            true,
                        );
                    }
                }
            }
        }
        for r in 0..rows {
            let inv = 1.0 / sum[r];
            // each task writes its own rows of its own head's columns
            let o = unsafe { std::slice::from_raw_parts_mut(out_ptr.get().add((q0 + r) * WIDTH + head * HEAD_DIM), HEAD_DIM) };
            for (o, a) in o.iter_mut().zip(&acc[r * HEAD_DIM..(r + 1) * HEAD_DIM]) {
                *o = a * inv;
            }
        }
    });
}

/// `row` (`kb` probabilities, at most 1) times `PROB_MAX` as int16 pairs at `dst[p * MR + ri]` for `pairs` pairs (zero
/// past the row); returns their sum.
fn pack_probs(row: &[f32], pairs: usize, dst: &mut [i32], ri: usize) -> i64 {
    assert!(dst.len() > (pairs - 1) * MR + ri && 2 * pairs >= row.len());
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        return unsafe { pack_probs_avx2(row, pairs, dst, ri) };
    }
    pack_probs_scalar(row, pairs, dst, ri, 0)
}

fn pack_probs_scalar(row: &[f32], pairs: usize, dst: &mut [i32], ri: usize, from: usize) -> i64 {
    let q = |j: usize| if j < row.len() { (row[j] * PROB_MAX).round_ties_even() as i16 } else { 0 };
    let mut total = 0i64;
    for p in from..pairs {
        let (lo, hi) = (q(2 * p), q(2 * p + 1));
        total += lo as i64 + hi as i64;
        dst[p * MR + ri] = (lo as u16 as u32 | ((hi as u16 as u32) << 16)) as i32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_probs_avx2(row: &[f32], pairs: usize, dst: &mut [i32], ri: usize) -> i64 {
    use std::arch::x86_64::*;
    let scale = _mm256_set1_ps(PROB_MAX);
    let full = row.len() / 16 * 16;
    let mut sums = _mm256_setzero_si256();
    let d = dst.as_mut_ptr().add(ri);
    for i in (0..full).step_by(16) {
        let q0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(row.as_ptr().add(i)), scale));
        let q1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(row.as_ptr().add(i + 8)), scale));
        sums = _mm256_add_epi32(sums, _mm256_add_epi32(q0, q1));
        let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(q0, q1));
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, p);
        let to = d.add(i / 2 * MR);
        for (j, v) in lanes.into_iter().enumerate() {
            *to.add(j * MR) = v;
        }
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sums);
    // per lane at most (KB / 16) * 2 * 32767: no overflow
    lanes.iter().map(|&v| v as i64).sum::<i64>() + pack_probs_scalar(row, pairs, dst, ri, full / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_agrees() {
        let row: Vec<f32> = (0..53).map(|i| ((i * 37) % 101) as f32 / 100.0).collect();
        let (mut a, mut b) = (vec![0; 40 * MR], vec![0; 40 * MR]);
        let sa = pack_probs(&row, 32, &mut a, 3);
        let sb = pack_probs_scalar(&row, 32, &mut b, 3, 0);
        assert_eq!((sa, a), (sb, b));
    }
}
