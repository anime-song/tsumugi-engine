//! Linear layers on int16 values, for CPUs with AVX2 but without AVX-VNNI: `vpmaddwd` multiplies 16 pairs of int16
//! and adds each pair into an int32, twice the multiply-adds per instruction of a float32 FMA.
//!
//! The weights are rounded at load to 12 bits per output channel (`quantize`); the rows of `a` to 13 bits per row and
//! slice of `KC` values of k, as they are packed.  A slice's int32 sums thus stay below 2^31
//! (`KC * 4095 * 2047 < 2^31`); each is scaled back to float32 (row scale times channel scale) and added up there.
//! The loop order is `gemm`'s otherwise: per slice of k the block's rows are packed once and every weight panel swept
//! over them by a 6 x 16 kernel (12 int32 accumulators).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cell::RefCell;

use rayon::prelude::*;

use crate::gemm::{Act, MR, NR, epilogue};
use crate::math;
use crate::simd;
use crate::weights::{Linear, Panels};

/// k per slice (even): 128 pairs.
pub const KC: usize = 256;
const A_MAX: f32 = 4095.0;
const W_MAX: f32 = 2047.0;
const _: () = assert!((KC as u64) * (A_MAX as u64) * (W_MAX as u64) < (1u64 << 31) && KC.is_multiple_of(2));

thread_local! {
    static SCRATCH: RefCell<(Vec<i32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// `lin` with its weights rounded to int16, 12 bits per output channel (its LoRA adapter kept as it is).
pub fn quantize(lin: &Linear) -> Linear {
    let (n, k) = (lin.n, lin.k);
    let panels = lin.panels();
    let pairs = k.div_ceil(2);
    let value = |i: usize| -> f32 {
        match &lin.w {
            Panels::F32(w) => w[i],
            Panels::F16(w) => half::f16::from_bits(w[i]).to_f32(),
            Panels::I8(w) => {
                let col = (i / (k * NR)) * NR + i % NR;
                if col < n { w[i] as f32 * lin.scale.expect("int8 weights without scales")[col] } else { 0.0 }
            }
            Panels::Q16(_) => panic!("already int16"),
        }
    };
    let mut q = vec![0i16; panels * pairs * NR * 2];
    let mut scale = vec![0.0f32; panels * NR];
    q.par_chunks_mut(pairs * NR * 2).zip(scale.par_chunks_mut(NR)).enumerate().for_each(|(panel, (q, scale))| {
        let at = |p: usize, j: usize| value((panel * k + p) * NR + j);
        for (j, s) in scale.iter_mut().enumerate() {
            let max = (0..k).map(|p| at(p, j).abs()).fold(0.0f32, f32::max);
            *s = max / W_MAX;
            let inv = if max > 0.0 { W_MAX / max } else { 0.0 };
            for p in 0..k {
                q[((p / 2) * NR + j) * 2 + p % 2] = (at(p, j) * inv).round_ties_even() as i16;
            }
        }
    });
    Linear {
        n,
        k,
        w: Panels::Q16(q.into()),
        scale: Some(Box::leak(scale.into_boxed_slice())),
        bias: lin.bias.clone(),
        lora: lin.lora.clone(),
    }
}

/// `gemm::product` for int16 weights `w`.
#[allow(clippy::too_many_arguments)]
pub fn product(
    a: &[f32],
    lda: usize,
    m: usize,
    lin: &Linear,
    w: &[i16],
    out: &mut [f32],
    ldo: usize,
    act: Act,
    accumulate: bool,
    a_scale: Option<&[f32]>,
) {
    let scale = lin.scale.expect("int16 weights without scales");
    let k = lin.k;
    let pairs = k.div_ceil(2);
    SCRATCH.with(|s| {
        let (apack, row_scale) = &mut *s.borrow_mut();
        let row_panels = m.div_ceil(MR);
        apack.resize(row_panels * MR * KC / 2, 0);
        row_scale.resize(row_panels * MR, 0.0);
        let mut tile = [0f32; MR * NR];
        for k0 in (0..k).step_by(KC) {
            let kc = KC.min(k - k0);
            let np = kc.div_ceil(2);
            let (first, last) = (k0 == 0, k0 + kc == k);
            for rp in 0..row_panels {
                let dst = &mut apack[rp * MR * np..(rp + 1) * MR * np];
                for r in 0..MR {
                    let i = rp * MR + r;
                    row_scale[i] = if i < m {
                        quantize_row(&a[i * lda + k0..][..kc], a_scale.map(|g| &g[k0..k0 + kc]), dst, r)
                    } else {
                        (0..np).for_each(|p| dst[p * MR + r] = 0);
                        0.0
                    };
                }
            }
            for panel in 0..lin.panels() {
                let b = &w[(panel * pairs + k0 / 2) * NR * 2..][..np * NR * 2];
                let n0 = panel * NR;
                let cols = NR.min(lin.n - n0);
                let cs = &scale[n0..n0 + NR];
                for rp in 0..row_panels {
                    let rows = MR.min(m - rp * MR);
                    let ap = &apack[rp * MR * np..(rp + 1) * MR * np];
                    let rs = &row_scale[rp * MR..(rp + 1) * MR];
                    if rows == MR && cols == NR {
                        let c = &mut out[rp * MR * ldo + n0..];
                        kernel(np, ap, b, rs, cs, c, ldo, !(first && !accumulate));
                        if last {
                            for r in 0..MR {
                                let o = &mut c[r * ldo..r * ldo + NR];
                                o.iter_mut().zip(&lin.bias[n0..n0 + NR]).for_each(|(o, b)| *o += b);
                                if act == Act::Gelu {
                                    math::gelu_slice(o);
                                }
                            }
                        }
                        continue;
                    }
                    kernel(np, ap, b, rs, cs, &mut tile, NR, false);
                    for r in 0..rows {
                        let o = &mut out[(rp * MR + r) * ldo + n0..][..cols];
                        epilogue(o, &tile[r * NR..r * NR + cols], &lin.bias[n0..n0 + cols], first && !accumulate, last, act);
                    }
                }
            }
        }
    });
}

/// One row's slice (times `gain`) as pairs of int16 at `dst[p * MR + r]` (`ceil(len / 2)` pairs, the odd tail paired
/// with 0), scaled to at most `A_MAX`; returns the scale.
fn quantize_row(x: &[f32], gain: Option<&[f32]>, dst: &mut [i32], r: usize) -> f32 {
    assert!(dst.len() >= x.len().div_ceil(2) * MR && r < MR);
    #[cfg(target_arch = "x86_64")]
    if simd::avx2() {
        return unsafe { quantize_row_avx2(x, gain, dst, r) };
    }
    quantize_row_scalar(x, gain, dst, r, 0)
}

/// From value `from` (even) on, with the row's maximum found here (`from == 0`) or by the caller.
fn quantize_row_scalar(x: &[f32], gain: Option<&[f32]>, dst: &mut [i32], r: usize, from: usize) -> f32 {
    let v = |i: usize| x[i] * gain.map_or(1.0, |g| g[i]);
    let max = (0..x.len()).map(|i| v(i).abs()).fold(0.0f32, f32::max);
    let inv = if max > 0.0 { A_MAX / max } else { 0.0 };
    for p in from / 2..x.len().div_ceil(2) {
        let lo = (v(2 * p) * inv).round_ties_even() as i16;
        let hi = if 2 * p + 1 < x.len() { (v(2 * p + 1) * inv).round_ties_even() as i16 } else { 0 };
        dst[p * MR + r] = (lo as u16 as u32 | ((hi as u16 as u32) << 16)) as i32;
    }
    max / A_MAX
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_row_avx2(x: &[f32], gain: Option<&[f32]>, dst: &mut [i32], r: usize) -> f32 {
    let len = x.len();
    let load = |i: usize| {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        match gain {
            Some(g) => _mm256_mul_ps(v, _mm256_loadu_ps(g.as_ptr().add(i))),
            None => v,
        }
    };
    let sign = _mm256_set1_ps(-0.0);
    let mut m = _mm256_setzero_ps();
    let full8 = len / 8 * 8;
    for i in (0..full8).step_by(8) {
        m = _mm256_max_ps(m, _mm256_andnot_ps(sign, load(i)));
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), m);
    let mut max = lanes.iter().fold(0.0f32, |a, &b| a.max(b));
    for i in full8..len {
        max = max.max((x[i] * gain.map_or(1.0, |g| g[i])).abs());
    }
    let inv = _mm256_set1_ps(if max > 0.0 { A_MAX / max } else { 0.0 });
    let full16 = len / 16 * 16;
    for i in (0..full16).step_by(16) {
        let q0 = _mm256_cvtps_epi32(_mm256_mul_ps(load(i), inv));
        let q1 = _mm256_cvtps_epi32(_mm256_mul_ps(load(i + 8), inv));
        // packs works per 128-bit lane: [q0 0..4, q1 0..4, q0 4..8, q1 4..8] -> in order
        let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(q0, q1));
        let mut pairs = [0i32; 8];
        _mm256_storeu_si256(pairs.as_mut_ptr() as *mut __m256i, p);
        let to = dst.as_mut_ptr().add(i / 2 * MR + r);
        for (j, v) in pairs.into_iter().enumerate() {
            *to.add(j * MR) = v;
        }
    }
    if full16 < len {
        quantize_row_scalar(x, gain, dst, r, full16);
    }
    max / A_MAX
}

/// `c[r][j] (+)= rs[r] * cs[j] * sum_p a[p][r] . b[p][j]` for a 6 x 16 tile: `a` `[np][MR]` and `b` `[np][16]` pairs of
/// int16 (`b` as `[np][16][2]`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn kernel(np: usize, a: &[i32], b: &[i16], rs: &[f32], cs: &[f32], c: &mut [f32], ldc: usize, add: bool) {
    assert!(a.len() >= MR * np && b.len() >= np * NR * 2 && rs.len() >= MR && cs.len() >= NR);
    assert!(c.len() >= (MR - 1) * ldc + NR);
    #[cfg(target_arch = "x86_64")]
    if simd::avx2() {
        unsafe { kernel_avx2(np, a.as_ptr(), b.as_ptr(), rs.as_ptr(), cs.as_ptr(), c.as_mut_ptr(), ldc, add) };
        return;
    }
    for r in 0..MR {
        for j in 0..NR {
            let mut acc = 0i32;
            for p in 0..np {
                let (lo, hi) = (a[p * MR + r] as i16 as i32, (a[p * MR + r] >> 16) as i16 as i32);
                acc += lo * b[(p * NR + j) * 2] as i32 + hi * b[(p * NR + j) * 2 + 1] as i32;
            }
            let v = acc as f32 * (cs[j] * rs[r]);
            let o = &mut c[r * ldc + j];
            *o = if add { *o + v } else { v };
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2(np: usize, a: *const i32, b: *const i16, rs: *const f32, cs: *const f32, c: *mut f32, ldc: usize, add: bool) {
    let z = _mm256_setzero_si256();
    let (mut c00, mut c01, mut c10, mut c11, mut c20, mut c21) = (z, z, z, z, z, z);
    let (mut c30, mut c31, mut c40, mut c41, mut c50, mut c51) = (z, z, z, z, z, z);
    // the rows' pairs broadcast as floats: a load only (`vpbroadcastd` costs a shuffle on some cores)
    let mut ap = a as *const f32;
    let mut bp = b as *const __m256i;
    for _ in 0..np {
        let b0 = _mm256_loadu_si256(bp);
        let b1 = _mm256_loadu_si256(bp.add(1));
        let x = _mm256_castps_si256(_mm256_broadcast_ss(&*ap));
        c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(x, b0));
        c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(x, b1));
        let x = _mm256_castps_si256(_mm256_broadcast_ss(&*ap.add(1)));
        c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(x, b0));
        c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(x, b1));
        let x = _mm256_castps_si256(_mm256_broadcast_ss(&*ap.add(2)));
        c20 = _mm256_add_epi32(c20, _mm256_madd_epi16(x, b0));
        c21 = _mm256_add_epi32(c21, _mm256_madd_epi16(x, b1));
        let x = _mm256_castps_si256(_mm256_broadcast_ss(&*ap.add(3)));
        c30 = _mm256_add_epi32(c30, _mm256_madd_epi16(x, b0));
        c31 = _mm256_add_epi32(c31, _mm256_madd_epi16(x, b1));
        let x = _mm256_castps_si256(_mm256_broadcast_ss(&*ap.add(4)));
        c40 = _mm256_add_epi32(c40, _mm256_madd_epi16(x, b0));
        c41 = _mm256_add_epi32(c41, _mm256_madd_epi16(x, b1));
        let x = _mm256_castps_si256(_mm256_broadcast_ss(&*ap.add(5)));
        c50 = _mm256_add_epi32(c50, _mm256_madd_epi16(x, b0));
        c51 = _mm256_add_epi32(c51, _mm256_madd_epi16(x, b1));
        ap = ap.add(MR);
        bp = bp.add(2);
    }
    let (s0, s1) = (_mm256_loadu_ps(cs), _mm256_loadu_ps(cs.add(8)));
    for (r, (lo, hi)) in [(c00, c01), (c10, c11), (c20, c21), (c30, c31), (c40, c41), (c50, c51)].into_iter().enumerate() {
        let g = _mm256_set1_ps(*rs.add(r));
        let lo = _mm256_mul_ps(_mm256_cvtepi32_ps(lo), _mm256_mul_ps(s0, g));
        let hi = _mm256_mul_ps(_mm256_cvtepi32_ps(hi), _mm256_mul_ps(s1, g));
        let p = c.add(r * ldc);
        if add {
            _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), lo));
            _mm256_storeu_ps(p.add(8), _mm256_add_ps(_mm256_loadu_ps(p.add(8)), hi));
        } else {
            _mm256_storeu_ps(p, lo);
            _mm256_storeu_ps(p.add(8), hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm;

    fn leak<T>(v: Vec<T>) -> &'static [T] {
        Box::leak(v.into_boxed_slice())
    }

    #[test]
    fn matches_float32() {
        // k with an odd tail and more than one slice; partial row and column panels; a gain on the columns of a
        for (m, n, k) in [(37usize, 45usize, 1101usize), (6, 16, 256), (13, 3, 7)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
            let w: Vec<f32> = (0..n * k).map(|i| ((i * 104729) % 97) as f32 / 48.0 - 1.0).collect();
            let gain: Vec<f32> = (0..k).map(|i| 0.5 + (i % 7) as f32 / 4.0).collect();
            let panels = n.div_ceil(NR);
            let mut packed = vec![0.0; panels * k * NR];
            for j in 0..n {
                for p in 0..k {
                    packed[((j / NR) * k + p) * NR + j % NR] = w[j * k + p];
                }
            }
            let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let lin = Linear {
                n,
                k,
                w: Panels::F32(leak(packed).into()),
                scale: None,
                bias: leak(bias.clone()).into(),
                lora: None,
            };
            let q = quantize(&lin);
            for (gain, accumulate) in [(None, false), (Some(&gain[..]), true)] {
                let mut want = vec![1.0; m * n];
                let mut got = want.clone();
                gemm::linear(&a, k, m, &lin, &mut want, n, Act::None, accumulate, gain);
                gemm::linear(&a, k, m, &q, &mut got, n, Act::None, accumulate, gain);
                let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs()));
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() < 2e-3 * scale, "{m} {n} {k}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn scalar_and_avx2_rows_agree() {
        let x: Vec<f32> = (0..53).map(|i| ((i * 31) % 17) as f32 - 8.3).collect();
        let mut a = vec![0; 27 * MR];
        let mut b = vec![0; 27 * MR];
        let sa = quantize_row(&x, None, &mut a, 2);
        let sb = quantize_row_scalar(&x, None, &mut b, 2, 0);
        assert_eq!((a, sa), (b, sb));
    }
}
