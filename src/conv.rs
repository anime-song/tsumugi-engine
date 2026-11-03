//! 2-D convolutions on channels-last maps `[h, w, c]`, straight from the (zero-padded) input: no patches are copied.
//! A tile of 6 output positions x 16 output channels sums, tap by tap, the input pixels the tap reads (6 positions
//! `stride` apart) times the tap's weights: with the weights of a panel stored `[taps, c, 16]` (`gemm`'s panels over
//! `k = tap * c + channel`), each tap is a 6 x 16 product over `c`, read in place.
//!
//! On int16 (`quantize`), the input map is rounded to 13 bits once (one scale for the map) and the channels padded to
//! an even count so that each tap's channels are whole pairs for `vpmaddwd`; the int32 sums are scaled back to float32
//! every `q16::KC` values of k.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rayon::prelude::*;

use crate::gemm::{Act, MR, NR};
use crate::math;
use crate::q16;
use crate::weights::{Linear, Panels};

const A_MAX: f32 = 4095.0;
const W_MAX: f32 = 2047.0;

/// The convolution's weights: float32 panels `[panels, taps * c, 16]`, or int16 `[panels, taps, cp / 2, 16, 2]` (`cp`
/// the channels padded to even) with a scale per output channel.
pub struct ConvWeights {
    pub lin: Linear,
    pub kernel: usize,
    pub stride: (usize, usize),
    pub channels: usize,
    int16: Option<(Vec<i16>, Vec<f32>)>,
    /// 3 x 3, stride 1: Winograd's F(2 x 2, 3 x 3) (`crate::winograd`).
    winograd: Option<crate::winograd::Winograd>,
}

impl ConvWeights {
    pub fn new(lin: Linear, kernel: usize, stride: (usize, usize)) -> Self {
        let channels = lin.k / (kernel * kernel);
        let winograd = if kernel == 3 && stride == (1, 1) {
            crate::winograd::Winograd::new(&lin, channels)
        } else {
            None
        };
        Self { lin, kernel, stride, channels, int16: None, winograd }
    }

    fn padded_channels(&self) -> usize {
        self.channels.div_ceil(2) * 2
    }

    /// Round the weights to int16, 12 bits per output channel.
    pub fn quantize(&mut self) {
        if let Some(w) = &mut self.winograd {
            w.quantize();
        }
        let lin = &self.lin;
        let (taps, c, cp) = (self.kernel * self.kernel, self.channels, self.padded_channels());
        assert!(cp <= q16::KC, "conv: more channels than a slice of k");
        let panels = lin.panels();
        let value = |i: usize| -> f32 {
            match &lin.w {
                Panels::F32(w) => w[i],
                Panels::F16(w) => half::f16::from_bits(w[i]).to_f32(),
                _ => panic!("conv: float32 or float16 weights expected"),
            }
        };
        let mut q = vec![0i16; panels * taps * cp * NR];
        let mut scale = vec![0.0f32; panels * NR];
        q.par_chunks_mut(taps * cp * NR).zip(scale.par_chunks_mut(NR)).enumerate().for_each(|(panel, (q, scale))| {
            let at = |kk: usize, j: usize| value((panel * lin.k + kk) * NR + j);
            for (j, s) in scale.iter_mut().enumerate() {
                let max = (0..lin.k).map(|kk| at(kk, j).abs()).fold(0.0f32, f32::max);
                *s = max / W_MAX;
                let inv = if max > 0.0 { W_MAX / max } else { 0.0 };
                for t in 0..taps {
                    for ch in 0..c {
                        q[((t * cp + ch) / 2 * NR + j) * 2 + ch % 2] = (at(t * c + ch, j) * inv).round_ties_even() as i16;
                    }
                }
            }
        });
        self.int16 = Some((q, scale));
    }

    /// `[h, w, c]` -> `[h', w', out]` (zero padding `kernel / 2`), then the bias and `act`.  AVX2 only (`None`
    /// without it).
    pub fn forward(&self, x: &[f32], h: usize, w: usize, act: Act) -> Option<(Vec<f32>, usize, usize)> {
        if let Some(wino) = &self.winograd {
            return Some((wino.forward(x, h, w, act), h, w));
        }
        self.forward_direct(x, h, w, act)
    }

    /// `forward` without Winograd's transform.
    pub fn forward_direct(&self, x: &[f32], h: usize, w: usize, act: Act) -> Option<(Vec<f32>, usize, usize)> {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::avx2() && (self.int16.is_some() || matches!(self.lin.w, Panels::F32(_))) {
            return Some(self.forward_avx2(x, h, w, act));
        }
        let _ = (x, h, w, act);
        None
    }

    #[cfg(target_arch = "x86_64")]
    fn forward_avx2(&self, x: &[f32], h: usize, w: usize, act: Act) -> (Vec<f32>, usize, usize) {
        let (k, c) = (self.kernel, self.channels);
        assert_eq!(x.len(), h * w * c);
        let pad = k / 2;
        let (sh, sw) = self.stride;
        let (ho, wo) = ((h + 2 * pad - k) / sh + 1, (w + 2 * pad - k) / sw + 1);
        let tiles = wo.div_ceil(MR);
        // padded so that every tile's taps read inside the map
        let (hp, wp) = (h + 2 * pad, (w + 2 * pad).max((tiles * MR - 1) * sw + k));
        let n = self.lin.n;
        let mut out = vec![0.0f32; ho * wo * n];
        let finish = |o: &mut [f32], tile: &[f32], n0: usize, cols: usize| {
            for (r, o) in o.chunks_mut(n).enumerate() {
                let o = &mut o[n0..n0 + cols];
                o.iter_mut().zip(&tile[r * NR..]).zip(&self.lin.bias[n0..]).for_each(|((o, t), b)| *o = t + b);
                if act == Act::Gelu {
                    math::gelu_slice(o);
                }
            }
        };
        match &self.int16 {
            None => {
                let Panels::F32(weights) = &self.lin.w else { panic!("conv: float32 weights expected") };
                let mut xp = vec![0.0f32; hp * wp * c];
                xp.par_chunks_mut(wp * c).skip(pad).zip(x.par_chunks(w * c)).for_each(|(d, s)| d[pad * c..][..w * c].copy_from_slice(s));
                let taps: Vec<usize> = (0..k * k).map(|t| ((t / k) * wp + t % k) * c).collect();
                let per_slice = (q16::KC / c).max(1);
                assert!(xp.len() >= ((ho - 1) * sh * wp + (tiles * MR - 1) * sw) * c + taps[k * k - 1] + c);
                out.par_chunks_mut(wo * n).enumerate().for_each(|(row, out)| {
                    // per panel and slice of taps (its weights stay in L1), over all the row's tiles
                    let mut acc = vec![0.0f32; tiles * MR * NR];
                    for panel in 0..self.lin.panels() {
                        acc.fill(0.0);
                        let b = &weights[panel * self.lin.k * NR..(panel + 1) * self.lin.k * NR];
                        for (i, slice) in taps.chunks(per_slice).enumerate() {
                            let b = &b[i * per_slice * c * NR..];
                            for (t, tile) in acc.chunks_exact_mut(MR * NR).enumerate() {
                                let a = &xp[((row * sh) * wp + t * MR * sw) * c..];
                                unsafe { taps_f32(slice, c, a.as_ptr(), sw * c, b.as_ptr(), tile) };
                            }
                        }
                        let n0 = panel * NR;
                        for (t, tile) in acc.chunks_exact(MR * NR).enumerate() {
                            let rows = MR.min(wo - t * MR);
                            finish(&mut out[t * MR * n..(t * MR + rows) * n], tile, n0, NR.min(n - n0));
                        }
                    }
                });
            }
            Some((weights, scale)) => {
                let cp = self.padded_channels();
                let max = x.par_iter().map(|v| v.abs()).reduce(|| 0.0f32, f32::max);
                let inv = if max > 0.0 { A_MAX / max } else { 0.0 };
                let x_scale = max / A_MAX;
                let mut xp = vec![0i16; hp * wp * cp];
                xp.par_chunks_mut(wp * cp).skip(pad).zip(x.par_chunks(w * c)).for_each(|(d, s)| {
                    for (d, s) in d[pad * cp..].chunks_mut(cp).zip(s.chunks(c)) {
                        d.iter_mut().zip(s).for_each(|(d, s)| *d = (s * inv).round_ties_even() as i16);
                    }
                });
                let taps: Vec<usize> = (0..k * k).map(|t| ((t / k) * wp + t % k) * cp / 2).collect();
                let per_slice = (q16::KC / cp).max(1);
                let pairs = cp / 2;
                assert!(xp.len() / 2 >= ((ho - 1) * sh * wp + (tiles * MR - 1) * sw) * pairs + taps[k * k - 1] + pairs);
                out.par_chunks_mut(wo * n).enumerate().for_each(|(row, out)| {
                    let mut acc = vec![0.0f32; tiles * MR * NR];
                    for panel in 0..self.lin.panels() {
                        acc.fill(0.0);
                        let n0 = panel * NR;
                        let b = &weights[panel * k * k * cp * NR..(panel + 1) * k * k * cp * NR];
                        let mut cs = [0.0f32; NR];
                        cs.iter_mut().zip(&scale[n0..]).for_each(|(c, s)| *c = s * x_scale);
                        for (i, slice) in taps.chunks(per_slice).enumerate() {
                            let b = &b[i * per_slice * cp * NR..];
                            for (t, tile) in acc.chunks_exact_mut(MR * NR).enumerate() {
                                let a = xp[((row * sh) * wp + t * MR * sw) * cp..].as_ptr() as *const i32;
                                unsafe { taps_i16(slice, pairs, a, sw * pairs, b.as_ptr(), &cs, tile) };
                            }
                        }
                        for (t, tile) in acc.chunks_exact(MR * NR).enumerate() {
                            let rows = MR.min(wo - t * MR);
                            finish(&mut out[t * MR * n..(t * MR + rows) * n], tile, n0, NR.min(n - n0));
                        }
                    }
                });
            }
        }
        (out, ho, wo)
    }
}

/// `tile[r][j] += sum over taps and channels a[tap + r * rsa + ch] * b[(tap, ch)][j]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn taps_f32(taps: &[usize], c: usize, a: *const f32, rsa: usize, b: *const f32, tile: &mut [f32]) {
    assert!(tile.len() >= MR * NR);
    let z = _mm256_setzero_ps();
    let (mut c00, mut c01, mut c10, mut c11, mut c20, mut c21) = (z, z, z, z, z, z);
    let (mut c30, mut c31, mut c40, mut c41, mut c50, mut c51) = (z, z, z, z, z, z);
    let mut bp = b;
    for &off in taps {
        let mut ap = a.add(off);
        for _ in 0..c {
            let b0 = _mm256_loadu_ps(bp);
            let b1 = _mm256_loadu_ps(bp.add(8));
            let x = _mm256_broadcast_ss(&*ap);
            c00 = _mm256_fmadd_ps(x, b0, c00);
            c01 = _mm256_fmadd_ps(x, b1, c01);
            let x = _mm256_broadcast_ss(&*ap.add(rsa));
            c10 = _mm256_fmadd_ps(x, b0, c10);
            c11 = _mm256_fmadd_ps(x, b1, c11);
            let x = _mm256_broadcast_ss(&*ap.add(2 * rsa));
            c20 = _mm256_fmadd_ps(x, b0, c20);
            c21 = _mm256_fmadd_ps(x, b1, c21);
            let x = _mm256_broadcast_ss(&*ap.add(3 * rsa));
            c30 = _mm256_fmadd_ps(x, b0, c30);
            c31 = _mm256_fmadd_ps(x, b1, c31);
            let x = _mm256_broadcast_ss(&*ap.add(4 * rsa));
            c40 = _mm256_fmadd_ps(x, b0, c40);
            c41 = _mm256_fmadd_ps(x, b1, c41);
            let x = _mm256_broadcast_ss(&*ap.add(5 * rsa));
            c50 = _mm256_fmadd_ps(x, b0, c50);
            c51 = _mm256_fmadd_ps(x, b1, c51);
            ap = ap.add(1);
            bp = bp.add(NR);
        }
    }
    let t = tile.as_mut_ptr();
    for (r, (lo, hi)) in [(c00, c01), (c10, c11), (c20, c21), (c30, c31), (c40, c41), (c50, c51)].into_iter().enumerate() {
        let p = t.add(r * NR);
        _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), lo));
        _mm256_storeu_ps(p.add(8), _mm256_add_ps(_mm256_loadu_ps(p.add(8)), hi));
    }
}

/// `tile[r][j] += cs[j] * sum over taps and pairs a[tap + r * rsa + p] . b[(tap, p)][j]` (pairs of int16).
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn taps_i16(taps: &[usize], pairs: usize, a: *const i32, rsa: usize, b: *const i16, cs: &[f32; NR], tile: &mut [f32]) {
    assert!(tile.len() >= MR * NR);
    let z = _mm256_setzero_si256();
    let (mut c00, mut c01, mut c10, mut c11, mut c20, mut c21) = (z, z, z, z, z, z);
    let (mut c30, mut c31, mut c40, mut c41, mut c50, mut c51) = (z, z, z, z, z, z);
    let mut bp = b as *const __m256i;
    for &off in taps {
        let mut ap = a.add(off);
        for _ in 0..pairs {
            let b0 = _mm256_loadu_si256(bp);
            let b1 = _mm256_loadu_si256(bp.add(1));
            let x = _mm256_set1_epi32(ap.read_unaligned());
            c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(x, b0));
            c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(x, b1));
            let x = _mm256_set1_epi32(ap.add(rsa).read_unaligned());
            c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(x, b0));
            c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(x, b1));
            let x = _mm256_set1_epi32(ap.add(2 * rsa).read_unaligned());
            c20 = _mm256_add_epi32(c20, _mm256_madd_epi16(x, b0));
            c21 = _mm256_add_epi32(c21, _mm256_madd_epi16(x, b1));
            let x = _mm256_set1_epi32(ap.add(3 * rsa).read_unaligned());
            c30 = _mm256_add_epi32(c30, _mm256_madd_epi16(x, b0));
            c31 = _mm256_add_epi32(c31, _mm256_madd_epi16(x, b1));
            let x = _mm256_set1_epi32(ap.add(4 * rsa).read_unaligned());
            c40 = _mm256_add_epi32(c40, _mm256_madd_epi16(x, b0));
            c41 = _mm256_add_epi32(c41, _mm256_madd_epi16(x, b1));
            let x = _mm256_set1_epi32(ap.add(5 * rsa).read_unaligned());
            c50 = _mm256_add_epi32(c50, _mm256_madd_epi16(x, b0));
            c51 = _mm256_add_epi32(c51, _mm256_madd_epi16(x, b1));
            ap = ap.add(1);
            bp = bp.add(2);
        }
    }
    let (s0, s1) = (_mm256_loadu_ps(cs.as_ptr()), _mm256_loadu_ps(cs.as_ptr().add(8)));
    let t = tile.as_mut_ptr();
    for (r, (lo, hi)) in [(c00, c01), (c10, c11), (c20, c21), (c30, c31), (c40, c41), (c50, c51)].into_iter().enumerate() {
        let p = t.add(r * NR);
        _mm256_storeu_ps(p, _mm256_fmadd_ps(_mm256_cvtepi32_ps(lo), s0, _mm256_loadu_ps(p)));
        _mm256_storeu_ps(p.add(8), _mm256_fmadd_ps(_mm256_cvtepi32_ps(hi), s1, _mm256_loadu_ps(p.add(8))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leak<T>(v: Vec<T>) -> &'static [T] {
        Box::leak(v.into_boxed_slice())
    }

    /// The naive convolution, weights `[out, kh, kw, c]`.
    #[allow(clippy::too_many_arguments)]
    fn naive(x: &[f32], h: usize, w: usize, c: usize, wt: &[f32], n: usize, k: usize, (sh, sw): (usize, usize)) -> Vec<f32> {
        let pad = k / 2;
        let (ho, wo) = ((h + 2 * pad - k) / sh + 1, (w + 2 * pad - k) / sw + 1);
        let mut out = vec![0.0; ho * wo * n];
        for y in 0..ho {
            for xx in 0..wo {
                for o in 0..n {
                    let mut s = 0.5f64; // the bias
                    for dy in 0..k {
                        for dx in 0..k {
                            let (iy, ix) = ((y * sh + dy) as isize - pad as isize, (xx * sw + dx) as isize - pad as isize);
                            if iy < 0 || ix < 0 || iy as usize >= h || ix as usize >= w {
                                continue;
                            }
                            for ch in 0..c {
                                s += x[(iy as usize * w + ix as usize) * c + ch] as f64 * wt[((o * k + dy) * k + dx) * c + ch] as f64;
                            }
                        }
                    }
                    out[(y * wo + xx) * n + o] = s as f32;
                }
            }
        }
        out
    }

    #[test]
    fn matches_naive() {
        if !crate::simd::avx2() {
            return;
        }
        for (h, w, c, n, k, stride) in [(9usize, 20usize, 15usize, 20usize, 7usize, (1usize, 1usize)), (8, 13, 6, 16, 3, (2, 2)), (7, 11, 64, 33, 3, (2, 1)), (5, 6, 130, 16, 3, (1, 1))] {
            let x: Vec<f32> = (0..h * w * c).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
            let wt: Vec<f32> = (0..n * k * k * c).map(|i| ((i * 104729) % 97) as f32 / 48.0 - 1.0).collect();
            let kk = k * k * c;
            let mut packed = vec![0.0; n.div_ceil(NR) * kk * NR];
            for o in 0..n {
                for p in 0..kk {
                    packed[((o / NR) * kk + p) * NR + o % NR] = wt[o * kk + p];
                }
            }
            let lin = Linear { n, k: kk, w: Panels::F32(leak(packed).into()), scale: None, bias: leak(vec![0.5; n]).into(), lora: None };
            let want = naive(&x, h, w, c, &wt, n, k, stride);
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let mut conv = ConvWeights::new(lin, k, stride);
            for (tol, label) in [(1e-5, "f32"), (2e-3, "int16")] {
                if label == "int16" {
                    conv.quantize();
                }
                let (got, _, _) = conv.forward(&x, h, w, Act::None).unwrap();
                assert_eq!(got.len(), want.len());
                for (g, v) in got.iter().zip(&want) {
                    assert!((g - v).abs() <= tol * scale, "{label} {h} {w} {c} {n} {k}: {g} vs {v}");
                }
            }
        }
    }
}
