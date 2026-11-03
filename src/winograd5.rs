//! 5 x 5 convolutions (stride 1, zero padding 2) on channels-last maps by Winograd's F(2 x 2, 5 x 5): each 2 x 2 block
//! of outputs from the 6 x 6 input tile around it, transformed (`B^T d B`, rows then columns) into 36 points; per
//! point one matrix product `[tiles, c] x [c, out]` with the transformed weights (`G g G^T`), and the 36 products
//! transformed back (`A^T m A`).  36 products of `c * out` per 4 outputs in place of 25 per output: 2.8 times fewer
//! multiply-adds.  The interpolation points are 0, 1, -1, 2, -2 and infinity.  The transforms' coefficients (up to 5
//! in `B^T`, 2 in `A^T`) widen the values' range, so the products stay float32 also for `--precision int16`: rounded to
//! int16 they lose several times the direct convolution's precision.

use rayon::prelude::*;

use crate::gemm::{self, Act, NR};
use crate::math;
use crate::weights::{Floats, Linear, Panels};

const R: usize = 5;
const T: usize = 6; // tile side
const POINTS: usize = T * T;

const BT: [[f32; T]; T] = [
    [4.0, 0.0, -5.0, 0.0, 1.0, 0.0],
    [0.0, -4.0, -4.0, 1.0, 1.0, 0.0],
    [0.0, 4.0, -4.0, -1.0, 1.0, 0.0],
    [0.0, -2.0, -1.0, 2.0, 1.0, 0.0],
    [0.0, 2.0, -1.0, -2.0, 1.0, 0.0],
    [0.0, 4.0, 0.0, -5.0, 0.0, 1.0],
];
const G: [[f64; R]; T] = [
    [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
    [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
    [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0],
    [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0, -1.0 / 3.0, 2.0 / 3.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
];
const AT: [[f32; T]; 2] = [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, 2.0, -2.0, 1.0]];

/// Tiles per matrix product: their transformed inputs and products (`TILES * 36 * c` floats each) stay in L2.
const TILES: usize = 24;

pub struct Winograd5 {
    u: Vec<Linear>, // per point (`6 * row + column`), `[out, c]` as panels
    bias: Floats,
    c: usize,
    n: usize,
}

/// `dst = coef * src` (`add`: `dst += coef * src`).
#[inline]
fn axpy(dst: &mut [f32], coef: f32, src: &[f32], add: bool) {
    let src = &src[..dst.len()];
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        return unsafe { axpy_avx2(dst, coef, src, add) };
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = if add { *d + coef * s } else { coef * s };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(dst: &mut [f32], coef: f32, src: &[f32], add: bool) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let k = _mm256_set1_ps(coef);
    let (d, s) = (dst.as_mut_ptr(), src.as_ptr());
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(s.add(i));
        let r = if add { _mm256_fmadd_ps(k, v, _mm256_loadu_ps(d.add(i))) } else { _mm256_mul_ps(k, v) };
        _mm256_storeu_ps(d.add(i), r);
        i += 8;
    }
    for j in i..n {
        *d.add(j) = if add { *d.add(j) + coef * *s.add(j) } else { coef * *s.add(j) };
    }
}

/// `dst = sum_k coefs[k] * src(k)` over the non-zero coefficients (at least one).
#[inline]
fn combine<'a>(dst: &mut [f32], coefs: &[f32; T], src: impl Fn(usize) -> &'a [f32]) {
    let mut first = true;
    for (k, &b) in coefs.iter().enumerate() {
        if b != 0.0 {
            axpy(dst, b, src(k), !first);
            first = false;
        }
    }
}

impl Winograd5 {
    /// For a 5 x 5 convolution's weights as `gemm` panels over `k = (5 * dy + dx) * c + channel`.
    pub fn new(lin: &Linear, c: usize) -> Option<Self> {
        if lin.k != R * R * c || !matches!(lin.w, Panels::F32(_) | Panels::F16(_)) {
            return None;
        }
        let n = lin.n;
        let value = |o: usize, kk: usize| -> f64 {
            let i = ((o / NR) * lin.k + kk) * NR + o % NR;
            match &lin.w {
                Panels::F32(w) => w[i] as f64,
                Panels::F16(w) => half::f16::from_bits(w[i]).to_f64(),
                _ => unreachable!(),
            }
        };
        let mut u = vec![vec![0.0f32; lin.panels() * c * NR]; POINTS];
        for o in 0..n {
            for ch in 0..c {
                let g = |dy: usize, dx: usize| value(o, (R * dy + dx) * c + ch);
                // G g G^T
                let mut gg = [[0.0f64; R]; T];
                for (i, row) in gg.iter_mut().enumerate() {
                    for (dx, v) in row.iter_mut().enumerate() {
                        *v = (0..R).map(|dy| G[i][dy] * g(dy, dx)).sum();
                    }
                }
                for i in 0..T {
                    for j in 0..T {
                        let v: f64 = (0..R).map(|dx| gg[i][dx] * G[j][dx]).sum();
                        u[T * i + j][((o / NR) * c + ch) * NR + o % NR] = v as f32;
                    }
                }
            }
        }
        let zeros: Floats = vec![0.0f32; n].into();
        let u = u
            .into_iter()
            .map(|w| Linear { n, k: c, w: Panels::F32(w.into()), scale: None, bias: zeros.clone(), lora: None })
            .collect();
        Some(Self { u, bias: lin.bias.clone(), c, n })
    }

    /// `[h, w, c]` -> `[h, w, out]`, then the bias and `act`: a row of tiles (two rows of outputs) per task, `TILES`
    /// tiles per product.
    pub fn forward(&self, x: &[f32], h: usize, w: usize, act: Act) -> Vec<f32> {
        let (c, n) = (self.c, self.n);
        assert_eq!(x.len(), h * w * c);
        let tw = w.div_ceil(2);
        let zero = vec![0.0f32; c];
        let mut out = vec![0.0f32; h * w * n];
        out.par_chunks_mut(2 * w * n).enumerate().for_each(|(ty, out)| {
            let rows = out.len() / (w * n);
            let mut v = vec![0.0f32; TILES * POINTS * c];
            let mut m = vec![0.0f32; TILES * POINTS * n];
            let mut e = vec![0.0f32; POINTS * c];
            let mut a = vec![0.0f32; 2 * T * n];
            let mut y = vec![0.0f32; n];
            for t0 in (0..tw).step_by(TILES) {
                let tiles = TILES.min(tw - t0);
                for (k, v) in v.chunks_mut(POINTS * c).take(tiles).enumerate() {
                    let tx = t0 + k;
                    let d: [&[f32]; POINTS] = std::array::from_fn(|ij| {
                        let (yy, xx) = ((2 * ty + ij / T) as isize - 2, (2 * tx + ij % T) as isize - 2);
                        if yy < 0 || xx < 0 || yy as usize >= h || xx as usize >= w {
                            &zero[..]
                        } else {
                            &x[(yy as usize * w + xx as usize) * c..][..c]
                        }
                    });
                    // B^T d: along the rows, then (B^T d) B: along the columns
                    for (i, bt) in BT.iter().enumerate() {
                        for j in 0..T {
                            combine(&mut e[(T * i + j) * c..][..c], bt, |r| d[T * r + j]);
                        }
                    }
                    for i in 0..T {
                        for (j, bt) in BT.iter().enumerate() {
                            combine(&mut v[(T * i + j) * c..][..c], bt, |s| &e[(T * i + s) * c..][..c]);
                        }
                    }
                }
                for (p, u) in self.u.iter().enumerate() {
                    gemm::linear_block(&v[p * c..], POINTS * c, tiles, u, &mut m[p * n..], POINTS * n, Act::None, false, None);
                }
                for k in 0..tiles {
                    let tx = t0 + k;
                    let m = &m[k * POINTS * n..][..POINTS * n];
                    // A^T m: a[o][j]
                    a.fill(0.0);
                    for (o, at) in AT.iter().enumerate() {
                        for j in 0..T {
                            let dst = &mut a[(T * o + j) * n..][..n];
                            for (i, &coef) in at.iter().enumerate() {
                                if coef != 0.0 {
                                    axpy(dst, coef, &m[(T * i + j) * n..][..n], true);
                                }
                            }
                        }
                    }
                    // (A^T m) A, the bias and act
                    for o in 0..rows {
                        for (o2, at) in AT.iter().enumerate() {
                            if 2 * tx + o2 >= w {
                                continue;
                            }
                            y.copy_from_slice(&self.bias);
                            for (j, &coef) in at.iter().enumerate() {
                                if coef != 0.0 {
                                    axpy(&mut y, coef, &a[(T * o + j) * n..][..n], true);
                                }
                            }
                            let dst = &mut out[(o * w + 2 * tx + o2) * n..][..n];
                            dst.copy_from_slice(&y);
                            if act == Act::Gelu {
                                math::gelu_slice(dst);
                            }
                        }
                    }
                }
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::ConvWeights;

    fn leak<T>(v: Vec<T>) -> &'static [T] {
        Box::leak(v.into_boxed_slice())
    }

    #[test]
    fn matches_direct() {
        if !crate::simd::avx2() {
            return;
        }
        for (h, w, c, n) in [(9usize, 13usize, 20usize, 37usize), (8, 6, 64, 16), (5, 61, 15, 64)] {
            let x: Vec<f32> = (0..h * w * c).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
            let kk = R * R * c;
            let mut packed = vec![0.0; n.div_ceil(NR) * kk * NR];
            for (i, p) in packed.iter_mut().enumerate() {
                *p = ((i * 104729) % 97) as f32 / 48.0 - 1.0;
            }
            let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let lin = Linear { n, k: kk, w: Panels::F32(leak(packed).into()), scale: None, bias: leak(bias).into(), lora: None };
            let direct = ConvWeights::new(lin.clone(), R, (1, 1)).forward_direct(&x, h, w, Act::Gelu).unwrap().0;
            let wino = Winograd5::new(&lin, c).unwrap();
            let scale = direct.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let got = wino.forward(&x, h, w, Act::Gelu);
            for (g, d) in got.iter().zip(&direct) {
                assert!((g - d).abs() <= 2e-5 * scale, "{h} {w} {c} {n}: {g} vs {d}");
            }
        }
    }
}
