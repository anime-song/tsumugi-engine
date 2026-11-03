//! 3 x 3 convolutions (stride 1, zero padding 1) on channels-last maps by Winograd's F(2 x 2, 3 x 3): each 2 x 2 block
//! of outputs comes from the 4 x 4 input tile around it, transformed (`B^T d B`) into 16 points; per point the
//! transformed tiles times the transformed weights (`G g G^T`) is one matrix product `[tiles, c] x [c, out]`, and the
//! 16 products transform back (`A^T m A`).  16 products of `c * out` per 4 outputs in place of 9 per output: 2.25
//! times fewer multiply-adds, the rest additions.  The products are `gemm::linear` (or `q16`'s on int16 weights).

use rayon::prelude::*;

use crate::gemm::{self, Act, NR};
use crate::math;
use crate::weights::{Floats, Linear, Panels};

pub struct Winograd {
    /// Per transformed point (`4 * row + column`), the weights `[out, c]` as panels.
    u: Vec<Linear>,
    bias: Floats,
    c: usize,
    n: usize,
}

const POINTS: usize = 16;

impl Winograd {
    /// For a 3 x 3 convolution's weights as `gemm` panels over `k = (3 * dy + dx) * c + channel`.
    #[allow(clippy::needless_range_loop)] // the index tables read as the formulas do
    pub fn new(lin: &Linear, c: usize) -> Option<Self> {
        if lin.k != 9 * c {
            return None;
        }
        let n = lin.n;
        let panels = lin.panels();
        let value = |o: usize, kk: usize| -> f64 {
            let i = ((o / NR) * lin.k + kk) * NR + o % NR;
            match &lin.w {
                Panels::F32(w) => w[i] as f64,
                Panels::F16(w) => half::f16::from_bits(w[i]).to_f64(),
                _ => unreachable!(),
            }
        };
        if !matches!(lin.w, Panels::F32(_) | Panels::F16(_)) {
            return None;
        }
        let mut u = vec![vec![0.0f32; panels * c * NR]; POINTS];
        for o in 0..n {
            for ch in 0..c {
                let g = |dy: usize, dx: usize| value(o, (3 * dy + dx) * c + ch);
                // G g: [4][3]
                let mut gg = [[0.0f64; 3]; 4];
                for dx in 0..3 {
                    let (g0, g1, g2) = (g(0, dx), g(1, dx), g(2, dx));
                    gg[0][dx] = g0;
                    gg[1][dx] = 0.5 * (g0 + g1 + g2);
                    gg[2][dx] = 0.5 * (g0 - g1 + g2);
                    gg[3][dx] = g2;
                }
                for (r, row) in gg.iter().enumerate() {
                    let (g0, g1, g2) = (row[0], row[1], row[2]);
                    let t = [g0, 0.5 * (g0 + g1 + g2), 0.5 * (g0 - g1 + g2), g2];
                    for (s, v) in t.into_iter().enumerate() {
                        u[4 * r + s][((o / NR) * c + ch) * NR + o % NR] = v as f32;
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

    /// The products on int16 values (`q16`).
    pub fn quantize(&mut self) {
        for u in &mut self.u {
            *u = crate::q16::quantize(u);
        }
    }

    /// `[h, w, c]` -> `[h, w, out]`, then the bias and `act`.  Two rows of tiles (four rows of outputs) per task, so
    /// what a task transforms, multiplies and transforms back stays in its caches: per point, its transformed tiles
    /// (each from 4 of the tile's 16 inputs), their product with the point's weights, and that product added into
    /// the outputs it reaches.
    pub fn forward(&self, x: &[f32], h: usize, w: usize, act: Act) -> Vec<f32> {
        let (c, n) = (self.c, self.n);
        assert_eq!(x.len(), h * w * c);
        let tw = w.div_ceil(2);
        const TILE_ROWS: usize = 2;
        let zero = vec![0.0f32; c];
        let mut out = vec![0.0f32; h * w * n];
        out.par_chunks_mut(2 * TILE_ROWS * w * n).enumerate().for_each(|(task, out)| {
            let rows = out.len() / (w * n); // output rows of this task
            let tile_rows = rows.div_ceil(2);
            let m = tile_rows * tw;
            let y0 = task * 2 * TILE_ROWS;
            let mut acc = vec![0.0f32; 2 * tile_rows * 2 * tw * n]; // [rows][2 * tw][n]
            let mut a = vec![0.0f32; m * c];
            let mut prod = vec![0.0f32; m * n];
            let at = |y: isize, xx: isize| -> &[f32] {
                if y < 0 || xx < 0 || y as usize >= h || xx as usize >= w {
                    &zero
                } else {
                    &x[(y as usize * w + xx as usize) * c..][..c]
                }
            };
            for (p, u) in self.u.iter().enumerate() {
                let (r, s) = (p / 4, p % 4);
                // B^T's row r: the two inputs it takes and their signs
                let (ri, rs) = TERMS[r];
                let (si, ss) = TERMS[s];
                for (t, a) in a.chunks_mut(c).enumerate() {
                    let (ty, tx) = (t / tw, t % tw);
                    let (top, left) = ((y0 + 2 * ty) as isize - 1, (2 * tx) as isize - 1);
                    let d = |i: usize, j: usize| at(top + ri[i] as isize, left + si[j] as isize);
                    let (d00, d01, d10, d11) = (d(0, 0), d(0, 1), d(1, 0), d(1, 1));
                    let (s00, s01, s10, s11) = (rs[0] * ss[0], rs[0] * ss[1], rs[1] * ss[0], rs[1] * ss[1]);
                    for (ch, a) in a.iter_mut().enumerate() {
                        *a = s00 * d00[ch] + s01 * d01[ch] + s10 * d10[ch] + s11 * d11[ch];
                    }
                }
                gemm::linear_block(&a, c, m, u, &mut prod, n, Act::None, false, None);
                // A^T's columns r and s: the outputs this point reaches and their signs
                for (t, pr) in prod.chunks(n).enumerate() {
                    let (ty, tx) = (t / tw, t % tw);
                    for (i, &ai) in BACK[r].iter().enumerate() {
                        if ai == 0.0 {
                            continue;
                        }
                        for (j, &aj) in BACK[s].iter().enumerate() {
                            if aj == 0.0 {
                                continue;
                            }
                            let o = &mut acc[((2 * ty + i) * 2 * tw + 2 * tx + j) * n..][..n];
                            if ai * aj > 0.0 {
                                o.iter_mut().zip(pr).for_each(|(o, v)| *o += v);
                            } else {
                                o.iter_mut().zip(pr).for_each(|(o, v)| *o -= v);
                            }
                        }
                    }
                }
            }
            for (i, out) in out.chunks_mut(w * n).enumerate() {
                for (xx, o) in out.chunks_mut(n).enumerate() {
                    let v = &acc[(i * 2 * tw + xx) * n..][..n];
                    o.iter_mut().zip(v).zip(self.bias.iter()).for_each(|((o, v), b)| *o = v + b);
                    if act == Act::Gelu {
                        math::gelu_slice(o);
                    }
                }
            }
        });
        out
    }
}

/// `B^T` (`[[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]`) by rows: the two inputs each takes, with signs.
const TERMS: [([usize; 2], [f32; 2]); 4] =
    [([0, 2], [1.0, -1.0]), ([1, 2], [1.0, 1.0]), ([2, 1], [1.0, -1.0]), ([1, 3], [1.0, -1.0])];

/// `A^T` (`[[1, 1, 1, 0], [0, 1, -1, -1]]`) by columns: what point r (or s) adds to output 0 and 1.
const BACK: [[f32; 2]; 4] = [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0]];

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
        for (h, w, c, n) in [(9usize, 13usize, 20usize, 37usize), (8, 6, 64, 16)] {
            let x: Vec<f32> = (0..h * w * c).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
            let kk = 9 * c;
            let mut packed = vec![0.0; n.div_ceil(NR) * kk * NR];
            for (i, p) in packed.iter_mut().enumerate() {
                *p = ((i * 104729) % 97) as f32 / 48.0 - 1.0;
            }
            let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let lin = Linear { n, k: kk, w: Panels::F32(leak(packed).into()), scale: None, bias: leak(bias).into(), lora: None };
            let direct = ConvWeights::new(lin.clone(), 3, (1, 1)).forward_direct(&x, h, w, Act::Gelu).unwrap().0;
            let mut wino = Winograd::new(&lin, c).unwrap();
            let scale = direct.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (i, tol) in [1e-5, 2e-3].into_iter().enumerate() {
                if i == 1 {
                    wino.quantize();
                }
                let got = wino.forward(&x, h, w, Act::Gelu);
                for (g, d) in got.iter().zip(&direct) {
                    assert!((g - d).abs() <= tol * scale, "{h} {w} {c} {n}: {g} vs {d}");
                }
            }
        }
    }
}
