//! The chord model (`tsumugi_chords.note_chords.NoteChordModel`, the query reader) on one 30 s window: the frontend,
//! the convolutional stem (time / 8, frequency / 4), a row of tokens per 6.25 Hz step (78 frequency bands, 88 key
//! queries, the key token, the beat token, 4 chord queries) through 6 axis-Transformer layers (along the row, then
//! along time), and the chord queries' outputs up-sampled to 25 Hz: bass (13), root x quality (745) and the chord
//! boundary.

use rayon::prelude::*;

use super::frontend::Frontend;
use crate::Result;
use crate::gemm::{self, Act};
use crate::math;
use crate::profile;
use crate::weights::{Linear, Weights};

pub const BEAT_FEATURES: usize = 11;
pub const KEY_FEATURES: usize = 25;

fn lin(x: &[f32], m: usize, l: &Linear, act: Act) -> Vec<f32> {
    let mut out = vec![0.0f32; m * l.n];
    gemm::linear(x, l.k, m, l, &mut out, l.n, act, false, None);
    out
}

/// A 2-D convolution on a channels-last map `[h, w, c]` as a linear layer over patches (`[out, kh, kw, c]`).
struct Conv {
    lin: Linear,
    kernel: usize,
    stride: (usize, usize),
}

impl Conv {
    fn load(w: &Weights, name: &str, kernel: usize, stride: (usize, usize)) -> Result<Self> {
        Ok(Self { lin: w.linear(name)?, kernel, stride })
    }

    /// `[h, w, c]` -> `[h', w', out]` (padding `kernel / 2`).
    fn forward(&self, x: &[f32], h: usize, w: usize, act: Act) -> (Vec<f32>, usize, usize) {
        let k = self.kernel;
        let c = self.lin.k / (k * k);
        let pad = k / 2;
        let (sh, sw) = self.stride;
        let (ho, wo) = ((h + 2 * pad - k) / sh + 1, (w + 2 * pad - k) / sw + 1);
        let n = self.lin.n;
        let mut out = vec![0.0f32; ho * wo * n];
        out.par_chunks_mut(wo * n).enumerate().for_each(|(row, out)| {
            let mut patches = vec![0.0f32; wo * self.lin.k];
            for (col, patch) in patches.chunks_mut(self.lin.k).enumerate() {
                for dy in 0..k {
                    let y = (row * sh + dy) as isize - pad as isize;
                    for dx in 0..k {
                        let xx = (col * sw + dx) as isize - pad as isize;
                        let to = &mut patch[(dy * k + dx) * c..][..c];
                        if y >= 0 && (y as usize) < h && xx >= 0 && (xx as usize) < w {
                            to.copy_from_slice(&x[(y as usize * w + xx as usize) * c..][..c]);
                        } else {
                            to.fill(0.0);
                        }
                    }
                }
            }
            gemm::linear_block(&patches, self.lin.k, wo, &self.lin, out, n, act, false, None);
        });
        (out, ho, wo)
    }
}

struct GroupNorm {
    gamma: &'static [f32],
    beta: &'static [f32],
    groups: usize,
}

impl GroupNorm {
    fn load(w: &Weights, name: &str, groups: usize) -> Result<Self> {
        Ok(Self { gamma: w.f32(&format!("{name}.g"))?, beta: w.f32(&format!("{name}.b"))?, groups })
    }

    /// Over a channels-last map `[positions, c]`, then GELU if `gelu`.
    fn forward(&self, x: &mut [f32], gelu: bool) {
        let c = self.gamma.len();
        let per = c / self.groups;
        let sums: Vec<(f64, f64)> = x
            .par_chunks(c * 64)
            .map(|block| {
                let mut s = vec![(0.0f64, 0.0f64); self.groups];
                for row in block.chunks(c) {
                    for (g, part) in row.chunks(per).enumerate() {
                        let (a, b) = part.iter().fold((0.0f32, 0.0f32), |(a, b), &v| (a + v, b + v * v));
                        s[g].0 += a as f64;
                        s[g].1 += b as f64;
                    }
                }
                s
            })
            .reduce(
                || vec![(0.0, 0.0); self.groups],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(a, b)| {
                        a.0 += b.0;
                        a.1 += b.1;
                    });
                    a
                },
            );
        let count = (x.len() / c * per) as f64;
        let (mut scale, mut shift) = (vec![0.0f32; c], vec![0.0f32; c]);
        for (g, &(s, q)) in sums.iter().enumerate() {
            let mean = s / count;
            let rstd = 1.0 / ((q / count - mean * mean).max(0.0) + 1e-5).sqrt();
            for ch in g * per..(g + 1) * per {
                scale[ch] = (rstd * self.gamma[ch] as f64) as f32;
                shift[ch] = (self.beta[ch] as f64 - mean * rstd * self.gamma[ch] as f64) as f32;
            }
        }
        x.par_chunks_mut(c * 64).for_each(|block| {
            for row in block.chunks_mut(c) {
                for ((v, s), b) in row.iter_mut().zip(&scale).zip(&shift) {
                    *v = *v * s + b;
                }
                if gelu {
                    math::gelu_slice(row);
                }
            }
        });
    }
}

struct Block {
    qkvg: Linear, // q (scaled), k, v, the gates; RMSNorm's scale folded in
    out: Linear,
    ff1: Linear,
    ff2: Linear,
}

/// RMSNorm without its scale (folded into the next layer), eps 1e-6.
fn rms(x: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    out.par_chunks_mut(dim).zip(x.par_chunks(dim)).for_each(|(o, x)| {
        let mean = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let r = 1.0 / (mean + 1e-6).sqrt();
        o.iter_mut().zip(x).for_each(|(o, v)| *o = v * r);
    });
    out
}

impl Block {
    fn load(w: &Weights, name: &str) -> Result<Self> {
        Ok(Self {
            qkvg: w.linear(&format!("{name}.qkvg"))?,
            out: w.linear(&format!("{name}.out"))?,
            ff1: w.linear(&format!("{name}.ff1"))?,
            ff2: w.linear(&format!("{name}.ff2"))?,
        })
    }

    /// `x` `[steps, row, dim]`; `along_row`: the sequences are the rows (else the columns, through time).
    fn forward(&self, x: &mut [f32], steps: usize, row: usize, heads: usize, along_row: bool, rope: &Rope) {
        let n = steps * row;
        let qkvg = self.project(x, n, |t| if along_row { t % row } else { t / row }, heads, rope);
        let (seqs, len) = if along_row { (steps, row) } else { (row, steps) };
        let token = move |s: usize, p: usize| if along_row { s * row + p } else { p * row + s };
        let y = attention(&qkvg, self.out.n, heads, seqs, len, token, 0..len, n);
        drop(qkvg);
        self.finish(x, &y, n);
    }

    /// The last row block for the outputs of tokens `first..row` of each row only (the chord queries; all tokens
    /// are keys): `[steps, row - first, dim]`.
    fn forward_tail(&self, x: &[f32], steps: usize, row: usize, heads: usize, rope: &Rope, first: usize) -> Vec<f32> {
        let dim = self.out.n;
        let qkvg = self.project(x, steps * row, |t| t % row, heads, rope);
        let token = move |s: usize, p: usize| s * row + p;
        let tail = row - first;
        let y = attention(&qkvg, dim, heads, steps, row, token, first..row, steps * row);
        drop(qkvg);
        let mut out = vec![0.0f32; steps * tail * dim];
        let mut ys = vec![0.0f32; steps * tail * dim];
        for s in 0..steps {
            let (a, b) = ((s * row + first) * dim, (s + 1) * row * dim);
            out[s * tail * dim..(s + 1) * tail * dim].copy_from_slice(&x[a..b]);
            ys[s * tail * dim..(s + 1) * tail * dim].copy_from_slice(&y[a..b]);
        }
        self.finish(&mut out, &ys, steps * tail);
        out
    }

    /// RMSNorm, q / k / v / gates and RoPE on q and k by each token's `position`.
    fn project(&self, x: &[f32], n: usize, position: impl Fn(usize) -> usize + Sync, heads: usize, rope: &Rope) -> Vec<f32> {
        let dim = self.out.n;
        let width = self.qkvg.n;
        let h = rms(x, dim);
        let mut qkvg = profile::time("chord linear", || lin(&h, n, &self.qkvg, Act::None));
        let hd = dim / heads;
        qkvg.par_chunks_mut(width).enumerate().for_each(|(t, r)| {
            let p = position(t);
            for part in 0..2 {
                for head in 0..heads {
                    rope.apply(&mut r[part * dim + head * hd..][..hd], p);
                }
            }
        });
        qkvg
    }

    /// The attention's output projection and the feed-forward block, residual, on `n` tokens.
    fn finish(&self, x: &mut [f32], y: &[f32], n: usize) {
        let dim = self.out.n;
        let o = profile::time("chord linear", || lin(y, n, &self.out, Act::None));
        x.par_iter_mut().zip(&o).for_each(|(x, o)| *x += o);
        let h = rms(x, dim);
        let f = profile::time("chord linear", || lin(&h, n, &self.ff1, Act::Gelu));
        let o = profile::time("chord linear", || lin(&f, n, &self.ff2, Act::None));
        x.par_iter_mut().zip(&o).for_each(|(x, o)| *x += o);
    }
}

/// Gated attention per sequence and head over `qkvg` rows (`[tokens, 3 * dim + heads]`, q pre-scaled): the queries at
/// `queries` of each sequence attend to all `len` tokens; `token(s, p)` is the token at position p of sequence s.
/// Returns `[tokens, dim]` (rows of other tokens left at 0), each head's output times its sigmoid gate.
#[allow(clippy::too_many_arguments)]
fn attention(
    qkvg: &[f32],
    dim: usize,
    heads: usize,
    seqs: usize,
    len: usize,
    token: impl Fn(usize, usize) -> usize + Sync,
    queries: std::ops::Range<usize>,
    tokens: usize,
) -> Vec<f32> {
    let width = qkvg.len() / tokens;
    let hd = dim / heads;
    let padded = len.div_ceil(8) * 8;
    let mut y = vec![0.0f32; tokens * dim];
    let y_ptr = y.as_mut_ptr() as usize;
    profile::time("chord attention", || {
        (0..seqs * heads).into_par_iter().for_each_init(
            || (vec![0.0f32; queries.len() * hd], vec![0.0f32; hd * padded], vec![0.0f32; len * hd], vec![0.0f32; queries.len() * hd], vec![0.0f32; 4 * padded]),
            |(q, kt, v, o, scores), job| {
                let (s, head) = (job / heads, job % heads);
                for p in 0..len {
                    let r = &qkvg[token(s, p) * width..];
                    for d in 0..hd {
                        kt[d * padded + p] = r[dim + head * hd + d];
                    }
                    v[p * hd..(p + 1) * hd].copy_from_slice(&r[2 * dim + head * hd..][..hd]);
                }
                for (i, p) in queries.clone().enumerate() {
                    q[i * hd..(i + 1) * hd].copy_from_slice(&qkvg[token(s, p) * width + head * hd..][..hd]);
                }
                attend(q, kt, v, len, padded, hd, o, scores);
                for (i, p) in queries.clone().enumerate() {
                    let t = token(s, p);
                    let gate = 1.0 / (1.0 + (-qkvg[t * width + 3 * dim + head]).exp());
                    // each (token, head) slice is written by one job only
                    let out = unsafe { std::slice::from_raw_parts_mut((y_ptr as *mut f32).add(t * dim + head * hd), hd) };
                    out.iter_mut().zip(&o[i * hd..(i + 1) * hd]).for_each(|(out, v)| *out = v * gate);
                }
            },
        );
    });
    y
}

/// `softmax(q k^T) v` for one sequence and head: `q` `[queries, hd]`, `kt` `[hd, padded]` (keys transposed, zero
/// past `len`), `v` `[len, hd]` -> `o` `[queries, hd]`.
#[allow(clippy::too_many_arguments)]
fn attend(q: &[f32], kt: &[f32], v: &[f32], len: usize, padded: usize, hd: usize, o: &mut [f32], scores: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() && hd == 32 {
        return unsafe { attend_avx2(q, kt, v, len, padded, o, scores) };
    }
    for (q, o) in q.chunks(hd).zip(o.chunks_mut(hd)) {
        let mut max = f32::NEG_INFINITY;
        for j in 0..len {
            let s: f32 = (0..hd).map(|d| q[d] * kt[d * padded + j]).sum();
            scores[j] = s;
            max = max.max(s);
        }
        let mut sum = 0.0f32;
        for s in &mut scores[..len] {
            *s = (*s - max).exp();
            sum += *s;
        }
        o.fill(0.0);
        for j in 0..len {
            let p = scores[j] / sum;
            o.iter_mut().zip(&v[j * hd..(j + 1) * hd]).for_each(|(o, v)| *o += p * v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attend_avx2(q: &[f32], kt: &[f32], v: &[f32], len: usize, padded: usize, o: &mut [f32], scores: &mut [f32]) {
    use std::arch::x86_64::*;
    const HD: usize = 32;
    const QB: usize = 4; // queries per pass: each key and value vector loaded once for all of them
    let kt = kt.as_ptr();
    let queries = q.len() / HD;
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut q0 = 0;
    while q0 < queries {
        let nq = QB.min(queries - q0);
        let qs = |i: usize| q.as_ptr().add((q0 + i.min(nq - 1)) * HD);
        // scores [QB][padded]
        let mut j = 0;
        while j < padded {
            let mut acc = [_mm256_setzero_ps(); QB];
            for d in 0..HD {
                let k = _mm256_loadu_ps(kt.add(d * padded + j));
                for (i, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(_mm256_broadcast_ss(&*qs(i).add(d)), k, *a);
                }
            }
            for (i, a) in acc.iter().enumerate() {
                _mm256_storeu_ps(scores.as_mut_ptr().add(i * padded + j), *a);
            }
            j += 8;
        }
        let mut inv = [0.0f32; QB];
        for (i, inv) in inv.iter_mut().enumerate() {
            let row = &mut scores[i * padded..(i + 1) * padded];
            let max = _mm256_set1_ps(row[..len].iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s)));
            let mut sum = _mm256_setzero_ps();
            let mut j = 0;
            while j < padded {
                let mut e = crate::math::exp8(_mm256_sub_ps(_mm256_loadu_ps(row.as_ptr().add(j)), max));
                if j + 8 > len {
                    let keep = _mm256_cmpgt_epi32(_mm256_set1_epi32((len - j) as i32), lanes);
                    e = _mm256_and_ps(e, _mm256_castsi256_ps(keep)); // the padding's keys
                }
                _mm256_storeu_ps(row.as_mut_ptr().add(j), e);
                sum = _mm256_add_ps(sum, e);
                j += 8;
            }
            let mut s = [0.0f32; 8];
            _mm256_storeu_ps(s.as_mut_ptr(), sum);
            *inv = 1.0 / s.iter().sum::<f32>();
        }
        let mut acc = [[_mm256_setzero_ps(); 4]; QB];
        for jj in 0..len {
            let row = v.as_ptr().add(jj * HD);
            let vv = [_mm256_loadu_ps(row), _mm256_loadu_ps(row.add(8)), _mm256_loadu_ps(row.add(16)), _mm256_loadu_ps(row.add(24))];
            for (i, a) in acc.iter_mut().enumerate() {
                let p = _mm256_broadcast_ss(&scores[i * padded + jj]);
                for (a, v) in a.iter_mut().zip(&vv) {
                    *a = _mm256_fmadd_ps(p, *v, *a);
                }
            }
        }
        for (i, a) in acc.iter().take(nq).enumerate() {
            let out = o.as_mut_ptr().add((q0 + i) * HD);
            let s = _mm256_set1_ps(inv[i]);
            for (k, a) in a.iter().enumerate() {
                _mm256_storeu_ps(out.add(8 * k), _mm256_mul_ps(*a, s));
            }
        }
        q0 += QB;
    }
}

/// Rotary tables: `cos`, `sin` `[positions, head_dim / 2]` (pairs interleaved).
struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl Rope {
    fn new(head_dim: usize, positions: usize) -> Self {
        let half = head_dim / 2;
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for p in 0..positions {
            for i in 0..half {
                let inv = 1.0 / 10000f32.powf((2 * i) as f32 / head_dim as f32);
                let a = p as f32 * inv;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Self { cos, sin, half }
    }

    fn apply(&self, x: &mut [f32], p: usize) {
        let (c, s) = (&self.cos[p * self.half..][..self.half], &self.sin[p * self.half..][..self.half]);
        for i in 0..self.half {
            let (e, o) = (x[2 * i], x[2 * i + 1]);
            x[2 * i] = e * c[i] - o * s[i];
            x[2 * i + 1] = o * c[i] + e * s[i];
        }
    }
}

pub struct ChordNet {
    pub frontend: Frontend,
    conv1: Conv,
    conv2: Conv,
    freq_embed: &'static [f32], // [bins, channels]
    blocks: Vec<(Conv, GroupNorm, bool)>,
    extra: Vec<[(GroupNorm, Conv); 2]>,
    band: &'static [f32],
    keys: &'static [f32],   // [88, dim]
    chords: &'static [f32], // [chords, dim]
    key_in: Linear,
    beat_in: Linear,
    rows: Vec<Block>,
    times: Vec<Block>,
    merge: Linear,
    up: Linear,
    adapter: (Linear, Linear),
    out: Linear,
    dim: usize,
    heads: usize,
    pub rate: usize,
}

/// One window's outputs at 25 Hz: logits `[frames, 13 + 745 + 1]` (bass, root x quality, chord boundary).
pub const OUTPUT: usize = 13 + 745 + 1;

impl ChordNet {
    pub fn load(w: &Weights) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_str(w.metadata.get("config").map_or("{}", |s| s))?;
        let get = |k: &str| config[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("chords: no {k}"));
        let blocks = [("block1", (2, 1), true), ("block2", (2, 2), true), ("block3", (2, 2), true), ("block4", (1, 1), false)]
            .into_iter()
            .map(|(name, stride, gelu)| {
                Ok((
                    Conv::load(w, &format!("stem.{name}.conv"), 3, stride)?,
                    GroupNorm::load(w, &format!("stem.{name}.norm"), 4)?,
                    gelu,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let extra = (0..get("stem_extra")?)
            .map(|i| {
                let part = |j: usize| -> Result<(GroupNorm, Conv)> {
                    Ok((
                        GroupNorm::load(w, &format!("extra.{i}.norm{j}"), 8)?,
                        Conv::load(w, &format!("extra.{i}.conv{j}"), 3, (1, 1))?,
                    ))
                };
                Ok([part(1)?, part(2)?])
            })
            .collect::<Result<Vec<_>>>()?;
        let layers = get("layers")?;
        let load_blocks =
            |kind: &str| (0..layers).map(|i| Block::load(w, &format!("{kind}.{i}"))).collect::<Result<Vec<_>>>();
        if get("chords")? < 2 {
            return Err("chords: a model with one chord query is not supported".into());
        }
        Ok(Self {
            frontend: Frontend::load(w, &config)?,
            conv1: Conv::load(w, "stem.conv1", 7, (1, 1))?,
            conv2: Conv::load(w, "stem.conv2", 5, (1, 1))?,
            freq_embed: w.f32("stem.freq_embed")?,
            blocks,
            extra,
            band: w.f32("tokens.band")?,
            keys: w.f32("tokens.keys")?,
            chords: w.f32("tokens.chords")?,
            key_in: w.linear("tokens.key_in")?,
            beat_in: w.linear("tokens.beat_in")?,
            rows: load_blocks("row")?,
            times: load_blocks("time")?,
            merge: w.linear("merge")?,
            up: w.linear("up")?,
            adapter: (w.linear("adapter.1")?, w.linear("adapter.2")?),
            out: w.linear("out")?,
            dim: get("dim")?,
            heads: get("heads")?,
            rate: get("rate")?,
        })
    }

    /// The stem's output `[steps, bands, dim]` for the frontend's `[frames, bins, channels]`.
    pub fn stem(&self, spec: &[f32], frames: usize, bins: usize) -> (Vec<f32>, usize, usize) {
        let (mut x, h, w) = self.conv1.forward(spec, frames, bins, Act::None);
        let c = self.conv1.lin.n;
        x.par_chunks_mut(w * c).for_each(|row| {
            for (col, v) in row.chunks_mut(c).enumerate() {
                v.iter_mut().zip(&self.freq_embed[col * c..(col + 1) * c]).for_each(|(v, e)| *v += e);
                math::gelu_slice(v);
            }
        });
        let (mut x, mut h, mut w) = self.conv2.forward(&x, h, w, Act::Gelu);
        for (conv, norm, gelu) in &self.blocks {
            let (y, h2, w2) = conv.forward(&x, h, w, Act::None);
            (x, h, w) = (y, h2, w2);
            norm.forward(&mut x, *gelu);
        }
        for [(n1, c1), (n2, c2)] in &self.extra {
            let mut y = x.clone();
            n1.forward(&mut y, true);
            let (mut y, _, _) = c1.forward(&y, h, w, Act::None);
            n2.forward(&mut y, true);
            let (y, _, _) = c2.forward(&y, h, w, Act::None);
            x.iter_mut().zip(&y).for_each(|(x, y)| *x += y);
        }
        (x, h, w)
    }

    /// Logits `[frames, OUTPUT]` of one window: `wave` `[2][samples]` at `rate`, `beats` `[frames, 11]`, `keys`
    /// `[frames, 25]` (25 Hz, `frames = samples / (rate / 25)`).
    pub fn forward(&self, wave: &[&[f32]; 2], beats: &[f32], keys: &[f32]) -> Vec<f32> {
        let (spec, spec_frames) = profile::time("chord frontend", || self.frontend.forward(wave));
        let bins = spec.len() / (spec_frames * Frontend::CHANNELS);
        let (stem, steps, bands) = profile::time("chord stem", || self.stem(&spec, spec_frames, bins));
        drop(spec);
        let frames = beats.len() / BEAT_FEATURES;
        let dim = self.dim;
        let chords = self.chords.len() / dim;
        let keys_n = self.keys.len() / dim;
        let row = bands + keys_n + 2 + chords;
        // the key and beat tokens: 4 frames each (zero-padded to 4 * steps)
        let patches = |x: &[f32], width: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; 4 * steps * width];
            let n = x.len().min(out.len());
            out[..n].copy_from_slice(&x[..n]);
            out
        };
        let key_tokens = lin(&patches(keys, KEY_FEATURES), steps, &self.key_in, Act::None);
        let beat_tokens = lin(&patches(beats, BEAT_FEATURES), steps, &self.beat_in, Act::None);
        let mut x = vec![0.0f32; steps * row * dim];
        x.par_chunks_mut(row * dim).enumerate().for_each(|(s, r)| {
            for b in 0..bands {
                let to = &mut r[b * dim..(b + 1) * dim];
                to.iter_mut().zip(&stem[(s * bands + b) * dim..]).zip(self.band).for_each(|((t, v), e)| *t = v + e);
            }
            r[bands * dim..(bands + keys_n) * dim].copy_from_slice(self.keys);
            let at = (bands + keys_n) * dim;
            r[at..at + dim].copy_from_slice(&key_tokens[s * dim..(s + 1) * dim]);
            r[at + dim..at + 2 * dim].copy_from_slice(&beat_tokens[s * dim..(s + 1) * dim]);
            r[at + 2 * dim..].copy_from_slice(self.chords);
        });
        drop(stem);
        let rope = Rope::new(dim / self.heads, row.max(steps));
        // the last layer only for what the chord queries read: their row attention (all tokens as keys), then
        // the time attention among them
        let c = profile::time("chord layers", || {
            let last = self.rows.len() - 1;
            for (r, t) in self.rows[..last].iter().zip(&self.times) {
                r.forward(&mut x, steps, row, self.heads, true, &rope);
                t.forward(&mut x, steps, row, self.heads, false, &rope);
            }
            let mut c = self.rows[last].forward_tail(&x, steps, row, self.heads, &rope, row - chords);
            self.times[last].forward(&mut c, steps, chords, self.heads, false, &rope);
            c
        });
        drop(x);
        // the chord queries: RMSNorm (scale in `merge`), merged, up-sampled x4, the adapter, the outputs
        let c = rms(&c, dim);
        let merged = lin(&c, steps, &self.merge, Act::None);
        let mut g = lin(&merged, steps, &self.up, Act::None); // [steps, 4 * dim] = [4 * steps, dim]
        g.truncate(frames * dim);
        let mut ln = vec![0.0f32; g.len()];
        ln.par_chunks_mut(dim).zip(g.par_chunks(dim)).for_each(|(o, x)| math::normalize(x, o, 1e-5));
        let a = lin(&ln, frames, &self.adapter.0, Act::Gelu);
        let a = lin(&a, frames, &self.adapter.1, Act::None);
        g.iter_mut().zip(&a).for_each(|(g, a)| *g += a);
        lin(&g, frames, &self.out, Act::None)
    }
}
