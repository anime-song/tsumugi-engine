//! Repeated bars hear each other (`tsumugi_chords.repeat`): the chord model's root x quality probabilities of a bar
//! mixed with those of the bars that repeat it.
//!
//! Each bar is cut into `SLOTS` equal slots and described by what the model hears in each: the chord tones it expects
//! (12 pitch classes) and its bass (12 + N).  Two bars a and b are compared under every transposition s (b moved up s
//! semitones); the pair's transposition is the one under which the `K` bars on either side match best (the context).
//! b repeats a when that context similarity reaches `context_min`, the two bars themselves `local_min`, and they are
//! at least `MIN_LAG` bars apart.  a's probabilities are then mixed with b's at the same place in the bar, transposed
//! by s: `p'_a = (p_a + beta * sum_b w_b p_b) / (1 + beta * sum_b w_b)`, `w_b` the context similarity.  The bass is
//! not mixed.

use rayon::prelude::*;

use super::FRAME_RATE;

const SLOTS: usize = 8;
const BASS_WEIGHT: f32 = 0.7;
const K: usize = 2;
const MIN_LAG: usize = 2;
const DIM: usize = 12 + 13; // chord tones, bass

/// `[bars]` frame spans from downbeat to downbeat.
fn bar_spans(downbeats: &[f64], frames: usize) -> Vec<(usize, usize)> {
    let db: Vec<i64> = downbeats
        .iter()
        .map(|&d| (d * FRAME_RATE).round_ties_even() as i64)
        .filter(|&d| d >= 0 && (d as usize) < frames)
        .collect();
    db.windows(2).filter(|w| w[1] > w[0]).map(|w| (w[0] as usize, w[1] as usize)).collect()
}

/// The slots' frame ranges of a bar (`np.linspace(a, b, SLOTS + 1).round()`, at least one frame each).
fn slot_edges(a: usize, b: usize) -> [(usize, usize); SLOTS] {
    let step = (b - a) as f64 / SLOTS as f64;
    let edge = |s: usize| if s == SLOTS { b } else { (a as f64 + s as f64 * step).round_ties_even() as usize };
    std::array::from_fn(|s| {
        let lo = edge(s);
        (lo, edge(s + 1).max(lo + 1))
    })
}

/// `x / (|x| + 1e-6)`.
fn unit(x: &mut [f32]) {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
    x.iter_mut().for_each(|v| *v /= norm);
}

/// Root x quality probabilities `[12 * qualities + 1]` of one frame moved up `shift` semitones into `out` (N stays).
fn roll_detail(x: &[f32], shift: usize, out: &mut [f32]) {
    let n = x.len() - 1;
    for q in (0..n).step_by(12) {
        for r in 0..12 {
            out[q + (r + shift) % 12] = x[q + r];
        }
    }
    out[n] = x[n];
}

/// `repeat.smooth`: `detail` `[frames, classes]` with each bar mixed with the bars that repeat it (`bass`
/// `[frames, 13]`, `tones` `[classes - 1, 12]` (the chord tones of each root x quality) and the downbeats in seconds
/// only find them).
#[allow(clippy::too_many_arguments)]
pub fn smooth(
    detail: &[f32],
    classes: usize,
    bass: &[f32],
    tones: &[f32],
    downbeats: &[f64],
    beta: f64,
    context_min: f64,
    local_min: f64,
) -> Option<Vec<f32>> {
    let frames = detail.len() / classes;
    let spans = bar_spans(downbeats, frames);
    let n = spans.len();
    if beta <= 0.0 || n < 2 * K + 2 {
        return None;
    }
    // the chord tones of each class, as lists
    let tone_lists: Vec<Vec<usize>> = tones.chunks(12).map(|t| (0..12).filter(|&p| t[p] != 0.0).collect()).collect();
    // per bar and slot: the mean chord tones (`detail @ tones`) and bass
    let slots: Vec<[[f32; DIM]; SLOTS]> = spans
        .par_iter()
        .map(|&(a, b)| {
            let mut out = [[0.0f32; DIM]; SLOTS];
            for (slot, (lo, hi)) in out.iter_mut().zip(slot_edges(a, b)) {
                for t in lo..hi {
                    let row = &detail[t * classes..(t + 1) * classes];
                    for (c, tl) in tone_lists.iter().enumerate() {
                        let p = row[c];
                        for &pc in tl {
                            slot[pc] += p;
                        }
                    }
                    for (o, v) in slot[12..].iter_mut().zip(&bass[t * 13..(t + 1) * 13]) {
                        *o += v;
                    }
                }
                let count = (hi - lo) as f32;
                slot.iter_mut().for_each(|v| *v /= count);
                unit(&mut slot[..12]);
                slot[12..].iter_mut().for_each(|v| *v *= BASS_WEIGHT);
            }
            out
        })
        .collect();
    let feature = |bar: usize, shift: usize| -> [[f32; DIM]; SLOTS] {
        let mut out = slots[bar];
        for (o, s) in out.iter_mut().zip(&slots[bar]) {
            for p in 0..12 {
                o[(p + shift) % 12] = s[p];
                o[12 + (p + shift) % 12] = s[12 + p];
            }
            unit(o);
        }
        out
    };
    // sim[s][a][b]: the slots' mean cosine of bar a and bar b moved up s semitones
    let sim: Vec<f32> = (0..12)
        .into_par_iter()
        .flat_map_iter(|s| {
            let a: Vec<_> = (0..n).map(|i| feature(i, 0)).collect();
            let b: Vec<_> = (0..n).map(|i| feature(i, s)).collect();
            let mut out = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    let dot: f32 =
                        a[i].iter().zip(&b[j]).map(|(x, y)| x.iter().zip(y).map(|(x, y)| x * y).sum::<f32>()).sum();
                    out[i * n + j] = dot / SLOTS as f32;
                }
            }
            out
        })
        .collect();
    // context: the mean of sim[s][a + d][b + d] over 0 < |d| <= K
    let mut ctx = vec![0.0f32; 12 * n * n];
    let mut count = vec![0.0f32; n * n];
    for d in (-(K as isize)..=K as isize).filter(|&d| d != 0) {
        let (lo, hi) = ((-d).max(0) as usize, (n as isize).min(n as isize - d) as usize);
        for s in 0..12 {
            for i in lo..hi {
                for j in lo..hi {
                    let (ii, jj) = ((i as isize + d) as usize, (j as isize + d) as usize);
                    ctx[(s * n + i) * n + j] += sim[(s * n + ii) * n + jj];
                }
            }
        }
        for i in lo..hi {
            for j in lo..hi {
                count[i * n + j] += 1.0;
            }
        }
    }
    for s in 0..12 {
        for (c, &m) in ctx[s * n * n..(s + 1) * n * n].iter_mut().zip(&count) {
            *c /= m.max(1.0);
        }
    }
    let mut out = detail.to_vec();
    let mut rolled = vec![0.0f32; classes];
    for i in 0..n {
        let mut partners = Vec::new();
        for j in 0..n {
            let shift = (0..12).fold(0, |b, s| if ctx[(s * n + i) * n + j] > ctx[(b * n + i) * n + j] { s } else { b });
            let weight = ctx[(shift * n + i) * n + j];
            if i.abs_diff(j) >= MIN_LAG
                && weight as f64 >= context_min
                && sim[(shift * n + i) * n + j] as f64 >= local_min
            {
                partners.push((j, shift, weight));
            }
        }
        if partners.is_empty() {
            continue;
        }
        let (a0, a1) = spans[i];
        let len = a1 - a0;
        let mut mixed = vec![0.0f32; len * classes];
        let mut total = 0.0f32;
        for &(j, shift, weight) in &partners {
            let (b0, b1) = spans[j];
            for (f, m) in mixed.chunks_mut(classes).enumerate() {
                let place = f as f64 / len as f64;
                let t = ((b0 as f64 + place * (b1 - b0) as f64).round_ties_even() as usize).min(frames - 1);
                roll_detail(&detail[t * classes..(t + 1) * classes], shift, &mut rolled);
                m.iter_mut().zip(&rolled).for_each(|(m, r)| *m += weight * r);
            }
            total += weight;
        }
        let beta = beta as f32;
        let norm = 1.0 + beta * total;
        for (o, m) in out[a0 * classes..a1 * classes].iter_mut().zip(&mixed) {
            *o = (*o + beta * m) / norm;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_edges_match_numpy() {
        // np.linspace(10, 31, 9).round(): 10, 12.625 -> 13, 15.25 -> 15, 17.875 -> 18, 20.5 -> 20, 23.125 -> 23,
        // 25.75 -> 26, 28.375 -> 28, 31
        let e = slot_edges(10, 31);
        let lo: Vec<usize> = e.iter().map(|r| r.0).collect();
        assert_eq!(lo, vec![10, 13, 15, 18, 20, 23, 26, 28]);
        assert_eq!(e[7].1, 31);
    }

    #[test]
    fn repeated_bars_pull_together() {
        // 8 bars of 25 frames, classes: 2 roots x 1 quality padded to 12 roots + N; bars 2 and 6 repeat each other
        let classes = 13;
        let bars = 8;
        let frames = 25 * bars + 1;
        let mut tones = vec![0.0f32; 12 * 12];
        for r in 0..12 {
            for k in [0, 4, 7] {
                tones[r * 12 + (r + k) % 12] = 1.0;
            }
        }
        let mut detail = vec![0.0f32; frames * classes];
        let mut bass = vec![0.0f32; frames * 13];
        for t in 0..frames {
            let bar = t / 25;
            let root = [0, 5, 7, 0, 9, 5, 7, 0, 0][bar];
            detail[t * classes + root] = 0.9;
            detail[t * classes + 12] = 0.1;
            bass[t * 13 + root] = 1.0;
        }
        // bar 6's middle frames lean elsewhere a little
        for t in 160..165 {
            detail[t * classes + 7] = 0.6;
            detail[t * classes + 2] = 0.3;
        }
        let downbeats: Vec<f64> = (0..=bars).map(|b| b as f64).collect();
        let out = smooth(&detail, classes, &bass, &tones, &downbeats, 0.5, 0.5, 0.5).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        // bar 2 (root 7) repeats bar 6: its frames stay on 7
        assert!(out[55 * classes + 7] > 0.8);
    }
}
