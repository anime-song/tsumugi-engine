//! Tsumugi-Chords' decoder (`tsumugi_chords.decode.ChordDecoder`): the semi-Markov search over candidate boundaries
//! on the beat grid (`semi_crf.decode`, see there for the scores), one quality per segment (`chords.segments`) and
//! the mir_eval labels.

use serde_json::Value;

use super::{FRAME_RATE, Info, Probs, parent_tonic};
use crate::Result;
use crate::weights::Weights;

const NEG: f64 = -1e9;
const DETAIL: usize = 745;
const SHARP: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const DEGREES: [&str; 12] = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"];

struct Params {
    w: f64,
    penalty: f64,
    bass_penalty: f64,
    nct_penalty: f64,
    boundary_weight: f64,
    lam: f64,
    gamma_dur: f64,
    gamma_pos: f64,
    peak: Option<f64>,
    divisions: Vec<usize>,
    max_beats: f64,
    long_beats: f64,
    max_candidates: usize,
    prune: f32,
}

pub struct Decoder {
    params: Params,
    dur_chord: Vec<f64>,
    dur_bass: Vec<f64>,
    pos_chord: Vec<f64>,
    pos_bass: Vec<f64>,
    lm: &'static [f32], // [states, states]
    group: Vec<usize>,  // [745] detail class -> root x family state
    quality_family: Vec<usize>,
    tones: &'static [f32], // [745, 12]
    states: usize,
    qualities: Vec<String>,
    labels: Vec<String>,
    dur_edges: Vec<f64>,
    subs: Vec<f64>,
    sub_tolerance: f64,
}

/// A chord of the output: times in seconds, root and bass pitch classes (None for no chord or no separate bass).
pub struct Chord {
    pub start: f64,
    pub end: f64,
    pub label: String,
    pub root: Option<usize>,
    pub quality: Option<String>,
    pub bass: Option<usize>,
}

impl Decoder {
    pub fn load(w: &Weights) -> Result<Self> {
        let meta: Value = serde_json::from_str(w.metadata.get("decoder").map_or("{}", |s| s))?;
        let p = &meta["params"];
        let f = |k: &str| p[k].as_f64().ok_or_else(|| format!("chords decoder: no {k}"));
        if p["refine"].as_u64().unwrap_or(0) != 0 {
            return Err("chords decoder: refine is not supported".into());
        }
        let params = Params {
            w: f("w")?,
            penalty: f("penalty")?,
            bass_penalty: f("bass_penalty")?,
            nct_penalty: f("nct_penalty")?,
            boundary_weight: f("boundary_weight")?,
            lam: f("lam")?,
            gamma_dur: f("gamma_dur")?,
            gamma_pos: f("gamma_pos")?,
            peak: p["peak"].as_f64(),
            divisions: p["divisions"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect()).unwrap_or_default(),
            max_beats: f("max_beats")?,
            long_beats: f("long_beats")?,
            max_candidates: f("max_candidates")? as usize,
            prune: f("prune")? as f32,
        };
        let v64 = |name: &str| -> Result<Vec<f64>> { Ok(w.f32(name)?.iter().map(|&v| v as f64).collect()) };
        let strings = |k: &str| -> Vec<String> {
            meta[k].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default()
        };
        let floats = |k: &str| -> Vec<f64> { meta[k].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default() };
        Ok(Self {
            params,
            dur_chord: v64("decoder.dur_chord")?,
            dur_bass: v64("decoder.dur_bass")?,
            pos_chord: v64("decoder.pos_chord")?,
            pos_bass: v64("decoder.pos_bass")?,
            lm: w.f32("decoder.lm")?,
            group: w.f32("decoder.group")?.iter().map(|&v| v as usize).collect(),
            quality_family: w.f32("decoder.quality_family")?.iter().map(|&v| v as usize).collect(),
            tones: w.f32("decoder.tones")?,
            states: meta["states"].as_u64().ok_or("chords decoder: no states")? as usize,
            qualities: strings("qualities"),
            labels: strings("labels"),
            dur_edges: floats("dur_edges"),
            subs: floats("subs"),
            sub_tolerance: meta["sub_tolerance"].as_f64().unwrap_or(0.07),
        })
    }

    /// Root x family probabilities `[frames, states]` (the qualities of a family summed).
    fn family(&self, p: &Probs) -> Vec<f32> {
        let s = self.states;
        let mut out = vec![0.0f32; p.frames * s];
        for (o, d) in out.chunks_mut(s).zip(p.detail.chunks(DETAIL)) {
            for (c, &v) in d.iter().enumerate() {
                o[self.group[c]] += v;
            }
        }
        out
    }

    pub fn decode(&self, p: &Probs, info: &Info) -> Vec<Chord> {
        let family = self.family(p);
        let (chord_path, bass_path) = self.search(p, &family, info);
        self.segments(p, &family, &chord_path, &bass_path)
    }

    /// `semi_crf.decode`: per frame the root x family state (N last) and the bass (12 under N).
    fn search(&self, p: &Probs, family: &[f32], info: &Info) -> (Vec<usize>, Vec<usize>) {
        let pr = &self.params;
        let (frames, s) = (p.frames, self.states);
        let n_state = s - 1;
        let mut keep: Vec<usize> =
            (0..s).filter(|&k| k == n_state || (0..frames).any(|t| family[t * s + k] >= pr.prune)).collect();
        keep.sort_unstable();
        let k_n = keep.len();
        let n_idx = k_n - 1;
        let chord_lp: Vec<f64> = (0..frames)
            .flat_map(|t| keep.iter().map(move |&k| (pr.w as f32 * family[t * s + k].max(1e-6).ln()) as f64))
            .collect();
        let bass_lp: Vec<f64> = (0..frames)
            .flat_map(|t| (0..12).map(move |b| (pr.w as f32 * p.bass[t * 13 + b].max(1e-6).ln()) as f64))
            .collect();
        let bnd: Vec<f64> = p.boundary.iter().map(|&b| (b as f64).clamp(1e-4, 1.0 - 1e-4)).collect();
        let change: Vec<f64> = bnd.iter().map(|b| -pr.boundary_weight * b.ln()).collect();
        let stay: Vec<f64> = bnd.iter().map(|b| -pr.boundary_weight * (-b).ln_1p()).collect();
        let seconds = frames as f64 / FRAME_RATE;
        let ext = extended_beats(&info.beats, seconds);
        let cand = self.candidates(&ext, &p.boundary, frames);
        let m = cand.len() - 1;
        let cumsum = |x: &[f64], width: usize| -> Vec<f64> {
            let mut out = vec![0.0f64; (frames + 1) * width];
            for t in 0..frames {
                for k in 0..width {
                    out[(t + 1) * width + k] = out[t * width + k] + x[t * width + k];
                }
            }
            out
        };
        let cp_all = cumsum(&chord_lp, k_n);
        let bp_all = cumsum(&bass_lp, 12);
        let cp: Vec<f64> = cand.iter().flat_map(|&c| cp_all[c * k_n..(c + 1) * k_n].to_vec()).collect();
        let bp: Vec<f64> = cand.iter().flat_map(|&c| bp_all[c * 12..(c + 1) * 12].to_vec()).collect();
        let mut sp = vec![0.0f64; frames + 1];
        for t in 0..frames {
            sp[t + 1] = sp[t] + stay[t];
        }
        let at: Vec<usize> = cand.iter().map(|&c| c.min(frames - 1)).collect();
        let g_chord: Vec<f64> =
            cand.iter().zip(&at).map(|(&c, &a)| sp[(c + 1).min(frames)] - if c > 0 { change[a] } else { 0.0 }).collect();
        let g_stay: Vec<f64> =
            cand.iter().zip(&at).map(|(&c, &a)| sp[(c + 1).min(frames)] - if c > 0 { stay[a] } else { 0.0 }).collect();
        let spc: Vec<f64> = cand.iter().map(|&c| sp[c]).collect();
        let u = beat_position(&ext, &cand.iter().map(|&c| c as f64 / FRAME_RATE).collect::<Vec<_>>());
        let in_bar = bar_beats(&ext, &info.downbeats);
        let pos = self.positions(&u, &in_bar);
        let pos_c: Vec<f64> = pos.iter().map(|&q| pr.gamma_pos * self.pos_chord[q]).collect();
        let pos_b: Vec<f64> = pos.iter().map(|&q| pr.gamma_pos * self.pos_bass[q]).collect();
        let dur_c: Vec<f64> = self.dur_chord.iter().map(|v| pr.gamma_dur * v).collect();
        let dur_b: Vec<f64> = self.dur_bass.iter().map(|v| pr.gamma_dur * v).collect();
        // chord-change transitions between kept states, per tonic (key-relative bigram, no self move)
        let tonic = key_frames(&info.keys, frames);
        let mut tonics: Vec<usize> = tonic.clone();
        tonics.sort_unstable();
        tonics.dedup();
        let trans: Vec<Vec<f64>> = tonics
            .iter()
            .map(|&k| {
                let rel: Vec<usize> =
                    keep.iter().map(|&st| if st == n_state { n_state } else { 12 * (st / 12) + (st % 12 + 12 - k) % 12 }).collect();
                let mut t = vec![0.0f64; k_n * k_n];
                for a in 0..k_n {
                    for b in 0..k_n {
                        t[a * k_n + b] = if a == b {
                            NEG
                        } else if pr.lam == 0.0 {
                            0.0
                        } else {
                            pr.lam * self.lm[rel[a] * s + rel[b]] as f64
                        };
                    }
                }
                t
            })
            .collect();
        let tonic_idx: Vec<usize> = at.iter().map(|&a| tonics.binary_search(&tonic[a]).unwrap()).collect();
        let mut nct = vec![0.0f64; k_n * 12];
        if pr.nct_penalty != 0.0 {
            for (idx, &st) in keep[..k_n - 1].iter().enumerate() {
                let q = self.quality_family.iter().position(|&f| f == st / 12).unwrap_or(0);
                for b in 0..12 {
                    if self.tones[(12 * q + st % 12) * 12 + b] == 0.0 {
                        nct[idx * 12 + b] = pr.nct_penalty;
                    }
                }
            }
        }
        // the forward over candidates (`semi_crf._forward`)
        let size = (m + 1) * k_n * 12;
        let at3 = |j: usize, k: usize, b: usize| (j * k_n + k) * 12 + b;
        let (mut start_c, mut start_b, mut start_n) = (vec![NEG; size], vec![NEG; size], vec![NEG; size]);
        let (mut end, mut end_long) = (vec![NEG; size], vec![NEG; size]);
        let (mut bp_i, mut bp_type) = (vec![0u32; size], vec![0u8; size]);
        let (mut bpl_i, mut bpl_type) = (vec![0u32; size], vec![0u8; size]);
        let (mut prev_k, mut prev_kb) = (vec![0u32; (m + 1) * k_n], vec![0u8; (m + 1) * k_n]);
        let (mut top1, mut top2) = (vec![0u8; (m + 1) * k_n], vec![0u8; (m + 1) * k_n]);
        let basses = |k: usize| if k != n_idx { 12 } else { 1 };
        for k in 0..k_n {
            for b in 0..basses(k) {
                start_c[at3(0, k, b)] = g_chord[0];
            }
        }
        let mut best_k = vec![0.0f64; k_n];
        let mut best_kb = vec![0usize; k_n];
        for j in 1..=m {
            for i in j.saturating_sub(pr.max_candidates)..j {
                let length = u[j] - u[i];
                if length > pr.max_beats && i != j - 1 {
                    continue;
                }
                let db = self.dur_edges.partition_point(|&e| e <= length);
                let (dc, dbb) = (dur_c[db], dur_b[db]);
                let long = length >= pr.long_beats;
                for k in 0..k_n {
                    let ce = cp[j * k_n + k] - cp[i * k_n + k];
                    for b in 0..basses(k) {
                        let e = ce + if k != n_idx { bp[j * 12 + b] - bp[i * 12 + b] } else { 0.0 };
                        let (mut v, mut t) = (start_c[at3(i, k, b)] + dc, 0u8);
                        let w = start_b[at3(i, k, b)] + dbb;
                        if w > v {
                            (v, t) = (w, 1);
                        }
                        let w = start_n[at3(i, k, b)];
                        if w > v {
                            (v, t) = (w, 2);
                        }
                        v += e;
                        let x = at3(j, k, b);
                        if v > end[x] {
                            (end[x], bp_i[x], bp_type[x]) = (v, i as u32, t);
                        }
                        if long && v > end_long[x] {
                            (end_long[x], bpl_i[x], bpl_type[x]) = (v, i as u32, t);
                        }
                    }
                }
            }
            for x in at3(j, 0, 0)..at3(j + 1, 0, 0) {
                end[x] -= spc[j];
                end_long[x] -= spc[j];
            }
            if j == m {
                break;
            }
            // segments starting at c_j
            for k in 0..k_n {
                let (mut mx, mut mb, mut m2, mut mb2) = (NEG, 0usize, NEG, 0usize);
                for b in 0..basses(k) {
                    let x = end[at3(j, k, b)];
                    if x > mx {
                        (m2, mb2, mx, mb) = (mx, mb, x, b);
                    } else if x > m2 {
                        (m2, mb2) = (x, b);
                    }
                }
                (best_k[k], best_kb[k]) = (mx, mb);
                (top1[j * k_n + k], top2[j * k_n + k]) = (mb as u8, mb2 as u8);
                if k != n_idx {
                    for b in 0..12 {
                        let other = if b == mb { m2 } else { mx };
                        start_b[at3(j, k, b)] = other - pr.bass_penalty - nct[k * 12 + b] + pos_b[j] + g_stay[j];
                    }
                }
                for b in 0..basses(k) {
                    start_n[at3(j, k, b)] = end_long[at3(j, k, b)] + g_stay[j];
                }
            }
            let tr = &trans[tonic_idx[j]];
            for k in 0..k_n {
                let (mut mx, mut mk) = (NEG, 0usize);
                for kp in 0..k_n {
                    let x = best_k[kp] + tr[kp * k_n + k];
                    if x > mx {
                        (mx, mk) = (x, kp);
                    }
                }
                (prev_k[j * k_n + k], prev_kb[j * k_n + k]) = (mk as u32, best_kb[mk] as u8);
                let sc = mx - pr.penalty + pos_c[j] + g_chord[j];
                for b in 0..basses(k) {
                    start_c[at3(j, k, b)] = sc;
                }
            }
        }
        // back-trace
        let mut chord_path = vec![0usize; frames];
        let mut bass_path = vec![0usize; frames];
        let (mut k, mut bb, mut best) = (0usize, 0usize, f64::NEG_INFINITY);
        for kk in 0..k_n {
            for b in 0..12 {
                let v = if kk == n_idx && b > 0 { NEG } else { end[at3(m, kk, b)] };
                if v > best {
                    (best, k, bb) = (v, kk, b);
                }
            }
        }
        let (mut j, mut use_long) = (m, false);
        while j > 0 {
            let x = at3(j, k, bb);
            let (i, typ) = if use_long { (bpl_i[x] as usize, bpl_type[x]) } else { (bp_i[x] as usize, bp_type[x]) };
            for t in cand[i]..cand[j] {
                chord_path[t] = keep[k];
                bass_path[t] = bb;
            }
            if i == 0 {
                break;
            }
            match typ {
                0 => {
                    (k, bb) = (prev_k[i * k_n + k] as usize, prev_kb[i * k_n + k] as usize);
                    use_long = false;
                }
                1 => {
                    bb = if top1[i * k_n + k] as usize == bb { top2[i * k_n + k] } else { top1[i * k_n + k] } as usize;
                    use_long = false;
                }
                _ => use_long = true,
            }
            j = i;
        }
        for (c, b) in chord_path.iter().zip(bass_path.iter_mut()) {
            if *c == n_state {
                *b = 12;
            }
        }
        (chord_path, bass_path)
    }

    /// Candidate boundary frames (0 and `frames` included), increasing: the beats cut in halves, thirds and quarters
    /// and the chord-boundary peaks.
    fn candidates(&self, ext: &[f64], boundary: &[f32], frames: usize) -> Vec<usize> {
        let mut points: Vec<f64> = ext.to_vec();
        for &n in &self.params.divisions {
            for k in 1..n {
                for w in ext.windows(2) {
                    points.push(w[0] + k as f64 / n as f64 * (w[1] - w[0]));
                }
            }
        }
        let mut out: Vec<i64> = points.iter().map(|&p| (p * FRAME_RATE).round_ties_even() as i64).collect();
        if let Some(peak) = self.params.peak {
            let p: Vec<f64> = boundary.iter().map(|&b| b as f64).collect();
            for i in 1..p.len().saturating_sub(1) {
                if p[i] >= p[i - 1] && p[i] > p[i + 1] && p[i] > peak {
                    out.push(i as i64);
                }
            }
        }
        let mut c: Vec<usize> =
            out.into_iter().map(|v| v.clamp(0, frames as i64) as usize).filter(|&v| v > 0 && v < frames).collect();
        c.push(0);
        c.push(frames);
        c.sort_unstable();
        c.dedup();
        c
    }

    /// `semi_crf.positions`: the position index (beat in the bar x place in the beat, off, unknown) of beat
    /// coordinates.
    fn positions(&self, u: &[f64], in_bar: &[i64]) -> Vec<usize> {
        let subs = self.subs.len();
        let unknown = 4 * (subs + 1);
        u.iter()
            .map(|&u| {
                let mut whole = (u + 1e-6).floor() as i64;
                let frac = u - whole as f64;
                let (mut sub, mut d) = (0usize, f64::INFINITY);
                for (i, s) in self.subs.iter().chain(&[1.0]).enumerate() {
                    let x = (frac - s).abs();
                    if x < d {
                        (sub, d) = (i, x);
                    }
                }
                let near = d <= self.sub_tolerance;
                let up = sub == subs;
                if up && near {
                    whole += 1;
                }
                if up {
                    sub = 0;
                }
                if !near {
                    sub = subs;
                }
                let bar = if whole >= 0 && (whole as usize) < in_bar.len() { in_bar[whole as usize] } else { -1 };
                if bar >= 0 { bar as usize * (subs + 1) + sub } else { unknown }
            })
            .collect()
    }

    /// `chords.segments`: the runs of the path, each with the quality of its family that has the highest mean share.
    fn segments(&self, p: &Probs, family: &[f32], chord_path: &[usize], bass_path: &[usize]) -> Vec<Chord> {
        let s = self.states;
        let n_family = s - 1;
        let frames = chord_path.len();
        let mut out = Vec::new();
        let mut a = 0;
        while a < frames {
            let mut b = a + 1;
            while b < frames && chord_path[b] == chord_path[a] && bass_path[b] == bass_path[a] {
                b += 1;
            }
            let (start, end) = (a as f64 / FRAME_RATE, b as f64 / FRAME_RATE);
            let c = chord_path[a];
            if c == n_family {
                out.push(Chord { start, end, label: "N".into(), root: None, quality: None, bass: None });
            } else {
                let (root, fam) = (c % 12, c / 12);
                let members: Vec<usize> = (0..self.quality_family.len()).filter(|&q| self.quality_family[q] == fam).collect();
                let mut best = (f32::NEG_INFINITY, 0usize);
                for &q in &members {
                    let share = (a..b)
                        .map(|t| p.detail[t * DETAIL + 12 * q + root] / family[t * s + c].max(1e-9))
                        .sum::<f32>()
                        / (b - a) as f32;
                    if share > best.0 {
                        best = (share, q);
                    }
                }
                let q = best.1;
                let bass = bass_path[a];
                let mut label = format!("{}:{}", SHARP[root], self.labels[q]);
                let bass = (bass < 12).then_some(bass);
                if let Some(bass) = bass {
                    let interval = (bass + 12 - root) % 12;
                    if interval != 0 {
                        label += &format!("/{}", DEGREES[interval]);
                    }
                }
                out.push(Chord { start, end, label, root: Some(root), quality: Some(self.qualities[q].clone()), bass });
            }
            a = b;
        }
        out
    }
}

fn median(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// numpy's `arange(start, stop, step)`.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// `semi_crf.extended_beats`: the median beat repeated before the first and after the last beat.
fn extended_beats(beats: &[f64], seconds: f64) -> Vec<f64> {
    let diffs: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    let step = median(&diffs);
    let first = beats[0];
    let last = beats[beats.len() - 1];
    let mut out: Vec<f64> = arange(first - step, -step, -step).into_iter().rev().collect();
    out.extend_from_slice(beats);
    out.extend(arange(last + step, seconds + step, step));
    out
}

/// `chord_crf.beat_position`: beat coordinates of `times` (fractional beat indices, extrapolated at the median beat).
fn beat_position(beats: &[f64], times: &[f64]) -> Vec<f64> {
    let diffs: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    let step = median(&diffs);
    let (first, last) = (beats[0], beats[beats.len() - 1]);
    times
        .iter()
        .map(|&t| {
            if t < first {
                (t - first) / step
            } else if t > last {
                (beats.len() - 1) as f64 + (t - last) / step
            } else {
                // np.interp
                let i = beats.partition_point(|&b| b <= t);
                if i >= beats.len() {
                    (beats.len() - 1) as f64
                } else {
                    let (a, b) = (beats[i - 1], beats[i]);
                    (i - 1) as f64 + (t - a) / (b - a)
                }
            }
        })
        .collect()
}

/// `semi_crf.bar_beats`: per beat its place in the bar (0..3, later beats 3; -1 before the first downbeat).
fn bar_beats(beats: &[f64], downbeats: &[f64]) -> Vec<i64> {
    if downbeats.is_empty() {
        return vec![-1; beats.len()];
    }
    let mut first: Vec<usize> = downbeats
        .iter()
        .map(|&d| {
            let mut best = 0;
            for (i, &b) in beats.iter().enumerate() {
                if (d - b).abs() < (d - beats[best]).abs() {
                    best = i;
                }
            }
            best
        })
        .collect();
    first.sort_unstable();
    first.dedup();
    (0..beats.len())
        .map(|i| {
            let last = first.partition_point(|&f| f <= i) as i64 - 1;
            if last >= 0 { ((i - first[last as usize]) as i64).min(3) } else { -1 }
        })
        .collect()
}

/// `chord_crf.key_frames`: per frame the parent major's tonic of the key there.
fn key_frames(keys: &[(f64, f64, String)], frames: usize) -> Vec<usize> {
    let mut out = vec![0usize; frames];
    let mut end = 0usize;
    for (start, stop, name) in keys {
        let Some(tonic) = parent_tonic(name) else { continue };
        let a = (start * FRAME_RATE).round_ties_even() as i64;
        let b = (stop * FRAME_RATE).round_ties_even() as i64;
        let (lo, hi) = (a.max(0) as usize, b.min(frames as i64).max(0) as usize);
        for v in out.iter_mut().take(hi).skip(lo) {
            *v = tonic;
        }
        end = end.max(b.min(frames as i64).max(0) as usize);
    }
    if end > 0 && end < frames {
        let v = out[end - 1];
        out[end..].fill(v);
    }
    out
}
