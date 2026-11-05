//! Tsumugi-Chords' decoder (`tsumugi_chords.decode.ChordDecoder`): repeated bars mixed (`repeat.smooth`), the
//! semi-Markov search over candidate boundaries on the beat grid (`semi_crf.decode`, see there for the scores), one
//! quality per segment (`chords.segments`), the merges after the search (`chords.merge_*`), and the mir_eval labels
//! and chord symbols.

use serde_json::Value;

use super::{FRAME_RATE, Info, Probs, decoder_parent, parent_tonic};
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
    bass_prior: f64,
    symmetric: bool,
    merge_fifth_bass: Option<String>, // "first", "longer" or "root"
    short_slash: f64,
    repeat_beta: f64,
    repeat_context: f64,
    repeat_local: f64,
    merge_similar: bool,
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
    bass_interval: Vec<f64>, // [families, 12]: log P(bass interval | family) over the family's likeliest
    symmetric: Vec<(usize, usize)>, // (family, the root step that keeps its notes): aug 4, dim7 3
}

/// A run of the path (`chords.Segment`): frames `[start, end)`, root x quality class (`NO_CHORD` for N), bass (12
/// for N).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Segment {
    start: usize,
    end: usize,
    detail: usize,
    bass: usize,
}

/// A chord of the output: times in seconds, root and bass pitch classes (None for no chord or no separate bass).
pub struct Chord {
    pub start: f64,
    pub end: f64,
    pub label: String,
    pub root: Option<usize>,
    pub quality: Option<String>,
    pub bass: Option<usize>,
    /// The chord symbol spelt in the chord's key, e.g. `Am7/G`.
    pub symbol: String,
}

impl Chord {
    /// As `<song>.chords.json` holds it.
    pub fn json(&self) -> Value {
        serde_json::json!({"start": self.start, "end": self.end, "label": self.label, "root": self.root,
                           "quality": self.quality, "bass": self.bass, "symbol": self.symbol})
    }
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
            // (exports from before Tsumugi-Chords 0.2.0 have none of these: off)
            bass_prior: p["bass_prior"].as_f64().unwrap_or(0.0),
            symmetric: p["symmetric"].as_bool().unwrap_or(false),
            merge_fifth_bass: match &p["merge_fifth_bass"] {
                Value::Bool(true) => Some("first".into()),
                Value::String(k) => Some(k.clone()),
                _ => None,
            },
            short_slash: p["short_slash"].as_f64().unwrap_or(0.0),
            repeat_beta: p["repeat_beta"].as_f64().unwrap_or(0.0),
            repeat_context: p["repeat_context"].as_f64().unwrap_or(0.9),
            repeat_local: p["repeat_local"].as_f64().unwrap_or(0.75),
            merge_similar: p["merge_similar"].as_bool().unwrap_or(false),
        };
        if params.bass_prior != 0.0 && !w.has("decoder.bass_interval") {
            return Err("chords decoder: no bass_interval table (export again with python/export_chords.py)".into());
        }
        let v64 = |name: &str| -> Result<Vec<f64>> { Ok(w.f32(name)?.iter().map(|&v| v as f64).collect()) };
        let strings = |k: &str| -> Vec<String> {
            meta[k].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default()
        };
        let floats = |k: &str| -> Vec<f64> { meta[k].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default() };
        let qualities = strings("qualities");
        let quality_family: Vec<usize> = w.f32("decoder.quality_family")?.iter().map(|&v| v as usize).collect();
        // `chords.SYMMETRIC`
        let symmetric = [("aug", 4), ("dim7", 3)]
            .iter()
            .filter_map(|&(name, step)| Some((quality_family[qualities.iter().position(|q| q == name)?], step)))
            .collect();
        let bass_interval = if w.has("decoder.bass_interval") { v64("decoder.bass_interval")? } else { Vec::new() };
        Ok(Self {
            params,
            dur_chord: v64("decoder.dur_chord")?,
            dur_bass: v64("decoder.dur_bass")?,
            pos_chord: v64("decoder.pos_chord")?,
            pos_bass: v64("decoder.pos_bass")?,
            lm: w.f32("decoder.lm")?,
            group: w.f32("decoder.group")?.iter().map(|&v| v as usize).collect(),
            quality_family,
            tones: w.f32("decoder.tones")?,
            states: meta["states"].as_u64().ok_or("chords decoder: no states")? as usize,
            qualities,
            labels: strings("labels"),
            dur_edges: floats("dur_edges"),
            subs: floats("subs"),
            sub_tolerance: meta["sub_tolerance"].as_f64().unwrap_or(0.07),
            bass_interval,
            symmetric,
        })
    }

    /// `chords.equivalent_roots`: the roots that spell the same notes as `root` in `family`.
    fn equivalent_roots(&self, family: usize, root: usize) -> Vec<usize> {
        match self.symmetric.iter().find(|&&(f, _)| f == family) {
            Some(&(_, step)) => (0..12 / step).map(|k| (root + k * step) % 12).collect(),
            None => vec![root],
        }
    }

    /// Root x family probabilities `[frames, states]` (the qualities of a family summed).
    fn family(&self, detail: &[f32]) -> Vec<f32> {
        let s = self.states;
        let mut out = vec![0.0f32; detail.len() / DETAIL * s];
        for (o, d) in out.chunks_mut(s).zip(detail.chunks(DETAIL)) {
            for (c, &v) in d.iter().enumerate() {
                o[self.group[c]] += v;
            }
        }
        out
    }

    /// `chords.merge_symmetric`: each symmetric chord's roots get the sum over the roots that spell its notes.
    fn merge_symmetric(&self, family: &mut [f32]) {
        for row in family.chunks_mut(self.states) {
            for &(fam, step) in &self.symmetric {
                let cols: [f32; 12] = std::array::from_fn(|r| row[12 * fam + r]);
                for r in 0..12 {
                    row[12 * fam + r] = (0..12 / step).fold(0.0, |acc, k| acc + cols[(r + 12 - k * step) % 12]);
                }
            }
        }
    }

    pub fn decode(&self, p: &Probs, info: &Info) -> Vec<Chord> {
        let pr = &self.params;
        let smoothed = if pr.repeat_beta != 0.0 && !info.downbeats.is_empty() {
            super::repeat::smooth(
                &p.detail,
                DETAIL,
                &p.bass,
                self.tones,
                &info.downbeats,
                pr.repeat_beta,
                pr.repeat_context,
                pr.repeat_local,
            )
        } else {
            None
        };
        let detail = smoothed.as_deref().unwrap_or(&p.detail);
        let family = self.family(detail);
        let mut searched = family.clone();
        if pr.symmetric {
            self.merge_symmetric(&mut searched);
        }
        let (chord_path, bass_path) = self.search(p, &searched, info);
        let mut segs = self.segments(detail, &family, &chord_path, &bass_path);
        if let Some(keep) = &pr.merge_fifth_bass {
            segs = merge_fifth_bass(&segs, keep);
        }
        if pr.short_slash != 0.0 {
            let beat_of = |f: usize| beat_position(&info.beats, &[f as f64 / FRAME_RATE])[0];
            segs = merge_short_slash(&segs, beat_of, pr.short_slash);
        }
        if pr.merge_similar {
            segs = self.merge_similar(&segs, detail);
        }
        segs.iter().map(|seg| self.chord(seg, &info.keys)).collect()
    }

    /// The output chord of a segment: its mir_eval label and its symbol in the key around its middle.
    fn chord(&self, seg: &Segment, keys: &[(f64, f64, String)]) -> Chord {
        let (start, end) = (seg.start as f64 / FRAME_RATE, seg.end as f64 / FRAME_RATE);
        if seg.detail >= NO_CHORD {
            return Chord { start, end, label: "N".into(), root: None, quality: None, bass: None, symbol: "N".into() };
        }
        let (root, q) = (seg.detail % 12, seg.detail / 12);
        let quality = &self.qualities[q];
        let bass = (seg.bass < 12).then_some(seg.bass);
        let mut label = format!("{}:{}", SHARP[root], self.labels[q]);
        if let Some(bass) = bass {
            let interval = (bass + 12 - root) % 12;
            if interval != 0 {
                label += &format!("/{}", DEGREES[interval]);
            }
        }
        let middle = (start + end) / 2.0;
        let key = keys
            .iter()
            .find(|(a, b, _)| *a <= middle && middle < *b)
            .or(keys.last())
            .map(|(_, _, k)| k.as_str());
        let symbol = chord_symbol(root, quality, bass, key);
        Chord { start, end, label, root: Some(root), quality: Some(quality.clone()), bass, symbol }
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
        // the chord-bass pair's prior, paid once per segment (not by a continuation)
        let mut prior = vec![0.0f64; k_n * 12];
        if pr.bass_prior != 0.0 {
            for (idx, &st) in keep[..k_n - 1].iter().enumerate() {
                let (fam, root) = (st / 12, st % 12);
                let table = &self.bass_interval[fam * 12..(fam + 1) * 12];
                // a symmetric chord: the bass's best reading over its spellings
                let rots: Vec<usize> = self.equivalent_roots(fam, 0);
                for b in 0..12 {
                    let best = rots.iter().map(|&r| table[(b + 24 - root - r) % 12]).fold(f64::NEG_INFINITY, f64::max);
                    prior[idx * 12 + b] = pr.bass_prior * best;
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
                start_c[at3(0, k, b)] = g_chord[0] + prior[k * 12 + b];
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
                        start_b[at3(j, k, b)] =
                            other - pr.bass_penalty - nct[k * 12 + b] + prior[k * 12 + b] + pos_b[j] + g_stay[j];
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
                    start_c[at3(j, k, b)] = sc + prior[k * 12 + b];
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

    /// `chords.segments`: the runs of the path, each with the quality of its family that has the highest mean share
    /// (`detail`: what the search read; `family`: its root x family sums, the symmetric chords not merged); a
    /// symmetric chord is spelt from the root its frames favour.
    fn segments(&self, detail: &[f32], family: &[f32], chord_path: &[usize], bass_path: &[usize]) -> Vec<Segment> {
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
            let mut c = chord_path[a];
            if c == n_family {
                out.push(Segment { start: a, end: b, detail: NO_CHORD, bass: 12 });
            } else {
                let (mut root, fam) = (c % 12, c / 12);
                let roots = if self.params.symmetric { self.equivalent_roots(fam, root) } else { vec![root] };
                if roots.len() > 1 {
                    let mass = |r: usize| (a..b).map(|t| family[t * s + 12 * fam + r]).sum::<f32>();
                    let mut best = (f32::NEG_INFINITY, root);
                    for &r in &roots {
                        let m = mass(r);
                        if m > best.0 {
                            best = (m, r);
                        }
                    }
                    root = best.1;
                    c = 12 * fam + root;
                }
                let members: Vec<usize> = (0..self.quality_family.len()).filter(|&q| self.quality_family[q] == fam).collect();
                let mut best = (f32::NEG_INFINITY, 0usize);
                for &q in &members {
                    let share = (a..b)
                        .map(|t| detail[t * DETAIL + 12 * q + root] / family[t * s + c].max(1e-9))
                        .sum::<f32>()
                        / (b - a) as f32;
                    if share > best.0 {
                        best = (share, q);
                    }
                }
                out.push(Segment { start: a, end: b, detail: 12 * best.1 + root, bass: bass_path[a] });
            }
            a = b;
        }
        out
    }

    /// `chords.merge_similar`: neighbouring segments over the same bass whose notes (bass included) are a subset of
    /// one another but whose roots differ joined into one, keeping the reading with the more probability over both.
    fn merge_similar(&self, segs: &[Segment], detail: &[f32]) -> Vec<Segment> {
        let notes = |seg: &Segment| -> u16 {
            let t = &self.tones[seg.detail * 12..(seg.detail + 1) * 12];
            (0..12).filter(|&p| t[p] != 0.0).fold(1u16 << seg.bass, |m, p| m | 1 << p)
        };
        let weight = |seg: &Segment, a: usize, b: usize| -> f32 {
            let (quality, root) = (seg.detail / 12, seg.detail % 12);
            let roots =
                if self.params.symmetric { self.equivalent_roots(self.quality_family[quality], root) } else { vec![root] };
            (a..b).map(|t| roots.iter().map(|&r| detail[t * DETAIL + 12 * quality + r]).sum::<f32>()).sum()
        };
        let mut out: Vec<Segment> = Vec::new();
        for seg in segs {
            if let Some(prev) = out.last_mut() {
                let subset = || {
                    let (p, n) = (notes(prev), notes(seg));
                    p & !n == 0 || n & !p == 0
                };
                if prev.detail != NO_CHORD
                    && seg.detail != NO_CHORD
                    && prev.bass == seg.bass
                    && prev.detail % 12 != seg.detail % 12
                    && subset()
                {
                    let keep = if weight(prev, prev.start, seg.end) >= weight(seg, prev.start, seg.end) { *prev } else { *seg };
                    *prev = Segment { start: prev.start, end: seg.end, ..keep };
                    continue;
                }
            }
            out.push(*seg);
        }
        out
    }
}

const NO_CHORD: usize = DETAIL - 1;

/// `chords.merge_fifth_bass`: a bass-only change between the chord's root and its fifth dropped; the joined segment
/// keeps the bass `keep` names (`first`, `longer` or `root`).
fn merge_fifth_bass(segs: &[Segment], keep: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for seg in segs {
        if let Some(prev) = out.last_mut()
            && prev.detail == seg.detail
            && seg.detail < NO_CHORD
        {
            let root = seg.detail % 12;
            let (a, b) = ((prev.bass + 12 - root) % 12, (seg.bass + 12 - root) % 12);
            if (a, b) == (0, 7) || (a, b) == (7, 0) {
                let bass = match keep {
                    "longer" if prev.end - prev.start < seg.end - seg.start => seg.bass,
                    "root" => root,
                    _ => prev.bass,
                };
                *prev = Segment { end: seg.end, bass, ..*prev };
                continue;
            }
        }
        out.push(*seg);
    }
    out
}

/// `chords.merge_short_slash`: a chord that opens with another bass for less than `max_beats` beats and then moves
/// to its root read as the root-position chord throughout (`beat_of`: frame -> beat).
fn merge_short_slash(segs: &[Segment], beat_of: impl Fn(usize) -> f64, max_beats: f64) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for seg in segs {
        let n = out.len();
        if let Some(&prev) = out.last() {
            let opening = n == 1 || out[n - 2].detail != prev.detail;
            if opening
                && prev.detail == seg.detail
                && seg.detail < NO_CHORD
                && prev.bass != seg.bass
                && seg.bass == seg.detail % 12
                && prev.bass != prev.detail % 12
                && beat_of(prev.end) - beat_of(prev.start) < max_beats
            {
                out[n - 1] = Segment { start: prev.start, ..*seg };
                continue;
            }
        }
        out.push(*seg);
    }
    out
}

const FLAT: [&str; 12] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
const LETTERS: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
const NATURAL: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
/// Letter steps of the 12 degrees above the tonic: 1, b2, 2, b3, 3, 4, #4, 5, b6, 6, b7, 7.
const DEGREE_STEP: [usize; 12] = [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6];

/// A letter with `shift` (-1, 0 or 1) as a note name.
fn with_accidental(letter: char, shift: i32) -> String {
    format!("{letter}{}", match shift {
        1 => "#",
        -1 => "b",
        _ => "",
    })
}

/// The letter `step` letters above `name`'s and how far `pc` lies from it (-6..5).
fn letter_above(name: &str, step: usize, pc: usize) -> (char, i32) {
    let first = name.chars().next().unwrap_or('C');
    let at = LETTERS.iter().position(|&l| l == first).unwrap_or(0);
    let letter_index = (at + step) % 7;
    (LETTERS[letter_index], (pc as i32 - NATURAL[letter_index] + 6).rem_euclid(12) - 6)
}

/// `decode.spell`: the name of pitch class `pc` in `key` (e.g. `A:minor`): the key's tonic spelt by its key
/// signature, the other notes by their degree above it; with `raised` a note outside the key is spelt as a raised
/// degree instead (the root of a diminished chord leads up).  Cb, Fb, E# and B# are written B, E, F and C.  No key:
/// C major's spelling.
fn spell(pc: usize, key: Option<&str>, raised: bool) -> String {
    let (tonic, mode) = key.unwrap_or("C:major").split_once(':').unwrap_or((key.unwrap_or("C:major"), ""));
    let index = |t: &str| SHARP.iter().position(|&n| n == t).or_else(|| FLAT.iter().position(|&n| n == t));
    let (t, mode) = match index(tonic) {
        Some(t) => (t, mode),
        None => (0, "major"),
    };
    let parent = (t as i64 + decoder_parent(mode)).rem_euclid(12) as usize;
    let names = if [1, 3, 5, 8, 10].contains(&parent) { FLAT } else { SHARP };
    let degree = (pc + 12 - t) % 12;
    let mut step = DEGREE_STEP[degree];
    if raised && !NATURAL.contains(&(((pc + 12 - parent) % 12) as i32)) && [1, 3, 8, 10].contains(&degree) {
        step -= 1;
    }
    let (letter, shift) = letter_above(names[t], step, pc);
    let name = if (-1..=1).contains(&shift) { with_accidental(letter, shift) } else { names[pc].to_string() };
    if ["Cb", "Fb", "E#", "B#"].contains(&name.as_str()) { names[pc].to_string() } else { name }
}

/// `decode.chord_symbol`: a chord symbol as in lead sheets, e.g. `Am7/G`, `FM7`, `G7(b9)`, `Bm7-5`.
fn chord_symbol(root: usize, quality: &str, bass: Option<usize>, key: Option<&str>) -> String {
    let name = spell(root, key, quality.starts_with("dim"));
    let symbol = format!("{name}{quality}");
    let Some(bass) = bass.filter(|&b| b != root) else { return symbol };
    // the bass by its interval above the root (B7/D#, C/Bb), else as the key spells it
    let interval = (bass + 12 - root) % 12;
    let mut step = DEGREE_STEP[interval];
    if interval == 6 && (quality.contains("-5") || quality.starts_with("dim")) {
        step = 4; // b5
    } else if interval == 8 && quality.contains("aug") {
        step = 4; // #5
    }
    let (letter, shift) = letter_above(&name, step, bass);
    let low = (-1..=1).contains(&shift).then(|| with_accidental(letter, shift));
    match low {
        Some(low) if !["Cb", "Fb", "E#", "B#"].contains(&low.as_str()) => format!("{symbol}/{low}"),
        _ => format!("{symbol}/{}", spell(bass, key, false)),
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
