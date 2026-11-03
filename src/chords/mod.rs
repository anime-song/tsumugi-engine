//! Tsumugi-Chords on the CPU: the chord model over a whole song in 30 s windows (`song_logits`), the beat and key
//! features it reads, and the semi-Markov decoder (`decode`) -> chords in mir_eval syntax.

pub mod decode;
pub mod frontend;
pub mod net;

use std::path::Path;

use rayon::prelude::*;

use crate::Result;
use crate::weights::Weights;
use net::{BEAT_FEATURES, ChordNet, KEY_FEATURES, OUTPUT};

pub const FRAME_RATE: f64 = 25.0;
const WINDOW: usize = 750; // frames (30 s, the training crops)
const MARGIN: usize = 125; // frames of context dropped at either side of a window

pub const FORMAT: &str = "tsumugi-chords-cpu/1";

/// What the chord model reads besides the audio: SheetSage2-Refined's beats, downbeats and keys.
pub struct Info {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    pub keys: Vec<(f64, f64, String)>, // start, end, "A:major"
}

pub struct Chords {
    pub net: ChordNet,
    pub decoder: decode::Decoder,
    pub weights: Weights,
}

/// The model's 25 Hz probabilities: `bass` `[frames, 13]`, `detail` `[frames, 745]`, `boundary` `[frames]`.
pub struct Probs {
    pub frames: usize,
    pub bass: Vec<f32>,
    pub detail: Vec<f32>,
    pub boundary: Vec<f32>,
}

impl Chords {
    /// `int16`: the model's matrix products on int16 values (`crate::q16`), else float32.
    pub fn load(path: &Path, int16: bool) -> Result<Self> {
        let weights = Weights::open_format(path, FORMAT, "python/export_chords.py")?;
        Ok(Self { net: ChordNet::load(&weights, int16)?, decoder: decode::Decoder::load(&weights)?, weights })
    }

    /// `tsumugi_chords.note_chords.song_logits`: the 25 Hz probabilities over a stereo song at the model's rate.
    pub fn probabilities(&self, wave: &[Vec<f32>; 2], info: &Info) -> Probs {
        let per_frame = self.net.rate / FRAME_RATE as usize;
        let frames = wave[0].len() / per_frame;
        let core = WINDOW - 2 * MARGIN;
        let size = WINDOW * per_frame;
        let padded: Vec<Vec<f32>> = wave
            .iter()
            .map(|w| {
                let mut w = w.clone();
                if w.len() < size {
                    w.resize(size, 0.0);
                }
                w
            })
            .collect();
        let mut logits = vec![0.0f32; frames * OUTPUT];
        for kept in (0..frames).step_by(core) {
            let start = (kept.saturating_sub(MARGIN)).min(frames.saturating_sub(WINDOW));
            let end = (kept + core).min(frames);
            let piece = [&padded[0][start * per_frame..(start + WINDOW) * per_frame], &padded[1][start * per_frame..(start + WINDOW) * per_frame]];
            let at = start as f64 / FRAME_RATE;
            let beats = beat_features(&info.beats, &info.downbeats, at, WINDOW);
            let keys = key_features(&info.keys, at, WINDOW);
            let out = self.net.forward(&piece, &beats, &keys);
            logits[kept * OUTPUT..end * OUTPUT].copy_from_slice(&out[(kept - start) * OUTPUT..(end - start) * OUTPUT]);
        }
        let mut probs = Probs {
            frames,
            bass: vec![0.0; frames * 13],
            detail: vec![0.0; frames * 745],
            boundary: vec![0.0; frames],
        };
        probs
            .bass
            .par_chunks_mut(13)
            .zip(probs.detail.par_chunks_mut(745))
            .zip(probs.boundary.par_iter_mut())
            .zip(logits.par_chunks(OUTPUT))
            .for_each(|(((bass, detail), boundary), l)| {
                softmax_into(&l[..13], bass);
                softmax_into(&l[13..13 + 745], detail);
                *boundary = 1.0 / (1.0 + (-l[758]).exp());
            });
        probs
    }
}

fn softmax_into(x: &[f32], out: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (o, v) in out.iter_mut().zip(x) {
        *o = (v - max).exp();
        sum += *o;
    }
    out.iter_mut().for_each(|o| *o /= sum);
}

const GAP: f64 = 8.0; // seconds: the cap of the since / until features

/// Seconds since the last mark and until the next, `log1p` of at most `GAP` scaled to 0..1 (1 where none).
fn since_until(t: f64, marks: &[f64]) -> (f64, f64) {
    let scale = GAP.ln_1p();
    if marks.is_empty() {
        return (1.0, 1.0);
    }
    let after = marks.partition_point(|&m| m <= t);
    let prev = if after > 0 { t - marks[after - 1] } else { GAP };
    let next = if after < marks.len() { marks[after] - t } else { GAP };
    (prev.clamp(0.0, GAP).ln_1p() / scale, next.clamp(0.0, GAP).ln_1p() / scale)
}

/// Each time's position between consecutive marks (0..1), if it lies between two.
fn phase(t: f64, marks: &[f64]) -> Option<f64> {
    if marks.len() < 2 {
        return None;
    }
    let index = marks.partition_point(|&m| m <= t) as isize - 1;
    if index < 0 || index as usize >= marks.len() - 1 {
        return None;
    }
    let i = index as usize;
    Some((t - marks[i]) / (marks[i + 1] - marks[i]))
}

/// `note_chords.beat_features`: `[frames, 11]` of a window from `start` (s).
pub fn beat_features(beats: &[f64], downbeats: &[f64], start: f64, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * BEAT_FEATURES];
    if beats.len() < 2 {
        return out;
    }
    for (column, marks) in [(0, beats), (1, downbeats)] {
        for &m in marks {
            let index = ((m - start) * FRAME_RATE).floor();
            if index >= 0.0 && (index as usize) < frames {
                out[index as usize * BEAT_FEATURES + column] = 1.0;
            }
        }
    }
    for f in 0..frames {
        let t = start + (f as f64 + 0.5) / FRAME_RATE;
        let row = &mut out[f * BEAT_FEATURES..(f + 1) * BEAT_FEATURES];
        for (column, marks) in [(2, beats), (4, downbeats)] {
            if let Some(v) = phase(t, marks) {
                let a = 2.0 * std::f64::consts::PI * v;
                row[column] = a.sin() as f32;
                row[column + 1] = a.cos() as f32;
            }
        }
        let (a, b) = since_until(t, beats);
        let (c, d) = since_until(t, downbeats);
        row[6..10].copy_from_slice(&[a as f32, b as f32, c as f32, d as f32]);
        row[10] = 1.0;
    }
    out
}

const SHARP: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

fn flat_to_sharp(name: &str) -> &str {
    match name {
        "Db" => "C#",
        "Eb" => "D#",
        "Gb" => "F#",
        "Ab" => "G#",
        "Bb" => "A#",
        "Cb" => "B",
        "Fb" => "E",
        "E#" => "F",
        "B#" => "C",
        other => other,
    }
}

/// The parent major's offset from a mode's tonic (`note_chords.PARENT`; minor is read as itself by `key_index`).
fn parent(mode: &str) -> i64 {
    match mode {
        "minor" | "harmonic_minor" => 3,
        "dorian" => -2,
        "phrygian" => -4,
        "lydian" => -5,
        "mixolydian" => -7,
        "locrian" => 1,
        _ => 0,
    }
}

/// `"E:major"` -> 4, `"A:minor"` -> 21 (tonic + 12 x minor; other modes by their parent major).
fn key_index(name: &str) -> Option<usize> {
    let (tonic, mode) = name.split_once(':').unwrap_or((name, ""));
    let tonic = SHARP.iter().position(|&s| s == flat_to_sharp(tonic))? as i64;
    if mode == "minor" {
        return Some(tonic as usize + 12);
    }
    Some((tonic + parent(mode)).rem_euclid(12) as usize)
}

/// The parent major's tonic (0..11) of a key name (`chord_crf.key_frames`).
pub fn parent_tonic(name: &str) -> Option<usize> {
    let (tonic, mode) = name.split_once(':').unwrap_or((name, ""));
    let tonic = SHARP.iter().position(|&s| s == tonic)? as i64;
    Some((tonic + parent(mode)).rem_euclid(12) as usize)
}

/// `note_chords.key_features`: `[frames, 25]` of a window from `start` (s).
pub fn key_features(keys: &[(f64, f64, String)], start: f64, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * KEY_FEATURES];
    for (a, b, name) in keys {
        let Some(index) = key_index(name) else { continue };
        for f in 0..frames {
            let t = start + (f as f64 + 0.5) / FRAME_RATE;
            if t >= *a && t < *b {
                out[f * KEY_FEATURES + index] = 1.0;
                out[f * KEY_FEATURES + 24] = 1.0;
            }
        }
    }
    out
}
