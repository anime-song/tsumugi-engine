//! The chord model's input (`tsumugi_chords.note_model.NoteFrontend`): a stereo window at 25.6 kHz -> per 50 Hz
//! frame and CQT bin the harmonic CQT of both channels (5 harmonics, normalised over the window), the onset saliency
//! of each channel's fundamental (`|dphi| * |X|`, normalised on its own) and three left-right cues, channels last:
//! `[frames, bins, 2 * 5 + 2 + 3]`.
//!
//! The CQT is Tsumugi's recursive one: per octave from the top down a 128-point STFT (Hann window, centred, reflect
//! padding) whose bins are multiplied by the octave's CQT kernels, the signal halved between octaves by a 65-tap
//! low-pass filter.

use rayon::prelude::*;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex32;

use crate::Result;
use crate::weights::Weights;

pub struct Frontend {
    octaves: Vec<Octave>,
    fir: &'static [f32],
    hop: usize,
    bins: usize,
    harmonics: usize,
    large: usize,
    actual: usize,
    lower: Vec<usize>, // [harmonics, bins]
    upper: Vec<usize>,
    alpha: &'static [f32],
}

struct Octave {
    re: &'static [f32], // [bins, size / 2 + 1]
    im: &'static [f32],
    window: &'static [f32], // [size]
}

impl Octave {
    fn bins(&self) -> usize {
        self.re.len() / (self.window.len() / 2 + 1)
    }
}

/// `[channels][bins][frames]` complex CQT bins, low first.
type Cqt = Vec<Vec<Complex32>>;

impl Frontend {
    pub fn load(w: &Weights, config: &serde_json::Value) -> Result<Self> {
        let get = |k: &str| config[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("chords: no {k}"));
        let octaves = (0..get("octaves")?)
            .map(|i| {
                Ok(Octave {
                    re: w.f32(&format!("cqt.{i}.re"))?,
                    im: w.f32(&format!("cqt.{i}.im"))?,
                    window: w.f32(&format!("cqt.{i}.window"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let index = |name: &str| -> Result<Vec<usize>> { Ok(w.f32(name)?.iter().map(|&v| v as usize).collect()) };
        Ok(Self {
            octaves,
            fir: w.f32("cqt.fir")?,
            hop: get("hop")?,
            bins: get("bins")?,
            harmonics: get("harmonics")?,
            large: get("large")?,
            actual: get("actual")?,
            lower: index("hcqt.lower")?,
            upper: index("hcqt.upper")?,
            alpha: w.f32("hcqt.alpha")?,
        })
    }

    pub const CHANNELS: usize = 15;

    /// One channel's complex CQT `[actual bins][frames]`, low bins first.
    fn cqt(&self, wave: &[f32], frames: usize) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); self.actual * frames];
        let mut x = wave.to_vec();
        let mut top = self.actual; // one past the octave's highest bin
        let mut planner = RealFftPlanner::<f32>::new();
        for (i, octave) in self.octaves.iter().enumerate() {
            let size = octave.window.len();
            let half = size / 2;
            let hop = self.hop >> i;
            let n = octave.bins();
            let fft = planner.plan_fft_forward(size);
            let count = x.len() / hop + 1; // torch.stft's frames with center=True
            let keep = count.min(frames);
            let reflect = |k: isize| -> f32 {
                let len = x.len() as isize;
                let k = if k < 0 { -k } else if k >= len { 2 * len - 2 - k } else { k };
                x[k as usize]
            };
            let lo = top - n;
            let columns: Vec<Vec<Complex32>> = (0..keep)
                .into_par_iter()
                .map_init(
                    || (fft.make_input_vec(), fft.make_output_vec(), fft.make_scratch_vec()),
                    |(input, spectrum, scratch), f| {
                        let start = (f * hop) as isize - half as isize;
                        for (j, v) in input.iter_mut().enumerate() {
                            *v = reflect(start + j as isize) * octave.window[j];
                        }
                        fft.process_with_scratch(input, spectrum, scratch).expect("fft");
                        (0..n)
                            .map(|b| {
                                let (re, im) = (&octave.re[b * (half + 1)..][..half + 1], &octave.im[b * (half + 1)..][..half + 1]);
                                let (mut sr, mut si) = (0.0f32, 0.0f32);
                                for k in 0..=half {
                                    let s = spectrum[k];
                                    sr += s.re * re[k] - s.im * im[k];
                                    si += s.re * im[k] + s.im * re[k];
                                }
                                Complex32::new(sr, si)
                            })
                            .collect()
                    },
                )
                .collect();
            for (f, column) in columns.iter().enumerate() {
                for (b, v) in column.iter().enumerate() {
                    out[(lo + b) * frames + f] = *v;
                }
            }
            top = lo;
            if i + 1 < self.octaves.len() {
                x = self.halve(&x);
            }
        }
        out
    }

    /// `conv1d(x, fir, stride=2, padding=taps / 2)`.
    fn halve(&self, x: &[f32]) -> Vec<f32> {
        let taps = self.fir.len();
        let pad = taps / 2;
        let len = (x.len() + 2 * pad - taps) / 2 + 1;
        (0..len)
            .into_par_iter()
            .map(|m| {
                let mut acc = 0.0f32;
                for (j, w) in self.fir.iter().enumerate() {
                    let k = (2 * m + j) as isize - pad as isize;
                    if k >= 0 && (k as usize) < x.len() {
                        acc += w * x[k as usize];
                    }
                }
                acc
            })
            .collect()
    }

    /// `[frames, bins, CHANNELS]` for a stereo window (`[2][samples]`), `frames = ceil(samples / hop)`.
    pub fn forward(&self, wave: &[&[f32]; 2]) -> (Vec<f32>, usize) {
        let frames = wave[0].len().div_ceil(self.hop);
        let (bins, harmonics) = (self.bins, self.harmonics);
        let z: Cqt = wave.par_iter().map(|w| self.cqt(w, frames)).collect();
        let actual = self.actual;
        // the magnitudes `[channel][bin][frame]`
        let mag: Vec<Vec<f32>> = z.iter().map(|z| z.par_iter().map(|v| v.norm()).collect()).collect();
        // the harmonics, read off the magnitudes by interpolation, normalised over both channels:
        // `[channel * harmonic][bin][frame]`
        let mut spec = vec![0.0f32; 2 * harmonics * bins * frames];
        spec.par_chunks_mut(frames).enumerate().for_each(|(i, row)| {
            let (plane, b) = (i / bins, i % bins);
            let (channel, h) = (plane / harmonics, plane % harmonics);
            let (lo, hi, a) = (self.lower[h * bins + b], self.upper[h * bins + b], self.alpha[h * bins + b]);
            debug_assert!(hi < self.large);
            let zero = vec![0.0f32; frames];
            let at = |b: usize| if b < actual { &mag[channel][b * frames..(b + 1) * frames] } else { &zero[..] };
            for ((o, &l), &u) in row.iter_mut().zip(at(lo)).zip(at(hi)) {
                *o = l + a * (u - l);
            }
        });
        let spec_moments = moments(&spec);
        // the onset saliency of each channel's fundamental: `[channel][bin][frame]`
        let mut sal = vec![0.0f32; 2 * bins * frames];
        sal.par_chunks_mut(frames).enumerate().for_each(|(i, out)| {
            let (channel, b) = (i / bins, i % bins);
            let row = &z[channel][b * frames..(b + 1) * frames];
            for f in 1..frames {
                let d = row[f] * row[f - 1].conj();
                out[f] = d.im.atan2(d.re).abs() * mag[channel][b * frames + f];
            }
        });
        let sal_moments = moments(&sal);
        // the left-right cues: phase difference (sine, cosine) weighted by the energy, level difference
        let count = (bins * frames) as f64;
        let cross: Vec<Complex32> = z[0][..bins * frames].par_iter().zip(&z[1][..bins * frames]).map(|(l, r)| l * r.conj()).collect();
        let energy: Vec<f32> = cross.par_iter().map(|c| c.norm()).collect();
        let energy_mean = (energy.par_iter().map(|&e| e as f64).sum::<f64>() / count).max(1e-12);
        let level_mean =
            ((0..bins * frames).into_par_iter().map(|i| (mag[0][i] + mag[1][i]) as f64).sum::<f64>() / count).max(1e-12);
        let floor = (1e-2 * level_mean) as f32;
        // channels last, a block of frames per task
        const FB: usize = 16;
        let c = Self::CHANNELS;
        let mut out = vec![0.0f32; frames * bins * c];
        out.par_chunks_mut(FB * bins * c).enumerate().for_each(|(block, out)| {
            let f0 = block * FB;
            let nf = out.len() / (bins * c);
            for b in 0..bins {
                let i0 = b * frames + f0;
                for (ch, plane) in spec.chunks(bins * frames).enumerate() {
                    for (j, &v) in plane[i0..i0 + nf].iter().enumerate() {
                        out[(j * bins + b) * c + ch] = ((v as f64 - spec_moments.0) / spec_moments.1) as f32;
                    }
                }
                for (channel, plane) in sal.chunks(bins * frames).enumerate() {
                    for (j, &v) in plane[i0..i0 + nf].iter().enumerate() {
                        out[(j * bins + b) * c + 2 * harmonics + channel] = ((v as f64 - sal_moments.0) / sal_moments.1) as f32;
                    }
                }
                for j in 0..nf {
                    let i = i0 + j;
                    let e = energy[i];
                    let weight = ((e as f64 / energy_mean).ln_1p().min(4.0) / 4.0) as f32;
                    let unit = cross[i] / e.max(1e-12);
                    let (al, ar) = (mag[0][i], mag[1][i]);
                    let ild = ((al + floor) / (ar + floor)).ln().clamp(-3.0, 3.0) / 3.0;
                    let o = &mut out[(j * bins + b) * c + 2 * harmonics + 2..][..3];
                    o[0] = unit.im * weight;
                    o[1] = unit.re * weight;
                    o[2] = if weight > 0.0 { ild } else { 0.0 };
                }
            }
        });
        (out, frames)
    }
}

/// Mean and (unbiased) standard deviation, as `HCQT.normalise` (the deviation at least 1e-8).
fn moments(x: &[f32]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.par_iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.par_iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt().max(1e-8))
}
