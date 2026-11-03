//! chords_check CHORDS.safetensors WAVE.f32 INFO.json OUT.safetensors: the chord model's frontend and stem on the first
//! 30 s window and the song's probabilities and chords, for comparing with PyTorch (`WAVE.f32`: the stereo waveform
//! at the model's rate, the left channel's samples then the right's; `INFO.json`: beats, downbeats and keys).

use std::collections::HashMap;
use std::time::Instant;

use safetensors::tensor::{Dtype, TensorView};
use sheetsage2_cpu::chords::{Chords, Info};

fn main() -> sheetsage2_cpu::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = Chords::load(args[0].as_ref())?;
    let bytes = std::fs::read(&args[1])?;
    let samples: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let n = samples.len() / 2;
    let wave = [samples[..n].to_vec(), samples[n..].to_vec()];
    let info: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let floats = |k: &str| info[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
    let keys = info["keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| (k["start"].as_f64().unwrap(), k["end"].as_f64().unwrap(), k["key"].as_str().unwrap().to_string()))
        .collect();
    let info = Info { beats: floats("beats"), downbeats: floats("downbeats"), keys };
    let mut out: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let size = 750 * 1024;
    let mut first = [wave[0].clone(), wave[1].clone()];
    for w in first.iter_mut() {
        w.resize(size.max(w.len()), 0.0);
        w.truncate(size);
    }
    let t = Instant::now();
    let (spec, frames) = model.net.frontend.forward(&[&first[0], &first[1]]);
    eprintln!("frontend {:.2} s", t.elapsed().as_secs_f64());
    let bins = spec.len() / frames / 15;
    let t = Instant::now();
    let (stem, h, w) = model.net.stem(&spec, frames, bins);
    eprintln!("stem {:.2} s", t.elapsed().as_secs_f64());
    out.push(("spec".into(), vec![frames, bins, 15], spec));
    out.push(("stem".into(), vec![h, w, stem.len() / (h * w)], stem));
    let t = Instant::now();
    let p = model.probabilities(&wave, &info);
    eprintln!("probabilities {:.2} s", t.elapsed().as_secs_f64());
    sheetsage2_cpu::profile::report();
    let t = Instant::now();
    let chords = model.decoder.decode(&p, &info);
    eprintln!("decode {:.3} s, {} chords", t.elapsed().as_secs_f64(), chords.len());
    out.push(("detail".into(), vec![p.frames, 745], p.detail));
    out.push(("bass".into(), vec![p.frames, 13], p.bass));
    out.push(("boundary".into(), vec![p.frames], p.boundary));
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> =
        out.into_iter().map(|(n, s, v)| (n, s, v.iter().flat_map(|x| x.to_le_bytes()).collect())).collect();
    let views: Vec<(String, TensorView)> =
        bytes.iter().map(|(n, s, b)| (n.clone(), TensorView::new(Dtype::F32, s.clone(), b).unwrap())).collect();
    let labels: Vec<serde_json::Value> = chords
        .iter()
        .map(|c| serde_json::json!({"start": c.start, "end": c.end, "label": c.label, "root": c.root, "quality": c.quality, "bass": c.bass}))
        .collect();
    let meta = HashMap::from([("chords".to_string(), serde_json::to_string(&labels)?)]);
    safetensors::serialize_to_file(views, Some(meta), args[3].as_ref())?;
    Ok(())
}
