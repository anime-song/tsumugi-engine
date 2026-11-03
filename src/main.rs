//! tsumugi-engine --weights cpu.safetensors [--adapter LORA] [--threads N] [--out-dir DIR]
//!                [--heads beat,melody,section,key] [--device cpu|gpu] [--logits] [--cpu-heads N] [--tempo-weight W]
//!                [--hold P] AUDIO...
//!
//! Writes `<out-dir>/<stem>.beats.json`, `<stem>.melody.json`, `<stem>.sections.json` and `<stem>.keys.json` (those
//! `--heads` names; default `beat,melody`) as `python -m sheetsage2_refined.beats` / `.melody` / `.sections` / `.keys`
//! do, and with `--logits` also `<stem>.heads.safetensors` with each head's logits (`<head>.<output>`,
//! `[frames, width]`) and, for the section and key models, the encoder states' block means (`encoder.states`).
//! AUDIO may be a WAV / FLAC / MP3 / Ogg file, or a `.f32` file of raw little-endian float32 samples, mono at 24 kHz.
//! `--adapter` applies a LoRA adapter (`export_cpu --adapter`) to the encoder at run time.
//! `--cpu-heads N` (with `--device gpu`) computes the last N of the 16 attention heads on the CPU while the GPU
//! computes the others.

use std::path::{Path, PathBuf};
use std::time::Instant;

use safetensors::tensor::{Dtype, TensorView};
use tsumugi_engine::chords::{Chords, Info};
use tsumugi_engine::{Frames, MemoryWatch, Model, Outputs, Result, audio, bars, encoder::SAMPLE_RATE, post};

fn usage() -> ! {
    eprintln!(
        "usage: tsumugi-engine --weights FILE [--adapter LORA] [--threads N] [--out-dir DIR] [--heads beat,melody,section,key] [--device cpu|gpu] [--logits] \
         [--cpu-heads N] [--tempo-weight W] [--hold P] [--chords FILE] [--precision f32|int16] AUDIO..."
    );
    std::process::exit(2)
}

fn load_wave(path: &Path) -> Result<Vec<f32>> {
    if path.extension().is_some_and(|e| e == "f32") {
        let bytes = std::fs::read(path)?;
        return Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    audio::load(path, SAMPLE_RATE)
}

/// A `--logits` file back as the heads' outputs and block means, and its duration.
fn read_logits(path: &Path) -> Result<(Frames, f64)> {
    let bytes = std::fs::read(path)?;
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes)?;
    let duration: f64 = meta.metadata().as_ref().and_then(|m| m.get("duration")).ok_or("no duration")?.parse()?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut names: Vec<String> = tensors.names().into_iter().map(str::to_string).collect();
    names.sort();
    let mut heads: Vec<(String, Outputs)> = Vec::new();
    let mut states = Vec::new();
    for key in names {
        let view = tensors.tensor(&key)?;
        if key == "encoder.states" {
            states = view.data().chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            continue;
        }
        let (head, layer) = key.split_once('.').ok_or("bad tensor name")?;
        let data: Vec<f32> =
            view.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let width = view.shape()[1];
        match heads.iter_mut().find(|(h, _)| h == head) {
            Some((_, outs)) => outs.push((layer.to_string(), width, data)),
            None => heads.push((head.to_string(), vec![(layer.to_string(), width, data)])),
        }
    }
    Ok((Frames { heads, states }, duration))
}

fn write_logits(path: &Path, frames: &Frames, duration: f64) -> Result<()> {
    let mut tensors = Vec::new();
    for (head, outs) in &frames.heads {
        for (name, width, logits) in outs {
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            tensors.push((format!("{head}.{name}"), Dtype::F32, vec![logits.len() / width, *width], bytes));
        }
    }
    if !frames.states.is_empty() {
        let bytes: Vec<u8> = frames.states.iter().flat_map(|v| v.to_le_bytes()).collect();
        let width = bars::GROUPS * tsumugi_engine::encoder::WIDTH;
        tensors.push(("encoder.states".into(), Dtype::F16, vec![frames.states.len() / width, width], bytes));
    }
    let views: Vec<(String, TensorView)> =
        tensors.iter().map(|(n, d, s, b)| (n.clone(), TensorView::new(*d, s.clone(), b).expect("tensor"))).collect();
    let meta = std::collections::HashMap::from([("duration".to_string(), duration.to_string())]);
    safetensors::serialize_to_file(views, Some(meta), path)?;
    Ok(())
}

/// Tsumugi-Chords' `<song>.chords.json` and `.chords.lab` from the stereo waveform and the song's `keys.json`.
fn chord_outputs(model: &Chords, wave: &[Vec<f32>; 2], keys: &serde_json::Value) -> Result<(serde_json::Value, String)> {
    let floats = |k: &str| -> Vec<f64> { keys[k].as_array().map(|a| a.iter().filter_map(|v| v.as_f64()).collect()).unwrap_or_default() };
    let segments = keys["keys"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|k| Some((k["start"].as_f64()?, k["end"].as_f64()?, k["key"].as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();
    let info = Info { beats: floats("beats"), downbeats: floats("downbeats"), keys: segments };
    let found = if info.beats.len() < 2 { Vec::new() } else { model.decoder.decode(&model.probabilities(wave, &info), &info) };
    let lab: String = found.iter().map(|c| format!("{:.3}\t{:.3}\t{}\n", c.start, c.end, c.label)).collect();
    let chords: Vec<serde_json::Value> = found
        .iter()
        .map(|c| serde_json::json!({"start": c.start, "end": c.end, "label": c.label, "root": c.root, "quality": c.quality, "bass": c.bass}))
        .collect();
    let json = serde_json::json!({"audio": keys["audio"], "duration": keys["duration"], "beats": keys["beats"],
                                  "downbeats": keys["downbeats"], "keys": keys["keys"], "chords": chords});
    Ok((json, lab))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut weights, mut threads, mut out_dir) = (None, None, PathBuf::from("outputs/cpu"));
    let (mut heads, mut logits, mut tempo_weight, mut inputs) = ("beat,melody".to_string(), false, 4.0, Vec::new());
    let mut hold = 0.0f64; // the beat tracker's fermata holds (`beats --hold`)
    let mut from_logits = false;
    let mut device = "cpu".to_string(); // inputs are `--logits` files: run the trackers only (for testing them)
    let mut adapter = None;
    let mut cpu_heads = 0usize;
    let mut chords_file = None;
    let mut precision = "f32".to_string();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--weights" => weights = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--chords" => chords_file = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--precision" => precision = args.next().unwrap_or_else(|| usage()),
            "--adapter" => adapter = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--threads" => threads = Some(args.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or_else(|| usage())),
            "--out-dir" => out_dir = PathBuf::from(args.next().unwrap_or_else(|| usage())),
            "--heads" => heads = args.next().unwrap_or_else(|| usage()),
            "--logits" => logits = true,
            "--device" => device = args.next().unwrap_or_else(|| usage()),
            "--from-logits" => from_logits = true,
            "--cpu-heads" => cpu_heads = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--hold" => hold = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--tempo-weight" => tempo_weight = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "-h" | "--help" => usage(),
            _ => inputs.push(PathBuf::from(arg)),
        }
    }
    let Some(weights) = weights else { usage() };
    if inputs.is_empty() {
        usage();
    }
    tsumugi_engine::simd::init_threads(threads)?;
    let started = Instant::now();
    let heads: Vec<&str> = heads.split(',').filter(|h| !h.is_empty()).collect();
    // the chord model reads the beats, downbeats and keys
    let mut load: Vec<&str> = heads.clone();
    if chords_file.is_some() && !load.contains(&"key") {
        load.push("key");
    }
    let int16 = match precision.as_str() {
        "f32" => false,
        "int16" => true,
        _ => usage(),
    };
    let chords = chords_file.as_deref().map(|p| Chords::load(p, int16)).transpose()?;
    let mut model = match device.as_str() {
        "cpu" => Model::load(&weights, &load)?,
        #[cfg(feature = "gpu")]
        "gpu" => Model::load_gpu(&weights, &load)?,
        #[cfg(not(feature = "gpu"))]
        "gpu" => return Err("built without the GPU encoder: cargo build --release --features gpu".into()),
        _ => usage(),
    };
    if int16 {
        let t = Instant::now();
        model.quantize()?;
        eprintln!("encoder on int16 ({:.2} s)", t.elapsed().as_secs_f64());
    }
    if let Some(path) = &adapter {
        let adapter = tsumugi_engine::lora::Adapter::open(path)?;
        model.set_adapter(Some(&adapter))?;
        eprintln!(
            "adapter {}: {} layers, {:.1} MiB",
            path.display(),
            adapter.layers.len(),
            adapter.bytes() as f64 / (1 << 20) as f64
        );
    }
    #[cfg(feature = "gpu")]
    if let Some(gpu) = &mut model.gpu {
        gpu.set_cpu_heads(cpu_heads)?;
    } else if cpu_heads > 0 {
        return Err("--cpu-heads needs --device gpu".into());
    }
    #[cfg(not(feature = "gpu"))]
    if cpu_heads > 0 {
        return Err("--cpu-heads needs --device gpu".into());
    }
    #[cfg(feature = "gpu")]
    if let Some(gpu) = &model.gpu {
        let mib = |b: u64| b as f64 / (1 << 20) as f64;
        eprintln!(
            "encoder on {}: {:.0} MiB of weights and {:.0} MiB of activations in GPU memory",
            gpu.adapter,
            mib(gpu.memory.0),
            mib(gpu.memory.1)
        );
    }
    let decoding: serde_json::Value =
        serde_json::from_str(model.weights.metadata.get("melody_decoding").map_or("{}", |s| s))?;
    eprintln!(
        "loaded {} ({}) in {:.2} s, AVX2: {}",
        weights.display(),
        model.weights.metadata.get("dtype").map_or("?", |s| s),
        started.elapsed().as_secs_f64(),
        tsumugi_engine::simd::avx2()
    );
    let tracker_options = tsumugi_engine::tracker::Options { p_hold: hold, ..Default::default() };
    std::fs::create_dir_all(&out_dir)?;
    for input in &inputs {
        let watch = MemoryWatch::start();
        let t0 = Instant::now();
        let mut stereo = None;
        let (frames, duration, t1) = if from_logits {
            let (frames, duration) = read_logits(input)?;
            (frames, duration, Instant::now())
        } else {
            let wave = match &chords {
                Some(c) if input.extension().is_none_or(|e| e != "f32") => {
                    let (both, mono) = audio::load_both(input, c.net.rate, SAMPLE_RATE)?;
                    stereo = Some(both);
                    mono
                }
                Some(c) => {
                    let wave = load_wave(input)?;
                    let up = audio::resample(&wave, SAMPLE_RATE, c.net.rate);
                    stereo = Some([up.clone(), up]);
                    wave
                }
                None => load_wave(input)?,
            };
            let t1 = Instant::now();
            (model.frames(&wave)?, wave.len() as f64 / SAMPLE_RATE as f64, t1)
        };
        let t2 = Instant::now();
        tsumugi_engine::profile::report();
        let stem = input.file_stem().map_or("audio".into(), |s| s.to_string_lossy().to_string());
        let stem = stem.strip_suffix(".heads").map_or(stem.clone(), str::to_string);
        let name = input.to_string_lossy().to_string();
        let head = |name: &str| frames.heads.iter().find(|(h, _)| h == name).map(|(_, o)| o);
        let mut results = Vec::new();
        let mut chord_files = Vec::new();
        if let Some(outs) = head("beat") {
            let probs = post::beat_probs(outs)?;
            let tracked = tsumugi_engine::tracker::track_beats(&probs, &tracker_options, tempo_weight);
            let result = post::tracked_json(&tracked, &name, duration);
            if model.sections.is_some() || model.keys.is_some() {
                let melody = head("melody").ok_or("the bar models need the melody head")?;
                let inputs = bars::FrameInputs::new(melody, head("chord"), &frames.states)?;
                let beats: Vec<f64> = tracked.beats.iter().map(|&b| post::r4(b)).collect();
                let downbeats: Vec<f64> = tracked.downbeats.iter().map(|&d| post::r4(d)).collect();
                let t = Instant::now();
                if let Some(sections) = &model.sections {
                    let mut r = result.clone();
                    r["sections"] = bars::sections_json(sections, &inputs, &beats, &downbeats, duration);
                    results.push(("sections", r));
                }
                if let Some(keys) = &model.keys {
                    let mut r = result.clone();
                    r["keys"] = bars::keys_json(keys, &inputs, &beats, &downbeats, duration);
                    if heads.contains(&"key") {
                        results.push(("keys", r.clone()));
                    }
                    if let (Some(model), Some(wave)) = (&chords, &stereo) {
                        let t = Instant::now();
                        let (json, lab) = chord_outputs(model, wave, &r)?;
                        tsumugi_engine::profile::add("chords", t.elapsed().as_secs_f64());
                        let path = out_dir.join(format!("{stem}.chords.lab"));
                        std::fs::write(&path, lab)?;
                        chord_files.push(path);
                        results.push(("chords", json));
                    }
                }
                tsumugi_engine::profile::add("bar models", t.elapsed().as_secs_f64());
            }
            if heads.contains(&"beat") {
                results.insert(0, ("beats", result));
            }
        }
        if let Some(outs) = head("melody").filter(|_| heads.contains(&"melody")) {
            results.push(("melody", post::melody_json(outs, &name, duration, &decoding)?));
        }
        let mut written = chord_files;
        for (suffix, result) in results {
            let path = out_dir.join(format!("{stem}.{suffix}.json"));
            std::fs::write(&path, serde_json::to_string(&result)?)?;
            written.push(path);
        }
        if logits && !from_logits {
            let path = out_dir.join(format!("{stem}.heads.safetensors"));
            write_logits(&path, &frames, duration)?;
            written.push(path);
        }
        tsumugi_engine::profile::report();
        let t3 = Instant::now();
        let (anon, file) = watch.stop();
        drop(frames);
        tsumugi_engine::release_memory();
        if std::env::var_os("SHEETSAGE_PROFILE").is_some() {
            let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
            eprintln!("  after the request: {}", status.lines().find(|l| l.starts_with("RssAnon")).unwrap_or(""));
        }
        let encode = (t2 - t1).as_secs_f64();
        eprintln!(
            "{}: {duration:.1} s of audio, decode {:.2} s, encoder + heads {encode:.2} s ({:.1}x real time), trackers {:.2} s, \
             memory {anon:.0} MiB + weights {file:.0} MiB mapped -> {}",
            input.display(),
            (t1 - t0).as_secs_f64(),
            duration / encode,
            (t3 - t2).as_secs_f64(),
            written.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}
