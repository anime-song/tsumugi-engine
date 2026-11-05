//! chords_decode CHORDS.safetensors PROBS.safetensors INFO.json: Tsumugi-Chords' decoder on given probabilities
//! (`detail` `[frames, 745]`, `bass` `[frames, 13]`, `boundary` `[frames]`), the chords as JSON on stdout (for
//! comparing with `ChordDecoder`).

use tsumugi_engine::chords::{Chords, Info, Probs};

fn main() -> tsumugi_engine::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = Chords::load(args[0].as_ref(), false)?;
    let bytes = std::fs::read(&args[1])?;
    let probs = safetensors::SafeTensors::deserialize(&bytes)?;
    let floats = |name: &str| -> tsumugi_engine::Result<Vec<f32>> {
        let data = probs.tensor(name)?.data().to_vec();
        Ok(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let boundary = floats("boundary")?;
    let p = Probs { frames: boundary.len(), bass: floats("bass")?, detail: floats("detail")?, boundary };
    let info: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let times = |k: &str| info[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
    let keys = info["keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| (k["start"].as_f64().unwrap(), k["end"].as_f64().unwrap(), k["key"].as_str().unwrap().to_string()))
        .collect();
    let info = Info { beats: times("beats"), downbeats: times("downbeats"), keys };
    let chords: Vec<serde_json::Value> = model.decoder.decode(&p, &info).iter().map(|c| c.json()).collect();
    println!("{}", serde_json::to_string(&chords)?);
    Ok(())
}
