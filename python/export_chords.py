"""Export Tsumugi-Chords (the chord model and its decoder's tables) for the engine.

    python python/export_chords.py chords.safetensors [--model REPO_OR_FOLDER] [--revision REV]

writes one safetensors file next to SheetSage2-Refined's export (``python/export_sheetsage.py``), which the
engine reads for the beats, downbeats and keys the chord model takes.  As there, constants are folded where the
engine would otherwise compute them per song:

* the RMSNorm scales go into the linear layers they feed, attention's ``1/sqrt(head_dim)`` into q;
* convolutions become linear layers over channels-last patches (``[out, kh, kw, in]``), the 4-frame key / beat token
  convolutions likewise (``[out, 4, in]``), the up-sampling transposed convolution one linear layer to 4 frames;
* the key queries (an MLP of constants), the chord queries and the token types are stored as the tokens they make;
* the adapter's LayerNorm goes into its first linear layer;
* of the outputs only the bass, root x quality and chord-boundary rows are kept (what the decoder reads);
* the CQT's kernels, windows and resampling filter, the harmonic interpolation and the decoder's tables (length,
  position, the key-relative bigram, the quality families, chord tones and labels) are stored as they are computed.

Weights are float32 (``--dtype f16`` stores the linear layers as float16, half the file).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from tsumugi_chords import chords as chord_tables
from tsumugi_chords import semi_crf
from tsumugi_chords.chord_crf import GROUP, NS
from tsumugi_chords.decode import bigram
from tsumugi_chords.note_chords import OUTPUTS, load_note_chords
from tsumugi_chords.theory import QUALITIES, chord_to_mir_eval, chord_tone_templates
from tsumugi_chords.tracker import DEFAULT_MODEL, model_file

PANEL = 16
FORMAT = "tsumugi-chords-cpu/1"


class Writer:
    def __init__(self, dtype: str):
        self.dtype, self.tensors = dtype, {}

    def linear(self, name: str, weight, bias=None) -> None:
        """``weight`` ``[N, K]`` in panels of 16 outputs ``[ceil(N / 16), K, 16]`` (the engine's layout)."""

        weight = np.asarray(weight, dtype=np.float32)
        n, k = weight.shape
        bias = np.zeros(n, np.float32) if bias is None else np.asarray(bias, dtype=np.float32)
        padded = np.zeros(((n + PANEL - 1) // PANEL * PANEL, k), dtype=np.float32)
        padded[:n] = weight
        if self.dtype == "f16":
            padded = padded.astype(np.float16)
        self.tensors[f"{name}.w"] = np.ascontiguousarray(padded.reshape(-1, PANEL, k).transpose(0, 2, 1))
        self.tensors[f"{name}.b"] = bias

    def vector(self, name: str, value) -> None:
        self.tensors[name] = np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def t(x) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def conv2d(out: Writer, name: str, conv) -> None:
    w = t(conv.weight)  # [out, in, kh, kw]
    out.linear(name, w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1), t(conv.bias))


def group_norm(out: Writer, name: str, norm) -> None:
    out.vector(f"{name}.g", t(norm.weight))
    out.vector(f"{name}.b", t(norm.bias))


def block(out: Writer, name: str, layer, heads: int) -> None:
    """An axis-Transformer ``Block``: RMSNorm scales folded, q scaled by ``1 / sqrt(head_dim)``."""

    a = layer.attention
    gamma = t(a.norm.gamma)
    qkv_w, qkv_b = t(a.qkv.weight) * gamma[None], t(a.qkv.bias)
    dim = qkv_w.shape[1]
    scale = 1.0 / math.sqrt(dim // heads)
    qkv_w[:dim] *= scale
    qkv_b[:dim] *= scale
    # q, k, v and the gates in one product
    out.linear(f"{name}.qkvg", np.concatenate([qkv_w, t(a.gates.weight) * gamma[None]]), np.concatenate([qkv_b, t(a.gates.bias)]))
    out.linear(f"{name}.out", t(a.out.weight), t(a.out.bias))
    ffn_gamma = t(layer.ffn[0].gamma)
    out.linear(f"{name}.ff1", t(layer.ffn[1].weight) * ffn_gamma[None], t(layer.ffn[1].bias))
    out.linear(f"{name}.ff2", t(layer.ffn[4].weight), t(layer.ffn[4].bias))


def export_model(out: Writer, model) -> dict:
    notes, config = model.notes, model.notes.config
    if model.reader != "query" or model.note_head or config.loops != 1:
        raise ValueError("the engine runs the query reader without the note head, one pass of the layers")
    cqt, hcqt = notes.frontend.hcqt.cqt, notes.frontend.hcqt
    for i in range(cqt.octaves):
        kernel = getattr(cqt, f"kernel_{i}")  # [bins, size / 2 + 1] complex
        out.vector(f"cqt.{i}.re", kernel.real)
        out.vector(f"cqt.{i}.im", kernel.imag)
        out.vector(f"cqt.{i}.window", t(getattr(cqt, f"window_{i}")))
    out.vector("cqt.fir", t(cqt.fir).ravel())
    out.tensors["hcqt.lower"] = t(hcqt.lower).astype(np.int32).astype(np.float32)
    out.tensors["hcqt.upper"] = t(hcqt.upper).astype(np.int32).astype(np.float32)
    out.vector("hcqt.alpha", t(hcqt.alpha)[..., 0])
    stem = notes.stem
    conv2d(out, "stem.conv1", stem.conv1)
    conv2d(out, "stem.conv2", stem.conv2)
    out.vector("stem.freq_embed", t(stem.freq_embed)[0, :, 0].T)  # [bins, channels]
    for name in ("block1", "block2", "block3", "block4"):
        seq = getattr(stem, name)
        conv2d(out, f"stem.{name}.conv", seq[0])
        group_norm(out, f"stem.{name}.norm", seq[1])
    for i, res in enumerate(notes.stem_extra):
        group_norm(out, f"extra.{i}.norm1", res.net[0])
        conv2d(out, f"extra.{i}.conv1", res.net[2])
        group_norm(out, f"extra.{i}.norm2", res.net[3])
        conv2d(out, f"extra.{i}.conv2", res.net[5])
    if not isinstance(notes.stem_proj, torch.nn.Identity):
        raise ValueError("a stem projection is not supported")
    out.vector("tokens.band", t(notes.types[0]))
    out.vector("tokens.keys", t(notes.pitch_query() + notes.types[1]))  # [88, D]
    out.vector("tokens.chords", t((model.chord_token + model.types[2]).reshape(-1, config.dim)))  # [chords, D]
    for name, conv, kind in (("tokens.key_in", model.key_in, 0), ("tokens.beat_in", model.beat_in, 1)):
        w = t(conv.weight)  # [out, in, 4]
        out.linear(name, w.transpose(0, 2, 1).reshape(w.shape[0], -1), t(conv.bias) + t(model.types[kind]))
    for i in range(config.layers):
        block(out, f"row.{i}", notes.row_layers[i], config.heads)
        block(out, f"time.{i}", notes.time_layers[i], config.heads)
    chords = model.chords
    gamma = np.tile(t(notes.norm.gamma), chords)
    if model.merge is not None:
        out.linear("merge", t(model.merge.weight) * gamma[None], t(model.merge.bias))
    else:
        out.vector("merge.gamma", gamma)
    w = t(model.up.weight)  # [in, out, 4]
    out.linear("up", w.transpose(2, 1, 0).reshape(-1, w.shape[0]), np.tile(t(model.up.bias), 4))
    ln, a1, a2 = model.adapter[0], model.adapter[1], model.adapter[4]
    out.linear("adapter.1", t(a1.weight) * t(ln.weight)[None], t(a1.bias) + t(a1.weight) @ t(ln.bias))
    out.linear("adapter.2", t(a2.weight), t(a2.bias))
    offsets, at = {}, 0
    for name, n in OUTPUTS:
        offsets[name] = (at, at + n)
        at += n
    rows = [np.arange(*offsets[name]) for name in ("bass", "detail", "chord_boundary")]
    keep = np.concatenate(rows)
    out.linear("out", t(model.out.weight)[keep], t(model.out.bias)[keep])
    return {"rate": config.rate, "hop": config.hop, "channels": config.channels, "bins": config.bins,
            "harmonics": len(config.harmonics), "phase": config.phase, "stereo": config.stereo, "dim": config.dim,
            "heads": config.heads, "layers": config.layers, "stem_extra": config.stem_extra, "chords": chords,
            "octaves": cqt.octaves, "fft_sizes": cqt.fft_sizes, "large": hcqt.large, "actual": hcqt.actual,
            "bands": notes.bands, "keys": 88, "outputs": {"bass": 13, "detail": len(keep) - 14, "boundary": 1}}  # fmt: skip


def export_decoder(out: Writer, folder_file, tracker: dict) -> dict:
    tables = semi_crf.tables_from_counts(dict(np.load(folder_file(tracker["tables"]))))
    for name, value in tables.items():
        out.vector(f"decoder.{name}", value)
    out.vector("decoder.lm", bigram(folder_file(tracker["bigram"])))  # [NS, NS]
    out.vector("decoder.group", np.asarray(GROUP, dtype=np.float32))  # [745] detail class -> root x family state
    out.vector("decoder.quality_family", chord_tables.QUALITY_FAMILY.astype(np.float32))  # [62]
    out.vector("decoder.tones", chord_tone_templates())  # [745, 12]
    params = semi_crf.Params(**tracker["decoding"])
    labels = [chord_to_mir_eval(0, q, None).split(":", 1)[1] for q in QUALITIES]  # mir_eval shorthand after "C:"
    return {"params": {k: v for k, v in vars(params).items() if k != "extra"}, "qualities": list(QUALITIES),
            "labels": labels, "states": NS, "dur_edges": semi_crf.DUR_EDGES.tolist(), "subs": list(semi_crf.SUBS),
            "sub_tolerance": semi_crf.SUB_TOLERANCE}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local model folder")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32", help="the linear layers' weights")
    args = parser.parse_args()
    folder_file = lambda name: model_file(args.model, name, args.revision)  # noqa: E731
    config = json.loads(folder_file("config.json").read_text(encoding="utf-8"))
    tracker = config["chord_tracker"]
    out = Writer(args.dtype)
    model = load_note_chords(folder_file(tracker["model"]))
    with torch.no_grad():
        meta = export_model(out, model)
    decoder = export_decoder(out, folder_file, tracker)
    metadata = {"format": FORMAT, "dtype": args.dtype, "model": str(args.model), "revision": str(args.revision),
                "config": json.dumps(meta), "decoder": json.dumps(decoder),
                "beats_keys": json.dumps(config.get("beats_keys") or {})}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out.tensors, str(args.out), metadata=metadata)
    size = sum(v.nbytes for v in out.tensors.values())
    print(f"{args.out}: {len(out.tensors)} tensors, {size / 2**20:.0f} MiB ({args.dtype})")


if __name__ == "__main__":
    main()
