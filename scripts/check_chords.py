"""Check the engine's chords against Tsumugi-Chords (PyTorch, CPU) on the same audio, beats and keys.

    python scripts/check_chords.py --out-dir DIR [--model REPO_OR_FOLDER] AUDIO ...

reads ``<DIR>/<song>.chords.json`` (``tsumugi-engine --chords``), runs ``HeadChords.probabilities`` and the decoder
on the beats, downbeats and keys stored there, and compares the chords (times, labels, roots, qualities, basses).
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from tsumugi_chords.tracker import DEFAULT_MODEL, HeadChords


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    tracker = HeadChords.load(args.model, "cpu")
    failures = 0
    for audio in args.audio:
        got = json.loads((args.out_dir / f"{audio.stem}.chords.json").read_text(encoding="utf-8"))
        info = {k: got[k] for k in ("beats", "downbeats", "keys")}
        p = tracker.probabilities(audio, info)
        want = [asdict(c) for c in tracker.decoder(p["detail"], p["bass"], p["boundary"], info["beats"], info["downbeats"], info["keys"])]
        same = want == got["chords"]
        failures += not same
        print(f"{audio.name}: {len(want)} chords, {'equal' if same else 'DIFFERENT'}")
        if not same:
            for a, b in zip(want + [None], got["chords"] + [None]):
                if a != b:
                    print(f"  first difference:\n  python {a}\n  engine {b}")
                    break
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
