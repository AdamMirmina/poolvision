"""Could identity work WITHOUT caps? Measured, not guessed.

review asked whether this will eventually be feasible bare-headed. The project has
assumed not since the beginning, on research rather than on evidence from this
pool, and an assumption that old deserves a number.

The obstacle to measuring it is that nobody has labeled who is who in the
bare-headed footage, and doing so is exactly the tedious job this project exists
to avoid. But the measurement does not need labels:

  two people visible in the SAME frame are definitely different people
  one track at two different moments is definitely the same person

So compare appearance embeddings within a track against embeddings across
simultaneous tracks. If the same person at two moments looks no more alike than
two different people do, re-identification cannot work here, and no amount of
model shopping changes that -- the information is not in the pixels.

The capped footage is the positive control. If the same measurement separates
cleanly there and not bare-headed, the difference is the caps, which is the
whole question.

    python src/nocaps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
POOL = (1150, 250, 3500, 1750)
PERSON = 0
FRAMES = 40          # sampled across the middle of the recording
STRIDE = 12          # frames between samples: people move, lighting does not


def collect(video: str, model, cv2, np, torch, net, mean, std):
    """Person crops with an appearance embedding, grouped into crude tracks."""
    cap = cv2.VideoCapture(str(ROOT / "footage" / video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(total * 0.45)           # the middle is what the recording is about
    dets = []
    for i in range(FRAMES):
        f = start + i * STRIDE
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        r = model.predict(fr, conf=0.30, verbose=False, classes=[PERSON], imgsz=1280)[0]
        batch, boxes = [], []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not (POOL[0] <= cx <= POOL[2] and POOL[1] <= cy <= POOL[3]):
                continue
            crop = fr[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if crop.size == 0 or crop.shape[0] < 40:
                continue
            im = cv2.resize(crop, (96, 192))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            batch.append(((im - mean) / std).transpose(2, 0, 1))
            boxes.append((f, cx, cy))
        if not batch:
            continue
        with torch.no_grad():
            emb = net(torch.from_numpy(np.stack(batch))).numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        for (f_, cx, cy), e in zip(boxes, emb):
            dets.append({"f": f_, "x": cx, "y": cy, "e": e})
    cap.release()

    # Crude tracking: nearest neighbour between consecutive sampled frames.
    # Good enough -- this measures appearance, and a track only has to be mostly
    # one person for "same track" to mean "same person" on average.
    tracks: list[list[dict]] = []
    for d in dets:
        best, bd = None, 1e9
        for t in tracks:
            last = t[-1]
            if d["f"] <= last["f"]:
                continue
            gap = (d["f"] - last["f"]) / STRIDE
            if gap > 2:
                continue
            dist = ((d["x"] - last["x"]) ** 2 + (d["y"] - last["y"]) ** 2) ** 0.5
            if dist < 260 * gap and dist < bd:
                best, bd = t, dist
        if best is None:
            tracks.append([d])
        else:
            best.append(d)
    return dets, [t for t in tracks if len(t) >= 3]


def main():
    import cv2
    import numpy as np
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    from ultralytics import YOLO

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    model = YOLO("yolo11s.pt")

    out = {}
    for video, tag in (("IMG_2483.MOV", "bare-headed"), ("IMG_2482.MOV", "caps on")):
        dets, tracks = collect(video, model, cv2, np, torch, net, mean, std)
        print(f"\n=== {video} ({tag}) ===")
        print(f"{len(dets)} people detected, {len(tracks)} tracks of 3+ sightings")
        if len(tracks) < 2:
            print("not enough tracks to compare")
            continue

        # same person, different moments
        same = []
        for t in tracks:
            for i in range(len(t)):
                for k in range(i + 1, len(t)):
                    if t[k]["f"] - t[i]["f"] >= STRIDE * 2:
                        same.append(float(t[i]["e"] @ t[k]["e"]))

        # different people, same instant -- guaranteed different, no labels needed
        byf: dict[int, list] = {}
        for ti, t in enumerate(tracks):
            for d in t:
                byf.setdefault(d["f"], []).append((ti, d))
        diff = []
        for f, group in byf.items():
            for i in range(len(group)):
                for k in range(i + 1, len(group)):
                    if group[i][0] != group[k][0]:
                        diff.append(float(group[i][1]["e"] @ group[k][1]["e"]))

        if not same or not diff:
            print("not enough pairs")
            continue
        s, d = np.array(same), np.array(diff)
        print(f"  same person, different moments : {s.mean():.3f} +- {s.std():.3f}  ({len(s)} pairs)")
        print(f"  different people, same instant : {d.mean():.3f} +- {d.std():.3f}  ({len(d)} pairs)")
        gap = s.mean() - d.mean()
        pooled = ((s.std() ** 2 + d.std() ** 2) / 2) ** 0.5
        print(f"  separation: {gap:+.3f}  ({gap / (pooled + 1e-9):.2f} standard deviations)")
        # How often would a nearest-neighbour match pick the right person?
        thr = (s.mean() + d.mean()) / 2
        acc = ((s > thr).sum() + (d <= thr).sum()) / (len(s) + len(d))
        print(f"  a best-guess threshold would be right {acc:.0%} of the time")
        out[tag] = {"same": s.mean(), "diff": d.mean(), "sep": gap, "acc": acc}

    if len(out) == 2:
        b, c = out["bare-headed"], out["caps on"]
        print("\n---")
        print(f"bare-headed separation {b['sep']:+.3f} vs capped {c['sep']:+.3f}")
        if b["acc"] < 0.75:
            print("Bare-headed identity does not separate. Caps are doing the work,")
            print("and this is not a model-shopping problem: the appearance of a")
            print("shirtless swimmer at this distance simply does not carry it.")
    (ROOT / "out/nocaps.json").write_text(json.dumps(
        {k: {kk: float(vv) for kk, vv in v.items()} for k, v in out.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
