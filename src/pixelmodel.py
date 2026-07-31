"""Does watching the clip beat watching the dot?

The trajectory model tops out at AUC 0.875, and its learning curve went flat at
about a hundred labels: accuracy still creeps up, ranking quality does not. That
is the signature of an input limit rather than a data limit. A ball reduced to a
moving centroid throws away the net, the backboard and the moment of contact,
which is where a make and a near-miss actually differ.

So before asking review to hand-judge another 185 clips, measure whether pixels
carry signal the centroid does not -- on the clips he has ALREADY judged. If a
pixel model clears 0.875 on the same data, more labeling is worth his time. If
it does not, more labeling of the same kind will not rescue it either and the
answer is to change the input, not the volume.

Deliberately a frozen pretrained backbone plus a linear head, not a fine-tuned
video network. With roughly two hundred examples, fine-tuning would mostly
measure how well it memorises them. Frozen features keep the test honest and it
runs on the CPU in minutes.

    python src/pixelmodel.py --fetch     # download the judged clips once
    python src/pixelmodel.py             # embed and score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
CLIPDIR = ROOT / "out/judged_clips"
FRAMES = 10          # evenly spaced across the clip
SIDE = 160           # what the backbone sees

MISS_LABELS = {"offiron", "behind", "airball", "airnet", "inout"}


def fetch():
    """Pull the exact mp4 review watched for every judged clip.

    Deliberately the uploaded file rather than a freshly cut one: IMG_2481's
    clips were re-cut after he judged them, so a local re-cut no longer
    corresponds one-to-one with his answers. Training on a clip he did not
    actually watch would pair the wrong pixels with the label.
    """
    import urllib.request

    PB = "https://poolean-api.adammirmina.com"
    CLIPDIR.mkdir(parents=True, exist_ok=True)
    recs = json.loads((ROOT / "labels/judged_records.json").read_text(encoding="utf-8"))
    got = 0
    for r in recs:
        dest = CLIPDIR / f"{r['id']}.mp4"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{PB}/api/files/vision_shots/{r['id']}/{r['clip']}"
        try:
            urllib.request.urlretrieve(url, dest)
            got += 1
        except Exception as e:
            print(f"  failed {r['id']}: {e}")
    print(f"{got} newly downloaded, {len(list(CLIPDIR.glob('*.mp4')))} clips on disk")


def rim_in_clip(video: str, hoop: str, frame_w: int, frame_h: int):
    """Where the rim sits inside a rendered clip.

    clips.py cuts the source crop region for that hoop and resizes it to OUT_W
    wide, preserving aspect, so the mapping is a single scale factor. Without
    this the whole frame goes to the backbone and the rim is a handful of
    pixels in a picture that is mostly pool and sky -- which is the flaw in the
    first version of this test.
    """
    import hoops
    rig = hoops.rig_for(video)
    cx1, cy1, cx2, cy2 = rig.crops[hoop]
    rx1, ry1, rx2, ry2 = rig.rims[hoop]
    sx = frame_w / max(1, cx2 - cx1)
    sy = frame_h / max(1, cy2 - cy1)
    return ((rx1 - cx1) * sx, (ry1 - cy1) * sy, (rx2 - cx1) * sx, (ry2 - cy1) * sy)


def embed_all(tight: bool = False):
    import cv2
    import numpy as np
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    recs = json.loads((ROOT / "labels/judged_records.json").read_text(encoding="utf-8"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    X, y, meta = [], [], []
    for i, r in enumerate(recs):
        if r["label"] not in MISS_LABELS and r["label"] != "make":
            continue
        path = CLIPDIR / f"{r['id']}.mp4"
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < FRAMES:
            cap.release()
            continue
        idx = [int(k * (total - 1) / (FRAMES - 1)) for k in range(FRAMES)]
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        box = None
        if tight:
            rx1, ry1, rx2, ry2 = rim_in_clip(r["video"], r["hoop"], fw, fh)
            rw = max(8.0, rx2 - rx1)
            pad = 2.2 * rw          # enough to hold the approach and the exit
            ccx, ccy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            box = (max(0, int(ccx - pad)), max(0, int(ccy - pad)),
                   min(fw, int(ccx + pad)), min(fh, int(ccy + pad)))
            if box[2] - box[0] < 24 or box[3] - box[1] < 24:
                box = None
        frames = []
        for f in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                break
            if box:
                fr = fr[box[1]:box[3], box[0]:box[2]]
            fr = cv2.resize(fr, (SIDE, SIDE))
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append((fr - mean) / std)
        cap.release()
        if len(frames) < FRAMES:
            continue
        batch = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2))
        with torch.no_grad():
            emb = net(batch).numpy()          # FRAMES x 512
        # Mean says what is in the clip; max says whether it ever happened; the
        # successive differences say what CHANGED, which is the part a static
        # frame cannot express and the centroid model throws away entirely.
        d = np.diff(emb, axis=0)
        X.append(np.concatenate([emb.mean(0), emb.max(0), np.abs(d).mean(0)]))
        y.append(1 if r["label"] == "make" else 0)
        meta.append(r)
        if (i + 1) % 25 == 0:
            print(f"  embedded {len(X)}...")
    return np.array(X), np.array(y), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--tight", action="store_true",
                    help="crop to the rim's own neighbourhood instead of the whole clip")
    args = ap.parse_args()
    if args.fetch:
        fetch()
        return

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, meta = embed_all(tight=args.tight)
    print(f"\n{len(y)} clips embedded ({int(y.sum())} makes, {len(y) - int(y.sum())} misses)")
    print(f"feature width {X.shape[1]}, which is far wider than the sample count -- "
          f"so everything below is squeezed through PCA first and scored out-of-fold")

    base = max(y.mean(), 1 - y.mean())
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    print(f"\nbaseline: {base:.1%}        trajectory model, same question: 82.2% / AUC 0.875\n")

    best = (0.0, None)
    for k in (16, 32, 64):
        m = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                          LogisticRegression(max_iter=4000, C=0.1))
        p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
        acc = ((p >= 0.5).astype(int) == y).mean()
        auc = roc_auc_score(y, p)
        print(f"pixels, {k:>2} components:  accuracy {acc:.1%}   AUC {auc:.3f}")
        if auc > best[0]:
            best = (auc, p)

    np.save(ROOT / "out/pixel_scores.npy", best[1])
    print(f"\nbest pixel AUC {best[0]:.3f} vs trajectory 0.875")
    if best[0] > 0.90:
        print("Pixels clearly carry more. More labeling is worth the time.")
    elif best[0] > 0.875:
        print("Pixels are ahead, but not by much on this sample size.")
    else:
        print("Pixels do NOT beat the trajectory here. More of the same labeling "
              "will not lift it either -- the input has to change.")


if __name__ == "__main__":
    main()
