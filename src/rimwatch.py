"""Look only at the hoops, at full resolution.

The full-frame pass tracks the ball fine in open water and never once sees it
within 150px of a rim. Two explanations fit that: the detector loses the ball
in the rim region, or there simply weren't many shots in the window. This tells
them apart, and if it's the former it is also most of the fix.

Downscaling a 3840-wide frame to 1920 for inference halves the ball. Cropping a
box around a hoop and running the model on that crop instead keeps native
pixels, so the ball is twice the size to the detector and the background is a
fraction of the scene. It is also cheaper per frame than the full-frame pass,
not more expensive, because the crop is small.

Output is a candidate list of moments the ball was near a rim, which is what
turns two hours of footage into a short list to confirm by eye.

Run: python src/rimwatch.py footage/x.MOV --from 280 --to 520
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hoops import rig_for  # noqa: E402

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent
SPORTS_BALL = 32
FINE_CONF = 0.05
# How many consecutive decode failures before a scan is declared incomplete.
MAX_BAD_GRABS = 90   # the fine-tune's scores never exceed 0.25



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    # THE WHOLE VIDEO by default. These used to default to 280-520s, a window
    # left over from an early experiment, so every run invoked without --from/--to
    # scanned four minutes of a fifteen-minute recording and reported success.
    #
    # I then diagnosed the missing 60% as a decoder failure and wrote a fix for a
    # bug that did not exist. The tool had printed "rimwatch 280s-520s" as its
    # FIRST LINE on every run, and I never read it -- I read the summary at the
    # end instead, which is exactly the habit that made the other four failures
    # today survive.
    #
    # 0 means "to the end".
    p.add_argument("--from", dest="t0", type=float, default=0.0)
    p.add_argument("--to", dest="t1", type=float, default=0.0)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--fine", action="store_true",
                   help="second look with the fine-tuned ball detector")
    # 1600. Measured on the minute marked by hand, at the five shots he
    # confirmed are real, with the widened crop:
    #
    #   imgsz  640    256 sightings in the window, 1 of his 5 shots seen
    #   imgsz  960    306 sightings, 5 of 5
    #   imgsz 1600    858 sightings, 5 of 5
    #
    # The crop and this are ONE tradeoff. A wider crop sees the ball's approach
    # and is then downscaled harder for the same imgsz, so the ball arrives
    # smaller -- which is why widening the crop alone barely helped, and why the
    # original "tight crop is better" measurement was reading the resolution half
    # of a tradeoff while missing the coverage half.
    #
    # 640 had never been compared against anything.
    p.add_argument("--imgsz", type=int, default=1600)
    p.add_argument("--step", type=int, default=1, help="frame stride")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any checkpoint and rescan from the start")
    # Settable so the resume path can actually be TESTED. A feature whose only
    # trigger is "wait an hour then pull the plug" does not get tested, and an
    # untested resume is worse than none: it would restore silently wrong state.
    p.add_argument("--ckpt-every", type=int, default=1800,
                   help="frames between checkpoints")
    p.add_argument("--out", type=Path, default=Path("out/rimwatch.json"))
    # Override the rig's detector padding, so the crop's effect on how much of a
    # descent is visible can be measured rather than argued about. A tighter crop
    # makes the ball bigger to the model and shortens the vertical span the
    # flight is seen over, and those pull in opposite directions.
    p.add_argument("--pad", default="", help="PX_X,PX_Y to override the detector crop")
    return p.parse_args()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    args = parse_args()
    import cv2
    import numpy as np
    from ultralytics import YOLO

    rig = rig_for(args.video)
    if args.pad:
        import hoops
        px, py = (int(v) for v in args.pad.split(","))
        rig = hoops.Rig(rims=rig.rims, crops=rig.crops, pool=rig.pool,
                        dets={k: hoops._crop_around(v, px, py) for k, v in rig.rims.items()})
    model = YOLO(args.model)
    fine = None
    if args.fine:
        fw = ROOT / 'out/balltrain/ball/weights/best.pt'
        if fw.exists():
            fine = YOLO(str(fw))
            print(f'second look: {fw.name}')
        else:
            print('no fine-tuned weights; stock only')
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 0 means the end of the file, so the default really is "all of it".
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start = int(args.t0 * fps)
    end = int(args.t1 * fps) if args.t1 > 0 else total
    # RESUME. A scan is hours long and this machine restarts on its own: one
    # Windows-initiated reboot on 2026-08-10 killed a scan at 3600 of 4500 frames
    # and another before its first progress line, and because the output is only
    # written at the very end, all of it was lost. Worse, a dead scan and a
    # running one look identical from the log file, so it went unnoticed for a
    # day.
    #
    # The checkpoint is written beside the output as <out>.partial, holding
    # everything found so far plus the frame reached. On restart the hits are
    # loaded back and the decoder seeks past them.
    ckpt = args.out.with_suffix(args.out.suffix + ".partial")
    hits_seed, resume_at = {}, start
    if ckpt.exists() and not args.fresh:
        try:
            d = json.loads(ckpt.read_text(encoding="utf-8"))
            # Only trust a checkpoint describing the SAME window and settings.
            # Resuming one scan's work into another's output is the silent-wrong
            # failure this project keeps producing.
            same = (d.get("window_s") == [args.t0, args.t1]
                    and d.get("conf") == args.conf
                    and d.get("imgsz") == args.imgsz
                    and d.get("video") == str(args.video))
            if same and isinstance(d.get("at_frame"), int):
                hits_seed = d.get("hits") or {}
                resume_at = max(start, int(d["at_frame"]))
                n = sum(len(v) for v in hits_seed.values())
                log(f"resuming from {resume_at} ({(resume_at-start)/max(1,end-start)*100:.0f}%), "
                    f"{n} detections already found")
            elif not same:
                log("checkpoint is for a different window or settings, ignoring it")
        except Exception as e:
            log(f"checkpoint unreadable, starting over ({e})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, resume_at)
    log(f"rimwatch {start/fps:.0f}s-{end/fps:.0f}s of {total/fps:.0f}s "
        f"({end-start} frames), conf {args.conf}, imgsz {args.imgsz}")

    # Both hoops in ONE inference, stacked into a single canvas.
    #
    # The obvious version runs the model once per hoop, which is two calls a
    # frame, and on this laptop each call is ~170ms -- six hours for the shootout
    # alone, before any of the steps that actually answer anything. Stacking the
    # two tight crops vertically costs one call instead of two AND lands at a
    # better scale than the wide crops did (0.74 against 0.60), because imgsz
    # fits the LONG side and two tall-ish crops stacked are still shorter than
    # one wide crop is wide. Half the time and a bigger ball, from the same model.
    det = rig.det_boxes()
    order = sorted(det)
    cw = max(det[k][2] - det[k][0] for k in order)
    offs, y = {}, 0
    for k in order:
        offs[k] = y
        y += det[k][3] - det[k][1]
    canvas = np.zeros((y, cw, 3), dtype=np.uint8)
    log(f"stacked canvas {cw}x{y} from {len(order)} hoops: "
        + ", ".join(f"{k} {det[k][2]-det[k][0]}x{det[k][3]-det[k][1]}" for k in order))

    hits = {k: list(hits_seed.get(k, [])) for k in det}
    f = resume_at
    bad = 0
    # Count the frames a CHECKPOINT already covered, or a resumed scan reports
    # its own slice as the whole job: the 2770 scan resumed at 73% and finished
    # with "scanned 5878 of 22078 frames (27%) <-- INCOMPLETE" on a run whose
    # data was complete. An honest coverage line is the whole point of this
    # counter, and one that under-reports after a resume would get a good scan
    # thrown away and re-run.
    covered = max(0, resume_at - start) // max(1, args.step)
    t0 = time.time()
    while f < end:
        # grab() advances the decoder without paying for the color conversion
        # and copy; retrieve() only happens on frames actually looked at. The
        # earlier version called read() every iteration while advancing `f` by
        # `step`, so with any stride above 1 it predicted on consecutive frames
        # but stamped them with strided frame numbers -- every timestamp came out
        # scaled by the stride. Harmless at the old default of 1, and it would
        # have quietly wrecked a whole night at 2.
        # A single decode hiccup used to END THE SCAN, and the script then wrote
        # its partial results and reported success. On IMG_2529 that stopped at
        # 8:40 of a 15:08 video: two of the five shots marked by hand were
        # never scanned at all, and I reported them as detector misses and quoted
        # a recall figure computed over 58% of the recording.
        #
        # Tolerate a short run of bad grabs -- long 4K files hiccup -- and count
        # what actually got looked at so a truncated scan cannot pass as a
        # complete one.
        if not cap.grab():
            bad += 1
            if bad > MAX_BAD_GRABS:
                log(f"decoder gave up at frame {f} ({f/fps:.0f}s) after "
                    f"{bad} consecutive failures -- SCAN IS INCOMPLETE")
                break
            f += 1
            continue
        bad = 0
        if (f - start) % args.step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                bad += 1
                f += 1
                continue
            covered += 1
            canvas[:] = 0
            for name in order:
                x1, y1, x2, y2 = det[name]
                canvas[offs[name]:offs[name] + (y2 - y1), 0:x2 - x1] = fr[y1:y2, x1:x2]
            r = model.predict(canvas, conf=args.conf, verbose=False,
                              classes=[SPORTS_BALL], imgsz=args.imgsz)[0]
            # Second look with the fine-tune where stock found nothing.
            #
            # Four of the twelve shots marked as missed had NO ball sighting
            # at either rim, and three more had one or two -- no rule change can
            # reach those, the ball simply was not seen. The fine-tune recovers 60
            # of 60 frames the stock detector misses, and this canvas is the
            # native-resolution rim crop it was trained on, so it is on home
            # ground here rather than fighting a downscaled full frame.
            #
            # It has one class, so the COCO filter above would drop everything;
            # and its scores never exceed 0.25, so it needs its own low threshold.
            if fine is not None and (r.boxes is None or not len(r.boxes)):
                rf = fine.predict(canvas, conf=FINE_CONF, verbose=False,
                                  imgsz=args.imgsz)[0]
                if rf.boxes is not None and len(rf.boxes):
                    r = rf
            if r.boxes is not None and len(r.boxes):
                for bx in r.boxes:
                    bx1, by1, bx2, by2 = bx.xyxy[0].tolist()
                    cy = (by1 + by2) / 2
                    # Which hoop's band did this land in, and where in the frame
                    # was that band taken from.
                    name = order[0]
                    for k in order:
                        if cy >= offs[k]:
                            name = k
                    x1, y1, x2, y2 = det[name]
                    hits[name].append({
                        "frame": f,
                        "t": round(f / fps, 2),
                        "x": round(x1 + (bx1 + bx2) / 2, 1),
                        "y": round(y1 + cy - offs[name], 1),
                        "conf": round(float(bx.conf.item()), 3),
                        "w": round(bx2 - bx1, 1),
                    })
        f += 1
        if (f - start) % args.ckpt_every == 0:
            done, tot = f - start, end - start
            n = sum(len(v) for v in hits.values())
            log(f"  {done}/{tot} frames, {time.time()-t0:.0f}s, {n} rim-region detections")
            # Checkpoint on the same cadence: one small write per 1800 frames,
            # which is about a minute of footage and roughly an hour of work on
            # this laptop. Written to a temp file and replaced, so a crash
            # DURING the write cannot leave a half-written checkpoint that the
            # next run would trust.
            try:
                tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps({
                    "video": str(args.video),
                    "window_s": [args.t0, args.t1],
                    "conf": args.conf,
                    "imgsz": args.imgsz,
                    "at_frame": f,
                    "hits": hits,
                }), encoding="utf-8")
                tmp.replace(ckpt)
            except Exception as e:
                log(f"  (could not write checkpoint: {e})")
    cap.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "window_s": [args.t0, args.t1],
        "conf": args.conf,
        "boxes": rig.crops,
        "det_boxes": det,
        "hits": hits,
    }, indent=1), encoding="utf-8")
    for k, v in hits.items():
        log(f"{k}: {len(v)} detections near the rim")
    # Say what fraction of the video was actually looked at. A scan that stops
    # early is otherwise indistinguishable from one that found nothing there.
    want = max(1, (end - start) // max(1, args.step))
    pct = 100.0 * covered / want
    log(f"scanned {covered} of {want} frames ({pct:.0f}%)"
        + ("" if pct > 99 else "   <-- INCOMPLETE, results cover only part of the video"))
    log(f"wrote {args.out}")
    # The scan finished, so the checkpoint is now a trap: leaving it would make a
    # future run resume a completed scan and report partial coverage.
    try:
        if ckpt.exists():
            ckpt.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()
