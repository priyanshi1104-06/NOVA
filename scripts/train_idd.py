"""
Fine-tune YOLOv8n on IDD - LOCALLY, on the 4050. No Colab, no upload.

    python scripts\\train_idd.py                 # ~2 h, writes best.pt
    python scripts\\train_idd.py --resume        # after a crash or Ctrl+C
    python scripts\\train_idd.py --epochs 10     # a quick proof it works

WHY LOCAL RATHER THAN COLAB
---------------------------
The dataset is 735 MB / 12,000 files and it is ALREADY on this disk. Getting
it to Colab means uploading it, and that failed twice: a folder upload to
Drive left 6,793 of 10,000 images, with a different number of labels, which
then failed the notebook's own consistency check. A 4050 is in the same class
as Colab's T4 for a nano model, and this way there is no upload, no Drive
mount and no session timeout.

The Colab notebook still works and is still the fallback if this machine is
needed for CARLA - but upload the ZIP there, not the folder. One 735 MB file
transfers far more reliably than 12,000 small ones.

THIS TAKES THE GPU. CARLA cannot run at the same time; 6 GB does not hold
both. Ctrl+C is safe - ultralytics writes last.pt every epoch, and --resume
picks up from it.

WHAT IT PRODUCES
----------------
runs/idd/weights/best.pt, copied to C:\\NOVA\\best.pt on success. That is the
file nova_video.py and nova_drive.py take via --weights.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "idd_yolo"),
                    help="the folder idd_to_yolo.py produced")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16,
                    help="16 fits 6 GB at 640 with AMP. Raise it only if "
                         "nvidia-smi shows headroom - an OOM two hours in "
                         "costs more than the speed is worth.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Windows spawns processes rather than forking, so "
                         "more workers is not automatically faster")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="idd")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data)
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"no data.yaml at {yaml_path}\n"
                         f"Run scripts/idd_to_yolo.py first.")

    n_train = len(list((data_dir / "images" / "train").glob("*.jpg")))
    n_val = len(list((data_dir / "images" / "val").glob("*.jpg")))
    l_train = len(list((data_dir / "labels" / "train").glob("*.txt")))
    l_val = len(list((data_dir / "labels" / "val").glob("*.txt")))
    print(f"train {n_train} images / {l_train} labels")
    print(f"val   {n_val} images / {l_val} labels")
    # The Drive copy silently truncated. Catch that here, in two seconds,
    # rather than after ultralytics has scanned the whole set.
    if n_train != l_train or n_val != l_val:
        raise SystemExit("image/label counts differ - the dataset is "
                         "incomplete. Re-run scripts/idd_to_yolo.py.")
    if not n_train or not n_val:
        raise SystemExit("dataset is empty")

    # ultralytics resolves a relative `path:` against its own config dir, not
    # against the yaml. Passing the absolute yaml AND rewriting `path` to an
    # absolute one removes both ways that can go wrong.
    text = yaml_path.read_text(encoding="utf-8")
    if text.lstrip().startswith("path: ."):
        text = text.replace("path: .", f"path: {data_dir.as_posix()}", 1)
        yaml_path.write_text(text, encoding="utf-8")
        print(f"data.yaml path -> {data_dir}")

    from ultralytics import YOLO

    if args.resume:
        last = ROOT / "runs" / args.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} does not exist")
        print(f"resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=10,          # stop early if val mAP plateaus
            project=str(ROOT / "runs"),
            name=args.name,
            exist_ok=True,
            # Indian street scenes are dense and heavily occluded. Mosaic is
            # what teaches the model the small, overlapping two-wheelers that
            # dominate this dataset; turning it off for the last 10 epochs
            # lets the model settle on realistic, un-stitched images.
            mosaic=1.0,
            close_mosaic=10,
            # cache=False on purpose: 10k images at 640 is roughly 12 GB
            # decoded, which does not fit in 8 GB of system RAM.
            cache=False,
        )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, ROOT / "best.pt")
        print(f"\nbest.pt -> {ROOT / 'best.pt'}")
        print("\nUse it:")
        print(r"  python scripts\nova_video.py --video road.mp4 "
              r"--weights best.pt --fov 70 --loop")
    else:
        print(f"\ntraining ended without a best.pt at {best}", file=sys.stderr)


if __name__ == "__main__":
    main()
