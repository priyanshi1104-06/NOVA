"""
IDD Detection (Pascal VOC) -> YOLO dataset, subsetted and resized.

RUN THIS ON YOUR LAPTOP, where the 22.8 GB IDD download lives. It produces a
~1 GB folder you can upload to Google Drive and train from in Colab. Uploading
the raw 22.8 GB would take most of a day and will not fit in a free 15 GB Drive.

    python scripts/idd_to_yolo.py --src "D:/IDD_Detection" --dst "D:/idd_yolo"

WHY THE CLASS NAMES MATTER
--------------------------
nova/perception.py does:

    label = names[int(ci)].lower()
    cls   = NAME_TO_CLASS.get(label)

so the trained model's class NAMES are the contract between YOLO and NOVA.
The names below are exactly the keys already present in NAME_TO_CLASS, which
is why swapping yolov8n.pt for best.pt needs no code change anywhere else.
Rename a class here and NOVA silently stops seeing that class - the detection
still happens, NAME_TO_CLASS.get() just returns None and the box is dropped.

STRUCTURE IS DISCOVERED, NOT ASSUMED
------------------------------------
IDD ships as Annotations/<camera>/<sequence>/<frame>.xml with a matching
JPEGImages/<camera>/<sequence>/<frame>.jpg, but the exact nesting has varied
between releases. This walks for *.xml and pairs each one with an image of the
same relative path, trying several extensions - so it keeps working if the
layout differs from what you expected.
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

# IDD label -> (class id, canonical name written into data.yaml).
# Only classes NOVA reasons about. Traffic lights and signs are deliberately
# excluded: the planner does not consume them, and every extra class costs
# training capacity a nano model does not have to spare.
CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "autorickshaw",
    "person",
    "rider",
    "animal",
    "vehicle fallback",
]
NAME_TO_ID = {n: i for i, n in enumerate(CLASSES)}
# IDD spelling variants seen in the wild, mapped onto the canonical names.
ALIASES = {
    "motorcyle": "motorcycle",       # IDD's own misspelling, present in the XML
    "auto-rickshaw": "autorickshaw",
    "autorickshaw": "autorickshaw",
    "vehicle_fallback": "vehicle fallback",
    "vehiclefallback": "vehicle fallback",
    "caravan": "truck",
    "trailer": "truck",
}

IMG_EXTS = (".jpg", ".jpeg", ".png")

# IDD was captured with seven cameras. NOVA sees ONE, pointing forward, so
# only these three are useful:
#
#     frontFar   4499 frames    frontNear  5060    highquality_16k 14753
#     sideLeft   7560           sideRight  7850    rearNear        2135
#
# A car photographed from a side camera at 90 degrees shares almost no
# appearance with the same car ahead of a dashcam, and a nano model has very
# little capacity to spend on a viewpoint it will never be shown. Left
# unfiltered, over half of a capped 8000-frame sample comes from side and rear
# views. 24312 front frames is already three times the cap.
FRONT_CAMERAS = ("frontFar", "frontNear", "highquality_16k")


def find_image(src: Path, rel_no_ext: Path) -> Path | None:
    for stem_root in ("JPEGImages", "Images", "leftImg8bit"):
        base = src / stem_root / rel_no_ext
        for ext in IMG_EXTS:
            p = base.with_suffix(ext)
            if p.exists():
                return p
    # Fallback: same tree as the annotation, different extension.
    for ext in IMG_EXTS:
        p = (src / rel_no_ext).with_suffix(ext)
        if p.exists():
            return p
    return None


def convert_one(xml_path: Path, img_path: Path, out_img: Path, out_lbl: Path,
                imgsz: int) -> bool:
    """Returns False if the frame has no usable objects - those are skipped.

    A frame with zero labels is legal in YOLO (it is a background image) but a
    dataset made mostly of them teaches the model to predict nothing.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return False

    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h0, w0 = img.shape[:2]
    if w0 == 0 or h0 == 0:
        return False

    lines = []
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip().lower()
        name = ALIASES.get(raw, raw)
        if name not in NAME_TO_ID:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        # Clamp: IDD boxes occasionally run a pixel past the image edge, and
        # ultralytics rejects the whole label file if any value exceeds 1.0.
        x1, x2 = max(0.0, min(x1, w0)), max(0.0, min(x2, w0))
        y1, y2 = max(0.0, min(y1, h0)), max(0.0, min(y2, h0))
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue
        lines.append(f"{NAME_TO_ID[name]} "
                     f"{((x1 + x2) / 2) / w0:.6f} {((y1 + y2) / 2) / h0:.6f} "
                     f"{bw / w0:.6f} {bh / h0:.6f}")

    if not lines:
        return False

    # Resize so the long side is imgsz. YOLO letterboxes to 640 internally, so
    # storing anything larger costs upload time and disk for no accuracy.
    scale = imgsz / max(w0, h0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                         interpolation=cv2.INTER_AREA)

    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="the extracted IDD_Detection folder")
    ap.add_argument("--dst", required=True, help="output YOLO dataset folder")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cameras", default=",".join(FRONT_CAMERAS),
                    help="comma-separated camera folders to include, or "
                         "'all'. Defaults to the forward-facing ones.")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    ann_root = src / "Annotations"
    search_root = ann_root if ann_root.exists() else src
    print(f"scanning {search_root} for annotations...")
    xmls = sorted(search_root.rglob("*.xml"))
    if not xmls:
        raise SystemExit(
            "no .xml annotation files found. Check --src points at the folder "
            "that CONTAINS Annotations/ and JPEGImages/.")
    print(f"found {len(xmls)} annotation files")

    if args.cameras.strip().lower() != "all":
        keep = {c.strip() for c in args.cameras.split(",") if c.strip()}
        before = len(xmls)
        xmls = [x for x in xmls
                if x.relative_to(search_root).parts[0] in keep]
        print(f"cameras {sorted(keep)}: kept {len(xmls)} of {before}")
        if not xmls:
            raise SystemExit(
                f"no annotations under {sorted(keep)}. Folders present: "
                f"{sorted({p.relative_to(search_root).parts[0] for p in search_root.rglob('*.xml')})}")

    rng = random.Random(args.seed)
    rng.shuffle(xmls)

    # PREFER IDD'S OWN SPLIT. train.txt and val.txt list frames as
    # "<camera>/<sequence>/<frame>", and they were checked here to share ZERO
    # sequences - so the official split is already leak-free, and using it
    # makes our val mAP directly comparable to published IDD numbers instead
    # of to a private split nobody else can reproduce.
    split_of: dict[str, str] = {}          # keyed by annotation stem, not seq
    official = {}
    for split, fname in (("train", "train.txt"), ("val", "val.txt")):
        f = src / fname
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    official[line] = split
    use_official = len(official) > 100
    if use_official:
        print(f"using IDD's own split: {len(official)} listed frames")
    else:
        # Fallback for a release without the split files. Split by SEQUENCE,
        # not by frame: IDD frames come from continuous drives, so
        # neighbouring frames are near-identical and a per-frame split leaks
        # the validation set into training, making val mAP meaningless.
        by_seq: dict[str, list[Path]] = {}
        for x in xmls:
            by_seq.setdefault(x.parent.as_posix(), []).append(x)
        seqs = sorted(by_seq)
        rng.shuffle(seqs)
        cut = max(1, int(len(seqs) * 0.9))
        for sq in seqs[:cut]:
            split_of[sq] = "train"
        for sq in seqs[cut:]:
            split_of[sq] = "val"
        print(f"no split files - {len(seqs)} sequences -> {cut} train / "
              f"{len(seqs) - cut} val")

    if dst.exists():
        shutil.rmtree(dst)
    counts = {"train": 0, "val": 0}
    caps = {"train": args.max_train, "val": args.max_val}
    skipped = 0

    for i, xml_path in enumerate(xmls):
        rel = xml_path.relative_to(search_root).with_suffix("")
        if use_official:
            split = official.get(rel.as_posix())
            if split is None:
                skipped += 1      # test.txt frames, and unlisted extras
                continue
        else:
            split = split_of[xml_path.parent.as_posix()]
        if counts[split] >= caps[split]:
            continue
        img_path = find_image(src, rel)
        if img_path is None:
            skipped += 1
            continue
        flat = rel.as_posix().replace("/", "__")
        ok = convert_one(
            xml_path, img_path,
            dst / "images" / split / f"{flat}.jpg",
            dst / "labels" / split / f"{flat}.txt",
            args.imgsz)
        if ok:
            counts[split] += 1
        else:
            skipped += 1
        if i % 500 == 0:
            print(f"  {i}/{len(xmls)}  train={counts['train']} "
                  f"val={counts['val']} skipped={skipped}")
        if all(counts[s] >= caps[s] for s in caps):
            break

    yaml = ["path: .", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    (dst / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"\ndone. train={counts['train']}  val={counts['val']}  "
          f"skipped={skipped}")
    print(f"wrote {dst / 'data.yaml'}")
    print("\nNext: zip this folder and upload the zip to Google Drive.")
    print(f'  Compress-Archive -Path "{dst}" -DestinationPath "{dst}.zip"')


if __name__ == "__main__":
    main()
