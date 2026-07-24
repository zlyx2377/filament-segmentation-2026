"""EDA + data-path auto-discovery for the Filament Segmentation 2026 data.

Two concerns this module solves:

1. **We don't know the exact on-Kaggle layout yet** (data is still being
   compressed). So instead of hard-failing on a guessed path, the training /
   inference entrypoints call :func:`resolve_train_paths` /
   :func:`resolve_test_paths`, which fall back to scanning the competition
   mount for the COCO json and the image directory.

2. **Printing the real structure** (:func:`print_structure`) at the very start
   of a Kaggle run means the kernel log always contains ground truth — even if
   a later step fails on a path/format mismatch, we can adapt from the log.
"""
from __future__ import annotations

import os
import json
from typing import Optional


def _load_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa
        return None


IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _is_coco(d) -> bool:
    return isinstance(d, dict) and "images" in d and "annotations" in d


def find_coco_json(mount_root: str) -> Optional[str]:
    """Return the path to the COCO json with the most annotations under mount."""
    best = None
    for dp, _, files in os.walk(mount_root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            d = _load_json(os.path.join(dp, fn))
            if _is_coco(d):
                n = len(d["annotations"])
                if best is None or n > best[1]:
                    best = (os.path.join(dp, fn), n)
    return best[0] if best else None


def _count_images(dp: str) -> int:
    if not os.path.isdir(dp):
        return 0
    return sum(1 for f in os.listdir(dp) if f.lower().endswith(IMG_EXTS))


def find_image_dir(mount_root: str, near: Optional[str] = None) -> Optional[str]:
    """Return the directory holding the most image files (optionally under `near`)."""
    best = None
    for dp, _, _ in os.walk(mount_root):
        if near and not (dp == near or dp.startswith(near + os.sep)):
            continue
        s = _count_images(dp)
        if s > 0 and (best is None or s > best[1]):
            best = (dp, s)
    return best[0] if best else None


def resolve_train_paths(cfg: dict, mount_root: str) -> dict:
    """Return a (possibly overridden) copy of cfg['data'] that points at the
    real annotation file + image directory. Falls back to scanning mount_root
    only when the configured path is missing."""
    data = dict(cfg["data"])
    root = data["root"]
    ann_rel = data["train_ann_file"]
    img_rel = data["train_images_dir"]

    configured_ann = os.path.join(root, ann_rel)
    if os.path.exists(configured_ann):
        return data  # configured paths look correct; trust them

    coco = find_coco_json(mount_root)
    if coco is None:
        print(f"[resolve] WARN: no COCO json found under {mount_root}; "
              f"using configured root={root}")
        return data

    new_root = os.path.dirname(coco)
    data["root"] = new_root
    data["train_ann_file"] = os.path.basename(coco)
    imgdir = find_image_dir(mount_root, near=new_root)
    if imgdir:
        data["train_images_dir"] = os.path.relpath(imgdir, new_root)
    print(f"[resolve] train ann auto-detected: root={new_root} "
          f"ann={os.path.basename(coco)} img_dir={data['train_images_dir']}")
    return data


def resolve_test_paths(cfg: dict, mount_root: str) -> dict:
    """Like :func:`resolve_train_paths` but for the (annotation-less) test set.

    Prefers a directory whose path contains "test" (so we don't accidentally
    point the submission at the training images).
    """
    data = dict(cfg["data"])
    root = data["root"]
    img_rel = data["test_images_dir"]
    configured = os.path.join(root, img_rel)
    if os.path.isdir(configured):
        return data

    # Prefer a directory whose *path segment starts with* "test" (e.g.
    # .../test_images). Using startswith (not substring) avoids false matches
    # like a parent dir ".../edatest/...". Falls back to the largest image dir.
    test_dir = None
    best = -1
    for dp, _, _ in os.walk(mount_root):
        segs = [s.lower() for s in dp.split(os.sep)]
        if any(s.startswith("test") for s in segs):
            c = _count_images(dp)
            if c > best:
                best = c
                test_dir = dp
    if test_dir is None:
        test_dir = find_image_dir(mount_root)
    if test_dir:
        data["root"] = mount_root
        data["test_images_dir"] = os.path.relpath(test_dir, mount_root)
        print(f"[resolve] test images auto-detected: dir={test_dir}")
    return data


# --------------------------------------------------------------------------- #
# EDA printer
# --------------------------------------------------------------------------- #
def _walk_tree(root: str, max_depth: int = 3):
    lines = []
    root = os.path.abspath(root)
    for dp, dirs, files in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        lines.append(f"{indent}{os.path.basename(dp)}/")
        if depth < max_depth:
            for fn in sorted(files)[:8]:
                lines.append(f"{indent}  {fn}")
            if len(files) > 8:
                lines.append(f"{indent}  ... ({len(files)} files)")
    return lines


def print_structure(mount_root: str):
    print("=" * 70)
    print("EDA: competition data structure @", mount_root)
    print("=" * 70)
    if not os.path.isdir(mount_root):
        print("!! mount_root does not exist:", mount_root)
        return

    print("\n-- top-level tree (depth<=3) --")
    print("\n".join(_walk_tree(mount_root, 3)))

    print("\n-- COCO-like JSON files --")
    found = False
    for dp, _, files in os.walk(mount_root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            p = os.path.join(dp, fn)
            d = _load_json(p)
            if not _is_coco(d) and not (
                isinstance(d, dict) and "images" in d
                and any("spine" in k.lower() for k in d.keys())
            ):
                continue
            found = True
            imgs = d.get("images", [])
            anns = d.get("annotations", [])
            cats = d.get("categories", [])
            print(f"\nJSON: {os.path.relpath(p, mount_root)}")
            print(f"  keys: {list(d.keys())}")
            print(f"  #images={len(imgs)}  #annotations={len(anns)}  "
                  f"#categories={len(cats)}")
            if imgs:
                print(f"  sample image: {imgs[0]}")
            if cats:
                print(f"  categories: {cats}")
            spine_keys = [k for k in d.keys() if "spine" in k.lower()]
            print(f"  top-level spine keys: {spine_keys}")
            if anns:
                a0 = anns[0]
                print(f"  sample annotation keys: {list(a0.keys())}")
                print("  sample annotation (trimmed): "
                      f"{ {k: str(v)[:80] for k, v in a0.items()} }")
                has_spine = sum(1 for a in anns[:200] if a.get("spine"))
                print(f"  annotations w/ 'spine' field (of first 200): {has_spine}")
            for sk in spine_keys:
                sp = d.get(sk)
                if isinstance(sp, list):
                    print(f"  {sk}: list len={len(sp)}; sample="
                          f"{sp[0] if sp else None}")
    if not found:
        print("  (no COCO-like JSON found)")
    print("\nEDA done.")
