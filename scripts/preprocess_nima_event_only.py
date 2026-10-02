from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import accumulate_to_rgb


def convert_one(
    path_str: str,
    height: int,
    width: int,
    percentile: float,
    overwrite: bool,
):
    path = Path(path_str)
    output = path.with_suffix(".png")

    if output.exists() and not overwrite:
        return "skip", str(path), None

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as z:
            if "event_data" not in z.files:
                return (
                    "error",
                    str(path),
                    "missing event_data",
                )

            events = z["event_data"]

        required = {"x", "y", "t", "p"}

        names = set(
            events.dtype.names or ()
        )

        if not required.issubset(names):
            return (
                "error",
                str(path),
                f"unexpected fields: {events.dtype.names}",
            )

        if len(events) == 0:
            return (
                "error",
                str(path),
                "empty event array",
            )

        x = events["x"]
        y = events["y"]
        p = events["p"]

        image = accumulate_to_rgb(
            x,
            y,
            p,
            (height, width),
            pct=percentile,
        )

        Image.fromarray(
            image,
            mode="RGB",
        ).save(
            output,
            format="PNG",
            optimize=False,
        )

        return "ok", str(path), None

    except Exception as exc:
        return (
            "error",
            str(path),
            f"{type(exc).__name__}: {exc}",
        )


def collect_files(
    root: Path,
    split: str,
):
    if split == "train":
        roots = [
            root / "extracted_train",
        ]
    elif split == "val":
        roots = [
            root / "extracted_val",
        ]
    else:
        roots = [
            root / "extracted_train",
            root / "extracted_val",
        ]

    files = []

    for split_root in roots:
        if not split_root.is_dir():
            raise FileNotFoundError(
                split_root
            )

        files.extend(
            sorted(
                split_root.rglob("*.npz")
            )
        )

    return files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(args.root)

    files = collect_files(
        root,
        args.split,
    )

    if args.max_files is not None:
        files = files[:args.max_files]

    print("root       :", root)
    print("split      :", args.split)
    print("files      :", len(files))
    print(
        "geometry   :",
        f"{args.height}x{args.width}",
    )
    print(
        "percentile :",
        args.percentile,
    )
    print("workers    :", args.workers)

    ok = 0
    skip = 0
    errors = []

    if args.workers <= 1:
        iterator = (
            convert_one(
                str(p),
                args.height,
                args.width,
                args.percentile,
                args.overwrite,
            )
            for p in files
        )

        for status, path, message in tqdm(
            iterator,
            total=len(files),
        ):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                errors.append(
                    (path, message)
                )

    else:
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:

            results = executor.map(
                convert_one,
                map(str, files),
                [args.height] * len(files),
                [args.width] * len(files),
                [args.percentile] * len(files),
                [args.overwrite] * len(files),
                chunksize=8,
            )

            for status, path, message in tqdm(
                results,
                total=len(files),
            ):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    errors.append(
                        (path, message)
                    )

    print()
    print("=" * 72)
    print("N-IMAGENET EVENT PREPROCESS SUMMARY")
    print("=" * 72)
    print("total :", len(files))
    print("ok    :", ok)
    print("skip  :", skip)
    print("error :", len(errors))

    if errors:
        print()
        print("First errors:")

        for path, message in errors[:20]:
            print(path)
            print("  ", message)

        raise SystemExit(1)

    print()
    print("Preprocessing: PASS")


if __name__ == "__main__":
    main()
