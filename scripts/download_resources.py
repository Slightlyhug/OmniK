#!/usr/bin/env python3
"""Download version-pinned Zenodo resources and verify size and MD5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MANIFEST = Path(__file__).resolve().parents[1] / "docs/zenodo-manifest.json"
MODEL = "OmniK_dense0n_checkpoint_last.pt"


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "md5:" + digest.hexdigest() == item["checksum"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=["benchmark", "model", "database", "predictions", "all"],
        default="benchmark",
    )
    parser.add_argument("--files", nargs="+", help="Exact filenames; overrides --group")
    parser.add_argument("--output-dir", type=Path, default=Path("data/zenodo"))
    parser.add_argument(
        "--list", action="store_true", help="List selected files without downloading"
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    items = manifest["files"]
    if args.files:
        unknown = set(args.files) - {f["name"] for f in items}
        if unknown:
            parser.error(f"Unknown files: {sorted(unknown)}")
        selected = [f for f in items if f["name"] in args.files]
    else:

        def belongs(name):
            if args.group == "all":
                return True
            if args.group == "benchmark":
                return name in {"train.csv", "val.csv", "test.csv"}
            if args.group == "model":
                return name == MODEL
            if args.group == "predictions":
                return name.startswith("OmniK_human_proteome_")
            return (
                name.endswith(".tsv")
                and not name.startswith("OmniK_")
                or name
                in {
                    "LysPTM-StateDB-Single.csv",
                    "PPI_human_all_high_confidence_interactions_score_gt_0.5.csv",
                    "README.txt",
                }
            )

        selected = [f for f in items if belongs(f["name"])]
    print(
        f"DOI: {manifest['doi']} | {len(selected)} files | {sum(f['size'] for f in selected) / 1e9:.3f} GB"
    )
    for item in selected:
        print(f"{item['name']} ({item['size']:,} bytes)", flush=True)
        if args.list:
            continue
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target = args.output_dir / item["name"]
        if verified(target, item):
            print("  already verified", flush=True)
            continue
        if target.exists():
            raise RuntimeError(
                f"Existing file failed verification: {target}; move it aside before retrying"
            )
        temporary = target.with_name(target.name + ".part")
        with urlopen(item["url"], timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                handle.write(block)
        if not verified(temporary, item):
            raise RuntimeError(f"Download failed size/MD5 check: {temporary}")
        temporary.replace(target)
        print("  verified", flush=True)


if __name__ == "__main__":
    main()
