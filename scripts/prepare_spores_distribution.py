"""Prepare SPORES manifests and optional ZIP archives for online distribution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calliope_nl_analysis.distribution import (  # noqa: E402
    distribution_summary,
    make_family_archives,
    make_single_archive,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spores-dir", type=Path, default=ROOT / "results" / "spores")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist" / "spores")
    parser.add_argument("--hash", action="store_true", help="Include SHA-256 hashes in the manifest.")
    parser.add_argument(
        "--archives",
        choices=("none", "family", "all", "both"),
        default="none",
        help="Create no archives, one archive per family, one archive with all files, or both.",
    )
    parser.add_argument(
        "--compression",
        choices=("stored", "deflated"),
        default="stored",
        help="Use stored ZIP entries for speed, or deflated entries for smaller archives.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_STORED if args.compression == "stored" else zipfile.ZIP_DEFLATED

    manifest_path = write_manifest(
        args.spores_dir,
        args.output_dir / "spores_manifest.csv",
        include_sha256=args.hash,
        root=ROOT,
    )
    print(f"Wrote manifest: {manifest_path}")

    summary = distribution_summary(args.spores_dir)
    print(f"Recognised SPORES files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_bytes'] / 1024**3:.2f} GiB")
    if summary["missing_or_incomplete"]:
        print(f"Missing or incomplete families: {summary['missing_or_incomplete']}")
    if summary["invalid_names"]:
        print(f"Invalid names: {summary['invalid_names']}")
    if summary["zero_byte_files"]:
        print(f"Zero-byte files: {summary['zero_byte_files']}")

    if args.archives in ("family", "both"):
        archives = make_family_archives(args.spores_dir, args.output_dir, compression=compression)
        for archive in archives:
            print(f"Wrote archive: {archive}")

    if args.archives in ("all", "both"):
        archive = make_single_archive(args.spores_dir, args.output_dir / "spores_all.zip", compression=compression)
        print(f"Wrote archive: {archive}")


if __name__ == "__main__":
    main()
