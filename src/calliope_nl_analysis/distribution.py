"""Helpers for preparing SPORES files for external distribution."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import zipfile

from .spores import build_spores_manifest, list_spore_records, validate_spore_inventory


def write_manifest(
    spores_dir: str | Path,
    output_path: str | Path,
    include_sha256: bool = False,
    root: str | Path | None = None,
) -> Path:
    """Write a CSV manifest for local or hosted SPORES files."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_spores_manifest(spores_dir, include_sha256=include_sha256, root=root)
    manifest.to_csv(output, index=False)
    return output


def make_family_archives(
    spores_dir: str | Path,
    output_dir: str | Path,
    families: list[str] | None = None,
    compression: int = zipfile.ZIP_STORED,
) -> list[Path]:
    """Create one ZIP archive per SPORES family."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_by_family = defaultdict(list)
    for record in list_spore_records(spores_dir):
        records_by_family[record.family].append(record)

    selected_families = families or sorted(records_by_family)
    archives = []
    for family in selected_families:
        records = records_by_family.get(family, [])
        if not records:
            continue
        archive_path = output / f"spores_{family}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=compression, allowZip64=True) as archive:
            for record in records:
                archive.write(record.path, arcname=f"results/spores/{record.path.name}")
        archives.append(archive_path)
    return archives


def make_single_archive(
    spores_dir: str | Path,
    output_path: str | Path,
    compression: int = zipfile.ZIP_STORED,
) -> Path:
    """Create one ZIP archive containing all SPORES files."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=compression, allowZip64=True) as archive:
        for record in list_spore_records(spores_dir):
            archive.write(record.path, arcname=f"results/spores/{record.path.name}")
    return output


def distribution_summary(spores_dir: str | Path) -> dict[str, object]:
    """Return validation plus per-family archive size estimates."""

    validation = validate_spore_inventory(spores_dir)
    records_by_family = defaultdict(list)
    for record in list_spore_records(spores_dir):
        records_by_family[record.family].append(record)

    family_sizes = {
        family: {
            "files": len(records),
            "size_bytes": sum(record.size_bytes for record in records),
            "size_gib": round(sum(record.size_bytes for record in records) / 1024**3, 3),
        }
        for family, records in sorted(records_by_family.items())
    }
    return {**validation, "family_sizes": family_sizes}
