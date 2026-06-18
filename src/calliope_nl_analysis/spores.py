"""Inventory and validation helpers for SPORES NetCDF result files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
from typing import Iterable

SPORE_NAME_RE = re.compile(
    r"^(?:(?P<direction>min|max)(?P<target>[a-z]+)_)?"
    r"spore(?:_(?P<number>\d+)|(?P<baseline>_baseline))?\.nc$"
)

SPORE_TARGETS = ("bat", "bio", "htp", "nuc", "off", "ons", "pv")
EXPECTED_SPORE_COUNTS = {
    "baseline": 1,
    "base": 10,
    **{f"{direction}{target}": 10 for direction in ("max", "min") for target in SPORE_TARGETS},
}

FAMILY_ORDER = {family: index for index, family in enumerate(EXPECTED_SPORE_COUNTS)}


@dataclass(frozen=True)
class SporeRecord:
    """Parsed metadata for one SPORES result file."""

    path: Path
    family: str
    direction: str | None
    target: str | None
    number: int | None
    is_baseline: bool
    size_bytes: int
    sha256: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    def as_dict(self, root: Path | None = None) -> dict[str, object]:
        path = self.path.resolve()
        display_path = str(path)
        relative_path = None
        if root is not None:
            relative_path = str(path.relative_to(root.resolve()))
            display_path = relative_path

        row = {
            "name": self.name,
            "stem": self.stem,
            "family": self.family,
            "direction": self.direction,
            "target": self.target,
            "spore_number": self.number,
            "is_baseline": self.is_baseline,
            "size_bytes": self.size_bytes,
            "size_mib": round(self.size_bytes / 1024 / 1024, 3),
            "path": display_path,
        }
        if relative_path is not None:
            row["relative_path"] = relative_path
        if self.sha256 is not None:
            row["sha256"] = self.sha256
        return row


def parse_spore_name(name: str) -> dict[str, object]:
    """Parse a SPORES filename into family metadata."""

    match = SPORE_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"Not a recognised SPORES filename: {name}")

    direction = match.group("direction")
    target = match.group("target")
    number = match.group("number")
    is_baseline = match.group("baseline") is not None

    if is_baseline:
        family = "baseline"
    elif direction is None:
        family = "base"
    else:
        family = f"{direction}{target}"

    return {
        "family": family,
        "direction": direction,
        "target": target,
        "number": int(number) if number is not None else None,
        "is_baseline": is_baseline,
    }


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_spore_records(spores_dir: str | Path, include_sha256: bool = False) -> list[SporeRecord]:
    """List recognised `.nc` SPORES files with parsed metadata."""

    directory = Path(spores_dir)
    records: list[SporeRecord] = []
    for path in directory.glob("*.nc"):
        parsed = parse_spore_name(path.name)
        records.append(
            SporeRecord(
                path=path,
                family=str(parsed["family"]),
                direction=parsed["direction"],  # type: ignore[arg-type]
                target=parsed["target"],  # type: ignore[arg-type]
                number=parsed["number"],  # type: ignore[arg-type]
                is_baseline=bool(parsed["is_baseline"]),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path) if include_sha256 else None,
            )
        )
    return sorted(records, key=_record_sort_key)


def build_spores_manifest(
    spores_dir: str | Path,
    include_sha256: bool = False,
    root: str | Path | None = None,
):
    """Build a pandas DataFrame manifest for the SPORES files."""

    import pandas as pd

    root_path = Path(root) if root is not None else Path(spores_dir).resolve().parents[1]
    rows = [record.as_dict(root=root_path) for record in list_spore_records(spores_dir, include_sha256)]
    manifest = pd.DataFrame(rows)
    if "spore_number" in manifest:
        manifest["spore_number"] = manifest["spore_number"].astype("Int64")
    return manifest


def validate_spore_inventory(
    spores_dir: str | Path,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Validate filename patterns, expected family counts, and empty files."""

    directory = Path(spores_dir)
    expected = expected_counts or EXPECTED_SPORE_COUNTS
    invalid_names = []
    records = []

    for path in directory.glob("*.nc"):
        try:
            parsed = parse_spore_name(path.name)
        except ValueError:
            invalid_names.append(path.name)
            continue
        records.append((path, parsed))

    counts: dict[str, int] = {}
    numbers_by_family: dict[str, set[int]] = {}
    zero_byte_files = []

    for path, parsed in records:
        family = str(parsed["family"])
        counts[family] = counts.get(family, 0) + 1
        if parsed["number"] is not None:
            numbers_by_family.setdefault(family, set()).add(int(parsed["number"]))
        if path.stat().st_size == 0:
            zero_byte_files.append(path.name)

    missing_or_incomplete = {}
    for family, expected_count in expected.items():
        actual_count = counts.get(family, 0)
        missing_numbers = []
        if family != "baseline":
            missing_numbers = sorted(set(range(1, expected_count + 1)) - numbers_by_family.get(family, set()))
        if actual_count != expected_count or missing_numbers:
            missing_or_incomplete[family] = {
                "expected_count": expected_count,
                "actual_count": actual_count,
                "missing_numbers": missing_numbers,
            }

    return {
        "directory": str(directory),
        "total_files": len(records),
        "total_size_bytes": sum(path.stat().st_size for path, _ in records),
        "counts": dict(sorted(counts.items(), key=lambda item: _family_sort_key(item[0]))),
        "missing_or_incomplete": missing_or_incomplete,
        "unexpected_families": sorted(set(counts) - set(expected)),
        "invalid_names": sorted(invalid_names),
        "zero_byte_files": sorted(zero_byte_files),
    }


def spore_paths_by_family(records: Iterable[SporeRecord]) -> dict[str, list[Path]]:
    """Group SPORES paths by parsed family."""

    grouped: dict[str, list[Path]] = {}
    for record in records:
        grouped.setdefault(record.family, []).append(record.path)
    return {family: sorted(paths) for family, paths in grouped.items()}


def _family_sort_key(family: str) -> tuple[int, str]:
    return (FAMILY_ORDER.get(family, len(FAMILY_ORDER)), family)


def _record_sort_key(record: SporeRecord) -> tuple[int, str, int]:
    return (*_family_sort_key(record.family), record.number or 0)
