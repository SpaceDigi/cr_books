#!/usr/bin/env python3
"""Convert all PDF files in a directory tree to a single concatenated text file.

The script walks through the provided directory (recursively by default),
extracts text from each ``.pdf`` file, and appends the contents to one output
``.txt`` file. Each document is separated with a header that contains the PDF
file name so that the resulting text file remains readable.

Example usage::

    python scripts/merge_pdfs_to_txt.py --source ./ --output merged_output.txt

Requirements:
    * pdfminer.six (``pip install pdfminer.six``)

"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger(Path(__file__).name)


def iter_pdf_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    """Yield PDF files within ``root``.

    Args:
        root: The starting directory.
        recursive: Whether to walk sub-directories as well.
    """

    if recursive:
        yield from (p for p in root.rglob("*.pdf") if p.is_file())
    else:
        yield from (p for p in root.glob("*.pdf") if p.is_file())


def natural_sort_key(text: str) -> List[object]:
    """Return a key for ordering strings in a human-friendly way."""

    key: List[object] = []
    for piece in re.split(r"(\d+)", text):
        if piece.isdigit():
            key.append(int(piece))
        elif piece:
            key.append(piece.lower())
    return key


def _structured_sort_key(text: str) -> Tuple[Tuple[int, object], ...]:
    """Convert ``text`` into a tuple that compares reliably across strings."""

    structured: List[Tuple[int, object]] = []
    for token in natural_sort_key(text):
        if isinstance(token, int):
            structured.append((0, token))
        else:
            structured.append((1, token))
    return tuple(structured)


def _strip_sw_band_prefix(name: str) -> str:
    return re.sub(r"^\[sw\.band\]\s*", "", name, flags=re.IGNORECASE)


def _canonical_title(name: str) -> str:
    """Create a normalized title suitable for de-duplication."""

    name = _strip_sw_band_prefix(name)
    name = re.sub(r"\s*-\s*8\.0(?:-combined(?:_compressed)?)?", " - 8.0", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*\((\d+)\)(?=\.[^.]+$)", "", name)
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _dedupe_score(path: Path) -> Tuple[int, int, int, int]:
    name = path.name.lower()
    return (
        1 if name.startswith("[sw.band]") else 0,
        1 if "combined" in name else 0,
        1 if "compressed" in name else 0,
        len(name),
    )


def dedupe_pdf_variants(paths: Iterable[Path]) -> List[Path]:
    """Prefer the cleanest filename when multiple variants represent the same title."""

    winners: Dict[str, Path] = {}
    for pdf in paths:
        key = _canonical_title(pdf.name.lower())
        current = winners.get(key)
        new_score = _dedupe_score(pdf)
        if current is None:
            winners[key] = pdf
            continue

        if new_score < _dedupe_score(current):
            LOGGER.info("Preferring %s over %s for title '%s'", pdf, current, key)
            winners[key] = pdf
        else:
            LOGGER.info("Skipping duplicate variant %s (keeping %s)", pdf, current)
    return list(winners.values())


def is_course_material(name: str) -> bool:
    return bool(re.match(r"^\s*\d+", name))


@dataclass(order=True)
class LearningOrderKey:
    category: int
    primary: int
    natural: Tuple[Tuple[int, object], ...]


def learning_order_key(path: Path) -> LearningOrderKey:
    name = _canonical_title(path.name)
    stripped = name.strip()
    match = re.match(r"^(\d+)", stripped)
    if match:
        number = int(match.group(1))
        return LearningOrderKey(0, number, _structured_sort_key(stripped))
    return LearningOrderKey(1, 0, _structured_sort_key(stripped))


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


def load_order_listing(listing_path: Path) -> List[str]:
    """Parse an order listing file into a sequence of file names."""

    entries: List[str] = []
    with listing_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^\s*\d+\.\s+(.+?)\s*$", line)
            if match:
                entries.append(match.group(1).strip())
    return entries


def plan_trading_learning_order(
    paths: Iterable[Path],
    desired_order: Optional[Sequence[str]] = None,
    include_unlisted: bool = False,
) -> List[Path]:
    unique_paths = dedupe_pdf_variants(paths)

    if not desired_order:
        return sorted(unique_paths, key=learning_order_key)

    by_name: Dict[str, Path] = {normalize_name(path.name): path for path in unique_paths}
    by_canonical: Dict[str, Path] = {
        normalize_name(_canonical_title(path.name)): path for path in unique_paths
    }

    ordered: List[Path] = []
    used: Set[Path] = set()
    missing: List[str] = []

    for name in desired_order:
        normalized = normalize_name(name)
        candidate = by_name.get(normalized)
        if candidate is None:
            candidate = by_canonical.get(normalized)
        if candidate is None:
            canonical_key = normalize_name(_canonical_title(name))
            candidate = by_canonical.get(canonical_key)

        if candidate and candidate not in used:
            ordered.append(candidate)
            used.add(candidate)
        else:
            missing.append(name)

    if missing:
        for name in missing:
            LOGGER.warning("Listed file not found among PDFs: %s", name)

    if include_unlisted:
        leftovers = [path for path in unique_paths if path not in used]
        if leftovers:
            LOGGER.info("Appending %d PDFs not present in the order listing", len(leftovers))
            ordered.extend(sorted(leftovers, key=learning_order_key))

    return ordered


def split_learning_groups(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {"course": [], "supplementary": []}
    for path in paths:
        bucket = "course" if is_course_material(_canonical_title(path.name)) else "supplementary"
        groups[bucket].append(path)
    return groups


def write_order_listing(paths: Sequence[Path], destination: Path) -> None:
    groups = split_learning_groups(paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("Course modules\n")
        for index, path in enumerate(groups["course"], start=1):
            handle.write(f"{index}. {path.name}\n")

        if groups["supplementary"]:
            handle.write("\nSupplementary materials\n")
            for index, path in enumerate(groups["supplementary"], start=1):
                handle.write(f"{index}. {path.name}\n")


def extract_pdf_text(path: Path) -> str:
    """Extract text from a single PDF file."""

    try:
        from pdfminer.high_level import extract_text
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit guidance
        raise SystemExit("pdfminer.six is required to extract PDF text.") from exc

    try:
        return extract_text(str(path))
    except Exception:  # pragma: no cover - provide context when extraction fails
        LOGGER.exception("Failed to extract text from %s", path)
        return ""


def merge_pdfs_to_text(pdf_files: Sequence[Path], output_file: Path) -> None:
    """Convert each PDF in ``pdf_files`` into a single text file."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Processing order:")
    for index, pdf_path in enumerate(pdf_files, start=1):
        LOGGER.info("  %3d. %s", index, pdf_path)

    with output_file.open("w", encoding="utf-8") as handle:
        for index, pdf_path in enumerate(pdf_files, start=1):
            LOGGER.info("Processing %s", pdf_path)
            text = extract_pdf_text(pdf_path).strip()

            header = f"{'=' * 80}\nDocument {index}: {pdf_path} \n{'=' * 80}\n"
            handle.write(header)
            if text:
                handle.write(text)
            else:
                handle.write("[No text extracted]\n")
            handle.write("\n\n")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.cwd(),
        help="Directory that contains PDF files (default: current working directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination .txt file that will store the merged content",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search for PDF files",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--order-output",
        type=Path,
        help="Optional path to save the planned processing order",
    )
    parser.add_argument(
        "--order-source",
        type=Path,
        help="Path to a text file that lists PDFs in the desired processing order",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only write the ordered list of files without extracting PDF text",
    )
    parser.add_argument(
        "--include-unlisted",
        action="store_true",
        help=(
            "Process PDFs not present in the order listing after the listed files using"
            " the default learning order"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    recursive = not args.no_recursive
    source = args.source.resolve()
    pdf_candidates = list(iter_pdf_files(source, recursive))

    desired_listing: Optional[Sequence[str]] = None
    order_source: Optional[Path] = args.order_source
    if order_source is None:
        default_order = Path(__file__).resolve().parent.parent / "trading_learning_order.txt"
        if default_order.exists():
            order_source = default_order

    if order_source is not None:
        if order_source.exists():
            desired_listing = load_order_listing(order_source)
            LOGGER.info(
                "Loaded %d entries from order listing %s", len(desired_listing), order_source
            )
        else:
            LOGGER.warning("Order listing %s not found", order_source)

    ordered_files = plan_trading_learning_order(
        pdf_candidates, desired_listing, include_unlisted=args.include_unlisted
    )

    if not ordered_files:
        if pdf_candidates:
            LOGGER.warning(
                "No PDFs matched the provided order listing; nothing to process."
            )
        else:
            LOGGER.warning("No PDF files found in %s", source)
        return

    if args.order_output:
        write_order_listing(ordered_files, args.order_output.resolve())
        LOGGER.info("Saved planned order to %s", args.order_output.resolve())

    if args.list_only:
        return

    if args.output is None:
        raise SystemExit("--output is required unless --list-only is used")

    merge_pdfs_to_text(ordered_files, args.output.resolve())


if __name__ == "__main__":
    main()
