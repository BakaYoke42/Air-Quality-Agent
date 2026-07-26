from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


# These values follow the production sizes suggested in Lab B1.
BASELINE_WORDS = 500
BASELINE_OVERLAP = 50
PARENT_WORDS = 800
PARENT_OVERLAP = 100
CHILD_WORDS = 200
CHILD_OVERLAP = 30
MIN_DOCUMENT_WORDS = 80

SUPPORTED_SUFFIXES = {".html", ".htm", ".pdf"}


@dataclass(frozen=True)
class SourceSpec:
    match: str
    doc_id: str
    title: str
    publisher: str
    publication_year: int | None
    data_year: int | None
    document_type: str
    evidence_status: str
    pollutants: tuple[str, ...]
    source_url: str


SOURCE_SPECS = (
    SourceSpec(
        match="eea_status_2026",
        doc_id="eea_air_quality_status_2026",
        title="Europe's air quality status 2026",
        publisher="European Environment Agency",
        publication_year=2026,
        data_year=2024,
        document_type="status_report",
        evidence_status="official_assessment",
        pollutants=("PM2.5", "NO2"),
        source_url=(
            "https://www.eea.europa.eu/en/analysis/publications/"
            "air-quality-status-report-2026"
        ),
    ),
    SourceSpec(
        match="eea_pm25_2026",
        doc_id="eea_pm25_status_2026",
        title="Particulate matter (PM2.5): air quality status 2026",
        publisher="European Environment Agency",
        publication_year=2026,
        data_year=2024,
        document_type="pollutant_assessment",
        evidence_status="official_assessment",
        pollutants=("PM2.5",),
        source_url=(
            "https://www.eea.europa.eu/en/analysis/publications/"
            "air-quality-status-report-2026/particulate-matter-pm2.5"
        ),
    ),
    SourceSpec(
        match="eea_no2_2026",
        doc_id="eea_no2_status_2026",
        title="Nitrogen dioxide (NO2): air quality status 2026",
        publisher="European Environment Agency",
        publication_year=2026,
        data_year=2024,
        document_type="pollutant_assessment",
        evidence_status="official_assessment",
        pollutants=("NO2",),
        source_url=(
            "https://www.eea.europa.eu/en/analysis/publications/"
            "air-quality-status-report-2026/nitrogen-dioxide-no2"
        ),
    ),
    SourceSpec(
        match="eea_methodology_2026",
        doc_id="eea_status_methodology_2026",
        title="Methodological approaches: air quality status 2026",
        publisher="European Environment Agency",
        publication_year=2026,
        data_year=2024,
        document_type="methodology",
        evidence_status="official_methodology",
        pollutants=("PM2.5", "NO2"),
        source_url=(
            "https://www.eea.europa.eu/en/analysis/publications/"
            "air-quality-status-report-2026/annex-methodological-approaches"
        ),
    ),
    SourceSpec(
        match="eea_exceedance_methodology",
        doc_id="eea_exceedance_methodology",
        title="Exceedance of air quality standards: methodology",
        publisher="European Environment Agency",
        publication_year=None,
        data_year=None,
        document_type="indicator_methodology",
        evidence_status="official_methodology",
        pollutants=("PM2.5", "NO2"),
        source_url=(
            "https://www.eea.europa.eu/en/analysis/indicators/"
            "exceedance-of-air-quality-standards"
        ),
    ),
    SourceSpec(
        match="etc_he_air_quality_2024_validated",
        doc_id="etc_he_air_quality_2024_validated",
        title="Status report of air quality in Europe for year 2024 using validated data",
        publisher="ETC HE / European Environment Agency",
        publication_year=2026,
        data_year=2024,
        document_type="technical_report",
        evidence_status="validated_data_assessment",
        pollutants=("PM2.5", "NO2"),
        source_url=(
            "https://www.eionet.europa.eu/etcs/etc-he/products/etc-he-products/"
            "etc-he-reports/etc-he-report-2026-1-status-report-of-air-quality-"
            "in-europe-for-year-2024-using-validated-data"
        ),
    ),
    SourceSpec(
        match="eu_directive_2024_2881",
        doc_id="eu_directive_2024_2881",
        title="Directive (EU) 2024/2881 on ambient air quality and cleaner air for Europe",
        publisher="European Union",
        publication_year=2024,
        data_year=None,
        document_type="legislation",
        evidence_status="binding_law",
        pollutants=("PM2.5", "NO2"),
        source_url="https://eur-lex.europa.eu/eli/dir/2024/2881/oj/eng",
    ),
    SourceSpec(
        match="eu_directive_2008_50",
        doc_id="eu_directive_2008_50",
        title="Directive 2008/50/EC on ambient air quality and cleaner air for Europe",
        publisher="European Union",
        publication_year=2008,
        data_year=None,
        document_type="legislation",
        evidence_status="binding_law",
        pollutants=("PM2.5", "NO2"),
        source_url=(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/"
            "?uri=CELEX:32008L0050"
        ),
    ),
    SourceSpec(
        match="who_air_quality_guidelines_summary",
        doc_id="who_air_quality_guidelines_2021_summary",
        title="WHO global air quality guidelines: executive summary",
        publisher="World Health Organization",
        publication_year=2021,
        data_year=None,
        document_type="health_guideline",
        evidence_status="non_binding_health_guideline",
        pollutants=("PM2.5", "NO2"),
        source_url="https://www.who.int/publications/i/item/9789240034433",
    ),
)


class _FallbackHTMLTextExtractor(HTMLParser):
    """Small dependency-free fallback used only if Trafilatura returns no text."""

    BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }
    IGNORED_TAGS = {"script", "style", "svg", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
        elif self._ignored_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif self._ignored_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def project_root() -> Path:
    """Work whether this file is in project_root/ or project_root/scripts/."""
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "data").is_dir():
        return script_dir
    if (script_dir.parent / "data").is_dir():
        return script_dir.parent
    return Path.cwd()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join words broken only because a PDF line ended with a hyphen.
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_html(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        import trafilatura
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'trafilatura'. Run: pip install trafilatura"
        ) from exc

    raw_html = path.read_text(encoding="utf-8", errors="replace")
    text = trafilatura.extract(
        raw_html,
        output_format="markdown",
        include_comments=False,
        include_links=True,
        include_tables=True,
        deduplicate=True,
        favor_precision=True,
    )
    used_fallback = False
    if not text or len(text.split()) < MIN_DOCUMENT_WORDS:
        parser = _FallbackHTMLTextExtractor()
        parser.feed(raw_html)
        fallback = parser.text()
        if len(fallback.split()) > len((text or "").split()):
            text = fallback
            used_fallback = True

    return normalize_text(text or ""), {
        "pages_total": None,
        "pages_indexed": None,
        "html_fallback_used": used_fallback,
    }


def extract_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'pypdf'. Run: pip install pypdf") from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise RuntimeError(f"PDF is encrypted and could not be opened: {path.name}") from exc

    sections: list[str] = []
    empty_pages: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            # Compatibility with older pypdf versions.
            page_text = page.extract_text() or ""
        page_text = normalize_text(page_text)
        if not page_text:
            empty_pages.append(page_number)
            continue
        # The marker survives word chunking and lets answers cite a PDF page.
        sections.append(f"[[PAGE {page_number}]]\n\n{page_text}")

    return normalize_text("\n\n".join(sections)), {
        "pages_total": len(reader.pages),
        "pages_indexed": len(reader.pages) - len(empty_pages),
        "empty_pages": empty_pages,
    }


def find_source_spec(path: Path) -> SourceSpec:
    name = path.name.lower()
    matches = [spec for spec in SOURCE_SPECS if spec.match in name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous source metadata for {path.name}")
    raise ValueError(
        f"No metadata is configured for {path.name}. Add a SourceSpec before indexing it."
    )


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Lab B1 word-window splitter, with production sizes and a safe final chunk."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and 0 <= overlap < size.")
    words = text.split()
    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
    return chunks


def page_range(text: str) -> tuple[int | None, int | None]:
    numbers = [int(n) for n in re.findall(r"\[\[PAGE\s+(\d+)\]\]", text)]
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def metadata_dict(spec: SourceSpec, local_file: str) -> dict[str, Any]:
    return {
        "doc_id": spec.doc_id,
        "title": spec.title,
        "publisher": spec.publisher,
        "publication_year": spec.publication_year,
        "data_year": spec.data_year,
        "document_type": spec.document_type,
        "evidence_status": spec.evidence_status,
        "pollutants": list(spec.pollutants),
        "source_url": spec.source_url,
        "local_file": local_file,
    }


def make_chunk_record(
    *,
    chunk_id: str,
    text: str,
    spec: SourceSpec,
    local_file: str,
    parent_id: str | None = None,
) -> dict[str, Any]:
    page_start, page_end = page_range(text)
    metadata_prefix = " | ".join(
        [
            spec.title,
            spec.publisher,
            spec.document_type,
            ", ".join(spec.pollutants),
        ]
    )
    record = {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        **metadata_dict(spec, local_file),
        "page_start": page_start,
        "page_end": page_end,
        "word_count": len(text.split()),
        "text": text,
        # Index this field. It gives BM25 and dense retrieval useful source context.
        "search_text": f"{metadata_prefix}\n{text}",
    }
    return record


def markdown_document(spec: SourceSpec, local_file: str, text: str) -> str:
    metadata = metadata_dict(spec, local_file)
    front_matter = ["---"]
    for key, value in metadata.items():
        front_matter.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    front_matter.extend(["---", "", f"# {spec.title}", "", text, ""])
    return "\n".join(front_matter)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_corpus(input_dir: Path, output_dir: Path) -> int:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    source_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not source_paths:
        raise FileNotFoundError(f"No PDF or HTML files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    document_dir = output_dir / "documents"
    document_dir.mkdir(parents=True, exist_ok=True)

    baseline_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    child_records: list[dict[str, Any]] = []
    child_to_parent: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    used_doc_ids: set[str] = set()
    found_spec_matches: set[str] = set()
    errors: list[str] = []

    print(f"Source documents found: {len(source_paths)}")
    for path in source_paths:
        print(f"Processing {path.name} ...")
        try:
            spec = find_source_spec(path)
            if spec.doc_id in used_doc_ids:
                raise ValueError(
                    f"More than one input file maps to doc_id '{spec.doc_id}'. "
                    "Keep only one copy."
                )
            used_doc_ids.add(spec.doc_id)
            found_spec_matches.add(spec.match)

            if path.suffix.lower() in {".html", ".htm"}:
                text, extraction = extract_html(path)
            else:
                text, extraction = extract_pdf(path)

            word_count = len(text.split())
            if word_count < MIN_DOCUMENT_WORDS:
                raise ValueError(
                    f"Only {word_count} words were extracted. The download may be an "
                    "error page or an image-only PDF."
                )

            markdown_path = document_dir / f"{spec.doc_id}.md"
            markdown_path.write_text(
                markdown_document(spec, path.name, text), encoding="utf-8"
            )

            document_baseline: list[dict[str, Any]] = []
            for index, chunk in enumerate(
                split_words(text, BASELINE_WORDS, BASELINE_OVERLAP)
            ):
                document_baseline.append(
                    make_chunk_record(
                        chunk_id=f"{spec.doc_id}_b{index:04d}",
                        text=chunk,
                        spec=spec,
                        local_file=path.name,
                    )
                )

            document_parents: list[dict[str, Any]] = []
            document_children: list[dict[str, Any]] = []
            for parent_index, parent_text in enumerate(
                split_words(text, PARENT_WORDS, PARENT_OVERLAP)
            ):
                parent_id = f"{spec.doc_id}_p{parent_index:04d}"
                document_parents.append(
                    make_chunk_record(
                        chunk_id=parent_id,
                        text=parent_text,
                        spec=spec,
                        local_file=path.name,
                    )
                )
                for child_index, child_text in enumerate(
                    split_words(parent_text, CHILD_WORDS, CHILD_OVERLAP)
                ):
                    child_id = f"{parent_id}_c{child_index:03d}"
                    document_children.append(
                        make_chunk_record(
                            chunk_id=child_id,
                            parent_id=parent_id,
                            text=child_text,
                            spec=spec,
                            local_file=path.name,
                        )
                    )
                    child_to_parent[child_id] = parent_id

            baseline_records.extend(document_baseline)
            parent_records.extend(document_parents)
            child_records.extend(document_children)

            manifest_rows.append(
                {
                    **metadata_dict(spec, path.name),
                    "raw_sha256": sha256_file(path),
                    "raw_bytes": path.stat().st_size,
                    "extracted_words": word_count,
                    "extracted_characters": len(text),
                    "pages_total": extraction.get("pages_total"),
                    "pages_indexed": extraction.get("pages_indexed"),
                    "empty_pdf_pages": json.dumps(extraction.get("empty_pages", [])),
                    "html_fallback_used": extraction.get("html_fallback_used", False),
                    "baseline_chunks": len(document_baseline),
                    "parent_chunks": len(document_parents),
                    "child_chunks": len(document_children),
                    "processed_markdown": str(markdown_path.relative_to(output_dir)),
                }
            )
            print(
                f"  {word_count:,} words -> {len(document_parents)} parents, "
                f"{len(document_children)} children"
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            print(f"  ERROR: {exc}")

    expected_matches = {spec.match for spec in SOURCE_SPECS}
    missing_sources = sorted(expected_matches - found_spec_matches)
    if missing_sources:
        errors.append("Missing expected source(s): " + ", ".join(missing_sources))

    if errors:
        print("\nCorpus preparation stopped because the source set is incomplete:")
        for error in errors:
            print(f"  - {error}")
        print("\nFix the listed files and run the same command again.")
        return 1

    write_jsonl(output_dir / "baseline_chunks.jsonl", baseline_records)
    write_jsonl(output_dir / "parents.jsonl", parent_records)
    write_jsonl(output_dir / "children.jsonl", child_records)
    (output_dir / "child_to_parent.json").write_text(
        json.dumps(child_to_parent, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "documents": len(manifest_rows),
        "baseline_chunks": len(baseline_records),
        "parent_chunks": len(parent_records),
        "child_chunks": len(child_records),
        "chunking": {
            "baseline": {
                "words": BASELINE_WORDS,
                "overlap": BASELINE_OVERLAP,
                "purpose": "basic retrieval and pre-improvement RAGAS baseline",
            },
            "parent": {"words": PARENT_WORDS, "overlap": PARENT_OVERLAP},
            "child": {"words": CHILD_WORDS, "overlap": CHILD_OVERLAP},
            "purpose": "retrieve children and return unique parents",
        },
        "documents_by_type": {
            kind: sum(row["document_type"] == kind for row in manifest_rows)
            for kind in sorted({row["document_type"] for row in manifest_rows})
        },
    }
    (output_dir / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\nCorpus preparation complete")
    print(f"Documents processed: {len(manifest_rows)}")
    print(f"Baseline chunks: {len(baseline_records)}")
    print(f"Parent chunks: {len(parent_records)}")
    print(f"Child chunks: {len(child_records)}")
    print(f"Output directory: {output_dir.resolve()}")
    return 0


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Extract the project's PDF and HTML evidence corpus and create Lab B1 "
            "baseline plus parent-child chunks."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "corpus_raw",
        help="Directory containing the downloaded PDF and HTML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "corpus_processed",
        help="Directory in which processed corpus files will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return process_corpus(args.input, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
