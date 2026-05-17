#!/usr/bin/env python3
"""
Fetch ArrayExpress studies from the BioStudies API and extract screening context.

By default this reads:
  - als-ftd-screened.txt
  - alzheimer-screened.txt
  - lbd-screened.txt

For each list it creates a same-named output directory, caches raw BioStudies
study JSON under raw_json/, and writes extracted inclusion/exclusion context
under context_json/, context_text/, and contexts.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


API_BASE = "https://www.ebi.ac.uk/biostudies/api/v1/studies"
DEFAULT_INPUTS = (
    "als-ftd-screened.txt",
    "alzheimer-screened.txt",
    "lbd-screened.txt",
)

SOURCE_CHARACTERISTICS = (
    "Organism",
    "Developmental stage",
    "Organism part",
    "Disease",
)

SOURCE_CHARACTERISTIC_ALIASES = {
    "Organism": ("Organism",),
    "Developmental stage": ("Developmental stage", "DevelopmentalStage"),
    "Organism part": ("Organism part", "OrganismPart"),
    "Disease": (
        "Disease",
        "Disease state",
        "Disease_state",
        "DiseaseState",
        "Diseasestate",
        "Disease stage",
        "DiseaseStage",
        "Disease staging",
        "DiseaseStaging",
        "Diseasestaging",
    ),
}

SAMPLE_ATTRIBUTE_ALIASES = {
    "Experimental Designs": ("Experimental Designs", "Experimental Design"),
    "Experimental Factors": ("Experimental Factors", "Experimental Factor"),
    "Disease state/staging": (
        "Disease state",
        "Disease_state",
        "DiseaseState",
        "Diseasestate",
        "Disease stage",
        "DiseaseStage",
        "Disease staging",
        "DiseaseStaging",
        "Diseasestaging",
    ),
    "Cell type/line": ("Cell type", "CellType", "Cell line", "CellLine", "Cellline"),
    "Clinical/familial": (
        "Clinical history",
        "Clinical information",
        "Clinical diagnosis",
        "Familial anamnesis",
        "Family history",
    ),
    "Treatment/compound": ("Treatment", "Compound"),
    "Quality controls": ("Quality Controls", "Quality Control"),
}

COUNT_ATTRIBUTE_ALIASES = {
    "Sample count": ("Sample count",),
    "Assay count": ("Assay count",),
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", str(value)).strip()


def unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def read_accessions(path: Path) -> list[str]:
    accessions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        accessions.append(line.split()[0])
    return accessions


def request_json(url: str, retries: int, timeout: int) -> dict[str, Any]:
    last_error = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "finding-datasets-with-LLMs ArrayExpress metadata fetcher",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def fetch_study(
    accession: str,
    raw_json_path: Path,
    api_base: str,
    force: bool,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    if raw_json_path.exists() and not force:
        return json.loads(raw_json_path.read_text(encoding="utf-8"))

    url = f"{api_base.rstrip('/')}/{urllib.parse.quote(accession)}"
    study = request_json(url, retries=retries, timeout=timeout)
    raw_json_path.write_text(
        json.dumps(study, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return study


def attribute_value(attribute: dict[str, Any]) -> str:
    if "value" in attribute:
        return clean_text(attribute.get("value"))
    if "val" in attribute:
        return clean_text(attribute.get("val"))
    if "text" in attribute:
        return clean_text(attribute.get("text"))
    return ""


def attribute_name(attribute: dict[str, Any]) -> str:
    return clean_text(attribute.get("name") or attribute.get("type") or attribute.get("key"))


def attributes_by_name(node: dict[str, Any]) -> dict[str, list[str]]:
    values = defaultdict(list)
    for attribute in node.get("attributes") or []:
        if not isinstance(attribute, dict):
            continue
        name = attribute_name(attribute)
        value = attribute_value(attribute)
        if name and value:
            values[name].append(value)
    return {name: unique(vals) for name, vals in values.items()}


def iter_sections(node: Any) -> list[dict[str, Any]]:
    sections = []
    if isinstance(node, dict):
        if "type" in node:
            sections.append(node)
        for key in ("section", "subsections"):
            child = node.get(key)
            if isinstance(child, list):
                for item in child:
                    sections.extend(iter_sections(item))
            elif isinstance(child, dict):
                sections.extend(iter_sections(child))
    elif isinstance(node, list):
        for item in node:
            sections.extend(iter_sections(item))
    return sections


def find_sections(study: dict[str, Any], section_type: str) -> list[dict[str, Any]]:
    wanted = normalize_name(section_type)
    return [
        section
        for section in iter_sections(study)
        if normalize_name(clean_text(section.get("type"))) == wanted
    ]


def descendant_sections(section: dict[str, Any]) -> list[dict[str, Any]]:
    descendants = []
    for child in section.get("subsections") or []:
        descendants.extend(iter_sections(child))
    return descendants


def all_attribute_values(study: dict[str, Any], wanted_names: tuple[str, ...] | list[str]) -> list[str]:
    wanted = {normalize_name(name) for name in wanted_names}
    values = []

    for node in [study, *iter_sections(study)]:
        if not isinstance(node, dict):
            continue
        for name, vals in attributes_by_name(node).items():
            if normalize_name(name) in wanted:
                values.extend(vals)
    return unique(values)


def first_attribute_value(study: dict[str, Any], wanted_names: tuple[str, ...] | list[str]) -> str:
    values = all_attribute_values(study, wanted_names)
    return values[0] if values else ""


def attribute_matches(name: str, target: str) -> bool:
    normalized_name = normalize_name(name)
    normalized_target = normalize_name(target)
    if normalized_name == normalized_target:
        return True

    bracketed = re.findall(r"\[([^\]]+)\]|\(([^)]+)\)", name)
    for left, right in bracketed:
        if normalize_name(left or right) == normalized_target:
            return True

    name_tokens = re.findall(r"[a-z0-9]+", name.lower())
    target_tokens = re.findall(r"[a-z0-9]+", target.lower())
    if not name_tokens or not target_tokens:
        return False

    if name_tokens == target_tokens:
        return True

    if len(name_tokens) > len(target_tokens) and name_tokens[-len(target_tokens) :] == target_tokens:
        return True
    return False


def collect_matching_attributes(sections: list[dict[str, Any]], targets: str | tuple[str, ...]) -> list[str]:
    if isinstance(targets, str):
        targets = (targets,)

    values = []
    for section in sections:
        for name, vals in attributes_by_name(section).items():
            if any(attribute_matches(name, target) for target in targets):
                values.extend(vals)
    return unique(values)


def collect_attribute_summary(
    sections: list[dict[str, Any]],
    targets: str | tuple[str, ...],
) -> dict[str, list[str]]:
    if isinstance(targets, str):
        targets = (targets,)

    summary = defaultdict(list)
    for section in sections:
        for name, values in attributes_by_name(section).items():
            if any(attribute_matches(name, target) for target in targets):
                summary[name].extend(values)
    return {name: unique(values) for name, values in summary.items()}


def extract_samples_context(study: dict[str, Any]) -> dict[str, Any]:
    sample_sections = find_sections(study, "Samples")
    sample_scope = []
    for section in sample_sections:
        sample_scope.append(section)
        sample_scope.extend(descendant_sections(section))

    if not sample_scope:
        sample_scope = iter_sections(study)

    source_values = {}
    for characteristic in SOURCE_CHARACTERISTICS:
        source_values[characteristic] = collect_matching_attributes(
            sample_scope,
            SOURCE_CHARACTERISTIC_ALIASES[characteristic],
        )

    additional_attributes = {
        label: collect_attribute_summary(sample_scope, aliases)
        for label, aliases in SAMPLE_ATTRIBUTE_ALIASES.items()
        if label not in {"Experimental Designs", "Experimental Factors"}
    }

    return {
        "Experimental Designs": collect_attribute_summary(
            sample_scope,
            SAMPLE_ATTRIBUTE_ALIASES["Experimental Designs"],
        ),
        "Experimental Factors": collect_attribute_summary(
            sample_scope,
            SAMPLE_ATTRIBUTE_ALIASES["Experimental Factors"],
        ),
        "Experimental Factors > Source Characteristics": source_values,
        "Additional screening-relevant attributes": additional_attributes,
    }


def extract_assay_context(study: dict[str, Any]) -> dict[str, list[str]]:
    assay_sections = find_sections(study, "Assays and Data")
    assay_scope = []
    for section in assay_sections:
        assay_scope.append(section)
        assay_scope.extend(descendant_sections(section))

    if not assay_scope:
        assay_scope = iter_sections(study)

    return {
        "Technology": collect_matching_attributes(assay_scope, "Technology"),
        "Assay by Molecule": collect_matching_attributes(assay_scope, "Assay by Molecule"),
    }


def extract_context(study: dict[str, Any], accession: str, source_list: str) -> dict[str, Any]:
    section = study.get("section") if isinstance(study.get("section"), dict) else {}
    title = first_attribute_value(study, ("Title", "Study Title")) or clean_text(study.get("title"))
    description = (
        first_attribute_value(study, ("Description", "Study Description", "Abstract"))
        or clean_text(study.get("description"))
    )
    study_type = first_attribute_value(study, ("Study Type", "Study type", "StudyType", "Experiment Type"))

    return {
        "Accession": accession,
        "Source list": source_list,
        "Title": title,
        "Study type": study_type,
        "Description": description,
        "Counts": {
            label: all_attribute_values(study, aliases)
            for label, aliases in COUNT_ATTRIBUTE_ALIASES.items()
        },
        "Samples": extract_samples_context(study),
        "Assays and Data": extract_assay_context(study),
        "BioStudies section type": clean_text(section.get("type")),
    }


def flatten_summary(summary: dict[str, list[str]]) -> str:
    return "; ".join(f"{name}: {', '.join(values)}" for name, values in summary.items())


def flatten_context_for_csv(context: dict[str, Any]) -> dict[str, str]:
    source = context["Samples"]["Experimental Factors > Source Characteristics"]
    assays = context["Assays and Data"]
    designs = context["Samples"]["Experimental Designs"]
    factors = context["Samples"]["Experimental Factors"]
    additional = context["Samples"]["Additional screening-relevant attributes"]
    counts = context["Counts"]

    return {
        "accession": context["Accession"],
        "source_list": context["Source list"],
        "title": context["Title"],
        "study_type": context["Study type"],
        "description": context["Description"],
        "sample_count": "; ".join(counts["Sample count"]),
        "assay_count": "; ".join(counts["Assay count"]),
        "experimental_designs": flatten_summary(designs),
        "experimental_factors": flatten_summary(factors),
        "organism": "; ".join(source["Organism"]),
        "developmental_stage": "; ".join(source["Developmental stage"]),
        "organism_part": "; ".join(source["Organism part"]),
        "disease": "; ".join(source["Disease"]),
        "disease_state_or_staging": flatten_summary(additional["Disease state/staging"]),
        "cell_type_or_line": flatten_summary(additional["Cell type/line"]),
        "clinical_or_familial": flatten_summary(additional["Clinical/familial"]),
        "treatment_or_compound": flatten_summary(additional["Treatment/compound"]),
        "quality_controls": flatten_summary(additional["Quality controls"]),
        "technology": "; ".join(assays["Technology"]),
        "assay_by_molecule": "; ".join(assays["Assay by Molecule"]),
    }


def format_block(title: str, value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    lines = [f"{prefix}{title}:"]
    if isinstance(value, dict):
        if not value:
            lines.append(f"{prefix}  Not found")
        for key, vals in value.items():
            if isinstance(vals, dict):
                lines.extend(format_block(key, vals, indent + 2))
            elif isinstance(vals, list):
                lines.append(f"{prefix}  {key}: {', '.join(vals) if vals else 'Not found'}")
            else:
                lines.append(f"{prefix}  {key}: {clean_text(vals) or 'Not found'}")
    elif isinstance(value, list):
        lines.append(f"{prefix}  {', '.join(value) if value else 'Not found'}")
    else:
        lines.append(f"{prefix}  {clean_text(value) or 'Not found'}")
    return lines


def context_to_text(context: dict[str, Any]) -> str:
    counts = context.get("Counts", {})
    lines = [
        f"Accession: {context['Accession']}",
        f"Source list: {context['Source list']}",
        f"Title: {context['Title'] or 'Not found'}",
        f"Study type: {context['Study type'] or 'Not found'}",
        f"Description: {context['Description'] or 'Not found'}",
        f"Sample count: {', '.join(counts.get('Sample count', [])) or 'Not found'}",
        f"Assay count: {', '.join(counts.get('Assay count', [])) or 'Not found'}",
        "",
    ]
    lines.extend(format_block("Samples", context["Samples"]))
    lines.append("")
    lines.extend(format_block("Assays and Data", context["Assays and Data"]))
    return "\n".join(lines) + "\n"


def process_list(
    input_path: Path,
    output_root: Path,
    api_base: str,
    force: bool,
    retries: int,
    timeout: int,
    sleep_seconds: float,
) -> tuple[int, int]:
    source_name = input_path.stem
    output_dir = output_root / source_name
    raw_dir = output_dir / "raw_json"
    context_json_dir = output_dir / "context_json"
    context_text_dir = output_dir / "context_text"
    for directory in (raw_dir, context_json_dir, context_text_dir):
        directory.mkdir(parents=True, exist_ok=True)

    accessions = read_accessions(input_path)
    rows = []
    failures = []

    for index, accession in enumerate(accessions, start=1):
        print(f"[{source_name}] {index}/{len(accessions)} {accession}")
        raw_path = raw_dir / f"{accession}.json"
        try:
            study = fetch_study(
                accession=accession,
                raw_json_path=raw_path,
                api_base=api_base,
                force=force,
                retries=retries,
                timeout=timeout,
            )
            context = extract_context(study, accession=accession, source_list=source_name)
            (context_json_dir / f"{accession}.json").write_text(
                json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (context_text_dir / f"{accession}.txt").write_text(
                context_to_text(context),
                encoding="utf-8",
            )
            rows.append(flatten_context_for_csv(context))
        except Exception as exc:
            failures.append((accession, str(exc)))
            print(f"  ERROR: {exc}", file=sys.stderr)
        if sleep_seconds:
            time.sleep(sleep_seconds)

    csv_path = output_dir / "contexts.csv"
    fieldnames = [
        "accession",
        "source_list",
        "title",
        "study_type",
        "description",
        "sample_count",
        "assay_count",
        "experimental_designs",
        "experimental_factors",
        "organism",
        "developmental_stage",
        "organism_part",
        "disease",
        "disease_state_or_staging",
        "cell_type_or_line",
        "clinical_or_familial",
        "treatment_or_compound",
        "quality_controls",
        "technology",
        "assay_by_molecule",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        error_path = output_dir / "errors.log"
        with error_path.open("w", encoding="utf-8") as handle:
            for accession, error in failures:
                handle.write(f"{accession}\t{error}\n")

    return len(rows), len(failures)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fetch BioStudies JSON for screened ArrayExpress accessions and extract screening context.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[script_dir / name for name in DEFAULT_INPUTS],
        help="Screened accession text files. Defaults to the three ArrayExpress/*-screened.txt files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir,
        help="Directory where per-list output directories are created. Default: ArrayExpress/",
    )
    parser.add_argument(
        "--api-base",
        default=API_BASE,
        help=f"BioStudies studies endpoint base URL. Default: {API_BASE}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch and overwrite cached raw JSON.",
    )
    parser.add_argument("--retries", type=int, default=3, help="HTTP retry count per accession.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API requests in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_ok = 0
    total_failed = 0

    for input_path in args.inputs:
        input_path = input_path if input_path.is_absolute() else Path.cwd() / input_path
        if not input_path.exists():
            print(f"Missing input file: {input_path}", file=sys.stderr)
            total_failed += 1
            continue
        ok, failed = process_list(
            input_path=input_path,
            output_root=args.output_root,
            api_base=args.api_base,
            force=args.force,
            retries=args.retries,
            timeout=args.timeout,
            sleep_seconds=args.sleep,
        )
        total_ok += ok
        total_failed += failed

    print(f"Done. Extracted {total_ok} studies; {total_failed} failures.")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
