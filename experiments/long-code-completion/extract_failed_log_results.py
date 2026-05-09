#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract compression results and available metrics from CodeCompressor logs.

The failed c_ff4/c_ff5 runs finished prompt compression but crashed before
generation/scoring. This script recovers what is truly present in the logs:

1. the 500 final fine-grained compressed contexts;
2. per-case compressed token counts and compression ratios;
3. ES/EM only when the log contains successful "Case ... => es=..., em=..."
   lines.

It deliberately does not borrow ES/EM from another run, because those scores
depend on the generated answers for this exact compressed context.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


TITLE = "整体细粒度压缩完后的结果"

LOG_PREFIX_RE = re.compile(
    r".*?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \| "
    r"[A-Z]+\s+\| .*? - ?"
)
ORIGINAL_RE = re.compile(
    r"CodeCompressor: Original tokens=(?P<original>\d+), "
    r"Target tokens=(?P<target>\d+), Calculated ratio=(?P<ratio>[0-9.]+)"
)
COMPRESSED_RE = re.compile(
    r"CodeCompressor: Compressed tokens=(?P<compressed>\d+), "
    r"Actual ratio=(?P<keep>[0-9.]+)"
)
FINE_TOTAL_RE = re.compile(
    r"\[HYBRID-FINE\]\[TOTAL\] original_tokens=(?P<original>\d+) "
    r"compressed_tokens=(?P<compressed>\d+) keep_ratio=(?P<keep>[0-9.]+) "
    r"compression_ratio=(?P<ratio>[0-9.]+)"
)
CASE_SCORE_RE = re.compile(
    r"Case\s+(?P<case_id>\d+)\s+=>\s+es=(?P<es>[-+0-9.eE]+),\s+"
    r"em=(?P<em>[-+0-9.eE]+),\s+ratio=(?P<ratio>[-+0-9.eE]+),\s+"
    r"orig_tokens=(?P<original>\d+),\s+comp_tokens=(?P<compressed>\d+)"
)
AVG_SCORE_RE = re.compile(
    r"Method\s+(?P<method>[^:]+):\s+Avg ES = (?P<avg_es>[-+0-9.eE]+),\s+"
    r"Avg EM = (?P<avg_em>[-+0-9.eE]+)\s+\((?P<valid>\d+)/(?P<total>\d+) scored\)"
)
TOTAL_TOKEN_RE = re.compile(
    r"Total token stats: original=(?P<original>\d+),\s+"
    r"compressed=(?P<compressed>\d+),\s+original/compressed=(?P<ratio>[-+0-9.eE]+)"
)


@dataclass
class TextBlock:
    case_id: int
    start_line: int
    end_line: int | None
    text: str


@dataclass
class OriginalEvent:
    case_id: int
    line: int
    original_tokens: int
    target_tokens: int
    calculated_keep_ratio: float


@dataclass
class CompressedEvent:
    case_id: int
    line: int
    compressed_tokens: int
    actual_keep_ratio: float


@dataclass
class FineTotalEvent:
    case_id: int
    line: int
    fine_original_tokens: int
    fine_compressed_tokens: int
    fine_keep_ratio: float
    fine_compression_ratio: float


@dataclass
class ScoreEvent:
    case_id: int
    line: int
    es: float
    em: float
    score_ratio: float
    score_original_tokens: int
    score_compressed_tokens: int


@dataclass
class CaseRecord:
    case_id: int
    compressed_context: str | None = None
    compressed_context_start_line: int | None = None
    compressed_context_end_line: int | None = None
    original_tokens: int | None = None
    target_tokens: int | None = None
    calculated_keep_ratio: float | None = None
    compressed_tokens: int | None = None
    keep_ratio: float | None = None
    compression_ratio: float | None = None
    fine_original_tokens: int | None = None
    fine_compressed_tokens: int | None = None
    fine_keep_ratio: float | None = None
    fine_compression_ratio: float | None = None
    es: float | None = None
    em: float | None = None
    score_ratio: float | None = None
    score_original_tokens: int | None = None
    score_compressed_tokens: int | None = None


def strip_log_prefix(line: str) -> str:
    """Remove a loguru prefix, preserving raw code lines inside multiline logs."""
    line = line.rstrip("\n")
    match = LOG_PREFIX_RE.match(line)
    if match:
        return line[match.end() :]
    return line


def has_log_prefix(line: str) -> bool:
    return LOG_PREFIX_RE.match(line.rstrip("\n")) is not None


def is_delimiter_line(line: str) -> bool:
    # A compressed code block can itself contain a visual separator made of
    # "=" characters. The real block terminator is the logger.info("=" * 100)
    # line, which has a loguru prefix; raw multiline payload lines do not.
    if not has_log_prefix(line):
        return False
    stripped = strip_log_prefix(line).strip()
    return len(stripped) >= 20 and set(stripped) == {"="}


def parse_final_compressed_blocks(log_path: Path) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    waiting_for_content = False
    collecting = False
    current_lines: list[str] = []
    start_line: int | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            msg = strip_log_prefix(raw_line)

            if TITLE in msg:
                waiting_for_content = True
                collecting = False
                current_lines = []
                start_line = None
                continue

            if waiting_for_content:
                if is_delimiter_line(raw_line):
                    continue
                if not collecting and msg == "":
                    continue
                collecting = True
                waiting_for_content = False
                start_line = line_no
                current_lines.append(msg)
                continue

            if collecting:
                if is_delimiter_line(raw_line):
                    blocks.append(
                        TextBlock(
                            case_id=len(blocks),
                            start_line=start_line if start_line is not None else line_no,
                            end_line=line_no,
                            text="\n".join(current_lines).rstrip(),
                        )
                    )
                    collecting = False
                    current_lines = []
                    start_line = None
                    continue
                current_lines.append(msg)

    if collecting:
        blocks.append(
            TextBlock(
                case_id=len(blocks),
                start_line=start_line if start_line is not None else 0,
                end_line=None,
                text="\n".join(current_lines).rstrip(),
            )
        )

    return blocks


def parse_metric_events(
    log_path: Path,
) -> tuple[list[OriginalEvent], list[CompressedEvent], list[FineTotalEvent], list[ScoreEvent], dict[str, Any]]:
    originals: list[OriginalEvent] = []
    compressed: list[CompressedEvent] = []
    fine_totals: list[FineTotalEvent] = []
    scores: list[ScoreEvent] = []
    logged_summary: dict[str, Any] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            msg = strip_log_prefix(raw_line)

            if match := ORIGINAL_RE.search(msg):
                originals.append(
                    OriginalEvent(
                        case_id=len(originals),
                        line=line_no,
                        original_tokens=int(match.group("original")),
                        target_tokens=int(match.group("target")),
                        calculated_keep_ratio=float(match.group("ratio")),
                    )
                )
                continue

            if match := COMPRESSED_RE.search(msg):
                compressed.append(
                    CompressedEvent(
                        case_id=len(compressed),
                        line=line_no,
                        compressed_tokens=int(match.group("compressed")),
                        actual_keep_ratio=float(match.group("keep")),
                    )
                )
                continue

            if match := FINE_TOTAL_RE.search(msg):
                fine_totals.append(
                    FineTotalEvent(
                        case_id=len(fine_totals),
                        line=line_no,
                        fine_original_tokens=int(match.group("original")),
                        fine_compressed_tokens=int(match.group("compressed")),
                        fine_keep_ratio=float(match.group("keep")),
                        fine_compression_ratio=float(match.group("ratio")),
                    )
                )
                continue

            if match := CASE_SCORE_RE.search(msg):
                scores.append(
                    ScoreEvent(
                        case_id=int(match.group("case_id")),
                        line=line_no,
                        es=float(match.group("es")),
                        em=float(match.group("em")),
                        score_ratio=float(match.group("ratio")),
                        score_original_tokens=int(match.group("original")),
                        score_compressed_tokens=int(match.group("compressed")),
                    )
                )
                continue

            if match := AVG_SCORE_RE.search(msg):
                logged_summary["logged_avg_es"] = float(match.group("avg_es"))
                logged_summary["logged_avg_em"] = float(match.group("avg_em"))
                logged_summary["logged_scored_cases"] = int(match.group("valid"))
                logged_summary["logged_total_cases"] = int(match.group("total"))
                logged_summary["logged_method"] = match.group("method").strip()
                continue

            if match := TOTAL_TOKEN_RE.search(msg):
                logged_summary["logged_total_original_tokens"] = int(match.group("original"))
                logged_summary["logged_total_compressed_tokens"] = int(match.group("compressed"))
                logged_summary["logged_total_compression_ratio"] = float(match.group("ratio"))

    return originals, compressed, fine_totals, scores, logged_summary


def safe_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def build_case_records(log_path: Path) -> tuple[list[CaseRecord], dict[str, Any]]:
    blocks = parse_final_compressed_blocks(log_path)
    originals, compressed, fine_totals, scores, logged_summary = parse_metric_events(log_path)

    max_count = max(
        [len(blocks), len(originals), len(compressed), len(fine_totals)]
        + ([max(score.case_id for score in scores) + 1] if scores else [0])
    )
    records = [CaseRecord(case_id=i) for i in range(max_count)]

    for block in blocks:
        records[block.case_id].compressed_context = block.text
        records[block.case_id].compressed_context_start_line = block.start_line
        records[block.case_id].compressed_context_end_line = block.end_line

    for event in originals:
        records[event.case_id].original_tokens = event.original_tokens
        records[event.case_id].target_tokens = event.target_tokens
        records[event.case_id].calculated_keep_ratio = event.calculated_keep_ratio

    for event in compressed:
        records[event.case_id].compressed_tokens = event.compressed_tokens
        records[event.case_id].keep_ratio = event.actual_keep_ratio
        records[event.case_id].compression_ratio = safe_ratio(
            records[event.case_id].original_tokens,
            event.compressed_tokens,
        )

    for event in fine_totals:
        records[event.case_id].fine_original_tokens = event.fine_original_tokens
        records[event.case_id].fine_compressed_tokens = event.fine_compressed_tokens
        records[event.case_id].fine_keep_ratio = event.fine_keep_ratio
        records[event.case_id].fine_compression_ratio = event.fine_compression_ratio

    for event in scores:
        if event.case_id >= len(records):
            records.extend(CaseRecord(case_id=i) for i in range(len(records), event.case_id + 1))
        records[event.case_id].es = event.es
        records[event.case_id].em = event.em
        records[event.case_id].score_ratio = event.score_ratio
        records[event.case_id].score_original_tokens = event.score_original_tokens
        records[event.case_id].score_compressed_tokens = event.score_compressed_tokens

    parse_counts = {
        "compressed_context_cases": len(blocks),
        "original_token_events": len(originals),
        "compressed_token_events": len(compressed),
        "fine_total_events": len(fine_totals),
        "score_events": len(scores),
    }
    parse_counts.update(logged_summary)
    return records, parse_counts


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize(log_path: Path, records: list[CaseRecord], parse_counts: dict[str, Any]) -> dict[str, Any]:
    scored_records = [r for r in records if r.es is not None and r.em is not None]
    token_records = [r for r in records if r.original_tokens is not None and r.compressed_tokens]
    fine_records = [
        r for r in records if r.fine_original_tokens is not None and r.fine_compressed_tokens
    ]
    score_token_records = [
        r for r in records if r.score_original_tokens is not None and r.score_compressed_tokens
    ]

    total_original_tokens = sum(r.original_tokens or 0 for r in token_records)
    total_compressed_tokens = sum(r.compressed_tokens or 0 for r in token_records)
    fine_total_original_tokens = sum(r.fine_original_tokens or 0 for r in fine_records)
    fine_total_compressed_tokens = sum(r.fine_compressed_tokens or 0 for r in fine_records)
    score_total_original_tokens = sum(r.score_original_tokens or 0 for r in score_token_records)
    score_total_compressed_tokens = sum(r.score_compressed_tokens or 0 for r in score_token_records)

    summary = {
        "log": str(log_path),
        "case_count": len(records),
        "parse_counts": parse_counts,
        "avg_es": mean_or_none([r.es for r in scored_records if r.es is not None]),
        "avg_em": mean_or_none([r.em for r in scored_records if r.em is not None]),
        "scored_cases": len(scored_records),
        "token_cases": len(token_records),
        "total_original_tokens": total_original_tokens,
        "total_compressed_tokens": total_compressed_tokens,
        "total_compression_ratio": safe_ratio(total_original_tokens, total_compressed_tokens),
        "avg_case_compression_ratio": mean_or_none(
            [r.compression_ratio for r in token_records if r.compression_ratio is not None]
        ),
        "avg_keep_ratio": mean_or_none([r.keep_ratio for r in token_records if r.keep_ratio is not None]),
        "fine_token_cases": len(fine_records),
        "fine_total_original_tokens": fine_total_original_tokens,
        "fine_total_compressed_tokens": fine_total_compressed_tokens,
        "fine_total_compression_ratio": safe_ratio(
            fine_total_original_tokens, fine_total_compressed_tokens
        ),
        "fine_avg_case_compression_ratio": mean_or_none(
            [
                r.fine_compression_ratio
                for r in fine_records
                if r.fine_compression_ratio is not None
            ]
        ),
        "score_token_cases": len(score_token_records),
        "score_total_original_tokens": score_total_original_tokens or None,
        "score_total_compressed_tokens": score_total_compressed_tokens or None,
        "score_total_compression_ratio": safe_ratio(
            score_total_original_tokens, score_total_compressed_tokens
        ),
    }
    return summary


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_jsonl(path: Path, records: list[CaseRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_summary_txt(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for summary in summaries:
        log_name = Path(summary["log"]).name
        lines.append(f"===== {log_name} =====")
        lines.append(
            "Method code_compressor: "
            f"Avg ES = {fmt_float(summary['avg_es'], 2)}, "
            f"Avg EM = {fmt_float(summary['avg_em'], 2)} "
            f"({summary['scored_cases']}/{summary['case_count']} scored)"
        )
        lines.append(
            "Log-only CodeCompressor token stats: "
            f"original={summary['total_original_tokens']}, "
            f"compressed={summary['total_compressed_tokens']}, "
            f"original/compressed={fmt_float(summary['total_compression_ratio'])}, "
            f"avg_case_ratio={fmt_float(summary['avg_case_compression_ratio'])}"
        )
        lines.append(
            "HYBRID-FINE internal token stats: "
            f"original={summary['fine_total_original_tokens']}, "
            f"compressed={summary['fine_total_compressed_tokens']}, "
            f"original/compressed={fmt_float(summary['fine_total_compression_ratio'])}, "
            f"avg_case_ratio={fmt_float(summary['fine_avg_case_compression_ratio'])}"
        )
        if summary.get("score_total_original_tokens"):
            lines.append(
                "Successful-score token stats: "
                f"original={summary['score_total_original_tokens']}, "
                f"compressed={summary['score_total_compressed_tokens']}, "
                f"original/compressed={fmt_float(summary['score_total_compression_ratio'])}"
            )
        if summary["scored_cases"] == 0:
            lines.append(
                "NOTE: this log has no Case => es/em lines, so ES/EM cannot be "
                "recovered from the failed generation log."
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract final fine-grained compressed contexts and metrics from c_ff*.log files."
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        default=["c_ff4.log", "c_ff5.log"],
        help="Failed or successful logs to parse. Default: c_ff4.log c_ff5.log",
    )
    parser.add_argument(
        "--reference-log",
        default=None,
        help="Optional successful log to parse as a format reference, e.g. c_ff3.log.",
    )
    parser.add_argument(
        "--output-dir",
        default="failed_log_extract",
        help="Directory for JSONL and summary outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_paths = [Path(item) for item in args.logs]
    if args.reference_log:
        log_paths.append(Path(args.reference_log))

    summaries: list[dict[str, Any]] = []
    for log_path in log_paths:
        if not log_path.exists():
            raise FileNotFoundError(f"Log not found: {log_path}")

        records, parse_counts = build_case_records(log_path)
        summary = summarize(log_path, records, parse_counts)
        summaries.append(summary)

        cases_path = output_dir / f"{log_path.stem}.cases.jsonl"
        write_jsonl(cases_path, records)

        print(f"===== {log_path.name} =====")
        print(f"cases_jsonl={cases_path}")
        print(
            "compressed_context_cases="
            f"{parse_counts['compressed_context_cases']}, "
            f"token_cases={summary['token_cases']}, "
            f"fine_token_cases={summary['fine_token_cases']}, "
            f"scored_cases={summary['scored_cases']}/{summary['case_count']}"
        )
        print(
            "Method code_compressor: "
            f"Avg ES = {fmt_float(summary['avg_es'], 2)}, "
            f"Avg EM = {fmt_float(summary['avg_em'], 2)} "
            f"({summary['scored_cases']}/{summary['case_count']} scored)"
        )
        print(
            "Log-only CodeCompressor token stats: "
            f"original={summary['total_original_tokens']}, "
            f"compressed={summary['total_compressed_tokens']}, "
            f"original/compressed={fmt_float(summary['total_compression_ratio'])}"
        )
        print(
            "HYBRID-FINE internal token stats: "
            f"original={summary['fine_total_original_tokens']}, "
            f"compressed={summary['fine_total_compressed_tokens']}, "
            f"original/compressed={fmt_float(summary['fine_total_compression_ratio'])}"
        )
        print()

    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path.write_text(
        json.dumps({"logs": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_txt(summary_txt_path, summaries)

    print(f"summary_json={summary_json_path}")
    print(f"summary_txt={summary_txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
