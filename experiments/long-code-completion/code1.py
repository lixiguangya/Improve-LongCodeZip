# -*- coding: utf-8 -*-
"""
Compare top case gaps for three code-compressor runs.

Groups:
- group1: code_compressor_ff20.py, log c_ff20.log
- group2: code_compressor_ff6.py, log c_ff6.log
- group3: code_compressor.py, log c0.log

For each requested comparison, this script prints top10 by ES difference.
Each case entry includes:
- scores for both groups and the difference
- both groups' final fine-grained compressed results from their logs
- problem/current-function context and answer from data.log

Usage:
    python code1.py
    python code1.py --output top10_cases.txt
    python code1.py --top-k 5 --max-section-chars 4000
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
DATA_LOG = BASE_DIR / "data.log"


@dataclass(frozen=True)
class GroupSpec:
    key: str
    label: str
    source: str
    log_path: Path
    final_result_title: str


@dataclass
class Score:
    case_id: int
    es: float
    em: float
    ratio: Optional[float] = None
    orig_tokens: Optional[int] = None
    comp_tokens: Optional[int] = None


@dataclass
class CaseData:
    context: str
    answer: str


GROUPS: Dict[str, GroupSpec] = {
    "g1": GroupSpec(
        key="g1",
        label="第一组",
        source="code_compressor_ff20.py",
        log_path=BASE_DIR / "c_ff20.log",
        final_result_title="整体细粒度压缩完后的结果",
    ),
    "g2": GroupSpec(
        key="g2",
        label="第二组",
        source="code_compressor_ff6.py",
        log_path=BASE_DIR / "c_ff6.log",
        final_result_title="整体细粒度压缩完后的结果",
    ),
    "g3": GroupSpec(
        key="g3",
        label="第三组",
        source="code_compressor.py",
        log_path=BASE_DIR / "c0.log",
        final_result_title="整体细粒度压缩完后的结果",
    ),
}

COMPARISONS: List[Tuple[str, str, str]] = [
    ("第一组 - 第二组", "g1", "g2"),
    ("第二组 - 第一组", "g2", "g1"),
    ("第一组 - 第三组", "g1", "g3"),
    ("第三组 - 第一组", "g3", "g1"),
    ("第二组 - 第三组", "g2", "g3"),
    ("第三组 - 第二组", "g3", "g2"),
]

SCORE_RE = re.compile(
    r"Case\s+(?P<case_id>\d+)\s*=>\s*"
    r"es=(?P<es>-?\d+(?:\.\d+)?),\s*"
    r"em=(?P<em>-?\d+(?:\.\d+)?),\s*"
    r"ratio=(?P<ratio>-?\d+(?:\.\d+)?),\s*"
    r"orig_tokens=(?P<orig_tokens>\d+),\s*"
    r"comp_tokens=(?P<comp_tokens>\d+)"
)
CASE_HEADER_RE = re.compile(r"^case(?P<case_id>\d+):\s*$", re.MULTILINE)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOG_PREFIX_RE = re.compile(
    r"^.*?\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2}\.\d+\s+\|\s+[A-Z]+\s+\|.*?-\s?"
)


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line.rstrip("\n").replace("\r", ""))


def strip_log_prefix(line: str) -> str:
    return LOG_PREFIX_RE.sub("", line, count=1)


def has_log_prefix(line: str) -> bool:
    return LOG_PREFIX_RE.match(line) is not None


def is_delimiter_line(line: str) -> bool:
    if not has_log_prefix(line):
        return False
    payload = strip_log_prefix(line).strip()
    return len(payload) >= 20 and set(payload) == {"="}


def maybe_truncate(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n... [TRUNCATED {omitted} chars]"


def strip_trailing_separators(text: str) -> str:
    lines = text.rstrip().splitlines()
    while lines:
        stripped = lines[-1].strip()
        if not stripped or set(stripped) == {"="}:
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def parse_scores(log_path: Path) -> Dict[int, Score]:
    scores: Dict[int, Score] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = clean_line(raw_line)
            match = SCORE_RE.search(line)
            if not match:
                continue
            case_id = int(match.group("case_id"))
            scores[case_id] = Score(
                case_id=case_id,
                es=float(match.group("es")),
                em=float(match.group("em")),
                ratio=float(match.group("ratio")),
                orig_tokens=int(match.group("orig_tokens")),
                comp_tokens=int(match.group("comp_tokens")),
            )
    return scores


def parse_final_compressed_results(spec: GroupSpec) -> Dict[int, str]:
    """Extract final fine-grained compressed result blocks by occurrence order."""
    results: Dict[int, str] = {}
    waiting_for_text = False
    collecting = False
    block_lines: List[str] = []
    case_id = 0

    with spec.log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = clean_line(raw_line)

            if collecting:
                if is_delimiter_line(line):
                    results[case_id] = "\n".join(block_lines).strip()
                    case_id += 1
                    block_lines = []
                    collecting = False
                    waiting_for_text = False
                    continue
                block_lines.append(strip_log_prefix(line))
                continue

            if waiting_for_text:
                if is_delimiter_line(line):
                    continue
                collecting = True
                block_lines = [strip_log_prefix(line)]
                continue

            if spec.final_result_title in line:
                waiting_for_text = True

    if collecting and block_lines:
        results[case_id] = "\n".join(block_lines).strip()

    return results


def parse_data_log(path: Path) -> Dict[int, CaseData]:
    text = read_text(path)
    matches = list(CASE_HEADER_RE.finditer(text))
    cases: Dict[int, CaseData] = {}

    for idx, match in enumerate(matches):
        case_id = int(match.group("case_id"))
        block_start = match.end()
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        context_marker = "[问题上下文 / 当前函数上下文]"
        fallback_context_marker = "[原始上下文]"
        answer_marker = "[答案]"
        context_start = block.find(context_marker)
        context_marker_len = len(context_marker)
        if context_start == -1:
            context_start = block.find(fallback_context_marker)
            context_marker_len = len(fallback_context_marker)
        answer_start = block.find(answer_marker)

        if context_start == -1 or answer_start == -1 or answer_start < context_start:
            cases[case_id] = CaseData(context="", answer="")
            continue

        context = block[context_start + context_marker_len : answer_start].strip()
        answer = strip_trailing_separators(block[answer_start + len(answer_marker) :])
        cases[case_id] = CaseData(context=context, answer=answer)

    return cases


def validate_inputs() -> None:
    missing = [str(DATA_LOG)] if not DATA_LOG.exists() else []
    for spec in GROUPS.values():
        if not spec.log_path.exists():
            missing.append(str(spec.log_path))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def score_line(score: Optional[Score]) -> str:
    if score is None:
        return "<missing score>"
    token_part = ""
    if score.ratio is not None:
        token_part = (
            f", ratio={score.ratio:.4f}, "
            f"orig_tokens={score.orig_tokens}, comp_tokens={score.comp_tokens}"
        )
    return f"ES={score.es:.6f}{token_part}"


def build_top_rows(
    left_scores: Dict[int, Score],
    right_scores: Dict[int, Score],
    metric: str,
    top_k: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case_id in sorted(set(left_scores) & set(right_scores)):
        left_score = left_scores[case_id]
        right_score = right_scores[case_id]
        left_value = getattr(left_score, metric)
        right_value = getattr(right_score, metric)
        rows.append(
            {
                "case_id": case_id,
                "diff": left_value - right_value,
                "left_score": left_score,
                "right_score": right_score,
            }
        )
    return sorted(rows, key=lambda item: item["diff"], reverse=True)[:top_k]


def format_case_entry(
    rank: int,
    row: Dict[str, object],
    metric_label: str,
    left_spec: GroupSpec,
    right_spec: GroupSpec,
    left_compressed_results: Dict[int, str],
    right_compressed_results: Dict[int, str],
    case_data: Dict[int, CaseData],
    max_section_chars: Optional[int],
) -> str:
    case_id = int(row["case_id"])
    diff = float(row["diff"])
    left_score = row["left_score"]
    right_score = row["right_score"]
    assert isinstance(left_score, Score)
    assert isinstance(right_score, Score)

    left_compressed = left_compressed_results.get(
        case_id,
        f"<missing {left_spec.label} compressed result for case{case_id}>",
    )
    right_compressed = right_compressed_results.get(
        case_id,
        f"<missing {right_spec.label} compressed result for case{case_id}>",
    )
    data = case_data.get(case_id, CaseData(context="", answer=""))

    parts = [
        "-" * 100,
        f"Rank {rank} | case{case_id} | {metric_label} diff = {diff:.6f}",
        f"{left_spec.label} ({left_spec.source}): {score_line(left_score)}",
        f"{right_spec.label} ({right_spec.source}): {score_line(right_score)}",
        "",
        f"[{left_spec.label}整体细粒度压缩完后的结果]",
        maybe_truncate(left_compressed, max_section_chars),
        "",
        f"[{right_spec.label}整体细粒度压缩完后的结果]",
        maybe_truncate(right_compressed, max_section_chars),
        "",
        "[问题上下文 / 当前函数上下文]",
        maybe_truncate(data.context, max_section_chars),
        "",
        "[答案]",
        maybe_truncate(data.answer, max_section_chars),
    ]
    return "\n".join(parts)


def generate_report(top_k: int, max_section_chars: Optional[int]) -> str:
    validate_inputs()

    all_scores = {key: parse_scores(spec.log_path) for key, spec in GROUPS.items()}
    all_compressed = {
        key: parse_final_compressed_results(spec) for key, spec in GROUPS.items()
    }
    case_data = parse_data_log(DATA_LOG)

    report: List[str] = []
    report.append("=" * 100)
    report.append("三组 case ES 差距 Top10 统计")
    report.append("=" * 100)
    report.append(f"data.log cases: {len(case_data)}")
    for key, spec in GROUPS.items():
        report.append(
            f"{spec.label}: {spec.source}, log={spec.log_path.name}, "
            f"scores={len(all_scores[key])}, compressed_results={len(all_compressed[key])}"
        )

    for title, left_key, right_key in COMPARISONS:
        left_spec = GROUPS[left_key]
        right_spec = GROUPS[right_key]
        report.append("\n" + "=" * 100)
        report.append(title)
        report.append("=" * 100)

        rows = build_top_rows(
            all_scores[left_key], all_scores[right_key], "es", top_k
        )
        report.append(f"\nTop {top_k} by ES difference ({title})")
        for rank, row in enumerate(rows, 1):
            report.append(
                format_case_entry(
                    rank=rank,
                    row=row,
                    metric_label="ES",
                    left_spec=left_spec,
                    right_spec=right_spec,
                    left_compressed_results=all_compressed[left_key],
                    right_compressed_results=all_compressed[right_key],
                    case_data=case_data,
                    max_section_chars=max_section_chars,
                )
            )

    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare top ES case gaps among c_ff20.log, c_ff6.log, c0.log."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top K cases to print.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--max-section-chars",
        type=int,
        default=0,
        help=(
            "Truncate each large text section to this many characters. "
            "0 means no truncation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_section_chars = args.max_section_chars or None
    report = generate_report(top_k=args.top_k, max_section_chars=max_section_chars)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
