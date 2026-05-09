from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path
from typing import List

from utils import load_data


BASE_DIR = Path(__file__).resolve().parents[1]
TOP10_PATH = BASE_DIR / "top10_cases.txt"


def parse_top10_case_ids(path: Path) -> List[int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    ids: List[int] = []
    for match in re.finditer(r"^Rank\s+\d+\s+\|\s+case(\d+)\s+\|", text, re.M):
        case_id = int(match.group(1))
        if case_id not in ids:
            ids.append(case_id)
    return ids


def parse_case_list(value: str) -> List[int]:
    if not value:
        return []
    ids: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test ff4/ff5 compression on selected top10 cases without generation."
    )
    parser.add_argument(
        "--modules",
        default="code_compressor_ff4,code_compressor_ff5",
        help="Comma-separated compressor modules.",
    )
    parser.add_argument(
        "--cases",
        default="175,494,231,225,45,165,266,482",
        help="Comma-separated filtered case ids. Empty means all ids from top10_cases.txt.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases after parsing.")
    parser.add_argument("--target-token", type=int, default=2000)
    parser.add_argument("--fine-ratio", type=float, default=0.8)
    parser.add_argument(
        "--compression-model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4",
    )
    args = parser.parse_args()

    case_ids = parse_case_list(args.cases)
    if not case_ids:
        case_ids = parse_top10_case_ids(TOP10_PATH)
    if args.limit > 0:
        case_ids = case_ids[: args.limit]
    if not case_ids:
        raise SystemExit("No cases selected.")

    dataset_size = max(case_ids) + 1
    dataset, _ = load_data(num_examples=dataset_size)
    instruction = "Complete the following code function given the context."

    for module_name in [m.strip() for m in args.modules.split(",") if m.strip()]:
        module = importlib.import_module(module_name)
        compressor = module.CodeCompressor(args.compression_model)
        print("\n" + "=" * 100)
        print(f"module={module_name}")
        print("=" * 100)
        for case_id in case_ids:
            example = dataset[case_id]
            background = str(example.get("background_context", ""))
            query = str(example.get("current_function_context", ""))
            result = compressor.compress_code_file(
                background,
                query=query,
                instruction=instruction,
                target_token=args.target_token,
                language="python",
                rank_only=False,
                fine_ratio=args.fine_ratio,
                fine_grained_importance_method="conditional_ppl",
                min_lines_for_fine_grained=5,
                importance_beta=0.5,
                use_knapsack=True,
            )
            original_tokens = int(result.get("original_tokens", 0) or 0)
            compressed_tokens = int(result.get("compressed_tokens", 0) or 0)
            ratio = original_tokens / max(1, compressed_tokens)
            print(
                f"case{case_id}: original_tokens={original_tokens}, "
                f"compressed_tokens={compressed_tokens}, ratio={ratio:.4f}"
            )


if __name__ == "__main__":
    main()
