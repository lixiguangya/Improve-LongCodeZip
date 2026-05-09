#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and score completions from failed compression logs.

Use case:
    c_ff4.log / c_ff5.log finished compression and printed 500 blocks titled
    "整体细粒度压缩完后的结果", but crashed before vLLM generation. This script
    treats those log blocks as a recovered compressed-background cache:

    compressed_background_from_log + "\n\n" + current_function_context

Then it follows main.py's generation/scoring logic to produce JSONL outputs,
average ES/EM, and prompt-level compression ratios.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from extract_failed_log_results import build_case_records


LOCAL_QWEN_7B_PATH = (
    "/home/nwpu_wyh/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-7B-Instruct/"
    "snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
)

DEFAULT_FAILED_LOGS = ["c_ff5.log"]
DEFAULT_EXPECTED_CASES = 500
DEFAULT_RECOVERED_METHOD_DIR = "method_recovered_failed_logs"


@dataclass
class DataExample:
    case_id: int
    background_context: str
    current_function_context: str
    gt: str
    context: str = ""


@dataclass
class PreparedCase:
    id: int
    gt: str
    original_background_context: str
    original_current_function_context: str
    compressed_background_context: str
    language: str
    original_tokens: int
    compressed_tokens: int
    ratio: float
    log_background_original_tokens: int | None
    log_background_compressed_tokens: int | None
    log_background_ratio: float | None
    fine_background_original_tokens: int | None
    fine_background_compressed_tokens: int | None
    fine_background_ratio: float | None


def configure_environment(cuda_visible_devices: str | None) -> None:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_local_model_path(model_name_or_path: str) -> str:
    if model_name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct":
        if os.path.exists(LOCAL_QWEN_7B_PATH):
            return LOCAL_QWEN_7B_PATH
        raise FileNotFoundError(
            f"本地模型路径不存在: {LOCAL_QWEN_7B_PATH}"
        )
    return model_name_or_path


def local_files_only_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "1") == "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE", "1"
    ) == "1"


def load_main_runtime_config() -> tuple[Any | None, dict[str, Any]]:
    """Read defaults from main.py so this recovery script matches the experiment entry."""
    try:
        import main as main_module

        signature = inspect.signature(main_module.evaluate_completion)
        defaults = {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        return main_module, defaults
    except Exception as exc:
        print(f"[WARN] failed to import main.py config, using fallback defaults: {exc}")
        return None, {}


def main_default(
    main_defaults: dict[str, Any],
    name: str,
    fallback: Any,
) -> Any:
    value = main_defaults.get(name, fallback)
    return fallback if value is inspect.Parameter.empty else value


def default_output_dir_from_main(main_defaults: dict[str, Any]) -> str:
    result_dir = str(
        main_default(main_defaults, "result_dir", "results1/completion_baselines")
    )
    return str(Path(result_dir) / DEFAULT_RECOVERED_METHOD_DIR)


def resolve_model_path_with_main(
    main_module: Any | None,
    model_name_or_path: str,
) -> str:
    if main_module is not None and hasattr(main_module, "resolve_local_model_path"):
        return main_module.resolve_local_model_path(model_name_or_path)
    return resolve_local_model_path(model_name_or_path)


def local_files_only_with_main(main_module: Any | None) -> bool:
    if main_module is not None and hasattr(main_module, "local_files_only_enabled"):
        return bool(main_module.local_files_only_enabled())
    return local_files_only_enabled()


def safe_model_filename(model_name_or_path: str) -> str:
    return (
        model_name_or_path.replace("/", "_slash_")
        .replace("\\", "_slash_")
        .replace(":", "_")
    )


def is_wide_delimiter(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 20 and set(stripped) == {"="}


def strip_blank_edges(lines: list[str]) -> str:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def load_examples_from_data_log(data_log: Path) -> list[DataExample]:
    """Parse data.log produced by utils.print_all_cases(dataset)."""
    field_map = {
        "[原始上下文]": "context",
        "[背景上下文]": "background_context",
        "[问题上下文 / 当前函数上下文]": "current_function_context",
        "[答案]": "gt",
    }
    case_re = re.compile(r"^case(?P<case_id>\d+):$")

    examples: list[DataExample] = []
    current: dict[str, Any] | None = None
    current_field: str | None = None

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return
        examples.append(
            DataExample(
                case_id=int(current["case_id"]),
                context=strip_blank_edges(current.get("context", [])),
                background_context=strip_blank_edges(
                    current.get("background_context", [])
                ),
                current_function_context=strip_blank_edges(
                    current.get("current_function_context", [])
                ),
                gt=strip_blank_edges(current.get("gt", [])),
            )
        )
        current = None

    with data_log.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if match := case_re.match(stripped):
                finish_current()
                current = {
                    "case_id": int(match.group("case_id")),
                    "context": [],
                    "background_context": [],
                    "current_function_context": [],
                    "gt": [],
                }
                current_field = None
                continue

            if current is None:
                continue

            if stripped in field_map:
                current_field = field_map[stripped]
                continue

            if is_wide_delimiter(line):
                continue

            if current_field:
                current[current_field].append(line)

    finish_current()
    examples.sort(key=lambda item: item.case_id)
    return examples


def load_examples_from_dataset(
    dataset_path: str,
    dataset_split: str,
    num_examples: int,
    filter_current_lines_max: int,
    filter_background_tokens_min: int,
) -> list[DataExample]:
    from utils import load_data

    dataset, _ = load_data(
        path=dataset_path,
        split=dataset_split,
        num_examples=num_examples,
        filter_current_lines_max=filter_current_lines_max,
        filter_background_tokens_min=filter_background_tokens_min,
    )
    examples: list[DataExample] = []
    for i, example in enumerate(dataset):
        examples.append(
            DataExample(
                case_id=int(example.get("id", i)),
                context=str(example.get("context", "")),
                background_context=str(example.get("background_context", "")),
                current_function_context=str(
                    example.get("current_function_context", "")
                ),
                gt=str(example.get("gt", "")),
            )
        )
    return examples


def load_examples(args: argparse.Namespace) -> list[DataExample]:
    data_log = Path(args.data_log)
    if args.data_source == "data_log" or (
        args.data_source == "auto" and data_log.exists()
    ):
        examples = load_examples_from_data_log(data_log)
        if not examples:
            raise RuntimeError(f"No examples parsed from data log: {data_log}")
        print(f"Loaded {len(examples)} examples from {data_log}")
        return examples

    examples = load_examples_from_dataset(
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        num_examples=args.num_examples,
        filter_current_lines_max=args.filter_current_lines_max,
        filter_background_tokens_min=args.filter_background_tokens_min,
    )
    print(f"Loaded {len(examples)} examples from dataset {args.dataset_path}")
    return examples


def prepare_cases_from_log(
    log_path: Path,
    examples: list[DataExample],
    tokenizer: Any,
    max_cases: int | None,
    expected_cases: int,
) -> tuple[list[str], list[PreparedCase], dict[str, Any]]:
    records, parse_counts = build_case_records(log_path)
    compressed_records = [r for r in records if r.compressed_context is not None]
    if len(compressed_records) < expected_cases and max_cases is None:
        raise RuntimeError(
            f"{log_path} only has {len(compressed_records)} compressed contexts; "
            f"expected {expected_cases}."
        )

    n_cases = min(len(compressed_records), len(examples), expected_cases)
    if max_cases is not None:
        n_cases = min(n_cases, max_cases)

    prompts: list[str] = []
    original_data: list[PreparedCase] = []
    total_original_tokens = 0
    total_compressed_tokens = 0

    for i in tqdm(range(n_cases), desc=f"Preparing prompts from {log_path.name}"):
        log_record = records[i]
        example = examples[i]
        compressed_bg = log_record.compressed_context or ""
        current_ctx = example.current_function_context
        original_prompt_text = example.background_context + "\n\n" + current_ctx
        compressed_prompt_text = (compressed_bg + "\n\n" + current_ctx).strip()

        original_tokens = len(tokenizer.encode(original_prompt_text))
        compressed_tokens = len(tokenizer.encode(compressed_prompt_text))
        ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0.0

        total_original_tokens += original_tokens
        total_compressed_tokens += compressed_tokens
        prompts.append(compressed_prompt_text)
        original_data.append(
            PreparedCase(
                id=i,
                gt=example.gt,
                original_background_context=example.background_context,
                original_current_function_context=current_ctx,
                compressed_background_context=compressed_bg,
                language="python",
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                ratio=ratio,
                log_background_original_tokens=log_record.original_tokens,
                log_background_compressed_tokens=log_record.compressed_tokens,
                log_background_ratio=log_record.compression_ratio,
                fine_background_original_tokens=log_record.fine_original_tokens,
                fine_background_compressed_tokens=log_record.fine_compressed_tokens,
                fine_background_ratio=log_record.fine_compression_ratio,
            )
        )

    summary = {
        "log": str(log_path),
        "parse_counts": parse_counts,
        "num_prepared_cases": n_cases,
        "total_original_tokens_all": total_original_tokens,
        "total_compressed_tokens_all": total_compressed_tokens,
        "compression_ratio_total_original_over_compressed": (
            total_original_tokens / total_compressed_tokens
            if total_compressed_tokens > 0
            else 0
        ),
        "avg_case_compression_ratio": (
            sum(item.ratio for item in original_data) / len(original_data)
            if original_data
            else 0
        ),
    }
    return prompts, original_data, summary


def save_prompt_cache(
    cache_path: Path,
    metadata: dict[str, Any],
    prompts: list[str],
    original_data: list[PreparedCase],
    summary: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "__metadata__": metadata,
                    "total_original_tokens_all": summary["total_original_tokens_all"],
                    "total_compressed_tokens_all": summary[
                        "total_compressed_tokens_all"
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for prompt, orig in zip(prompts, original_data):
            f.write(
                json.dumps(
                    {"prompt": prompt, "original_data": asdict(orig)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(tmp_path, cache_path)


def compute_scores(gt: str, output: str) -> tuple[float, int]:
    from utils import compute_EM, compute_ES

    return compute_ES(gt, output), compute_EM(gt, output)


def generate_batch_outputs(
    llm: Any,
    batch_prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=max_new_tokens,
    )
    batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
    return [item.outputs[0].text for item in batch_outputs]


def load_existing_result_count(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    count = 0
    with output_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_result_line(
    f_out: Any,
    original: PreparedCase,
    prompt: str,
    output: str,
) -> tuple[float, int, bool]:
    es = 0.0
    em = 0
    scored = False
    if output != "ERROR_GENERATING" and original.gt is not None:
        try:
            es, em = compute_scores(original.gt, output)
            scored = True
        except Exception as exc:  # keep batch evaluation moving
            print(f"[WARN] scoring failed for case {original.id}: {exc}")

    result = {
        **asdict(original),
        "prompt": prompt,
        "output": output,
        "es": es,
        "em": em,
    }
    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    f_out.flush()
    print(
        f"Case {original.id} => es={es}, em={em}, "
        f"ratio={original.ratio:.4f}, "
        f"orig_tokens={original.original_tokens}, "
        f"comp_tokens={original.compressed_tokens}"
    )
    return es, em, scored


def score_result_file(output_path: Path) -> dict[str, Any]:
    total_es = 0.0
    total_em = 0.0
    valid_scores = 0
    total_cases = 0
    total_original_tokens = 0
    total_compressed_tokens = 0

    with output_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            total_cases += 1
            total_original_tokens += int(item.get("original_tokens", 0) or 0)
            total_compressed_tokens += int(item.get("compressed_tokens", 0) or 0)
            if item.get("output") != "ERROR_GENERATING":
                total_es += float(item.get("es", 0) or 0)
                total_em += float(item.get("em", 0) or 0)
                valid_scores += 1

    return {
        "num_examples_scored": valid_scores,
        "num_examples_total": total_cases,
        "average_es": total_es / valid_scores if valid_scores > 0 else 0,
        "average_em": total_em / valid_scores if valid_scores > 0 else 0,
        "total_original_tokens_all": total_original_tokens,
        "total_compressed_tokens_all": total_compressed_tokens,
        "compression_ratio_total_original_over_compressed": (
            total_original_tokens / total_compressed_tokens
            if total_compressed_tokens > 0
            else 0
        ),
    }


def evaluate_one_log(
    log_path: Path,
    examples: list[DataExample],
    tokenizer: Any,
    model_load_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    log_stem = log_path.stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{log_stem}.jsonl"
    score_path = out_dir / f"{log_stem}-SCORES.json"
    prompt_cache_path = out_dir / f"{log_stem}-prepared-prompts-cache.jsonl"

    prompts, original_data, prepare_summary = prepare_cases_from_log(
        log_path=log_path,
        examples=examples,
        tokenizer=tokenizer,
        max_cases=args.max_cases,
        expected_cases=args.expected_cases,
    )

    metadata = {
        "source_log": str(log_path),
        "main_config_source": "main.py:evaluate_completion defaults",
        "model_name": args.model_name,
        "model_load_path": model_load_path,
        "data_source": args.data_source,
        "data_log": args.data_log,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "num_examples": args.num_examples,
        "max_cases": args.max_cases,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "algorithm": "failed-log compressed background + current_function_context",
    }
    save_prompt_cache(prompt_cache_path, metadata, prompts, original_data, prepare_summary)

    print(
        f"[{log_path.name}] prepared {len(prompts)} prompts; "
        f"prompt token ratio="
        f"{prepare_summary['compression_ratio_total_original_over_compressed']:.4f}; "
        f"cache={prompt_cache_path}"
    )

    if args.prepare_only:
        score_data = {
            **metadata,
            "method": "code_compressor_recovered_from_log",
            "num_examples_scored": 0,
            "num_examples_total": len(original_data),
            "average_es": None,
            "average_em": None,
            **prepare_summary,
            "raw_output_path": str(output_path),
            "prompt_cache_path": str(prompt_cache_path),
            "note": "prepare_only=True, generation/scoring not run.",
        }
        score_path.write_text(
            json.dumps(score_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return score_data

    existing_count = load_existing_result_count(output_path) if args.resume else 0
    if existing_count > len(prompts):
        raise RuntimeError(
            f"{output_path} has {existing_count} lines, more than current prompt count {len(prompts)}. "
            "Use --no-resume or a new --output-dir."
        )

    if existing_count < len(prompts):
        from vllm import LLM

        print(
            f"[{log_path.name}] initializing vLLM from {model_load_path}; "
            f"resume_existing={existing_count}"
        )
        llm = LLM(
            model=model_load_path,
            tokenizer=model_load_path,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=1,
            max_model_len=args.max_model_len,
        )
        mode = "a" if existing_count > 0 else "w"
        with output_path.open(mode, encoding="utf-8") as f_out:
            start_index = existing_count
            for batch_start in tqdm(
                range(start_index, len(prompts), args.batch_size),
                desc=f"Generating {log_path.name}",
            ):
                batch_prompts = prompts[batch_start : batch_start + args.batch_size]
                try:
                    batch_outputs = generate_batch_outputs(
                        llm,
                        batch_prompts,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    print(
                        f"[ERROR] generation failed for batch starting at {batch_start}: {exc}"
                    )
                    batch_outputs = ["ERROR_GENERATING"] * len(batch_prompts)

                for offset, output in enumerate(batch_outputs):
                    idx = batch_start + offset
                    write_result_line(f_out, original_data[idx], prompts[idx], output)

        del llm
        try:
            import gc
            import torch

            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
    else:
        print(f"[{log_path.name}] existing output already has {existing_count} cases.")

    score_summary = score_result_file(output_path)
    score_data = {
        **metadata,
        "method": "code_compressor_recovered_from_log",
        **score_summary,
        "parse_counts": prepare_summary.get("parse_counts", {}),
        "avg_case_compression_ratio": prepare_summary["avg_case_compression_ratio"],
        "raw_output_path": str(output_path),
        "score_output_path": str(score_path),
        "prompt_cache_path": str(prompt_cache_path),
    }
    score_path.write_text(
        json.dumps(score_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Method code_compressor_recovered_from_log({log_path.name}): "
        f"Avg ES = {score_data['average_es']:.2f}, "
        f"Avg EM = {score_data['average_em']:.2f} "
        f"({score_data['num_examples_scored']}/{score_data['num_examples_total']} scored)"
    )
    print(
        f"Total token stats: original={score_data['total_original_tokens_all']}, "
        f"compressed={score_data['total_compressed_tokens_all']}, "
        "original/compressed="
        f"{score_data['compression_ratio_total_original_over_compressed']:.4f}"
    )
    print(f"Scores saved to {score_path}")
    return score_data


def parse_args() -> argparse.Namespace:
    main_module, main_defaults = load_main_runtime_config()
    # main.py currently keeps num_examples small as a fire default, while these
    # failed logs contain 500 completed compression cases. Keep all runtime
    # settings aligned with main.py, but recover 500 cases unless the user
    # explicitly overrides it.
    main_num_examples = int(main_default(main_defaults, "num_examples", 0) or 0)
    default_num_examples = max(main_num_examples, DEFAULT_EXPECTED_CASES)
    default_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    parser = argparse.ArgumentParser(
        description=(
            "Recover compressed prompts from failed c_ff*.log files, generate completions, "
            "and compute ES/EM like main.py."
        )
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        default=DEFAULT_FAILED_LOGS,
        help="Log files. Defaults to c_ff4.log c_ff5.log.",
    )
    parser.add_argument(
        "--data-source",
        choices=["auto", "data_log", "dataset"],
        default="dataset",
        help="Default is dataset to match main.py load_data().",
    )
    parser.add_argument("--data-log", default="data.log")
    parser.add_argument(
        "--dataset-path",
        default=main_default(main_defaults, "dataset_path", "microsoft/LCC_python"),
    )
    parser.add_argument(
        "--dataset-split",
        default=main_default(main_defaults, "dataset_split", "test"),
    )
    parser.add_argument("--num-examples", type=int, default=default_num_examples)
    parser.add_argument("--expected-cases", type=int, default=DEFAULT_EXPECTED_CASES)
    parser.add_argument("--max-cases", type=int, default=None, help="For smoke tests; default uses 500.")
    parser.add_argument(
        "--filter-current-lines-max",
        type=int,
        default=main_default(main_defaults, "filter_current_lines_max", 50),
    )
    parser.add_argument(
        "--filter-background-tokens-min",
        type=int,
        default=main_default(main_defaults, "filter_background_tokens_min", 5000),
    )
    parser.add_argument(
        "--model-name",
        default=main_default(main_defaults, "model_name", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    )
    parser.add_argument("--output-dir", default=default_output_dir_from_main(main_defaults))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=main_default(main_defaults, "max_new_tokens", 128),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=main_default(main_defaults, "batch_size", 16),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=main_default(main_defaults, "tensor_parallel_size", 1),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=main_default(main_defaults, "gpu_memory_utilization", 0.6),
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--cuda-visible-devices", default=default_cuda_devices)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=main_default(main_defaults, "trust_remote_code", True),
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only extract/build prompts and token stats.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=main_default(main_defaults, "use_prompt_cache", True),
    )
    args = parser.parse_args()
    args._main_module = main_module
    args._main_defaults = main_defaults
    return args


def main() -> int:
    args = parse_args()
    configure_environment(args.cuda_visible_devices)

    from transformers import AutoTokenizer

    model_load_path = resolve_model_path_with_main(args._main_module, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only_with_main(args._main_module),
    )
    print(
        "Recovered-log evaluation config: "
        f"logs={args.logs}, dataset={args.dataset_path}/{args.dataset_split}, "
        f"num_examples={args.num_examples}, batch_size={args.batch_size}, "
        f"max_new_tokens={args.max_new_tokens}, cuda={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"output_dir={args.output_dir}"
    )
    examples = load_examples(args)

    summaries = []
    for log_item in args.logs:
        summaries.append(
            evaluate_one_log(
                log_path=Path(log_item),
                examples=examples,
                tokenizer=tokenizer,
                model_load_path=model_load_path,
                args=args,
            )
        )

    summary_path = Path(args.output_dir) / "SUMMARY.json"
    summary_path.write_text(
        json.dumps({"logs": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"All summaries saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
