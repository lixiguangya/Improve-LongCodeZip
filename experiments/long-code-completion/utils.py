import datasets
import editdistance
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import re
from tqdm import tqdm
from typing import Any, Dict, Optional


def compute_ES(target, prediction):
    """Compute edit similarity score"""
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = "\n".join(target_lines)
    prediction_lines = [
        line.strip()
        for line in prediction.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ][: len(target_lines)]
    prediction_str = "\n".join(prediction_lines)

    return (
        1
        - (
            editdistance.eval(target_str, prediction_str)
            / max(len(target_str), len(prediction_str))
        )
    ) * 100


def compute_EM(target, prediction):
    """Compute exact match score"""
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    prediction_lines = [
        line.strip()
        for line in prediction.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ][: len(target_lines)]

    if len(target_lines) != len(prediction_lines):
        return 0
    return int(target_lines == prediction_lines) * 100


def print_case_block(case_id: int, example: Dict[str, Any]) -> None:
    """
    打印单个 case 的原始上下文、问题上下文、答案。
    """
    original_context = str(example.get("context", ""))
    background_context = str(example.get("background_context", ""))
    current_function_context = str(example.get("current_function_context", ""))
    answer_text = str(example.get("gt", ""))

    print("\n" + "=" * 120)
    print(f"case{case_id}:")
    print("=" * 120)

    print("\n[原始上下文]")
    print(original_context)

    print("\n[背景上下文]")
    print(background_context if background_context.strip() else "(空)")

    print("\n[问题上下文 / 当前函数上下文]")
    print(current_function_context if current_function_context.strip() else "(空)")

    print("\n[答案]")
    print(answer_text if answer_text.strip() else "(未找到答案字段)")
    print("=" * 120)


def load_data(
    path="microsoft/LCC_python",
    split="test",
    num_examples=500,
    filter_current_lines_max=50,
    filter_background_tokens_min=5000,
):
    """
    Loads the dataset, processes it to split contexts, filters it based on context lengths,
    and returns the filtered dataset along with the tokenizer used.
    """
    print(f"Loading initial {num_examples} examples from {path} ({split} split)...")
    dataset = datasets.load_dataset(path, split=split)

    # keep 10 times of num_examples for testing
    dataset = dataset.select(range(num_examples * 10))
    original_size = len(dataset)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    print("Tokenizer Qwen/Qwen2.5-Coder-7B-Instruct initialized.")

    print("Splitting context into background and current function...")

    def add_split_context(example):
        background, current_func = split_context_ast(example["context"])
        example["background_context"] = background
        example["current_function_context"] = current_func
        return example

    processed_dataset = dataset.map(add_split_context, num_proc=4)

    filtered_dataset_list = []
    print(
        f"Filtering dataset: Keeping examples where current func lines <= {filter_current_lines_max} and background tokens >= {filter_background_tokens_min}."
    )

    for example in tqdm(processed_dataset):
        curr_ctx = example["current_function_context"]
        bg_ctx = example["background_context"]

        curr_line_count = len(curr_ctx.splitlines())

        bg_token_count = 0
        if bg_ctx and bg_ctx.strip():
            bg_token_count = len(tokenizer.encode(bg_ctx, add_special_tokens=False))

        if (
            curr_line_count <= filter_current_lines_max
            and bg_token_count >= filter_background_tokens_min
        ):
            filtered_dataset_list.append(example)

    filtered_dataset = datasets.Dataset.from_list(filtered_dataset_list)
    if num_examples > len(filtered_dataset):
        selected_dataset = filtered_dataset
    else:
        selected_dataset = filtered_dataset.select(range(num_examples))

    print(
        f"Filtering complete. Original size: {original_size}, Filtered size: {len(filtered_dataset)}. Retaining {min(num_examples, len(filtered_dataset))} examples."
    )

    return selected_dataset, tokenizer


def find_last_func_or_class_start(code_string):
    """
    Finds the starting line of the last top-level function or class definition
    using line-based heuristics, robust to syntax errors.
    Accounts for decorators.
    Returns the 1-based line number or None if not found.
    """
    lines = code_string.splitlines()
    if not lines:
        return None

    last_def_line_index = -1

    for i in range(len(lines) - 1, -1, -1):
        stripped_line = lines[i].lstrip()
        if re.match(r"^(def|async\s+def|class)\s+", stripped_line):
            last_def_line_index = i
            break

    if last_def_line_index != -1:
        start_line_index = last_def_line_index
        for i in range(last_def_line_index - 1, -1, -1):
            stripped_line = lines[i].lstrip()
            if stripped_line.startswith("@"):
                start_line_index = i
            elif stripped_line == "" or stripped_line.startswith("#"):
                continue
            else:
                break
        return start_line_index + 1
    else:
        return None


def split_context_ast(code_string):
    """
    Splits the code context into background and current function/class context using AST.
    """
    lines = code_string.splitlines()
    split_line_1_based = find_last_func_or_class_start(code_string)

    if split_line_1_based is not None and split_line_1_based > 0:
        background_lines = lines[: split_line_1_based - 1]
        current_func_lines = lines[split_line_1_based - 1 :]
        return "\n".join(background_lines), "\n".join(current_func_lines)
    else:
        return "", code_string


def print_all_cases(dataset):
    """
    打印 500 个样本的原始上下文、背景上下文、问题上下文、答案。
    """
    print("\n" + "#" * 120)
    print(f"开始打印样本信息，共 {len(dataset)} 个 case")
    print("#" * 120)

    for idx, example in enumerate(dataset):
        print_case_block(idx, example)


def analyze_dataset(dataset, tokenizer):
    """Analyzes and plots context length distributions, including function counts and token ratios."""
    background_lines = []
    current_func_lines = []
    background_tokens = []
    current_func_tokens = []
    background_func_counts = []
    bg_curr_token_ratios = []

    print(f"\nAnalyzing {len(dataset)} examples...")
    for example in tqdm(dataset):
        bg_ctx = example.get("background_context", "")
        curr_ctx = example.get("current_function_context", "")

        bg_token_count = 0
        curr_token_count = 0
        func_count = 0

        if bg_ctx:
            bg_lines = bg_ctx.splitlines()
            bg_line_count = len(bg_lines)
            background_lines.append(bg_line_count)
            bg_token_count = len(tokenizer.encode(bg_ctx, add_special_tokens=False))
            background_tokens.append(bg_token_count)

            for line in bg_lines:
                if re.match(r"^\s*def\s+", line):
                    func_count += 1
            background_func_counts.append(func_count)

        if curr_ctx:
            curr_line_count = len(curr_ctx.splitlines())
            current_func_lines.append(curr_line_count)
            curr_token_count = len(tokenizer.encode(curr_ctx, add_special_tokens=False))
            current_func_tokens.append(curr_token_count)

        if bg_token_count > 0 and curr_token_count > 0:
            bg_curr_token_ratios.append(bg_token_count / curr_token_count)

    if not any(
        [
            background_lines,
            current_func_lines,
            background_tokens,
            current_func_tokens,
            background_func_counts,
            bg_curr_token_ratios,
        ]
    ):
        print(
            "No data points found for analysis after filtering. Skipping plot generation."
        )
        return

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    tokenizer_name = (
        tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "Tokenizer"
    )
    fig.suptitle(
        f"Context Analysis (Filtered LCC Python Dataset - {len(dataset)} examples, Tokenizer: {tokenizer_name})"
    )

    if background_lines:
        axs[0, 0].hist(background_lines, bins=50, edgecolor="black")
        print(
            f"Background Lines: Min={np.min(background_lines)}, Max={np.max(background_lines)}, Avg={np.mean(background_lines):.2f}, Median={np.median(background_lines)}"
        )
    else:
        axs[0, 0].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, 0].transAxes,
        )
    axs[0, 0].set_title("Background Context (Lines)")
    axs[0, 0].set_ylabel("Count")

    if background_tokens:
        axs[0, 1].hist(background_tokens, bins=50, edgecolor="black")
        print(
            f"Background Tokens: Min={np.min(background_tokens)}, Max={np.max(background_tokens)}, Avg={np.mean(background_tokens):.2f}, Median={np.median(background_tokens)}"
        )
    else:
        axs[0, 1].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, 1].transAxes,
        )
    axs[0, 1].set_title("Background Context (Tokens)")
    axs[0, 1].set_ylabel("Count")

    if background_func_counts:
        max_funcs = np.max(background_func_counts)
        bins = min(50, max(1, int(max_funcs) + 1))
        axs[1, 0].hist(background_func_counts, bins=bins, edgecolor="black")
        print(
            f"Background Func Count: Min={np.min(background_func_counts)}, Max={max_funcs}, Avg={np.mean(background_func_counts):.2f}, Median={np.median(background_func_counts)}"
        )
    else:
        axs[1, 0].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[1, 0].transAxes,
        )
    axs[1, 0].set_title("Background Function Count")
    axs[1, 0].set_ylabel("Count")

    if bg_curr_token_ratios:
        axs[1, 1].hist(bg_curr_token_ratios, bins=50, edgecolor="black")
        print(
            f"BG/Current Token Ratio: Min={np.min(bg_curr_token_ratios):.2f}, Max={np.max(bg_curr_token_ratios):.2f}, Avg={np.mean(bg_curr_token_ratios):.2f}, Median={np.median(bg_curr_token_ratios):.2f}"
        )
        axs[1, 1].set_title("BG/Current Token Ratio")
    else:
        axs[1, 1].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[1, 1].transAxes,
        )
    axs[1, 1].set_ylabel("Count")

    if current_func_lines:
        axs[2, 0].hist(current_func_lines, bins=50, edgecolor="black")
        print(
            f"Current Func Lines: Min={np.min(current_func_lines)}, Max={np.max(current_func_lines)}, Avg={np.mean(current_func_lines):.2f}, Median={np.median(current_func_lines)}"
        )
    else:
        axs[2, 0].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[2, 0].transAxes,
        )
    axs[2, 0].set_title("Current Function Context (Lines)")
    axs[2, 0].set_xlabel("Number of Lines")
    axs[2, 0].set_ylabel("Count")

    if current_func_tokens:
        axs[2, 1].hist(current_func_tokens, bins=50, edgecolor="black")
        print(
            f"Current Func Tokens: Min={np.min(current_func_tokens)}, Max={np.max(current_func_tokens)}, Avg={np.mean(current_func_tokens):.2f}, Median={np.median(current_func_tokens)}"
        )
    else:
        axs[2, 1].text(
            0.5,
            0.5,
            "No Data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[2, 1].transAxes,
        )
    axs[2, 1].set_title("Current Function Context (Tokens)")
    axs[2, 1].set_xlabel("Number of Tokens")
    axs[2, 1].set_ylabel("Count")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("context_analysis_distributions_filtered.png")
    print("Saved figure to context_analysis_distributions_filtered.png")


if __name__ == "__main__":
    filtered_dataset, tokenizer = load_data(
        num_examples=500, filter_current_lines_max=50, filter_background_tokens_min=5000
    )

    # 先把 500 个 case 全部打印出来
    print_all_cases(filtered_dataset)

    # 再做统计和画图
    analyze_dataset(filtered_dataset, tokenizer)
