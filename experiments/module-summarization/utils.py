import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from loguru import logger
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoTokenizer
from tqdm import tqdm


def truncate_text(text, max_len=512):
    """Helper function to truncate long text for logging."""
    if len(text) <= max_len:
        return text
    return text[:max_len//2] + "\n...\n" + text[-max_len//2:]


def load_dataset_samples(
    dataset_name="JetBrains-Research/lca-module-summarization",
    split="test",
    max_examples=None,
    hf_api_key=None,
    max_tokens=32768,
    min_tokens=1024
):
    """
    Load dataset samples with optional limiting and filtering of long examples.
    After filtering, print:
      - Avg. Context Len.
      - Avg. Ground Truth Len.
    """
    dataset = load_dataset(dataset_name, token=hf_api_key)[split]

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Filter out examples with extremely long/short code
    if max_tokens > 0:
        filtered_indices = []
        skipped_count_long = 0
        skipped_count_short = 0

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

        # 用于统计保留下来的样本平均长度
        total_context_tokens = 0
        total_ground_truth_tokens = 0
        kept_count = 0

        for i, row in enumerate(tqdm(dataset, desc="Filtering long examples")):
            code = row["relevant_code_context"]
            gt = row.get("target_text", "")

            if len(code) > max_tokens * 10:
                logger.warning(
                    f"Skipping example {i} because it exceeds {max_tokens*10} characters "
                    f"({len(code)}/{max_tokens*10})"
                )
                skipped_count_long += 1
                continue

            context_tokens = tokenizer.encode(code, truncation=False)
            gt_tokens = tokenizer.encode(gt, truncation=False) if gt else []

            if len(context_tokens) > max_tokens:
                logger.warning(
                    f"Skipping example {i} because it exceeds {max_tokens} tokens "
                    f"({len(context_tokens)}/{max_tokens})"
                )
                skipped_count_long += 1
                continue

            if len(context_tokens) < min_tokens:
                logger.warning(
                    f"Skipping example {i} because it is too short "
                    f"({len(context_tokens)}/{min_tokens})"
                )
                skipped_count_short += 1
                continue

            filtered_indices.append(i)

            # 统计保留样本的平均 token 长度
            total_context_tokens += len(context_tokens)
            total_ground_truth_tokens += len(gt_tokens)
            kept_count += 1

        if skipped_count_long > 0:
            logger.info(f"Skipped {skipped_count_long} examples that exceeded token limit of {max_tokens}")
        if skipped_count_short > 0:
            logger.info(f"Skipped {skipped_count_short} examples that are too short ({min_tokens} tokens)")

        dataset = dataset.select(filtered_indices)

        # 打印平均 token 长度
        avg_context_len = total_context_tokens / kept_count if kept_count > 0 else 0.0
        avg_ground_truth_len = total_ground_truth_tokens / kept_count if kept_count > 0 else 0.0

        logger.info(f"Avg. Context Len. = {avg_context_len:.2f}")
        logger.info(f"Avg. Ground Truth Len. = {avg_ground_truth_len:.2f}")

    return dataset


def get_actual_token_lengths(dataset, tokenizer, output_path="./analysis"):
    """
    Calculate actual token lengths using the specified tokenizer.

    Args:
        dataset: The dataset containing code samples
        tokenizer: The tokenizer to use for counting tokens
        output_path: Path to save analysis results and plots

    Returns:
        Dict with statistics about token lengths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Extract actual token counts
    token_lengths = []

    # Print intent for each file
    logger.info("\nIntent for each example:")
    logger.info("======================")
    
    for i, row in enumerate(tqdm(dataset, desc="Calculating token lengths")):
        code = row['relevant_code_context']
        tokens = tokenizer.encode(code, truncation=False) if hasattr(tokenizer, 'encode') else []
        token_len = len(tokens)
        token_lengths.append(token_len)
        
        # Print the intent for each file
        docfile_name = row.get('docfile_name', f'file_{i}')
        intent = row.get('intent', 'unknown')
        logger.info(f"  Example {i}: {docfile_name} - Intent: {intent} - Token Length: {token_len}")

    # Calculate statistics
    stats = {
        'min': min(token_lengths),
        'max': max(token_lengths),
        'mean': np.mean(token_lengths),
        'median': np.median(token_lengths),
        'p90': np.percentile(token_lengths, 90),
        'p95': np.percentile(token_lengths, 95),
        'p99': np.percentile(token_lengths, 99),
    }

    # Plot token length histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, alpha=0.7)
    plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=1, label=f"Mean: {stats['mean']:.0f}")
    plt.axvline(stats['median'], color='green', linestyle='dashed', linewidth=1, label=f"Median: {stats['median']:.0f}")
    plt.axvline(stats['p90'], color='orange', linestyle='dashed', linewidth=1, label=f"90th %: {stats['p90']:.0f}")
    plt.axvline(stats['p95'], color='purple', linestyle='dashed', linewidth=1, label=f"95th %: {stats['p95']:.0f}")
    plt.title('Actual Code Length Distribution (Tokens)')
    plt.xlabel('Tokens')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'code_length_actual_tokens.png'))

    # Save statistics to a text file
    with open(os.path.join(output_path, 'token_length_stats.txt'), 'w') as f:
        f.write("Token Length Statistics\n")
        f.write("=====================\n\n")

        for key, value in stats.items():
            f.write(f"  {key}: {value:.2f}\n")

    # Return the statistics for further use
    return stats


if __name__ == "__main__":
    
    dataset = load_dataset_samples(dataset_name="JetBrains-Research/lca-module-summarization", split="test", max_examples=1000)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    token_stats = get_actual_token_lengths(dataset, tokenizer, "./analysis")

    # Print summary of findings using logger
    logger.info("\nSummary of Code Length Analysis:")
    logger.info("================================")
    logger.info(f"Number of examples analyzed: {len(dataset)}")

    logger.info("\nActual token-based statistics:")
    logger.info(f"  Mean length: {token_stats['mean']:.0f} tokens")
    logger.info(f"  Median length: {token_stats['median']:.0f} tokens")
    logger.info(f"  95th percentile: {token_stats['p95']:.0f} tokens")