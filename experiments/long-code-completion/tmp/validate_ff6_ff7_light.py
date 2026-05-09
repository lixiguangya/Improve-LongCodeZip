#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for ff6/ff7 without loading model weights."""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_top_case_ids(limit: int = 20) -> list[int]:
    text = (ROOT / "top10_cases.txt").read_text(encoding="utf-8", errors="replace")
    ids: list[int] = []
    for match in re.finditer(r"^Rank\s+\d+\s+\|\s+case(\d+)\s+\|", text, re.M):
        case_id = int(match.group(1))
        if case_id not in ids:
            ids.append(case_id)
        if len(ids) >= limit:
            break
    return ids


def validate_split_code2() -> None:
    import split_code2

    src = textwrap.dedent(
        """
        def g(x):
            value = helper(x)
            # return normalized result for caller
            return value
        """
    ).strip() + "\n"
    tree = split_code2.parse_python_ast(src)
    func = tree.body[0]
    _, stmt_by_ast_id, _ = split_code2.build_stmt_infos_for_function(
        src, src.splitlines(), func
    )
    blocks = split_code2.build_semantic_blocks(
        func, src, src.splitlines(), None, stmt_by_ast_id
    )
    assert blocks, "split_code2 returned no blocks"
    assert any(getattr(block, "split_code2_comment_attached", False) for block in blocks), (
        "split_code2 did not attach the useful return comment"
    )
    for prev, cur in zip(blocks, blocks[1:]):
        assert prev.end_line < cur.start_line, (
            f"overlapping blocks: {prev.start_line}-{prev.end_line} and "
            f"{cur.start_line}-{cur.end_line}"
        )
    print("split_code2_ok", len(blocks))


def validate_hybrid(module_name: str) -> None:
    module = __import__(module_name)
    obj = module.CodeCompressor.__new__(module.CodeCompressor)
    obj.get_token_length = lambda text, add_special_tokens=False: max(
        1, len((text or "").split())
    )
    blocks = [
        "def f(x):",
        "if x is None:\n    return default_value",
        "value = helper(x)",
        "return value",
        "# long unrelated license text",
    ]
    query = "def f(x):\n    return value"
    preserved = obj._select_preserved_completion_blocks(
        blocks, query=query, language="python"
    )
    scores = obj._build_hybrid_completion_block_scores(
        blocks,
        [0.1, 0.2, 0.3, 0.4, 0.0],
        [0, 1, 2, 0, 0],
        query=query,
        language="python",
    )
    selected, info = obj._hybrid_knapsack_block_selection(
        blocks,
        [0.1, 0.2, 0.3, 0.4, 0.0],
        scores,
        [0, 1, 2, 0, 0],
        target_tokens=12,
        preserved_block_indices=preserved,
        language="python",
    )
    assert selected, f"{module_name} selected no blocks"
    assert 0 in selected, f"{module_name} lost function signature"
    assert 3 in selected, f"{module_name} lost return block"
    assert 4 not in selected, f"{module_name} selected low-value comment block"
    assert info.get("total_weight", 0) <= int(info.get("target_weight", 12) * 1.2), (
        f"{module_name} budget escaped: {info}"
    )
    print(module_name, "ok", sorted(selected), info.get("total_weight"), info.get("target_weight"))


def main() -> None:
    logger.remove()
    print("top_case_ids", parse_top_case_ids())
    validate_split_code2()
    validate_hybrid("code_compressor_ff6")
    validate_hybrid("code_compressor_ff7")


if __name__ == "__main__":
    main()
