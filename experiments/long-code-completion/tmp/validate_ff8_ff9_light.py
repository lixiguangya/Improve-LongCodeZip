from pathlib import Path
import sys
from types import MethodType


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_dummy(module_name):
    module = __import__(module_name)
    compressor = module.CodeCompressor.__new__(module.CodeCompressor)
    compressor.get_token_length = MethodType(lambda self, text: max(1, len((text or "").split())), compressor)
    return compressor


def validate(module_name, expected_selector):
    compressor = build_dummy(module_name)
    query = "def modify_module(module_path):\n    return module_path"
    blocks = [
        "def modify_module(module_path):",
        "    with open(module_path, 'rb') as f:\n        data = f.read()",
        "    unrelated_cache = build_cache(other_name)",
        "    return module_path",
        "    # copyright license warranty",
    ]
    importances = [1.0, 8.0, 20.0, 4.0, 0.0]
    dependencies = [1.0, 4.0, 20.0, 2.0, 0.0]

    assert compressor._block_query_overlap_score("module_path module_path", query) == 1.0
    preserved = compressor._select_preserved_completion_blocks(blocks, query=query, language="python")
    placeholder = compressor._build_hybrid_completion_block_scores(
        blocks=blocks,
        block_importances=importances,
        block_dependency_counts=dependencies,
        query=query,
        language="python",
    )
    selected, info = compressor._hybrid_knapsack_block_selection(
        blocks=blocks,
        block_importances=importances,
        hybrid_scores=placeholder,
        block_dependency_counts=dependencies,
        target_tokens=20,
        preserved_block_indices=preserved,
        language="python",
    )

    assert info["selector"] == expected_selector
    assert info["objective_names"] == ["ppl_change", "dependency_count", "overlap_count"]
    assert info["solution_count"] >= info["pareto_count"] >= 1
    assert info["total_overlap_count"] >= 1
    assert 0 in selected
    assert 4 not in selected
    print(module_name, "ok", sorted(selected), info["method"], info["total_weight"], info["target_weight"])


if __name__ == "__main__":
    validate("code_compressor_ff8", "maximin")
    validate("code_compressor_ff9", "copeland")
