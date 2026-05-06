import json
from code_compressor import CodeCompressor
from loguru import logger


# ==========================
# 读取 RepoQA JSON
# ==========================
def load_repo_example(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 只取 python 第一个 repo
    repo = data["python"][0]

    repo_name = repo["repo"]
    files = repo["content"]

    code_context = ""

    for path, code in files.items():

        code_context += f"\n# FILE: {path}\n"
        code_context += code
        code_context += "\n"

    return repo_name, code_context


# ==========================
# 构造 RepoQA query
# ==========================
def build_query():

    description = """
Read the repository and find the function that reads the configuration
from pyproject.toml and loads it into the program.
"""

    return description


# ==========================
# 主函数
# ==========================
def main():

    # 读取仓库
    repo_name, code_context = load_repo_example("repoqa_sample.json")

    logger.info(f"Repo: {repo_name}")
    logger.info(f"Code length: {len(code_context)}")

    query = build_query()

    instruction = """
Based on the repository code context and the function description,
retrieve the exact function implementation from the code.
Return the function in a code block.
"""

    # ==========================
    # 初始化压缩器
    # ==========================

    compression_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    compressor = CodeCompressor(compression_model)

    # 计算token
    original_tokens = len(compressor.tokenizer.encode(code_context))

    target_token = 512

    ratio = min(1.0, target_token / original_tokens)

    logger.info(f"Original tokens: {original_tokens}")
    logger.info(f"Compression ratio: {ratio:.4f}")

    # ==========================
    # 压缩代码
    # ==========================

    result = compressor.compress_code_file(
        code=code_context,
        query=query,
        instruction=instruction,
        rate=ratio,
        language="python",
        rank_only=False
    )

    compressed_code = result["compressed_code"]

    logger.info(f"Compressed tokens: {result['compressed_tokens']}")
    logger.info(f"Compression ratio: {result['compression_ratio']:.4f}")

    # ==========================
    # 构造 prompt
    # ==========================

    prompt = f"""
{instruction}

Repository Code Context:

{compressed_code}

Description:

{query}

Answer:
"""

    logger.info("Prompt ready.")
    logger.info(prompt[:1000])


if __name__ == "__main__":
    main()