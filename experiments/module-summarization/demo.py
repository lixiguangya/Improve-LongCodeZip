from code_compressor import CodeCompressor
from loguru import logger
import argparse
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import json


# =============================
# 初始化本地模型（只加载一次）
# =============================
def load_local_model(model_name):

    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    return tokenizer, model


# =============================
# 生成摘要
# =============================
def generate_summary(
    prompt,
    tokenizer=None,
    model=None,
    generation_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_new_tokens=256,
    use_openai=False,
    api_key=None
):

    # =============================
    # OpenAI API
    # =============================
    if use_openai:

        api_key = api_key or os.getenv("OPENAI_API_KEY")

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=generation_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant that writes documentation for Python code."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_new_tokens,
            temperature=0
        )

        generated = response.choices[0].message.content.strip()

        return generated

    # =============================
    # 本地模型
    # =============================

    messages = [
        {
            "role": "system",
            "content": "You are an expert assistant that writes documentation for Python code."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return generated


# =============================
# 主程序
# =============================
if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_openai",
        action="store_true"
    )

    parser.add_argument(
        "--generation_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct"
    )

    args = parser.parse_args()

    # =============================
    # 加载模型
    # =============================
    tokenizer, model = None, None

    if not args.use_openai:
        tokenizer, model = load_local_model(args.generation_model)

    # =============================
    # 读取数据
    # =============================
    with open("data/repoqa_sample.json", "r") as f:
        dataset = json.load(f)

    with open("data/test.py", "r", encoding="utf-8") as f:
        context = f.read()

    sample = dataset[0]
    query = sample["intent"]

    instruction = "Generate documentation based on this code."

    # =============================
    # 初始化压缩器
    # =============================
    logger.info("Initializing CodeCompressor...")

    compression_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    compressor = CodeCompressor(compression_model)

    # =============================
    # 计算 token
    # =============================
    original_tokens = len(compressor.tokenizer.encode(context))

    # 建议提高 token 防止语义丢失
    target_token = 512

    target_ratio = min(1.0, max(0.0, target_token / original_tokens))

    logger.info(
        f"Original tokens={original_tokens}, target tokens={target_token}, ratio={target_ratio:.4f}"
    )

    # =============================
    # Coarse-grained compression
    # =============================
    logger.info("\nTesting Coarse-grained compression...")

    result = compressor.compress_code_file(
        code=context,
        query=query,
        instruction=instruction,
        rate=target_ratio,
        language="python",
        rank_only=True
    )

    compressed_code = result["compressed_code"]

    logger.info(f"Compression ratio: {result['compression_ratio']:.4f}")

    compressed_tokens = len(compressor.tokenizer.encode(compressed_code))

    logger.info(f"Compressed tokens: {compressed_tokens}")

    logger.debug("\nCompressed code preview:\n" + compressed_code[:800])

    # =============================
    # Prompt
    # =============================
    prompt = f"""
You are a software documentation generator.

Read the following Python code and produce a concise description of:

1. The purpose of the module
2. The main classes or functions
3. The key functionality

Code:

{compressed_code}

Documentation:
"""

    summary = generate_summary(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        generation_model=args.generation_model,
        use_openai=args.use_openai
    )

    logger.info(f"\nGenerated Documentation:\n{summary}")

    # =============================
    # Coarse + Fine compression
    # =============================
    logger.info("\nTesting Coarse + Fine-grained compression...")

    result = compressor.compress_code_file(
        code=context,
        query=query,
        instruction=instruction,
        rate=target_ratio,
        language="python",
        rank_only=False,
        fine_ratio=0.5
    )

    compressed_code = result["compressed_code"]

    logger.info(f"Compression ratio: {result['compression_ratio']:.4f}")

    compressed_tokens = len(compressor.tokenizer.encode(compressed_code))

    logger.info(f"Compressed tokens: {compressed_tokens}")

    logger.debug("\nCompressed code preview:\n" + compressed_code[:800])

    prompt = f"""
You are a software documentation generator.

Read the following Python code and produce a concise description of its purpose and functionality.

Code:

{compressed_code}

Documentation:
"""

    summary = generate_summary(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        generation_model=args.generation_model,
        use_openai=args.use_openai
    )

    logger.info(f"\nGenerated Documentation:\n{summary}")