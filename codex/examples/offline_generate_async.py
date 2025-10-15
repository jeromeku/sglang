"""
Asynchronous offline generation with sgl.Engine.

Usage:
  python -m asyncio codex/examples/offline_generate_async.py --model-path meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs


async def main_async(server_args: ServerArgs) -> None:
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    prompt = "List three applications of graph neural networks:"
    sampling_params = {"temperature": 0.2, "max_new_tokens": 64}

    # Non-streaming
    out = await llm.async_generate(prompt, sampling_params)
    print("Non-streaming:", out["text"])  # out keys: text, output_ids, meta_info

    # Streaming
    agen = await llm.async_generate(prompt, sampling_params, stream=True)
    async for chunk in agen:
        text = chunk.get("text", "")
        if text:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    asyncio.run(main_async(server_args))

