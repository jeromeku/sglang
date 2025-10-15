"""
Minimal synchronous, non-streaming offline batch inference with sgl.Engine.

Usage:
  python codex/examples/offline_generate_sync.py --model-path meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main(server_args: ServerArgs) -> None:
    llm = sgl.Engine(**dataclasses.asdict(server_args))

    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 64}

    outputs = llm.generate(prompts, sampling_params)
    for p, out in zip(prompts, outputs):
        print("---")
        print("Prompt:", p)
        print("Text:", out["text"])  # out keys: text, output_ids, meta_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)

