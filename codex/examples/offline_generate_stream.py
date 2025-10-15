"""
Streaming generation with sgl.Engine (sync wrapper pulls async generator).

Usage:
  python codex/examples/offline_generate_stream.py --model-path meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main(server_args: ServerArgs) -> None:
    llm = sgl.Engine(**dataclasses.asdict(server_args))

    prompt = "Write a short limerick about llamas."
    sampling_params = {"temperature": 0.8, "max_new_tokens": 64}

    for chunk in llm.generate(prompt, sampling_params, stream=True):
        text = chunk.get("text", "")
        if text:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)

