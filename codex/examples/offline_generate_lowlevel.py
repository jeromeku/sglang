"""
Drive the low-level offline engine directly without sgl.Engine.

This example uses internal helpers to launch TokenizerManager + Scheduler + Detokenizer,
then constructs GenerateReqInput and consumes the async generator.

Usage:
  python codex/examples/offline_generate_lowlevel.py --model-path meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import asyncio

from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.managers.io_struct import GenerateReqInput


def main(server_args: ServerArgs) -> None:
    # Allocate ports and launch subprocesses similarly to Engine.__init__
    port_args = PortArgs.init_new(server_args)
    tokenizer_manager, template_manager, sched_info = _launch_subprocesses(server_args, port_args)

    # Build a non-stream request
    obj = GenerateReqInput(
        text="Hello, my name is",
        sampling_params={"temperature": 0.7, "max_new_tokens": 64},
        stream=False,
    )
    gen = tokenizer_manager.generate_request(obj, request=None)
    out = asyncio.get_event_loop().run_until_complete(gen.__anext__())
    print("Non-streaming:", out["text"])  # { text, output_ids, meta_info }

    # Streaming request
    obj2 = GenerateReqInput(
        text="Write a short limerick about llamas.",
        sampling_params={"max_new_tokens": 64},
        stream=True,
    )
    agen = tokenizer_manager.generate_request(obj2, request=None)
    while True:
        try:
            chunk = asyncio.get_event_loop().run_until_complete(agen.__anext__())
            print(chunk.get("text", ""), end="", flush=True)
        except StopAsyncIteration:
            break
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)

