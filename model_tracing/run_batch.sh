#!/bin/bash

export SGLANG_LOGGING_CONFIG_PATH="logging_config.json"
MODEL_ID="Qwen/Qwen3-0.6B"
LOG_LEVEL="DEBUG"
python offline_batch_inference.py --model ${MODEL_ID} --log-level ${LOG_LEVEL}
