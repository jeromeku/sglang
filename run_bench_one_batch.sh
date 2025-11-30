MODELID=qwen/qwen3-0.6B
export SGLANG_LOGGING_CONFIG_PATH="sglang-logging.json"
python -m sglang.bench_one_batch --model-path ${MODELID} --load-format dummy
