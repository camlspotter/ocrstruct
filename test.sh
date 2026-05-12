uv run python -m ocrstruct.cli \
  _data/data.pdf \
  --outdir _data/data \
  --chunk-chars 800 \
  --chunk-overlap-chars 200 \
  --lazy \
  --with-image-understanding \
  --image-screening-base-url http://localhost:18000/v1 \
  --image-screening-model Qwen/Qwen3.6-27B-FP8 \
  --image-understanding-base-url http://localhost:18000/v1 \
  --image-understanding-model Qwen/Qwen3.6-27B-FP8 \
  --log-level INFO
