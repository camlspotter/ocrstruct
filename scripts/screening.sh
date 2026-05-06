# uv run python scripts/run_image_screening_eval.py \
#   --eval-set sample_images.imageref.json \
#   --out screening/image_eval_results_v2.jsonl \
#   --model gpt-5-nano \
#   --model gpt-5-mini \
#   --model gpt-5 \
#   --model gpt-5.2 \
#   --model gpt-5.4 \
#   --model gpt-5.4-mini \
#   --model gpt-5.4-nano

uv run python scripts/run_image_screening_eval.py \
  --eval-set sample_images.imageref.json \
  --out screening/image_eval_results_new.jsonl \
  --base-url http://localhost:18000/v1 \
  --no-thinking \
  --skip-existing \
  --model google/gemma-4-26B-A4B-it

# uv run python scripts/run_image_screening_eval.py \
#   --eval-set sample_images.imageref.json \
#   --out screening/image_eval_results_local.jsonl \
#   --base-url http://localhost:18000/v1 \
#   --skip-existing \
#   --no-thinking \
#   --model Qwen/Qwen3.6-27B-FP8
 
