uv run python scripts/run_image_understanding_eval.py \
  --screening-results understanding/screening_result_for_understanding.jsonl \
  --out understanding/image_understanding_results_v6_mtp.jsonl \
  --base-url http://localhost:18000/v1 \
  --model Qwen/Qwen3.6-27B-FP8 \
  --no-thinking \
  --skip-existing
 
# uv run python scripts/run_image_understanding_eval.py \
#   --screening-results understanding/screening_result_for_understanding.jsonl \
#   --out understanding/image_understanding_results_v5.jsonl \
#   --base-url http://localhost:18000/v1 \
#   --model Qwen/Qwen3.6-35B-A3B-FP8 \
#   --no-thinking \
#   --skip-existing
  
# uv run python scripts/run_image_understanding_eval.py \
#   --screening-results understanding/screening_result_for_understanding.jsonl \
#   --out understanding/image_understanding_results_v5.jsonl \
#   --base-url http://localhost:18000/v1 \
#   --model google/gemma-4-26B-A4B-it \
#   --no-thinking \
#   --skip-existing
 
# uv run python scripts/run_image_understanding_eval.py \
#   --screening-results understanding/screening_result_for_understanding.jsonl \
#   --out understanding/image_understanding_results_v5.jsonl \
#   --model gpt-5.4 \
#   --no-thinking \
#   --skip-existing
 
