# 画像 Screening 評価スクリプトの使い方

この文書では、以下の 2 つのスクリプトの使い方を説明します。

- [scripts/run_image_screening_eval.py](/Users/jun/mocrdown/scripts/run_image_screening_eval.py:1)
- [scripts/summarize_image_screening_eval.py](/Users/jun/mocrdown/scripts/summarize_image_screening_eval.py:1)

目的は、`ImageRef` の評価セットに対して複数モデルで screening を実行し、

- 判定結果
- レイテンシ
- usage
- 推定価格
- エラー

を記録し、あとで比較できるようにすることです。

## 1. 入力データ

評価セットは `ImageRef` の配列を JSON で保存したものを使います。

最も単純な形式はこれです。

```json
[
  {
    "pdf_path": "/path/to/doc.pdf",
    "middle_json_path": "/path/to/middle.json",
    "page_idx": 7,
    "block_index": 8,
    "block_type": "image",
    "image_path": "39189e383a1c930ef1bcaafd520d845452ca2e3fa922f4ed8688ee415eb4bb2e.jpg",
    "caption": "図 1.1.2 外為法と関係法令の体系"
  }
]
```

次の形式でも読み込めます。

```json
{
  "items": [
    {
      "pdf_path": "/path/to/doc.pdf",
      "middle_json_path": "/path/to/middle.json",
      "page_idx": 7,
      "block_index": 8,
      "block_type": "image",
      "image_path": "39189e383a1c930ef1bcaafd520d845452ca2e3fa922f4ed8688ee415eb4bb2e.jpg"
    }
  ]
}
```

## 2. 実行前の準備

OpenAI 互換 API を使う場合は、必要に応じて API キーを設定します。

```bash
export OPENAI_API_KEY=...
```

`--api-key` を明示して渡してもかまいません。

## 3. 評価の実行

基本形は次のとおりです。

```bash
uv run python scripts/run_image_screening_eval.py \
  --eval-set data/image_eval_set.json \
  --out data/image_eval_results.jsonl \
  --model gpt-5-mini
```

複数モデルを比較する場合は `--model` を複数回指定します。

```bash
uv run python scripts/run_image_screening_eval.py \
  --eval-set data/image_eval_set.json \
  --out data/image_eval_results.jsonl \
  --model gpt-5-mini \
  --model gpt-5 \
  --model gpt-4.1-mini
```

OpenAI 互換サーバを使う場合は `--base-url` を指定します。

```bash
uv run python scripts/run_image_screening_eval.py \
  --eval-set data/image_eval_set.json \
  --out data/image_eval_results.jsonl \
  --model some-vlm-model \
  --base-url http://localhost:8000/v1
```

API キーをコマンド引数で渡すこともできます。

```bash
uv run python scripts/run_image_screening_eval.py \
  --eval-set data/image_eval_set.json \
  --out data/image_eval_results.jsonl \
  --model gpt-5-mini \
  --api-key YOUR_API_KEY
```

## 4. 出力ファイル

`--out` で指定したファイルには JSONL 形式で 1 実行 1 行ずつ追記されます。

各行には概ね次の情報が入ります。

- `ref`
- `model`
- `base_url`
- `latency_sec`
- `status`
- `run.result`
- `run.usage`
- `run.price`

失敗した場合は `status.ok` が `false` になり、`status.error` にエラーメッセージが入ります。

## 5. 推定価格

OpenAI 本家の主要モデルについては、スクリプト内に既定の価格表があります。

- `gpt-5`
- `gpt-5-mini`
- `gpt-5-nano`
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4o`
- `gpt-4o-mini`

usage が返ってきた場合は、

- `input_tokens`
- `output_tokens`
- `total_tokens`

を保存し、それに基づいて推定価格を計算します。

注意:

- 価格はあくまで推定です
- OpenAI 互換 API では usage が返らないことがあります
- usage がない場合、価格も `null` になります

## 6. 価格表を上書きする

既定の価格表ではなく独自の価格表を使いたい場合は、`--pricing-json` を使います。

形式は次のとおりです。

```json
{
  "gpt-5-mini": {
    "input_per_million_usd": 0.25,
    "output_per_million_usd": 2.0
  },
  "my-local-model": {
    "input_per_million_usd": 0.0,
    "output_per_million_usd": 0.0
  }
}
```

実行例:

```bash
uv run python scripts/run_image_screening_eval.py \
  --eval-set data/image_eval_set.json \
  --out data/image_eval_results.jsonl \
  --model gpt-5-mini \
  --pricing-json data/model_pricing.json
```

## 7. 集計

実行結果をモデルごとに集計するには、次を使います。

```bash
uv run python scripts/summarize_image_screening_eval.py \
  data/image_eval_results.jsonl
```

モデルごとに、概ね次の項目が出ます。

- 総件数
- 成功件数
- 失敗件数
- 平均レイテンシ
- 総 input tokens
- 総 output tokens
- 総推定価格

## 8. おすすめの進め方

最初は小さい評価セットで回すのがおすすめです。

1. 20〜50 件の `ImageRef` を用意する
2. `gpt-5-mini` で一度回す
3. 同じセットを `gpt-5` や `gpt-4.1-mini` でも回す
4. 集計結果と実際の判定内容を見比べる

この段階で、

- 速度
- 価格
- 出力の安定性
- 分類の質

をかなり見比べられます。
