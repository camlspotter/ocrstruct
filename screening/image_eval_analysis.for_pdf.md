# VLM による画像分類実験

計測データ: `image_eval_results_v2.jsonl`
正解データ: `image_eval_results.gpt-5.4.goldset.json`

## Abstract

Multimodal RAG を作成するにあたり、文書から画像を取り出し、VLM によりそれらの
説明テキストを抽出したい。その前段階の screening として、画像の説明テキストを
コストと時間をかけて得る「テキスト化価値」があるかどうかの判別を
高速、低コストで行うこととした。

本実験では、ROIS にある文書中の約 100個の画像に対し、複数のモデルを使用して
テキスト化価値の推定を行い、モデルの評価を行った。

その結果 XXX TODO 現在結論をつける段階にない。

## テキスト化価値

テキスト化価値とは、画像について追加の説明文や OCR 結果を生成し、
RAG に登録するだけの価値があるかどうかを指す。

`kind`: 画像の種類。たとえば `diagram`, `text_as_image`, `seal`, `code_symbol` など。

`rag_value` の値は次のように解釈する:

- `high`: 画像そのものが情報の担い手であり、説明文や抽出テキストを RAG に入れる価値が高い。
- `medium`: 一定の情報価値はあるが、文書本文だけで大筋が把握できる可能性もある。
- `low`: 補助的な価値はあるが、RAG への寄与は限定的である。
- `none`: 説明文や OCR を追加しても、RAG 上の価値はほとんど期待できない。

`detail_level` の値は、screening 後にどこまでコストをかけて調べるべきかを表す:

- `skip`: 追加調査を行わない。
- `short`: 短い説明文だけを付ければ十分である。
- `long`: 構造や意味関係を含めた、やや詳しい説明文を作る価値がある。
- `extract_text`: 詳しい説明に加え、画像内テキストそのものを OCR などで抽出する価値が高い。

この2つは似ているが、役割が異なる。`rag_value` は「そもそも価値があるか」を表し、
`detail_level` は「価値があるとして、どこまで調べるか」を表す。

## 対象概要

画像: ROIS本部に存在する文書から人間が選んだ画像 106 枚

モデル

- Qwen/Qwen3.6-27B-FP8 (thinking=null)
- Qwen/Qwen3.6-27B-FP8 (thinking=false)
- Qwen/Qwen3.6-35B-A3B-FP8 (thinking=null)
- Qwen/Qwen3.6-35B-A3B-FP8 (thinking=false)
- gpt-5
- gpt-5-mini
- gpt-5-nano
- gpt-5.2
- gpt-5.4
- gpt-5.4-mini
- gpt-5.4-nano

ここで `thinking=null` は、thinking を明示指定せず、
モデル側のデフォルト動作に任せた条件を意味する。
Qwen 3.6 系では、この条件は実質的に `thinking=true` と解釈して差し支えない。
Qwen モデルについては ROIS 本部の DGX Spark 互換機を利用し、そこでの処理時間を計測し、費用は0とした。

OpenAI モデルについては `reasoning_effort` の指定を行っていない。
OpenAI のドキュメントによれば、この場合は多くのモデルで reasoning は行わない、
もしくは極小ということになっている。

これらを組み合わせ、合計 106 x 11 = 1166 の結果を得た。

計測データ: `image_eval_results_v2.jsonl`

## 正解データとスコアの定義

正解データとして、 `gpt-5.4` で得られた回答を人間がチェックし、修正を加えた物を使用する。

正解データ: `image_eval_results.gpt-5.4.goldset.json`

各モデルの出力を正解データと比べて採点した。

- `kind` は離散的なラベルなので、正解と完全に一致したかどうかで評価する。
- `rag_value` と `detail_level` は順序付きの尺度なので、以下の距離スコアを使う:
  - 完全一致なら 1.0
  - 1段階ずれなら 0.67
  - 2段階ずれなら 0.33
  - 最大ずれなら 0.0
  式で書くと $1 - |\mathit{pred} - \mathit{gold}| / 3$ である。

最終的な総合スコアは、次の重み付き平均で計算した。

- `kind` 完全一致: 40%
- `rag_value` 距離スコア: 30%
- `detail_level` 距離スコア: 30%

正解データの作成経緯から `gpt-5.4` でのスコアが高くなるバイアスが存在している可能性がある。

## 評価

### 総合スコア

![Overall Score](image_eval_gold_overall_score.pdf)

| モデル条件 | 総合スコア | kind 一致率 | rag 距離スコア | detail 距離スコア | 3項目完全一致率 |
|---|---:|---:|---:|---:|---:|
| `gpt-5.4` | 0.900 | 86.8% | 0.962 | 0.881 | 52.8% |
| `gpt-5` | 0.886 | 89.6% | 0.912 | 0.846 | 46.2% |
| `gpt-5.2` | 0.880 | 86.8% | 0.925 | 0.852 | 45.3% |
| `Qwen/Qwen3.6-27B-FP8 [thinking=false]` | 0.865 | 84.9% | 0.912 | 0.840 | 37.7% |
| `gpt-5-mini` | 0.864 | 87.7% | 0.928 | 0.783 | 38.7% |
| `gpt-5.4-mini` | 0.864 | 88.7% | 0.921 | 0.777 | 28.3% |
| `Qwen/Qwen3.6-27B-FP8 [thinking=null]` | 0.862 | 88.7% | 0.921 | 0.770 | 33.0% |
| `gpt-5.4-nano` | 0.848 | 85.8% | 0.915 | 0.767 | 34.9% |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=null]` | 0.842 | 84.0% | 0.915 | 0.770 | 36.8% |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=false]` | 0.826 | 77.4% | 0.909 | 0.814 | 34.9% |
| `gpt-5-nano` | 0.782 | 82.1% | 0.789 | 0.723 | 27.4% |

### 項目別スコア

![Gold Axis Accuracy](image_eval_gold_axis_accuracy.pdf)

| モデル条件 | kind 一致率 | rag 平均ずれ | rag 距離スコア | detail 平均ずれ | detail 距離スコア |
|---|---:|---:|---:|---:|---:|
| `gpt-5.4` | 86.8% | 0.113 | 0.962 | 0.358 | 0.881 |
| `gpt-5` | 89.6% | 0.264 | 0.912 | 0.462 | 0.846 |
| `gpt-5.2` | 86.8% | 0.226 | 0.925 | 0.443 | 0.852 |
| `Qwen/Qwen3.6-27B-FP8 [thinking=false]` | 84.9% | 0.264 | 0.912 | 0.481 | 0.840 |
| `gpt-5-mini` | 87.7% | 0.217 | 0.928 | 0.651 | 0.783 |
| `gpt-5.4-mini` | 88.7% | 0.236 | 0.921 | 0.670 | 0.777 |
| `Qwen/Qwen3.6-27B-FP8 [thinking=null]` | 88.7% | 0.236 | 0.921 | 0.689 | 0.770 |
| `gpt-5.4-nano` | 85.8% | 0.255 | 0.915 | 0.698 | 0.767 |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=null]` | 84.0% | 0.255 | 0.915 | 0.689 | 0.770 |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=false]` | 77.4% | 0.274 | 0.909 | 0.557 | 0.814 |
| `gpt-5-nano` | 82.1% | 0.632 | 0.789 | 0.830 | 0.723 |

### コストとレイテンシ

![Latency vs Cost](image_eval_latency_cost.pdf)

| モデル条件 | 平均レイテンシ | 中央レイテンシ | 1画像あたり平均コスト | このセット合計コスト |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3.6-27B-FP8 [thinking=null]` | 175.09s | 143.51s | $0.0 | $0.0 |
| `Qwen/Qwen3.6-27B-FP8 [thinking=false]` | 7.47s | 7.43s | $0.0 | $0.0 |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=null]` | 27.02s | 22.51s | $0.0 | $0.0 |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=false]` | 1.56s | 1.45s | $0.0 | $0.0 |
| `gpt-5` | 9.16s | 8.79s | $0.0071 | $0.756 |
| `gpt-5-mini` | 8.59s | 8.16s | $0.0013 | $0.133 |
| `gpt-5-nano` | 11.66s | 11.15s | $0.0006 | $0.060 |
| `gpt-5.2` | 2.97s | 2.78s | $0.0035 | $0.369 |
| `gpt-5.4` | 2.41s | 1.84s | $0.0045 | $0.482 |
| `gpt-5.4-mini` | 1.75s | 1.43s | $0.0013 | $0.141 |
| `gpt-5.4-nano` | 2.75s | 2.66s | $0.0004 | $0.040 |

### 主な所見

総合スコアが最も高かったのは `gpt-5.4` の 0.900。ただし正解データは `gpt-5.4` の結果を元にしているため、バイアスが入っている可能性がある。

最速は `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=false]` だった。

OpenAI の古いモデルでは意外と時間がかかる傾向がある。

Qwen では `Qwen/Qwen3.6-27B-FP8` の `thinking=null`（実質 `thinking=true`）と `thinking=false` で総合スコアはそれぞれ 0.862, 0.865 と大差なかったにも関わらず、
時間差が極端で、平均 175.1s と 7.5s であることが目立つ。

`Qwen/Qwen3.6-35B-A3B-FP8` は `thinking=null` で平均 27.0s、総合スコア 0.842 だった。
`thinking=false` では平均 1.56s まで高速化した一方、総合スコアは 0.826 まで低下した。

費用に関しては、当然 open weight model が最安である。

### ROIS 文書中の画像 3985件に外挿した概算

現在、ROIS のローカル文書中には screening 対象の画像が 3985 件ある。
今回の実験の 1件あたり平均値を単純外挿すると、これらを処理する際の目安は次のとおりである:

| モデル条件 | 3985件の概算コスト | 3985件を直列実行した場合の概算時間 |
|---|---:|---:|
| `Qwen/Qwen3.6-27B-FP8 [thinking=null]` | $0.00 | 193.8時間 |
| `Qwen/Qwen3.6-27B-FP8 [thinking=false]` | $0.00 | 8.3時間 |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=null]` | $0.00 | 29.9時間 |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=false]` | $0.00 | 1.7時間 |
| `gpt-5` | $28.43 | 10.1時間 |
| `gpt-5-mini` | $4.99 | 9.5時間 |
| `gpt-5-nano` | $2.27 | 12.9時間 |
| `gpt-5.2` | $13.86 | 3.3時間 |
| `gpt-5.4` | $18.13 | 2.7時間 |
| `gpt-5.4-mini` | $5.31 | 1.9時間 |
| `gpt-5.4-nano` | $1.50 | 3.0時間 |

時間については OpenAI モデルについては並列処理を行うことで短縮することができる。
Open weight model については ROIS 本部での機材の関係上、並列化は難しい。

## 推奨

総合評価で見ると、第一候補は `gpt-5.4`。ROIS 内文書のみに限ればコストも許容範囲だと思われる。

ローカル GPU で速度と品質のバランスが最も良いのは `Qwen/Qwen3.6-27B-FP8 [thinking=false]`。平均 7.5s で、総合スコアは 0.865。
`Qwen/Qwen3.6-35B-A3B-FP8 [thinking=false]` はさらに高速だが、品質はやや下がる。

`Qwen/Qwen3.6-27B-FP8` は `thinking` 条件で速度が大きく変わる。 Thinking はほぼ役に立っていない。

## 付録: kind の主な取り違え

| モデル条件 | 主な取り違え |
|---|---|
| `gpt-5.4` | `logo_or_mark`→`seal` (5)<br>`decorative`→`text_as_image` (1)<br>`decorative`→`diagram` (1)<br>`diagram`→`table_or_form` (1)<br>`diagram`→`code_symbol` (1) |
| `gpt-5` | `logo_or_mark`→`seal` (4)<br>`text_as_image`→`decorative` (3)<br>`logo_or_mark`→`decorative` (1)<br>`diagram`→`code_symbol` (1)<br>`logo_or_mark`→`code_symbol` (1) |
| `gpt-5.2` | `logo_or_mark`→`seal` (4)<br>`text_as_image`→`decorative` (3)<br>`arrow_only`→`diagram` (1)<br>`logo_or_mark`→`table_or_form` (1)<br>`text_as_image`→`seal` (1) |
| `Qwen/Qwen3.6-27B-FP8 [thinking=false]` | `diagram`→`table_or_form` (3)<br>`logo_or_mark`→`seal` (3)<br>`logo_or_mark`→`arrow_only` (1)<br>`decorative`→`diagram` (1)<br>`diagram`→`code_symbol` (1) |
| `gpt-5-mini` | `logo_or_mark`→`seal` (5)<br>`text_as_image`→`decorative` (3)<br>`logo_or_mark`→`table_or_form` (1)<br>`text_as_image`→`seal` (1)<br>`diagram`→`code_symbol` (1) |
| `gpt-5.4-mini` | `logo_or_mark`→`seal` (3)<br>`text_as_image`→`decorative` (2)<br>`logo_or_mark`→`table_or_form` (1)<br>`decorative`→`text_as_image` (1)<br>`diagram`→`table_or_form` (1) |
| `Qwen/Qwen3.6-27B-FP8` | `logo_or_mark`→`seal` (4)<br>`diagram`→`table_or_form` (2)<br>`ui_or_screenshot`→`table_or_form` (1)<br>`logo_or_mark`→`ui_or_screenshot` (1)<br>`decorative`→`diagram` (1) |
| `gpt-5.4-nano` | `logo_or_mark`→`seal` (3)<br>`decorative`→`text_as_image` (2)<br>`arrow_only`→`table_or_form` (1)<br>`logo_or_mark`→`text_as_image` (1)<br>`decorative`→`diagram` (1) |
| `Qwen/Qwen3.6-35B-A3B-FP8` | `logo_or_mark`→`seal` (5)<br>`diagram`→`table_or_form` (2)<br>`text_as_image`→`decorative` (2)<br>`ui_or_screenshot`→`table_or_form` (1)<br>`table_or_form`→`diagram` (1) |
| `Qwen/Qwen3.6-35B-A3B-FP8 [thinking=false]` | `diagram`→`table_or_form` (4)<br>`logo_or_mark`→`seal` (4)<br>`decorative`→`diagram` (2)<br>`table_or_form`→`diagram` (2)<br>`logo_or_mark`→`arrow_only` (1) |
| `gpt-5-nano` | `logo_or_mark`→`seal` (6)<br>`logo_or_mark`→`text_as_image` (2)<br>`decorative`→`text_as_image` (2)<br>`diagram`→`text_as_image` (2)<br>`arrow_only`→`table_or_form` (1) |
