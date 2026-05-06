# 画像評価分析

元データ: `image_eval_results.jsonl`

## 概要

- 画像数: 106
- モデル: gpt-5-nano, gpt-5-mini, gpt-5, gpt-5.2, gpt-5.4
- 結果行数: 530
- すべての画像について、5モデル分の結果がちょうど1回ずつ揃っています。

## 主な所見

- 今回の run では `gpt-5.2` と `gpt-5.4` が最も高速で、平均レイテンシはそれぞれ 2.48 秒、1.88 秒でした。
- `gpt-5-nano` は 1 画像あたり約 $0.0006 と最安でしたが、大きいモデル群とのズレも最も大きくなりました。
- 全体として最も近いペアは `gpt-5.2` と `gpt-5.4` で、`kind + rag_value + detail_level` の完全一致率は 55.7% でした。
- `kind` はモデル間で比較的安定しており、ペアごとの一致率はおおむね 82%〜95% でした。一方で `detail_level` は 44%〜75% とかなり揺れます。
- `gpt-5` は `extract_text` を最も積極的に選び（106 件中 62 件）、`gpt-5.4` は `long` を他モデルよりかなり多く使っています（106 件中 28 件）。
- `gpt-5-nano` は、小さなアイコン、QR コード、ページ断片に対して `text_as_image` や `table_or_form` を過剰に付ける傾向が見られました。

## 読み方の注意

- この評価にはまだ正解ラベルがありません。したがって、このレポートは最終的な精度順位ではなく、モデル挙動の比較として読むのが適切です。
- この GPT-5 系 vision リクエストでは、コストはローカルの画像サイズではなく API の `usage` フィールドから推定しています。課金比較としては、これを基準にするのが実務上もっとも自然です。
- 今回もっとも意外だったのは、安いモデルほど遅かったことです。これはこのデータセット上の実観測ですが、モデル価格に関する一般法則としてではなく、今回の run の経験的結果として扱うべきです。
- OpenAI の公開ガイダンスでは、より新しい小型モデルは通常 low-latency 側に位置づけられています。そのため今回の結果は、特に `gpt-5.4-mini` / `gpt-5.4-nano` や複数回の再測定で検証する価値がある観測と考えるべきです。
- `gpt-5` の pricing は公式価格表で再確認済みで、設定値は正しいです。今回の総コストが高いのは pricing バグではなく、実際の token usage が多かったためです。

## コストとレイテンシ

![Latency vs Cost](/Users/jun/mocrdown/image_eval_latency_cost.svg)

| モデル | 平均レイテンシ | 1画像あたり平均コスト | 106画像あたり平均コスト |
|---|---:|---:|---:|
| `gpt-5-nano` | 11.57s | $0.0006 | $0.062 |
| `gpt-5-mini` | 8.01s | $0.0012 | $0.131 |
| `gpt-5` | 9.47s | $0.0068 | $0.724 |
| `gpt-5.2` | 2.48s | $0.0033 | $0.353 |
| `gpt-5.4` | 1.88s | $0.0044 | $0.462 |

補足:

- `gpt-5` の結果は特に注意が必要です。token 単価は公式には `gpt-5.2` より低いのに、今回の総コストは高くなっています。理由は、平均 token 使用量がより大きかったためです。
- pricing は `usage` ベースなので、モデルが信頼できる `usage` を返す限り、画像サイズを別途手計算する必要はありません。

## detail_level の傾向

![Detail Distribution](/Users/jun/mocrdown/image_eval_detail_distribution.svg)

| モデル | skip | short | long | extract_text |
|---|---:|---:|---:|---:|
| `gpt-5-nano` | 9 | 35 | 0 | 62 |
| `gpt-5-mini` | 22 | 18 | 1 | 65 |
| `gpt-5` | 32 | 6 | 6 | 62 |
| `gpt-5.2` | 26 | 21 | 13 | 46 |
| `gpt-5.4` | 25 | 17 | 28 | 36 |

解釈:

- `gpt-5` はほぼ二極的で、`extract_text` と `skip` が多く、中間的な判断が少ないです。
- `gpt-5.4` は `long` を他モデルよりかなり多く使います。後段でより丁寧な enrichment をしたいなら有利かもしれません。
- `gpt-5-nano` は、他モデルが `skip` や `long` / `extract_text` を選ぶ場面で `short` に寄る傾向があります。

## 一致率

![Agreement Heatmap](/Users/jun/mocrdown/image_eval_agreement_heatmap.svg)

| ペア | kind 一致率 | rag 一致率 | detail 一致率 | 3項目完全一致率 |
|---|---:|---:|---:|---:|
| `gpt-5-nano` vs `gpt-5-mini` | 82.1% | 59.4% | 57.5% | 33.0% |
| `gpt-5-nano` vs `gpt-5` | 83.0% | 54.7% | 51.9% | 33.0% |
| `gpt-5-nano` vs `gpt-5.2` | 82.1% | 61.3% | 47.2% | 30.2% |
| `gpt-5-nano` vs `gpt-5.4` | 83.0% | 64.2% | 44.3% | 25.5% |
| `gpt-5-mini` vs `gpt-5` | 94.3% | 72.6% | 75.5% | 59.4% |
| `gpt-5-mini` vs `gpt-5.2` | 92.5% | 78.3% | 63.2% | 52.8% |
| `gpt-5-mini` vs `gpt-5.4` | 91.5% | 78.3% | 54.7% | 45.3% |
| `gpt-5` vs `gpt-5.2` | 91.5% | 72.6% | 68.9% | 50.9% |
| `gpt-5` vs `gpt-5.4` | 93.4% | 71.7% | 66.0% | 45.3% |
| `gpt-5.2` vs `gpt-5.4` | 95.3% | 86.8% | 64.2% | 55.7% |

解釈:

- 主に `kind` だけを見たいなら、このモデル群はすでにかなり一貫しています。
- 後段の作業計画まで含めて気にするなら、注目すべき軸は `detail_level` です。モデル差の多くはここから来ています。
- 現時点では `gpt-5.2` が最も良い基準点に見えます。`gpt-5.4` に近く、それでいてより安価です。

実務上の含意:

- 後段パイプラインが画像タイプの粗い振り分けだけを必要とするなら、複数モデルが実用候補になります。
- 後段コストの大半が OCR や詳細抽出を行うかどうかで決まるなら、`kind` より `detail_level` の不一致のほうがずっと重要です。

## 食い違いの大きい例

| 画像 | 注目理由 | 各モデルの出力 |
|---|---|---|
| `6e40acf242d69b693d9c44a0a6f65a7e763fe143459d1bfdd0acbe521493d60d.jpg` | score=16 | `gpt-5-nano`: table_or_form / high / extract_text<br>`gpt-5-mini`: arrow_only / low / skip<br>`gpt-5`: arrow_only / none / skip<br>`gpt-5.2`: diagram / medium / short<br>`gpt-5.4`: arrow_only / low / skip |
| `fcf6def197c69632cba86a1f6016a6652a7ad3bf9590371765865c1a8bce36d7.jpg` | score=14 | `gpt-5-nano`: text_as_image / high / extract_text<br>`gpt-5-mini`: logo_or_mark / high / extract_text<br>`gpt-5`: other / low / skip<br>`gpt-5.2`: logo_or_mark / medium / short<br>`gpt-5.4`: other / medium / short |
| `27d967c59825fa2ed7337de414a70602943058e8f6ca2b6cf23c080beb2d9926.jpg` | score=14 | `gpt-5-nano`: text_as_image / high / extract_text<br>`gpt-5-mini`: decorative / none / skip<br>`gpt-5`: logo_or_mark / none / skip<br>`gpt-5.2`: decorative / low / skip<br>`gpt-5.4`: text_as_image / low / short |
| `5ed229306d91d3a34dbfd3f2882450047870904ec5f942ae3c25518265c348a4.jpg` | score=13 | `gpt-5-nano`: table_or_form / high / extract_text<br>`gpt-5-mini`: decorative / none / skip<br>`gpt-5`: decorative / none / skip<br>`gpt-5.2`: logo_or_mark / low / skip<br>`gpt-5.4`: logo_or_mark / low / skip |
| `8d5d981863c3c4938f85a3826799cfb2cc2a85eb534becaf654344562b7fdabc.jpg` | score=13 | `gpt-5-nano`: text_as_image / medium / extract_text<br>`gpt-5-mini`: text_as_image / high / extract_text<br>`gpt-5`: other / low / skip<br>`gpt-5.2`: logo_or_mark / low / skip<br>`gpt-5.4`: logo_or_mark / low / skip |
| `39854df2b18c6b9d6ad36fc1e9721d310c9a225b4dc19df684aa85aa6f05a8cd.jpg` | score=12 | `gpt-5-nano`: text_as_image / low / extract_text<br>`gpt-5-mini`: other / medium / extract_text<br>`gpt-5`: other / low / skip<br>`gpt-5.2`: diagram / medium / short<br>`gpt-5.4`: diagram / medium / short |
| `7b911190ae8a9a56bff1407119946d31b60ced88f8c4145d9549a6b068fe034c.jpg` | score=11 | `gpt-5-nano`: text_as_image / high / extract_text<br>`gpt-5-mini`: decorative / none / skip<br>`gpt-5`: decorative / low / skip<br>`gpt-5.2`: text_as_image / medium / extract_text<br>`gpt-5.4`: text_as_image / medium / extract_text |
| `ff453133061a974354db176e303c7912b78ff893c54bb0ba74cde8a41343a426.jpg` | score=11 | `gpt-5-nano`: text_as_image / low / short<br>`gpt-5-mini`: decorative / low / short<br>`gpt-5`: logo_or_mark / none / skip<br>`gpt-5.2`: logo_or_mark / low / skip<br>`gpt-5.4`: logo_or_mark / low / skip |

目立つ傾向:

- 小さな矢印やチェックマークは、切り出しが曖昧だと周辺の大きな構造物と誤認されることがあります。
- QR コードはモデル間で大きく割れます。低価値なマークと見るものもあれば、読み取り価値のある情報資産と見るものもあります。
- 周辺に少しだけテキストを伴う装飾イラストは、特に `gpt-5-nano` で不安定です。

## 推奨

1. 当面の標準基準は `gpt-5.2` にするのがよさそうです。今回の run では高速で、`gpt-5.4` にかなり近く、それより明確に安価です。
2. 境界事例の確認や、より豊かな `detail_level` 判断がほしい場合は `gpt-5.4` を spot check 用に残すのがよいです。
3. 微妙な差よりコストを優先するなら `gpt-5-mini` が有力です。少なくとも `kind` については `gpt-5` / `gpt-5.2` に比較的よく追随しています。
4. QR コード、アイコン、小断片に対する二段目フィルタを入れない限り、このタスクで `gpt-5-nano` を唯一の分類器にするのは避けたほうがよさそうです。

## 次の一手

1. まず、食い違いの大きかった画像から小さな gold set を作る。
2. レイテンシについて強い結論を出す前に、少なくとももう一度は再測定する。
3. OpenAI は `gpt-5.4-mini` と `gpt-5.4-nano` を低レイテンシ寄りの新しい小型モデルとして位置づけているので、次の比較に加える。
