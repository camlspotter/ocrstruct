# MEMO

## PDFリンク矩形とテキスト対応づけの難しさ

- PDF のリンクは注釈レイヤー (`/Annots` の `/Rect`) にあり、OCR/抽出テキストの bbox とは別系統。
- そのため、リンク矩形とテキスト span bbox は一致が保証されない。
- 実際には以下が起きる:
  - リンク矩形が文字列より広い/狭い
  - 1つのリンク矩形に複数 span が入る
  - 1つの span が複数リンク矩形にまたがる
  - ページ座標系・抽出誤差・OCR分割差でズレる
- 内部リンク (`GoTo`/`Dest`) は destination が点 (`/XYZ`) やページ表示指定 (`/Fit*`) のことがあり、リンク先に矩形がない場合がある。

### 実装方針（現実解）

1. まずリンク矩形を正として保持する。
2. テキスト対応づけは重なり率 + 距離でスコアリングする。
3. しきい値未満は「あいまい」として扱い、必要なら追加OCR/再解釈を行う。

## HTMLレンダリングの source image link

- `source-image-link-row` は現在使われている。
- 用途は HTML 出力されたソース画像リンク行のラッパーで、CSS から余白と右寄せを与えること。
- 現在の使用箇所は table の HTML 出力。
- JavaScript からは参照されておらず、JS が使うのは `source-image-link` と `source-image-modal` 系。

## cross_page_table_merge 後の空テーブル

- `middle.json` 内の `type == "table"` で HTML を持たないものを調べると、`lines_deleted: true` を持つ `table_body` だけが残るケースが複数あった。
- このパターンは `cross_page_table_merge` の副作用で、前ページ側の table に統合された後ページ側の殻が残っている可能性が高い。
- 観測上の特徴:
  - 子 block がほぼ `table_body` のみ
  - `table_body.lines == []`
  - `lines_deleted: true`
  - 前ページに HTML を持つ近い table があることが多い
- 例:
  - `pdf/senryaku/__data/20260325_senryaku_keihi_qa.pdf/middle.json`
    - page 1 に HTML あり table
    - page 2 に `html_count = 0` かつ `lines_deleted: true` の table
  - `anzen/pdf/__data/tougou.pdf/middle.json`
    - page 103 に HTML あり table
    - page 104, 105 に空の table 残骸
- 一方で、統合後の table 側は HTML は残っても `image_path` が落ちることがあり、上流の後処理バグの可能性がある。
- renderer 側の応急策アイデア:
  - HTML のない table に遭遇したら、同じ page の `preproc_blocks` から対応する table を探す。
  - まずは近い bbox や同系統の table block を手がかりにして、そこに `image_path` があれば補う。
  - table HTML 自体を `preproc_blocks` から補完するかどうかは別判断として、少なくとも `image_path` の導入候補としては使える。

## 行政系文書では equation 認識を切る案

- 行政・規程・手続き系の PDF では、数式をほぼ含まない一方で、日本語本文の一部が `inline_equation` に誤認識されることがある。
- 例: `（以下では，「貨物」という。）` の `」という` 付近が壊れた LaTeX 断片として抽出され、MathJax 表示が崩れるケースがあった。
- この種の文書では equation 認識を止めるオプションを有効にすると、誤認識回避に有効な可能性が高い。
- 特に行政関連文書については「数式を探さない」設定を検討してよい。
