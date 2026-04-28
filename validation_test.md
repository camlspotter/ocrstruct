# Validation Test

`ocrstruct/middle.py` の妥当性確認は、`/Users/jun/rois-rag/_data/staff_page2` 配下の実データ `middle.json` をまとめて `Result` で validate する方法で行う。

## 実行方法

```bash
uv run python validation_test.py
```

成功時は次のような出力になる。

```text
files=78 failed=0
```

失敗時は先頭 10 件まで、対象パスと `ValidationError` の内容を表示する。

## スクリプトの意図

`validation_test.py` は通常の `import ocrstruct.middle` を使わず、`ocrstruct/utils.py` と `ocrstruct/middle.py` を直接ロードしている。

これは、作業中に `ocrstruct/__init__.py` や `ocrstruct/types.py` が壊れていても、`middle.py` 単体の validate を続けられるようにするため。

## 見直しループ

`middle.py` を更新したら、次の順で確認する。

1. `uv run pyright ocrstruct/middle.py`
2. `uv run python validation_test.py`

`validation_test.py` で失敗が出た場合は、実データの shape に合わせて `ocrstruct/middle.py` を修正し、再実行する。
