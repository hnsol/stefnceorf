# Stefnceorf

テキストを編集する感覚で音声を編集するCLIツール（Descriptの音声編集機能の日本語対応・ローカル・無料版に相当）。

音声を文字起こしし、生成されたテキストを編集（削除・行の並べ替え）すると、その結果が音声ファイルへ反映される。全処理ローカル・クラウド不使用。

正式コマンドは `stefnceorf`、2文字エイリアス `sc`（同一動作）。

## 前提

- Apple Silicon Mac / Python 3.11+
- ffmpeg インストール済み
- 入力は沈黙・ノイズ除去済みの wav（前処理は Audacity 等で）

## インストール

```sh
uv pip install -e . -p .venv/bin/python
```

依存: mlx-whisper, numpy, soundfile

## 使い方

### 1. 文字起こし

```sh
stefnceorf transcribe input.wav [--lang ja|en] [--model MODEL] [--no-filler-suggest]
# sc transcribe input.wav でも同じ
```

- `input.sc.txt`（人間が編集するテキスト）と `input.sc.json`（単語→時刻・信頼度の対応表、触らない）を生成する
- `--lang` 省略時は自動判定
- `--model` 既定 `mlx-community/whisper-large-v3-turbo`
- フィラー候補数を表示。`--no-filler-suggest` で提案を無効化

### 2. 編集して音声へ反映

```sh
stefnceorf render input.sc.txt [-o output.wav]
```

（render は次のバージョンで実装予定）

## 編集ルール（input.sc.txt）

- 1行 = 1セグメント。行頭 `[ID]` が対応表へのキー
- 行削除 → その発話が音声から消える
- 行の並べ替え → その順で音声が再構成される
- 行内の文字削除 → 対応する単語の音声が消える
- `〔...〕` フィラー提案 → render が既定で削除。残すなら括弧だけ消す（`〔まあ〕`→`まあ`）
- `◆` 低信頼マーカー → 信頼度が低い単語の直前に付与。聞き直し用。render は無視するので消さなくてよい

詳細な編集・renderの挙動は後続バージョンのドキュメントで補足する。
