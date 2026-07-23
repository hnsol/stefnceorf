<p align="center">
  <img src="assets/logo-light.svg#gh-light-mode-only" alt="Stefnceorf" width="160" height="160">
  <img src="assets/logo-dark.svg#gh-dark-mode-only" alt="Stefnceorf" width="160" height="160">
</p>

<h1 align="center">Stefnceorf</h1>

<p align="center">
  <strong>テキストを編集するだけで音声を編集 — ローカル・無料・AIエージェントで拡張可能</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black.svg" alt="Apple Silicon">
  <a href="README.md">English</a>
</p>

---

Stefnceorf は、**テキスト編集で音声を編集する**無料のCLIツール。ポッドキャストや音声コンテンツの収録後、テキストを編集するだけで音声ファイルの編集が完了する。全処理ローカル完結・クラウド不使用。[Descript](https://www.descript.com/) の音声編集機能の、ローカル・無料・日本語ネイティブ対応版。

正式コマンドは `stefnceorf`、短縮形 `sc`（同一動作）。

## クイックスタート

```sh
sc episode.wav                   # 文字起こし + Logic Pro 用 FCPXML を一括生成
```

WAV ファイルを渡すだけで、文字起こしと FCPXML 生成が完了します。生成された FCPXML を Logic Pro で開けばすぐに編集を始められます。

テキストを先に整理したい場合は、編集してから再書き出しします:

```sh
sc episode.wav                   # .sc.txt + .sc.json + .logic.fcpxml を生成
$EDITOR episode.sc.txt           # テキストを編集（削除・並べ替え）
sc logic episode.sc.txt          # 編集後の FCPXML を再生成
sc render episode.sc.txt         # または WAV へ直接書き出し
```

## 特徴

- **全処理ローカル完結・無料** — mlx-whisper による Apple Silicon GPU 文字起こし。クラウド不使用・APIキー不要
- **テキスト編集 = 音声編集** — テキストを消せば音声が消える。行を並べ替えれば音声も並び替わる
- **フィラー語の自動検出・削除提案** — 辞書ベースで「えー」「まあ」等を `〔〕` で提案。誤検出は括弧を外して却下
- **ポーズベース区切り** — 自然なポーズ境界でカット単位をスナップし、不自然な結合を防止
- **長い無音の自動切り詰め** — Whisper の幻覚対策＋出力の間延び防止
- **verbatim モード** — Whisper が通常吸収するフィラーも転写。フィラー削除ワークフロー用
- **幻覚検出＋レスキュー** — 繰り返し幻覚を自動検出し、安全設定で再認識して復旧
- **音質保持** — WAV render出力は元 wav と同一のサンプルレート・ビット深度。カット境界は等パワークロスフェード
- **日本語・英語対応** — 言語別フィラー辞書付き

## 前提

- Apple Silicon Mac（M1 / M2 / M3 / M4）
- Python 3.11+
- ffmpeg インストール済み（`brew install ffmpeg`）
- 入力はノイズ除去済みの wav（無音の切り詰めは stefnceorf が行う）

## インストール

```sh
# uv を使う場合（推奨）
uv venv
uv pip install -e .

# pip を使う場合
python -m venv .venv
.venv/bin/pip install -e .
```

`stefnceorf` と `sc` の2コマンドが登録されます。サブコマンドは省略可能です: `sc trans` = `sc transcribe`。`.wav` ファイルをサブコマンドなしで渡すと `auto` として実行されます。

依存: [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), numpy, soundfile

### どこからでも使えるようにする（任意）

editable インストール後、`.venv/bin/` にある実行ファイルへシンボリックリンクを張ると、どのディレクトリからでも実行できる。

```sh
ln -s "$(pwd)/.venv/bin/stefnceorf" /opt/homebrew/bin/stefnceorf
ln -s "$(pwd)/.venv/bin/sc" /opt/homebrew/bin/sc
```

## 使い方

### auto（既定の動作）

```sh
sc input.wav [--lang ja|en] [--no-verbatim] [--no-filler-suggest] [-o output.fcpxml]
```

文字起こしと FCPXML 書き出しを一括で実行します。`sc transcribe` → `sc logic` と同じ結果です。サブコマンドなしで WAV ファイルを渡すと自動的にこの動作になります（`sc input.wav` = `sc auto input.wav`）。

生成されるファイル:

- `input.sc.txt` — 編集可能なテキスト
- `input.sc.json` — 単語→時刻・信頼度の対応表（編集不要）
- `input.logic.fcpxml` — Logic Pro 用 FCPXML（未編集のテキストから生成）

テキスト編集後は `sc logic input.sc.txt` で FCPXML を再生成してください。

### 1. 文字起こし

```sh
sc transcribe input.wav [--lang ja|en] [--verbatim] [--no-filler-suggest] [--model MODEL] [--pause-threshold 0.15]
```

- `input.sc.txt`（人間が編集するテキスト）と `input.sc.json`（単語→時刻・信頼度の対応表）を生成
- `--lang` 既定 `ja`。英語は `--lang en`
- `--verbatim`（既定で有効）: フィラーも転写する。`--no-verbatim` で無効化。verbatim 時は自動で大きいモデル（`mlx-community/whisper-large-v3-mlx`）＋フィラー例文プロンプト＋`condition_on_previous_text=True` に切り替わる
- `--no-filler-suggest` でフィラー提案を無効化
- `--pause-threshold`（既定 0.15秒）: ブロック境界の最小ポーズ長。`0` で単語単位削除（旧挙動）
- 認識用の一時 wav で長い無音（1.2秒以上）を 0.7秒に切り詰め、Whisper の幻覚を抑止。単語時刻は元音源へ逆写像するため出力に影響なし

### 2. 編集

`input.sc.txt` を任意のテキストエディタで開く:

```
[0001 0:00] 結局面倒くさい作業があるなら／ツールを作ればいい
[0002 0:12] 〔まあ〕とか／そういう言葉を消して
[0003 0:25] 私ですね〔あのー〕／動画で喋りたいことは◆いくらでもある
```

- **行削除** → セグメント全体が音声から消える
- **行の並べ替え** → その順で音声が再構成される（行単位のみ。行内の語順入れ替えは非対応）
- **行内の文字削除** → ブロック単位で対応する単語の音声が消える
- **`〔...〕`（フィラー提案）** → **括弧ごと残す＝削除指定**。音声に残したいフィラーは括弧だけ外す（`〔まあ〕` → `まあ`）
- **`／`（ポーズ区切り）** → 安全なカット点。ブロック内を1文字でも消すとブロック全体が消える
- **`◆`（低信頼マーカー）** → 聞き直し用の印。render は無視する
- **追加・書き換えは不可** → 元音声が存在しないため警告のみ（削除のみ有効）
- **ID のない行・不明な ID** → render がエラーで停止する

### 3. 音声へ反映

```sh
sc render input.sc.txt [-o output.wav] [--gap-threshold 1.2] [--gap-max 0.7]
```

- 編集後 txt と `input.sc.json` を突き合わせ、**元の input.wav** から切り出して再構成
- `-o` 省略時の出力は `input.edited.wav`
- `--gap-threshold`（既定 1.2秒）以下のポーズはそのまま、超えるポーズは `--gap-max`（既定 0.7秒）に切り詰め
- 非破壊。何度でも再実行可能

### 4. Logic Proへ渡す

```sh
sc logic input.sc.txt [-o output.fcpxml] [--gap-threshold 1.2] [--gap-max 0.7]
```

- `-o` 省略時の出力は `input.logic.fcpxml`
- Logic Proで **File > Import > Final Cut Pro XML** を選び、このファイルを読み込む
- FCPXMLは元のWAVを直接参照し、テキストの削除・行の並べ替え・フィラー採否・長い無音の扱いを反映する
- 通常の削除は削除尺と同じ空白として残り、並べ替えたリージョンの境界には1秒の空白が入る。フェード／クロスフェードは含まれないため、必要に応じてLogicで設定する
- 元のWAVを移動すると、Logicで再リンクが必要になる場合がある

## フィラー辞書のカスタマイズ

フィラー候補は辞書との**完全一致**で `〔〕` に包まれる。辞書はパッケージ同梱ファイル（1行1語）で編集可能:

- 日本語: `stefnceorf/fillers_ja.txt`（既定: あのー, そのー, えー, えっと, えと, まあ, まー, うーん）
- 英語: `stefnceorf/fillers_en.txt`（既定: um, uh, uhm, er, ah）

## フォーク＆AIエージェントでカスタマイズ

このリポジトリは、AIコーディングエージェント（[Claude Code](https://docs.anthropic.com/en/docs/claude-code)、Cursor、GitHub Copilot 等）で**フォークして自分用に改造する**前提で設計されている。

同梱されているもの:

- **`CLAUDE.md`** — Claude Code 用のプロジェクトコンテキスト
- **`AGENTS.md`** — AIエージェント用のワークフロー指示
- **網羅的なテストスイート** — ユニットテスト、CLIテスト、受け入れテスト
- **シンプルな Python コードベース** — 4モジュール、約2,500行

AIエージェントへのプロンプト例:

> 「render コマンドに SRT 字幕出力を追加して」

> 「pyannote で話者分離を追加して」

> 「mlx-whisper を faster-whisper に置き換えて Linux で動くようにして」

## 制約と注意

- 日本語の文字起こし精度は 93–95% 程度。固有名詞・専門用語はブレやすい
- 単語タイムスタンプの誤差は前後 20ms のマージンとクロスフェードで緩和するが、**最終的に通しで1回聞くことを前提**とする
- `--verbatim` の繰り返し幻覚は自動検出・レスキューされるが、警告された区間は聞き直すこと
- 対象は音声のみ。動画・テロップ出力、ノイズ除去は対象外

## 開発・テスト

```sh
.venv/bin/python -m pytest -q
```

実モデル（mlx-whisper）や `say` コマンドを使う重いテストは `slow` マーカーが付き既定でスキップ。`-m slow` で実行。

## ロードマップ

- [ ] SRT 字幕出力
- [ ] 固有名詞辞書
- [ ] 話者分離
- [ ] 動画対応

## 名前の由来

*Stefnceorf* — 古英語 *stefn*（声）+ *ceorfan*（切る）から。

## ライセンス

[MIT License](LICENSE)
