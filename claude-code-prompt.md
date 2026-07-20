# Claude Codeに渡すプロンプト

同じフォルダに `audio-descript-design.md`（設計書）を置いてから、以下を貼り付ける。

---

音声編集CLIツール **Stefnceorf**（コマンド `stefnceorf`、2文字エイリアス `sc`）を実装してください。仕様は `audio-descript-design.md` に従うこと。設計書と矛盾する実装をしたくなった場合は、勝手に変えずに理由を提示して確認を取ってください。

## 前提環境

- Apple Silicon Mac / Python 3.11+ / ffmpegインストール済み
- mlx-whisperは既存venvに導入済み（モデル: mlx-community/whisper-large-v3-turbo キャッシュ済み）。本プロジェクトでは新しいvenvを作り `mlx-whisper numpy soundfile` を入れてよい
- ネットワークアクセスはモデルダウンロード以外不要。クラウドAPIは使用禁止

## 成果物

```
stefnceorf/
  pyproject.toml        # console_scripts: stefnceorf と sc（同一エントリポイント）
  stefnceorf/cli.py     # argparse: transcribe / render の2サブコマンド
  stefnceorf/transcribe.py  # mlx_whisper.transcribe(word_timestamps=True) → .sc.txt/.sc.json
  stefnceorf/fillers.py     # フィラー辞書ロードと照合（fillers_ja.txt / fillers_en.txt同梱）
  stefnceorf/render.py      # txtパース → 文字diff → 区間算出 → numpy/soundfileで結合
  tests/                # 下記テスト
  README.md             # 使い方（インストール、2コマンド、編集ルール）
```

## 実装上の要点（設計書の再掲・特に守ること）

1. transcribeは入力wavをffmpegで16kHz mono一時wavに変換して認識する。ただし**renderの切り出しは元wavから**行い、出力は元wavと同一フォーマットにする
2. `.sc.json` には セグメントID・元テキスト・単語ごとの {word, start, end, probability} を保存する。`.sc.txt` は人間の編集用で、`[ID] テキスト` 形式
3. probability < 0.5（定数で調整可）の単語の直前に `◆` を付与。renderは `◆` を無視する
4. フィラーは辞書と単語トークンの完全一致のみ `〔〕` で包む。renderは `〔...〕` を削除扱い。括弧が外されていれば残す
5. 行内diffはdifflibの文字単位比較。「追加・書き換え」文字は無視して警告表示。削除のみ音声に反映
6. 残す区間は前後20msマージン（隣の削除区間に食い込まない範囲で）＋境界5–10msの等パワークロスフェード
7. エラーは黙殺しない：不明ID、ID行の破損、jsonとtxtの不整合は具体的なメッセージで失敗させる

## 進め方

1. まず骨格（CLI・ファイルIO・型）を作り、次にtranscribe、最後にrenderの順で実装
2. 各段階でユニットテストを書く。特にrenderのdiff→区間算出は音声なしでテスト可能な純関数として切り出すこと
3. 設計書§12の受け入れテストを `tests/` に落とし込む。音声が必要なテストは、`say`コマンドかsine波合成で数秒のテストwavを生成して使う
4. 全テスト通過後、実際の使い方をREADMEに書いて完了報告

## 完了条件

- `stefnceorf transcribe sample.wav`（`sc`でも同じ）→ `.sc.txt` / `.sc.json` 生成、フィラー候補数の表示
- `.sc.txt` の行削除・行内文字削除・行入れ替え・〔〕採否が `stefnceorf render` で音声に反映される
- 設計書§12の受け入れテストが全て通る
