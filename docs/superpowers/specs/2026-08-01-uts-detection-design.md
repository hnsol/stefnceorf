# 未転写発話（UTS）検出 設計

日付: 2026-08-01
参考実装: [vidtighten](https://github.com/matsumotoryosuke-dev/vidtighten)（MIT）の
`src/preprod/audio.py` `detect_untranscribed_speech`

## 背景 / 課題

Whisper は 100–150ms 程度の短い発声（「え」「ん」「あ」等のヒゲ、息、相槌）を転写しないことがある。
stefnceorf は「転写に載った語」しか削除対象にできないため、無音トリム後もこれらが残る。

既存の `_preserve_unrecognized_gaps`（transcribe.py）は再認識不能な非無音区間を
`kind: "unrecognized"` として安全側で保持するが、内部 gap が `SILENCE_MIN_S`(1.2s) 以下だと
行を作らないため、短い未転写発声は可視化されない。

## 決定事項

1. **自動カットしない。検出・可視化のみ**
   - UTS が拾うのは相槌・笑い・小声・ゲストの短い発言も含み、フィラーとは限らない
   - 既存の「未認識区間は安全側で保持」の設計思想と整合させる
   - 実データ 3 本程度で精度確認後、誤検出が十分少なければ将来 `--auto-uts` を検討
2. **`--verbatim` とは独立して常時検出**
   - verbatim は「フィラーも転写するモード」で既定有効。連動させると既定動作で UTS が死ぬ
3. **新規 uts.py は作らず transcribe.py へ統合**。ただし検出は 2 純関数に分離する
4. **既存の安全側保持を弱めない**
   - UTS 島だけを候補行にし、残余 gap は従来どおり保持
   - 保持対象から除外するのは既検出無音で完全に説明できる部分のみ

## 検出アルゴリズム

2 純関数に分離（テスト容易性のため）:

### `detect_rms_silences()`
短時間 RMS 無音検出。既存の粗い無音検出（1s 級）とは別パス
（粗いパスだとフィラーが発話ブロックに融合され島境界が立たない）。
- 20ms RMS 窓、最小無音 0.1s、hangover なし
- 閾値は適応: `min(RMS_SAFE_DBFS, 発話中央値 − RMS_SAFE_REL_DB)`（render.py の既存規約）。
  適応閾値用の有効な単語が無い場合は `RMS_SAFE_DBFS`(−40dB) へフォールバック

### `detect_untranscribed_speech()`（vidtighten 移植、MIT 帰属コメント付き）
上記の短時間無音リスト＋単語タイムスタンプを受け取り、UTS 島 `[(start, end), ...]` を返す。
- 島条件（vidtighten 由来）: 長さ 0.10s–1.2s、隣接無音の大きい方 ≥ 0.25s、小さい方 ≥ 0.15s
- 既転写との重複判定は**単語**タイムスタンプ基準（segment は未転写音を跨いで覆うため不可）、許容 20ms
- ファイル先頭・末尾は無限長無音とみなす

### 統合
- **実行タイミング**: 16kHz 一時 wav が unlink される前（音声配列がメモリにある間）に検出する
- UTS span は挿入前にソート・マージ・音源範囲へクランプ
- `_preserve_unrecognized_gaps(..., uts_spans=...)` へ拡張し、通常 gap のスキップ判定より**前**に
  UTS 島を `kind: "unrecognized"` ＋ `reason: "uts"` の独立セグメントとして挿入。
  長い既存 gap は UTS 島と残余保持区間に分割し、残余は従来どおり保持。
  UTS 島が無い gap は現行動作を完全維持（uts_spans 空なら完全一致）

## 出力・UX

- `.sc.txt`: 候補を独立行として提示（既存 unrecognized 行と同形式）。
  行を残せば音声保持、行を削除すればカット — 既存の編集パラダイムそのまま
- `.sc.json`: 区間＋ `reason: "uts"` を記録（識別・集計用）
- stats: 「未転写発話候補 N 件」（削除秒数は出さない。削除は既定でしないため）

## テスト

- `detect_rms_silences`: 窓・最小長・適応閾値・フォールバックの単体テスト
- `detect_untranscribed_speech`: 島検出、境界値（0.10/0.25/0.15/1.2s）、単語重複除外（20ms 許容）
- `_preserve_unrecognized_gaps` 拡張: 短い gap 内の島挿入／長い gap の分割／
  uts_spans 空での現行動作一致（回帰）
- verbatim / 非 verbatim 両方で検出、`.sc.json` への reason 記録

## 不採用にした代替案

- **vidtighten への乗り換え**: FCP＋動画＋GUI 前提。Logic Pro 向けエッジスナップと
  テキスト編集ワークフローを失うため不採用
- **UTS の自動カット**: 決定事項 1 の理由で不採用（当初案から撤回）
- **verbatim 時のみ検出スキップ**: verbatim の意味の誤読に基づく案だったため撤回
- **`_preserve_unrecognized_gaps` 後段での細分化**: 短い gap は行が作られず対象が存在しないため不成立
