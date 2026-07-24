# Auto Silence Stats Design

## Goal

`sc <wav>` の処理段階を明確にし、Logic Pro向けFCPXMLで実際に短縮された無音の件数と時間を、生成後の統計として常に表示する。

## Behavior

`auto` は文字起こし完了後、FCPXML生成の直前に次を表示する。

```text
logic: exporting FCPXML for Logic Pro: <input.sc.txt>
```

文字起こし時の `無音切り詰め` は、Whisperへ渡す認識用コピーの前処理実績であり、現行位置と表記を維持する。

FCPXML生成後の統計は、実際の出力編集計画から集計する。無音短縮が0件でも表示する。

```text
生成: <input.logic.fcpxml>
Silence: trimmed 0 gaps, 0.0s
Fillers: removed 13 of 15 (2 kept: too close to speech)
Duration: 25:00 -> 24:50 (-10.1s)
```

短縮がある場合は、従来の英語表記を使う。

```text
Silence: trimmed 75 gaps, -85.1s
```

## Silence Trimming

現在の出力処理は、検出済み無音区間が発話間ギャップを完全に覆う場合だけ短縮する。実データでは無音検出時刻と単語時刻の端に僅かな差があり、安全な無音が存在しても対象外になる。

修正後は、発話間ギャップと検出済み無音区間の共通部分だけを短縮対象とする。検出済み無音の外側は保持し、未認識音声を削除しない。検出済み無音のうち既存の `gap_max` を残し、それを超える部分だけを除去する。

## Scope

- `auto` のFCPXML開始通知
- `render` / `logic` 共通編集計画の安全な無音短縮
- 無音統計の常時表示
- 実データを模した境界不一致の回帰テスト

文字起こし方式、フィラー処理、通常削除、並べ替え、FCPXML形式は変更しない。

## Verification

- 境界が完全一致しない検出済み無音だけが短縮される
- 検出済みでない区間は保持される
- 0件時も `Silence` 行が生成後に表示される
- `auto` のFCPXML開始通知が文字起こし結果の後、生成結果の前に表示される
- サンプルデータで末尾統計と出力時間を確認する
- CLI、render、FCPXMLの関連テストと全テストを実行する
