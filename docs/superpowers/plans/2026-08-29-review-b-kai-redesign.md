# --review 画面の再設計（案B改・全幅ワークスペース）

決定案: claude.ai/code/artifact/a4fa00b3-2d27-4ed3-bfc6-ad3c19812a4a（1 ページ目）。
参照モック: scratchpad の Main.dc.html（構造・寸法・色の見本。文言はサンプル）。

## 目的
コントロール3段＋凡例＋統計で画面上部が重く、本文の見える行数が少ない。
1トランスポートに統合して本文領域を最大化し、zelo-loop-design の語彙に合わせる。

## 変更対象
- stefnceorf/review.py の `_HTML_TEMPLATE` / `_SAVE_AREA` / `_EXPORT_AREA` / `_PREVIEW_PANEL` / `_SERVE_SCRIPT` と stats 文字列の整形
- tests/test_review.py・tests/test_serve.py のマークアップ依存箇所
- serve.py の HTTP API は変更しない

## デザイン仕様
1. フォント: Google Fonts（Zen Old Mincho / Zen Kaku Gothic New / IBM Plex Mono）を link。
   フォールバックは現行の -apple-system / Hiragino Sans。オフラインでも成立すること。
2. 色: 既存の Flexoki カスタムプロパティ体系を維持（dark 対応含む）。
   `--primary: #1A4F8C`（dark: #4385BE）を追加し、見出し・アクティブ状態・保存ボタンに使う。
   コンポーネントに hex 直書きしない。
3. レイアウト: max-width 80rem。ヘッダーは sticky で2行。
   - 1行目: タイトル（Zen Old Mincho・--primary）+ ファイル名等 subtitle（mono）+
     アクティブな audio（flex:1）+ 保存ボタン（--serve のみ、右端）
   - 2行目: モード segmented control（原音 / 保存済み、--serve では + 現在案）+
     シーク（−10s/−5s/+5s/+10s、mono）+ 速度（serve）+ 範囲（serve）+
     対象/現在案の絶対時刻（serve）+ XML出力（serve、右端）
4. audio 3要素は残し、モードに応じて1つだけをヘッダーのスロットに表示する
   （display 切替。カスタムスクラバーは作らない）。
   クリック再生の対象が src/out という既存セマンティクスは変えない。
   「現在案」segは preview audio を表示するだけで、試聴のトリガ（トグル→行クリック）は現行どおり。
5. ステータスバー（ヘッダー直下・1行・mono 11px）: 現在案の状態ラベル + stats。
   stats は複数行 pre をやめ、「削除 14 行 / 2:31.4 · フィラー 提案 38 · …」の1行に整形。
6. 凡例は1行のまま。✂ と ▲▼ の文字グリフを inline SVG（stroke 1.5）に置換。
7. トランスクリプト行: 現行構造を維持しつつ見た目を更新。
   ts/sid は IBM Plex Mono 11px、active 行は左に 2px の --primary バー（inset box-shadow）、
   unsafe 破線アウトライン等の状態表現は現行を踏襲。
8. キーボード操作（新規）: Space（既存）に加え J/K=行移動（active 行を移して再生）、
   --serve では X=active 行の削除トグル、P=現在案を試聴、S=保存。
   入力要素にフォーカスがあるときは発火しない。
   画面下部にヒント行（mono 10px、serve 専用キーは serve のみ表示）。
9. 挙動・API・保存/試聴/並び替え/XML出力のロジックは不変。マークアップと表示だけ変える。

## 検証
- `uv run pytest` 全通過（マークアップ依存のテストは新構造に合わせて更新、挙動のカバレッジは維持）
- 静的モード（--serve なし）でも成立すること（serve 専用 UI が出ない）
