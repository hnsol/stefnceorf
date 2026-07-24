## ワークフロー
- 設計・方針決定・レビューは自分が行う
- 実装は @agent-implementer または @agent-fast-implementer に委譲する
- 軽量タスク → fast-implementer、重要な実装 → implementer
- サブエージェントの出力が不十分なら指示を精緻化して再実行する

## 作業記録
- 作業開始時に前回の記録を読む
- コミット時に記録を書く。「ログ書いて」と指示された場合も書く
- 記録は日単位のファイル（YYYY-MM-DD.md）に書く
- 各エントリの先頭に記録時刻（HH:MM）を書く
- 各エントリに含める内容：
  - やったこと（変更の要約）
  - なぜそうしたか（設計判断の背景・採用理由・不採用にした代替案）
  - 未解決事項
  - 次のアクション
- 1エントリ10行以内
- 1行だけの追記はしない。毎回上記の構造で書く
- 記録先は .Codex/Codex.local.md を参照

## リリース手順
- バージョンは pyproject.toml の version 1箇所で管理する
  - stefnceorf/__init__.py の __version__ は importlib.metadata で自動取得。手で書き換えない
- リリース時の手順:
  1. pyproject.toml の version を上げてコミット・push
  2. `git tag vX.Y.Z` して `git push origin vX.Y.Z`（タグ名は pyproject の version と一致させる）
  3. `gh release create vX.Y.Z` でリリースノート付き Release を作成
