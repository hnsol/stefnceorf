# FCPXML リージョン端の自動補正でクリックノイズを防ぐ実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** FCPXML 経由で Logic へ読み込んだリージョン端に出るクリックノイズを、リージョン端を内側の静音点へ自動補正することで防ぐ。

**Background:** Logic 向け FCPXML はフェード／クロスフェードを持たない設計（[2026-07-23-logic-fcpxml-spacing.md](2026-07-23-logic-fcpxml-spacing.md)）のため、リージョン端が非ゼロ振幅のまま無音（タイムライン空白）へ接続され波形が不連続になり、カット箇所の 30-50% でクリックノイズが発生していた。WAV render は等パワークロスフェードで対策済みだが FCPXML は未対策だった。本計画はそのフェード無し設計の補完策として、境界サンプルを内側の静音点へ寄せる。

**Architecture:** 実装は `stefnceorf/fcpxml.py` に限定する。`EditPlan` は `audio`（float64 numpy、全サンプル）と `samplerate` を持つため fcpxml.py から参照できる。`export_fcpxml()` が区間をサンプル整数へ量子化した直後に補正を適用し、以降の `_timeline_offsets`・`_time`・duration はすべて補正後サンプルを使う。WAV render・CLI・JSON・公開 API・フェード無し仕様は変更しない。

**Tech Stack:** Python 3.10+、numpy、`xml.etree.ElementTree`、`fractions.Fraction`、pytest、FCPXML 1.8

## アルゴリズム要点

- 定数: `EDGE_SEARCH_S = 0.005`（最大補正幅 5ms）、`EDGE_SAFE_DBFS = -50.0`（安全閾値）。閾値振幅は `10 ** (EDGE_SAFE_DBFS / 20)`。
- `_snap_edge_inward(audio, samplerate, sample, direction, limit_sample)`
  - `sample` から `direction`（start端=+1 後方 / end端=-1 前方）へ最大 `round(EDGE_SEARCH_S * samplerate)` サンプル走査。`limit_sample` を越えない。
  - 全チャンネルの |振幅| が閾値以下になる **最短距離** の位置を採用（元 sample が既に安全なら距離 0）。
  - 安全位置が無ければ走査範囲内で **チャンネル横断の最大 |振幅|** が最小のサンプルを返す（片 ch だけ静かな点を採らない）。走査範囲が空なら sample をそのまま返す。
- `_snap_region(audio, samplerate, start_sample, end_sample)`
  - start 端を +1、end 端を -1 で補正。end_sample は排他境界なので評価対象は最終サンプル index `end_sample - 1`。
  - 外側へは動かさない（`new_start >= start_sample`, `new_end <= end_sample`）。互いの補正が交差しないよう start は区間中点で制約し、end は new_start を下限にする。
  - 補正後 `new_end - new_start >= 1` を割る場合は補正を諦め元値を返す。
- 削除境界の空白は `max(0, current_start - previous_end)` で計算されるため、内側補正で空白がわずかに広がるのは仕様どおり（削除分＋補正分）。並べ替え境界の固定 1 秒は不変。

## テスト（tests/test_fcpxml.py）

- 安全な端（既に −50dBFS 以下）は移動しない。
- 高振幅端（サイン波）が 5ms 以内で内側へ補正され、境界振幅が −50dBFS 以下になる。
- start は減らない・end は増えない（外側へ動かない）。
- ステレオ L=sin / R=cos: 片 ch だけ静かな点を採らず、ch 横断 max|振幅| 最小点を採用する。
- 極端に短いリージョン（数サンプル〜10ms）でも長さ ≥ 1 サンプルを維持。
- `export_fcpxml` 経由の統合テスト（生成 XML の asset-clip start/duration を Fraction でパースし、境界サンプル振幅が閾値以下）。
- 既存テスト（通常削除空白・並べ替え 1 秒空白・DTD/フォーマット検証）を壊さない。

## 検証

`uv run pytest tests/ -q` 全通過。

## 変更ファイル

- `stefnceorf/fcpxml.py` — 定数 2・関数 2・`export_fcpxml` へ組み込み（numpy import 追加）
- `tests/test_fcpxml.py` — 上記テスト追加
- `README.md` / `README_ja.md` — 自動補正の記載
