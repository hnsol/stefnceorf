# Logic FCPXML Timeline Spacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Logic Proへ読み込むFCPXMLでは編集順を維持し、通常削除を削除尺の空白、並べ替え境界を固定1秒の空白として配置する。

**Architecture:** `render.build_edit_plan()`の音声区間計算は変更せず、`stefnceorf.fcpxml`内でサンプル整数へ量子化した区間列からLogicタイムライン上の`offset`を計算する。現在の境界より前後の元音声区間に、別位置へ残された区間があれば並べ替え境界、それ以外は通常削除境界と判定する。

**Tech Stack:** Python 3.10+、`xml.etree.ElementTree`、`fractions.Fraction`、pytest、FCPXML 1.8

## Global Constraints

- `sc logic input.sc.txt`と公開関数`export_fcpxml(...) -> str`を維持する。
- FCPXMLは元WAVを絶対`file:` URLで直接参照し、コピーしない。
- FCPXMLのフレームレートは30fpsのままにする。
- リージョン順は編集後テキストの順序と一致させる。
- 通常削除境界の空白は、隣接する元WAV区間の間で除外されたサンプル数と一致させる。
- 元音声時間が逆行する境界、または境界間の元音声区間が別位置にも残る境界は並べ替え境界とし、空白を正確に1秒にする。
- 先頭リージョンの`offset`は`0s`とし、末尾へ追加の空白は設けない。
- FCPXMLへtransition、fade、crossfadeを含めない。
- WAV renderの出力配置、クロスフェード、警告を変更しない。

---

### Task 1: FCPXMLの空白配置を実装する

**Files:**
- Modify: `stefnceorf/fcpxml.py`
- Modify: `tests/test_fcpxml.py`
- Modify: `README.md`
- Modify: `README_ja.md`

**Interfaces:**
- Consumes: `EditPlan.output_intervals: list[tuple[float, float]]`、`EditPlan.samplerate: int`、`EditPlan.total_samples: int`
- Produces: `_timeline_offsets(bounds: list[tuple[int, int]], samplerate: int) -> tuple[list[int], int]`。戻り値は各リージョンのタイムライン開始サンプル列と末尾リージョン終了位置。

- [ ] **Step 1: 通常削除尺を空白として残す失敗テストを書く**

```python
def test_deleted_audio_becomes_same_length_timeline_gap(tmp_path):
    # 元区間0-1秒と2-3秒を残す。
    # offsetは0秒と2秒、sequence durationは3秒になることを検証する。
```

- [ ] **Step 2: 並べ替え境界を1秒空ける失敗テストを書く**

```python
def test_reordered_regions_have_one_second_boundary_gaps(tmp_path):
    # 元順A(0-1), B(1-2), C(2-3), D(3-4)をA,C,B,Dへ並べる。
    # retained audioが飛び越し区間に存在する3境界すべてを並べ替えと判定し、
    # offsetが0, 2, 4, 6秒、sequence durationが7秒になることを検証する。
```

- [ ] **Step 3: 対象テストを実行し、連続配置との差で失敗することを確認する**

Run: `pytest -q tests/test_fcpxml.py -k 'deleted_audio_becomes or reordered_regions_have'`

Expected: offsetまたはsequence durationの不一致で2件FAIL。

- [ ] **Step 4: サンプル整数で境界種別とoffsetを計算する**

```python
REORDER_GAP_S = 1


def _timeline_offsets(
    bounds: list[tuple[int, int]], samplerate: int
) -> tuple[list[int], int]:
    if not bounds:
        return [], 0
    offsets = [0]
    destination = bounds[0][1] - bounds[0][0]
    for index in range(1, len(bounds)):
        previous_start, previous_end = bounds[index - 1]
        current_start, current_end = bounds[index]
        source_gap_start = min(previous_end, current_start)
        source_gap_end = max(previous_end, current_start)
        retained_in_gap = any(
            other_index not in (index - 1, index)
            and other_start < source_gap_end
            and other_end > source_gap_start
            for other_index, (other_start, other_end) in enumerate(bounds)
        )
        reordered = current_start < previous_start or retained_in_gap
        gap = samplerate * REORDER_GAP_S if reordered else max(
            0, current_start - previous_end
        )
        destination += gap
        offsets.append(destination)
        destination += current_end - current_start
    return offsets, destination
```

`export_fcpxml()`は有効な区間を先に`bounds`へ量子化し、`_timeline_offsets()`が返した値を各`asset-clip`の`offset`と`sequence duration`へ使う。

- [ ] **Step 5: 対象テストを再実行して通ることを確認する**

Run: `pytest -q tests/test_fcpxml.py -k 'deleted_audio_becomes or reordered_regions_have'`

Expected: `2 passed`。

- [ ] **Step 6: 既存の連続配置前提テストを新仕様へ更新し、全FCPXMLテストを実行する**

Run: `pytest -q tests/test_fcpxml.py`

Expected: 全件PASS（Logic同梱DTDが無い環境だけSKIP可）。

- [ ] **Step 7: READMEの日英説明を空白配置へ更新する**

`README.md`は「通常削除は削除尺、並べ替え境界は1秒の空白。フェードはLogicで設定」と記載する。`README_ja.md`にも同じ仕様を記載し、「連続配置」の旧記述を削除する。

- [ ] **Step 8: 回帰確認を実行する**

Run: `pytest -q`

Expected: 全件PASS（環境依存テストのみSKIP可）。

Run: `git diff --check`

Expected: 出力なし、終了コード0。

- [ ] **Step 9: 実装をコミットする**

```bash
git add README.md README_ja.md stefnceorf/cli.py stefnceorf/render.py stefnceorf/fcpxml.py tests/test_cli.py tests/test_render.py tests/test_fcpxml.py docs/superpowers/plans/2026-07-23-logic-fcpxml-spacing.md
git commit -m "feat: add Logic Pro FCPXML export"
```
