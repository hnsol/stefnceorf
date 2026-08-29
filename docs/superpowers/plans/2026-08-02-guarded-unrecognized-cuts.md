# Guarded Unrecognized Cuts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 未認識区間行・未転写発話（UTS）行を削除するとき、元位置の検出済み無音を合計0.25秒残せる場合だけ削除し、足りなければ音声を保持する。

**Architecture:** フィラーの既存安全判定は変更しない。未認識/UTS専用の guarded cut 判定を `render.py` に追加し、削除対象の前後に連続する既知無音だけから0.25秒を割り当てる。デジタル無音の生成、別位置のルームトーン転用、長いクロスフェードによる疑似ポーズは行わない。

**Tech Stack:** Python 3.11+、NumPy、SoundFile、pytest、既存の EditPlan / Hlyst review data

## Global Constraints

- 対象は `kind: "unrecognized"` の行削除と、`reason: "uts"` の〔〕削除・行削除だけ。
- フィラー削除の `plan_filler_cut()` と現在の挙動は変更しない。
- 目標ポーズは合計 `0.25` 秒。基本は前後 `0.125` 秒ずつ、片側不足分はもう片側へ再配分する。
- 合計0.25秒を確保できない場合は削除を実行せず、対象音声全体を保持する。
- ポーズとして数えてよいのは `.sc.json` の検出済み無音のうち、対象の元位置で前後の発話へ連続している部分だけ。
- 単なる時間上の gap、未知区間、別位置のルームトーンをポーズとして数えない。
- デジタル無音を生成・挿入しない。
- クロスフェードは既存のクリック防止 `FADE_S` の用途に限定し、間を作るために延長しない。
- WAV、preview、Logic/FCPXML が同じ `EditPlan.output_intervals` を使う状態を維持する。
- 未コミットの `CLAUDE.md`、`uv.lock`、`_skill_transfer.md` は変更・stageしない。
- TDDで進め、Taskごとに独立コミットする。

---

### Task 1: 元位置の既知無音から0.25秒を配分する純関数を追加する

**Files:**
- Modify: `stefnceorf/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Produces: `UNRECOGNIZED_PAUSE_KEEP_S = 0.25`
- Produces: `plan_guarded_pause(...) -> tuple[tuple[float, float], tuple[float, float]] | None`
- Does not modify: `FILLER_PAUSE_KEEP_S`、`plan_filler_cut()`

- [ ] **Step 1: 失敗テストを書く**

`tests/test_render.py` に次を追加する。戻り値は「左側に残す元音声区間」「右側に残す元音声区間」とする。

```python
def test_guarded_pause_keeps_125ms_on_each_side():
    result = render.plan_guarded_pause(
        cut_start=1.30,
        cut_end=1.70,
        previous_end=1.00,
        next_start=2.00,
        silence_spans=[(1.00, 1.30), (1.70, 2.00)],
    )
    assert result == (
        (pytest.approx(1.00), pytest.approx(1.125)),
        (pytest.approx(1.875), pytest.approx(2.00)),
    )


def test_guarded_pause_redistributes_short_left_side():
    result = render.plan_guarded_pause(
        cut_start=1.05,
        cut_end=1.70,
        previous_end=1.00,
        next_start=2.00,
        silence_spans=[(1.00, 1.05), (1.70, 2.00)],
    )
    assert result == (
        (pytest.approx(1.00), pytest.approx(1.05)),
        (pytest.approx(1.80), pytest.approx(2.00)),
    )


def test_guarded_pause_rejects_when_total_known_silence_is_short():
    assert render.plan_guarded_pause(
        cut_start=1.10,
        cut_end=1.90,
        previous_end=1.00,
        next_start=2.00,
        silence_spans=[(1.00, 1.10), (1.90, 2.00)],
    ) is None


def test_guarded_pause_does_not_count_unknown_gap():
    assert render.plan_guarded_pause(
        cut_start=1.30,
        cut_end=1.70,
        previous_end=1.00,
        next_start=2.00,
        silence_spans=[(1.05, 1.30), (1.70, 1.95)],
    ) is None
```

最後のテストでは既知無音自体は合計0.5秒あるが、`previous_end→silence start` と
`silence end→next_start` が未知なので利用禁止とする。

- [ ] **Step 2: FAILを確認する**

```bash
uv run pytest \
  tests/test_render.py::test_guarded_pause_keeps_125ms_on_each_side \
  tests/test_render.py::test_guarded_pause_redistributes_short_left_side \
  tests/test_render.py::test_guarded_pause_rejects_when_total_known_silence_is_short \
  tests/test_render.py::test_guarded_pause_does_not_count_unknown_gap -v
```

Expected: `plan_guarded_pause` が存在せず FAIL。

- [ ] **Step 3: 最小実装を追加する**

```python
UNRECOGNIZED_PAUSE_KEEP_S = 0.25


def plan_guarded_pause(
    cut_start: float,
    cut_end: float,
    previous_end: float,
    next_start: float,
    silence_spans: list[tuple[float, float]] | None,
    pause_keep: float = UNRECOGNIZED_PAUSE_KEEP_S,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """元位置で発話境界に連続する既知無音を pause_keep 秒残す。

    左は previous_end から連続する既知無音、右は next_start まで連続する
    既知無音だけを容量に数える。基本は半分ずつ配分し、片側不足分を他方へ
    再配分する。合計を確保できなければ None。
    """
```

実装規則:

- `silence_spans is None` または空なら `None`。
- 左容量は `previous_end` に接する既知無音の連続部分だけ。右容量は `next_start` に接する連続部分だけ。
- 接続許容は既存 `SILENCE_EDGE_TOL_S` を使う。
- 重複・接触する silence span は先にマージする。
- 左右容量の合計が `pause_keep - 1e-6` 未満なら `None`。
- `pause_keep / 2` を左右へ割り当て、片側不足分を他方へ回す。
- 戻す区間は発話側へ接続したままにする。左は `previous_end` から、右は `next_start` までを残す。
- サンプル丸めは後段の既存 `plan_chunk_bounds()` に任せ、ここでは秒の純関数に留める。

- [ ] **Step 4: テストを通す**

```bash
uv run pytest tests/test_render.py -q
```

- [ ] **Step 5: コミットする**

```bash
git add stefnceorf/render.py tests/test_render.py
git commit -m "feat: plan source-local pauses for unrecognized cuts"
```

---

### Task 2: 未認識/UTSの削除を guarded cut として EditPlan に反映する

**Files:**
- Modify: `stefnceorf/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `plan_guarded_pause()`
- Produces: 未認識/UTS専用の effective cut / unsafe keep 判定
- Produces: exact cut gap を単純結合する `exact_cut_spans` 経路
- Produces: `EditPlan.guarded_cut_reasons: dict[str, str]`

- [ ] **Step 1: 失敗する統合テストを書く**

合成音声では、前の発話・削除対象・後の発話を異なる周波数にする。既知無音を左右に配置し、出力サンプル長と周波数列の両方を確認する。

```python
def test_deleted_unrecognized_row_keeps_full_250ms_source_pause(tmp_path):
    # 左0.125s + 右0.125sを元位置から残し、中間の未認識音声だけ消える。
    plan = _build_deleted_unrecognized_plan(
        tmp_path, left_silence=0.20, right_silence=0.20
    )
    assert _middle_tone_is_absent(plan)
    assert _pause_between_outer_tones(plan) == pytest.approx(0.25, abs=1/16000)


def test_deleted_unrecognized_row_redistributes_pause(tmp_path):
    plan = _build_deleted_unrecognized_plan(
        tmp_path, left_silence=0.05, right_silence=0.30
    )
    assert _pause_between_outer_tones(plan) == pytest.approx(0.25, abs=1/16000)


def test_deleted_unrecognized_row_is_kept_when_pause_is_short(tmp_path):
    plan = _build_deleted_unrecognized_plan(
        tmp_path, left_silence=0.10, right_silence=0.10
    )
    assert _middle_tone_is_present(plan)
    assert any("ポーズを0.25秒確保できないため残しました" in w for w in plan.warnings)


def test_deleted_uts_row_and_bracketed_uts_share_guard(tmp_path):
    line_deleted = _build_uts_plan(tmp_path / "line", mode="line-delete")
    bracketed = _build_uts_plan(tmp_path / "bracket", mode="bracket-delete")
    assert line_deleted.output_intervals == bracketed.output_intervals


def test_guarded_cut_does_not_use_unknown_gap_as_pause(tmp_path):
    plan = _build_deleted_unrecognized_plan(
        tmp_path,
        left_silence=0.20,
        right_silence=0.20,
        omit_silence_metadata=True,
    )
    assert _middle_tone_is_present(plan)
```

- [ ] **Step 2: FAILを確認する**

```bash
uv run pytest tests/test_render.py -k "deleted_unrecognized or deleted_uts_row_and_bracketed or guarded_cut_does_not" -v
```

- [ ] **Step 3: 削除要求を先に分類する**

`build_edit_plan_from_text()` 冒頭で、JSONの unrecognized ID と TXT の状態から次を分類する。

```python
requested_unrecognized_line_ids: set[str]  # TXTから行ごと消えた未認識/UTS
requested_uts_bracket_ids: set[str]        # 行はあるが〔〕が残るUTS
guarded_requested_ids = requested_unrecognized_line_ids | requested_uts_bracket_ids
```

通常speech行の削除はこの集合へ入れず、従来の無条件削除＋hot-boundary警告を維持する。

- [ ] **Step 4: guarded cut の順序と境界を確定する**

各対象について、JSON元順の直前・直後コンテンツ境界を `seg_bounds` から取得し、`plan_guarded_pause()` を呼ぶ。

- 成功時: 対象音声を `content_spans` から除外し、返された左右の元音声区間を `plan_intervals` の対象位置へ追加する。
- 失敗時: 対象の `source_start/source_end` 全体を `plan_intervals` の対象位置へ戻し、effective は保持とする。
- 行削除時も、対象位置はJSON元順の前後アンカー間とする。
- 前後アンカーが並べ替えで逆転し一意に挿入できない場合は安全側へ倒し、対象音声を元位置の隣接アンカー側へ保持して警告する。通常speech行の並べ替えは変更しない。

左右の保持区間を追加しただけでは短い gap が後段で再結合され、削除音声が復活する。そのため `plan_output_intervals()` に専用引数を追加する。

```python
def plan_output_intervals(
    intervals,
    word_spans,
    gap_threshold=GAP_THRESHOLD_S,
    gap_max=GAP_MAX_S,
    filler_spans=None,
    filler_pause_keep=FILLER_PAUSE_KEEP_S,
    silence_spans=None,
    exact_cut_spans: list[tuple[float, float]] | None = None,
):
```

`exact_cut_spans` と `(pe, ns)` が重なる場合は、`pending` と `cur` を `cf_flags=False` で単純結合し、gap音声を復活させない。ここでは追加のポーズ配分をしない。0.25秒は `plan_guarded_pause()` が返した左右区間として既に `intervals` に含まれている。

safe時に登録するexact spanは、元の対象spanそのものではなく
`(left_keep.end, right_keep.start)` とする。これにより対象音声と余分な周辺無音を削除し、
出力へ残る元位置無音を正確に0.25秒にする。

分岐順は次で固定する。

1. ソース逆行
2. exact guarded cut
3. 通常削除単語のクロスフェード
4. 既存フィラーポーズ
5. 通常無音トリム

`exact_cut_spans` は未認識/UTSだけに使い、`filler_spans` の意味とテストを変更しない。

- [ ] **Step 5: unsafe時の警告と統計を更新する**

`EditStats` に次を追加する。

```python
unrecognized_requested: int = 0
unrecognized_cut: int = 0
unrecognized_skipped_unsafe: int = 0
unrecognized_cut_removed_s: float = 0.0
```

UTSは既存 `uts_*` に集計し、通常未認識行は新しい `unrecognized_*` に集計する。`lines_deleted` はTXT上の削除要求数を維持し、`lines_deleted_removed_s` は実際に削除できた行だけを合算する。

`EditPlan` にはセグメント単位の安全判定理由を追加する。

```python
guarded_cut_reasons: dict[str, str]  # "pause" / "boundary"。安全に削除できたIDは含めない
```

今回追加するポーズ不足は `"pause"`。既存UTS境界判定で残す場合は `"boundary"`。
Hlystがwarning文字列を解析して理由を推測する実装は禁止する。

警告文は理由を明確にする。

```text
[0010] 未認識区間は元位置のポーズを0.25秒確保できないため残しました
[0010] 未転写発話〔1.2秒〕は元位置のポーズを0.25秒確保できないため残しました
```

- [ ] **Step 6: WAVとEditPlanの統合テストを通す**

```bash
uv run pytest tests/test_render.py -q
```

- [ ] **Step 7: コミットする**

```bash
git add stefnceorf/render.py tests/test_render.py
git commit -m "feat: guard unrecognized cuts with source pauses"
```

---

### Task 3: Hlystへ「ポーズ不足のため保持」を反映する

**Files:**
- Modify: `stefnceorf/review.py`
- Modify: `stefnceorf/serve.py`（必要な場合のみ。状態データの生成元がreview.pyだけなら変更しない）
- Test: `tests/test_review.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: EditPlanに復元された unrecognized/UTS 音声
- Produces: requested=true / effective=false の状態表示
- Consumes: `EditPlan.guarded_cut_reasons`
- Produces: review data各segmentの `unsafe_reason` と `state`

- [ ] **Step 1: 失敗テストを書く**

```python
def test_plain_unrecognized_uses_safety_label_when_pause_is_short():
    assert review.state_label(
        True, False, kind="unrecognized", reason=None,
        unsafe_reason="pause",
    ) == "ポーズ不足のため保持"


def test_hlyst_preview_reports_requested_but_kept_for_short_pause(tmp_path):
    session, _txt = _short_pause_unrecognized_session(tmp_path)
    data = session.review_data(session.raw_text, session.plan)
    seg = next(s for s in data["segments"] if s["kind"] == "unrecognized")
    assert seg["requested_delete"] is True
    assert seg["effective_delete"] is False
    assert seg["state"] == "ポーズ不足のため保持"
```

- [ ] **Step 2: FAILを確認する**

```bash
uv run pytest \
  tests/test_review.py::test_plain_unrecognized_uses_safety_label_when_pause_is_short \
  tests/test_serve.py::test_hlyst_preview_reports_requested_but_kept_for_short_pause -v
```

- [ ] **Step 3: 状態文言を安全判定へ揃える**

現在の plain unrecognized 用「削除（無条件）」を廃止する。requested/effective の組み合わせは次とする。

```text
requested=false, effective=false  -> 保持
requested=true,  effective=true   -> 削除
requested=true,  effective=false  -> ポーズ不足のため保持
```

境界RMSなど別理由のunsafeを既存どおり表示する必要があるため、EditPlanまたはreview dataに `unsafe_reason` を持たせる。値は最低限 `"pause"` / `"boundary"` / `None` とし、画面文言を推測で組み立てない。

`build_review_data()` は `plan.guarded_cut_reasons.get(seg_id)` を `unsafe_reason` として
各segmentへ載せ、同じ値を渡した `state_label()` の結果を `state` に載せる。

Hlystのチェックボックスは削除要求を表すためONのままにし、取消線は `effective_delete` のときだけ付ける。この方針は既存requested/effective分離を維持する。

- [ ] **Step 4: preview/saveの状態一致を確認する**

`POST /preview` と `POST /save` がどちらも `build_edit_plan_from_text()` を通り、同じ対象が同じeffective結果になることを統合テストで確認する。保存時にunsafe項目の行や〔〕を勝手に復元・解除しない。

```python
def test_preview_and_save_agree_on_pause_unsafe_state(tmp_path):
    session, _txt = _short_pause_unrecognized_session(tmp_path)
    preview = session.preview(_delete_toggle(), _target())
    saved = session.save(_delete_toggle(), expected_digest=session.digest)
    assert _state(preview) == _state(saved) == (True, False)
```

- [ ] **Step 5: テストとコミット**

```bash
uv run pytest tests/test_review.py tests/test_serve.py -q
git add stefnceorf/review.py stefnceorf/serve.py tests/test_review.py tests/test_serve.py
git commit -m "feat: show pause-guarded cuts in Hlyst"
```

---

### Task 4: 全経路の回帰検証と実音声確認を行う

**Files:**
- Modify: `docs/superpowers/specs/2026-08-01-uts-detection-design.md`
- Modify: `docs/superpowers/specs/2026-08-01-review-html-design.md`（状態表示の説明がある場合のみ）
- Test: `tests/test_render.py`
- Test: `tests/test_review.py`
- Test: `tests/test_serve.py`
- Test: `tests/test_fcpxml.py`

- [ ] **Step 1: フィラー回帰テストを追加する**

```python
def test_guarded_unrecognized_policy_does_not_change_filler_short_pause_behavior():
    out, flags = render.plan_output_intervals(
        [(0.0, 1.0), (1.15, 2.15)],
        [],
        filler_spans=[(1.05, 1.10)],
        filler_pause_keep=0.25,
    )
    kept = (out[0][1] - 1.0) + (1.15 - out[1][0])
    assert kept == pytest.approx(0.10)
    assert flags == [False]
```

これは、今回の「0.25秒未満なら保持」をフィラーへ誤適用していないことを固定する。

- [ ] **Step 2: 出力形式の一致を検証する**

同じ `EditPlan` について次を確認する。

- `render_plan()` のWAVが0.25秒の元位置ポーズを含む。
- `render_plan_window()` の該当窓が全体renderとサンプル一致する。
- Logic/FCPXML側が `output_intervals` と同じ保持区間を参照する。
- synthesized silence用の新規chunk・asset・APIが存在しない。

- [ ] **Step 3: 全テストを実行する**

```bash
uv run pytest -q
```

Expected: 全テストPASS。

- [ ] **Step 4: 実データで確認する**

ep22-4の問題箇所を `sc review --serve` で確認する。

1. 元位置に0.25秒の既知無音がある未認識/UTSをONにする。
2. 現在案previewで削除後に自然な間が残ることを聴く。
3. 0.25秒ない対象をONにする。
4. 「ポーズ不足のため保持」と表示され、音声が残ることを聴く。
5. 保存後のreview.wavとLogic/FCPXMLでも結果が一致することを確認する。

- [ ] **Step 5: specを更新してコミットする**

specには次の決定を記録する。

- 未認識/UTS削除は元位置の既知無音0.25秒を必須とする。
- 不足時はunsafe保持。
- フィラーの既存方針は変更しない。
- デジタル無音、別位置ルームトーン、長いクロスフェードは採用しない。

```bash
git add docs/superpowers/specs tests stefnceorf
git commit -m "docs: define guarded unrecognized cut policy"
```

## 前のHlyst UI計画との関係

この計画は `2026-08-02-hlyst-preview-ux.md` と統合せず、先に実行する。

1. 本計画 Task 1〜2: 音声編集結果を確定
2. 本計画 Task 3: Hlystへeffective/unsafe理由を公開
3. `2026-08-02-hlyst-preview-ux.md`: 3段UI・再生位置・排他再生を実装
4. 本計画 Task 4とHlyst計画の最終検証をまとめて実施

分離理由は、音声カット方針をUI変更とは独立してレビュー・差し戻しできるため。Hlyst計画側は表示文言を「安全判定により保持」から「ポーズ不足のため保持」へ読み替える。
