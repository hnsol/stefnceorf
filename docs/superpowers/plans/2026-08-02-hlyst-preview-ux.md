# Hlyst Preview UX and State Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Hlystを「原音・保存済み・現在案」の3段比較UIへ整理し、未保存状態でも位置と削除状態を取り違えず、音声が重ならない試聴操作を実現する。

**Architecture:** サーバーは現在案previewの音声と原音時刻・現在案時刻を返し、ブラウザは保存済み位置と未保存位置を別状態として保持する。通常クリックは現在案previewを再生しながら、原音・保存済みプレイヤーを同じ比較窓の開始位置へシークして待機させる。保存済み状態、現在案状態、ユーザーが指定した削除状態を混ぜない。

**Tech Stack:** Python 3.11+、stdlib HTTP server、インラインHTML/CSS/JavaScript、NumPy、SoundFile、pytest、Node.js

**Prerequisite:** `2026-08-02-guarded-unrecognized-cuts.md` の Task 1〜3を先に実施する。未認識/UTSの requested=true / effective=false は「ポーズ不足のため保持」と表示する。

## Global Constraints

- 未コミットの CLAUDE.md、uv.lock、_skill_transfer.md は変更・stageしない。
- sc review の静的HTMLモードは従来動作を維持し、変更は原則 --serve のUIに限定する。
- 3つのaudio要素が同時に再生されないこと。
- 行・単語の通常クリックは現在案preview専用とし、modifierクリックの再生機能は廃止する。
- 試聴範囲は ±2秒 / ±4秒 / ±8秒、既定は ±4秒。
- デスクトップでは3段目を1行表示し、狭い画面だけ折り返しを許可する。
- 3段目の絶対時刻は「対象 原音 M:SS.d」と「現在案 M:SS.d」だけを表示する。
- TDDで進め、Taskごとに独立コミットする。

---

### Task 1: 保存済み・現在案・削除指定の状態を分離する

**Files:**
- Modify: stefnceorf/review.py（build_review_data、_SERVE_SCRIPT）
- Test: tests/test_review.py

**Interfaces:**
- Produces: segment/wordの source_center、output_center
- Produces: ブラウザ内の保存済み位置表 savedPositions
- Produces: mergePreviewStates() と mergeSavedStates()

- [ ] **Step 1: 失敗テストを書く**

~~~python
def test_preview_merge_does_not_overwrite_requested_or_saved_positions(tmp_path):
    html, _rd = _serve_html(tmp_path)
    body = html.split("function mergePreviewStates(segments) {")[1].split("\n  }")[0]
    assert "effective_delete" in body
    assert "requested_delete" not in body
    assert "savedPositions" not in body


def test_review_data_exposes_center_positions(tmp_path):
    _html, rd = _serve_html(tmp_path)
    seg = rd["segments"][0]
    assert {"source_center", "output_center"} <= set(seg)
    if seg["words"]:
        assert {"center", "output_center"} <= set(seg["words"][0])
~~~

- [ ] **Step 2: FAILを確認する**

~~~bash
uv run pytest tests/test_review.py::test_preview_merge_does_not_overwrite_requested_or_saved_positions tests/test_review.py::test_review_data_exposes_center_positions -v
~~~

- [ ] **Step 3: build_review_dataへ中心位置を追加する**

segmentの中心を source_center とする。保持対象は map_source_sample() で output_centerへ写像し、削除対象は既存の output（結合境界）を output_centerとして使う。wordは既存centerを map_source_sample() で output_centerへ写像する。

画面の「対象 原音」はリストのタイムスタンプと一致させるため、中心ではなく既存の source（開始時刻）を表示する。「現在案」も既存の output（対応する開始時刻／削除時は結合境界）を表示する。source_center / output_center は比較窓のシーク計算専用とする。

- [ ] **Step 4: JavaScriptの状態更新を分離する**

~~~javascript
function capturePositions(segments) {
  // L<id> / W<id>:<word> をキーに保存済みsource/output中心位置を保存
}

function mergePreviewStates(segments) {
  // effective_delete と現在案output_centerだけ更新
  // requested_delete とsavedPositionsは変更しない
}

function mergeSavedStates(segments) {
  // /save成功時だけ全状態を更新しsavedPositionsを再構築
}
~~~

/preview成功時はmergePreviewStates()、/save成功時はmergeSavedStates()を使う。行を削除して戻しても、配下フィラーの個別ON/OFFを失わないこと。

- [ ] **Step 5: テストとコミット**

~~~bash
uv run pytest tests/test_review.py tests/test_serve.py -q
git add stefnceorf/review.py tests/test_review.py
git commit -m "fix: separate saved and preview state in Hlyst"
~~~

---

### Task 2: preview範囲変更の競合をなくす

**Files:**
- Modify: stefnceorf/review.py（_SERVE_SCRIPT）
- Test: tests/test_review.py

- [ ] **Step 1: 失敗テストを書く**

~~~python
def test_preview_freezes_pad_for_request_and_cache_key(tmp_path):
    html, _rd = _serve_html(tmp_path)
    body = html.split("function runPreview(target) {")[1].split("\n  }")[0]
    assert "var requestPad = currentPad;" in body
    assert "pad_s: requestPad" in body
    assert "previewKey(target, requestPad)" in body


def test_busy_disables_pad_buttons(tmp_path):
    html, _rd = _serve_html(tmp_path)
    body = html.split("function setControlsBusy(value")[1].split("\n  }")[0]
    assert ".padbtn" in body
~~~

- [ ] **Step 2: FAILを確認する**

~~~bash
uv run pytest tests/test_review.py::test_preview_freezes_pad_for_request_and_cache_key tests/test_review.py::test_busy_disables_pad_buttons -v
~~~

- [ ] **Step 3: requestPadを固定する**

runPreview()開始時に var requestPad = currentPad を取り、POST body、previewKey、応答entryに同じ値を使う。応答処理中にグローバルcurrentPadを参照しない。

- [ ] **Step 4: busy中の操作を止める**

setControlsBusy()でcheckboxとpadボタンをdisableする。busy中の行クリックは黙って対象を変更せず、「試聴を生成中です」と表示する。

- [ ] **Step 5: preview中と保存中を区別する**

busyKindを null / preview / save で管理する。preview中は保存ボタンの文言を維持したままdisable、save中だけ「保存中…」と表示する。

- [ ] **Step 6: テストとコミット**

~~~bash
uv run pytest tests/test_review.py tests/test_serve.py -q
git add stefnceorf/review.py tests/test_review.py
git commit -m "fix: prevent preview range races in Hlyst"
~~~

---

### Task 3: 3段UIを1行構成へ整理する

**Files:**
- Modify: stefnceorf/review.py（HTML template、preview panel、CSS、凡例）
- Test: tests/test_review.py

- [ ] **Step 1: 失敗テストを書く**

~~~python
def test_serve_ui_has_three_compact_player_rows(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert html.count('class="player-row') == 3
    assert "対象 原音" in html
    assert "現在案" in html
    assert 'id="abs-window"' not in html
    assert 'id="abs-pos"' not in html
    assert "Shift+クリック" not in html
    assert "Option(Alt)+クリック" not in html


def test_top_player_labels_are_short_and_equal_width(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert ">原音</button>" in html
    assert ">保存済み</button>" in html
    mode_css = html.split(".mode {")[1].split("}")[0]
    assert "width:" in mode_css
~~~

- [ ] **Step 2: FAILを確認する**

~~~bash
uv run pytest tests/test_review.py::test_serve_ui_has_three_compact_player_rows tests/test_review.py::test_top_player_labels_are_short_and_equal_width -v
~~~

- [ ] **Step 3: 次の3段へ変更する**

~~~text
[原音        ] [audio]                                      [変更を保存]
[保存済み    ] [audio]
[現在案：保持] [audio] [範囲 2秒 | 4秒 | 8秒] 対象 原音 3:53.9 → 現在案 3:47.3
~~~

実装規則:

- 上2段の左ボタンは同じ固定幅。
- 保存ボタンは1段目右端。原音操作とは余白・区切りで分離。
- 3段目はデスクトップでnowrap、狭い画面のmedia queryだけwrap。
- 秒数は各ボタンへ±を繰り返さず、前に「範囲」と一度だけ表示。
- 範囲選択は青いアクション色を使わず、中立色のsegmented controlにする。
- 選択中はborder、薄い背景、font-weightで表現する。
- 3段目の状態は「現在案：保持／削除／安全判定により保持」。
- 「対象 原音」はリストと同じsource開始時刻を表示し、「現在案」は対応するoutput開始時刻を表示する。
- 試聴範囲、再生位置、クリックで再生、modifierクリック凡例を削除する。
- serveモードへ固定の「前後2秒」を表示しない。

- [ ] **Step 4: 状態ラベルを結線する**

preview応答のmessageを3段目左ラベルへ設定する。保存通知は保存ボタン横、previewエラーは3段目内へ表示し、混ぜない。

- [ ] **Step 5: テストとコミット**

~~~bash
uv run pytest tests/test_review.py -q
git add stefnceorf/review.py tests/test_review.py
git commit -m "feat: simplify Hlyst into three comparison rows"
~~~

---

### Task 4: 行クリックで3プレイヤーを同じ比較位置へ揃える

**Files:**
- Modify: stefnceorf/review.py（_SERVE_SCRIPT）
- Test: tests/test_review.py

- [ ] **Step 1: 失敗テストを書く**

~~~python
def test_normal_click_primes_source_and_saved_then_plays_preview(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert "function primeComparisonPlayers(target, pad) {" in html
    assert "primeComparisonPlayers(target, currentPad);" in html
    assert "playCurrentPlan(target);" in html
    assert "ev.shiftKey" not in html
    assert "ev.altKey" not in html
~~~

- [ ] **Step 2: FAILを確認する**

~~~bash
uv run pytest tests/test_review.py::test_normal_click_primes_source_and_saved_then_plays_preview -v
~~~

- [ ] **Step 3: 同期シークを実装する**

~~~javascript
function comparisonStart(center, pad) {
  return Math.max(0, center - pad);
}

function primeComparisonPlayers(target, pad) {
  // 原音を target.sourceCenter - pad へシーク
  // 保存済みを savedPositions[targetKey].outputCenter - pad へシーク
  // 上2段ではplay()を呼ばない
}
~~~

通常クリックの順序:

1. 対象行をactive表示。
2. 原音・保存済みプレイヤーを比較窓先頭へシーク。
3. 現在案previewをキャッシュ再生。なければ生成して再生。

範囲変更時もcurrentTargetがあれば上2段を再シークしてから現在案previewを再生成する。保存済みシークに未保存プランのoutputを使わない。

- [ ] **Step 4: テストとコミット**

~~~bash
uv run pytest tests/test_review.py tests/test_serve.py -q
git add stefnceorf/review.py tests/test_review.py
git commit -m "feat: synchronize Hlyst comparison positions"
~~~

---

### Task 5: 排他再生とSpaceキー操作を追加する

**Files:**
- Modify: stefnceorf/review.py（_SERVE_SCRIPT）
- Test: tests/test_review.py

- [ ] **Step 1: 失敗テストを書く**

~~~python
def test_serve_script_pauses_other_audio_players(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert "function pauseOtherPlayers(active) {" in html
    assert 'player.addEventListener("play"' in html
    assert "other.pause();" in html


def test_space_toggles_last_player_but_ignores_controls(tmp_path):
    html, _rd = _serve_html(tmp_path)
    assert 'document.addEventListener("keydown"' in html
    assert 'ev.code !== "Space"' in html
    assert 'closest("input, button, select, textarea, audio' in html
    assert "ev.preventDefault();" in html
    assert "lastPlayer.pause();" in html
    assert "lastPlayer.play();" in html
~~~

- [ ] **Step 2: FAILを確認する**

~~~bash
uv run pytest tests/test_review.py::test_serve_script_pauses_other_audio_players tests/test_review.py::test_space_toggles_last_player_but_ignores_controls -v
~~~

- [ ] **Step 3: 排他制御を追加する**

~~~javascript
var allPlayers = [audioSrc, audioOut, preview];
var lastPlayer = preview;

function pauseOtherPlayers(active) {
  allPlayers.forEach(function (other) {
    if (other !== active) { other.pause(); }
  });
  lastPlayer = active;
}
~~~

各audioのplayイベントでpauseOtherPlayers(player)を呼ぶ。

- [ ] **Step 4: Spaceキー操作を追加する**

次の場合はSpaceを横取りしない:

- input、button、select、textarea、audio
- contenteditable内
- metaKey、ctrlKey、altKey
- key repeat

それ以外ではpreventDefault()し、lastPlayerをplay/pauseする。preview未生成なら何もしない。行クリック後はpreviewをlastPlayerにする。

- [ ] **Step 5: テストとコミット**

~~~bash
uv run pytest tests/test_review.py -q
git add stefnceorf/review.py tests/test_review.py
git commit -m "feat: add exclusive playback and space control to Hlyst"
~~~

---

### Task 6: 統合検証と文書同期

**Files:**
- Modify only if relevant: README.md、llms.txt
- Test: tests/test_review.py、tests/test_serve.py、tests/test_render.py、tests/test_cli.py

- [ ] **Step 1: 対象テストを実行する**

~~~bash
uv run pytest tests/test_review.py tests/test_serve.py tests/test_render.py tests/test_cli.py -q
~~~

Expected: 全件PASS。

- [ ] **Step 2: 全テストを実行する**

~~~bash
uv run pytest -q
~~~

Expected: 全件PASS。

- [ ] **Step 3: 実ブラウザで確認する**

実データで uv run sc review <episode.sc.txt> --serve を起動し、順に確認する。

1. 行クリックで現在案だけが再生される。
2. 同時に原音・保存済みの開始位置が比較窓先頭へ移動する。
3. 任意のaudioを再生すると他2つが停止する。
4. Spaceで最後に操作したaudioだけが再生・停止する。
5. フォーム操作中のSpaceは標準操作を壊さない。
6. 2/4/8秒変更時に3プレイヤーの開始位置が更新される。
7. 行を削除して復活しても、その行内のフィラー個別ON/OFFが変わらない。
8. 未保存状態でも保存済みプレイヤーは保存済み位置へシークする。
9. 3段目が通常のデスクトップ幅で1行表示される。
10. 保存ボタンがpreview中に「保存中…」と誤表示されない。

- [ ] **Step 4: README/llms.txtに旧UI説明がある場合だけ更新する**

記載内容:

- 通常クリック＝現在案preview
- 範囲2/4/8秒
- 原音・保存済みを同じ比較位置へ準備
- Spaceで最後のプレイヤーを再生・停止

- [ ] **Step 5: 文書変更がある場合だけコミットする**

~~~bash
git add README.md llms.txt
git commit -m "docs: update Hlyst comparison controls"
~~~

## Completion Criteria

- 保存済み位置と現在案位置を混同しない。
- preview応答でユーザーの削除指定状態を上書きしない。
- pad変更中に異なる長さの音声を誤キャッシュしない。
- デスクトップで原音・保存済み・現在案が3段に揃う。
- 通常クリックで現在案を試聴し、上2段は比較開始位置へ待機する。
- 常に鳴る音声は1つだけ。
- Spaceで最後のプレイヤーを停止・再開できる。
- 対象テスト、全テスト、実ブラウザ確認が完了している。
