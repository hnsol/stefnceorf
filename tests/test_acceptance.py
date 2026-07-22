"""設計書 §12 の受け入れテスト。

実モデル(mlx_whisper)は使わず、合成 wav と手書きの .sc.json / .sc.txt を
fixture として生成して render の挙動を検証する。

各「単語」は識別可能な別周波数の sine 波として合成し、周波数を見れば出力に
どの単語がどの順で含まれるかを判定できる。波形は全体を位相連続に合成する
（単語境界で値の不連続を作らない）ため、音質(不連続)検証の基準にできる。
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from stefnceorf import render

SR = 48000
WORD_DUR = 1.0  # 1単語=1秒


# ---- 合成プロジェクト生成 -------------------------------------------------

def _build_project(tmp_path, segments, sr=SR, subtype="PCM_16", blocks=None,
                   silences=None, zero_ranges=None):
    """合成 wav + .sc.json を生成し、(wav パス, 無編集の txt 本文) を返す。

    segments: [[(word_str, freq_hz), ...], ...]  各内側リストが1セグメント。
    単語は 1秒ずつ、ソース時間軸で連続配置する。波形は位相連続。
    blocks: segments と同形の [[block_idx, ...], ...]。指定時は各 word の json に
    "block" を付与する（ポーズ区切り＝ブロック単位削除の検証用）。
    silences: [[s, e], ...]。指定時は json に "silences" を付与する（フィラー
    精密カットの音響安全判定用）。
    zero_ranges: [[s, e], ...]。指定時は波形の該当秒区間を無音化する（json に
    silences を書かない旧形式で、境界 RMS を実際に下げて音響安全判定を通す用）。
    """
    inst_freq_parts: list[np.ndarray] = []
    json_segs: list[dict] = []
    default_lines: list[str] = []

    t_cursor = 0.0
    n_per_word = int(WORD_DUR * sr)
    for si, seg in enumerate(segments):
        seg_id = f"{si + 1:04d}"
        words = []
        text = ""
        for wi, (word, freq) in enumerate(seg):
            inst_freq_parts.append(np.full(n_per_word, float(freq)))
            wd = {
                "word": word,
                "start": t_cursor,
                "end": t_cursor + WORD_DUR,
                "probability": 0.9,
            }
            if blocks is not None:
                wd["block"] = blocks[si][wi]
            words.append(wd)
            text += word
            t_cursor += WORD_DUR
        json_segs.append({"id": seg_id, "text": text, "words": words})
        default_lines.append(f"[{seg_id}] {text}")

    inst_freq = np.concatenate(inst_freq_parts)
    # 位相連続合成: 瞬時周波数を積分して位相にする
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr
    audio = 0.3 * np.sin(phase)

    if zero_ranges is not None:
        for zs, ze in zero_ranges:
            a = max(0, int(round(zs * sr)))
            b = min(len(audio), int(round(ze * sr)))
            if b > a:
                audio[a:b] = 0.0

    wav = tmp_path / "input.wav"
    sf.write(str(wav), audio, sr, subtype=subtype)

    data = {
        "source_wav": str(wav.resolve()),
        "language": "ja",
        "model": "test",
        "segments": json_segs,
    }
    if silences is not None:
        data["silences"] = silences
    (tmp_path / "input.sc.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return wav, "\n".join(default_lines) + "\n"


# ---- 周波数解析ヘルパ -----------------------------------------------------

def _dominant_freq(chunk, sr):
    """chunk の支配周波数(DCを除く最大振幅ビン)を返す。"""
    spec = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
    spec[0] = 0.0
    return freqs[int(np.argmax(spec))]


def _freq_sequence(audio, sr, known, win=0.05, tol=40.0, amp=1e-3):
    """出力を窓ごとに解析し、支配周波数を known にスナップして
    連続重複を畳んだ「出現順」リストを返す。境界の曖昧窓は捨てる。
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    n = int(win * sr)
    seq: list[float] = []
    for i in range(0, len(audio) - n + 1, n):
        chunk = audio[i : i + n]
        if np.max(np.abs(chunk)) < amp:
            continue
        f = _dominant_freq(chunk, sr)
        snapped = min(known, key=lambda k: abs(k - f))
        if abs(snapped - f) > tol:
            continue  # クロスフェード等の曖昧窓は無視
        if not seq or seq[-1] != snapped:
            seq.append(snapped)
    return seq


def _present_freqs(audio, sr, known):
    return set(_freq_sequence(audio, sr, known))


# ==========================================================================
# §12-1 往復テスト: 無編集 render → ほぼ同尺・全単語が順に存在
# ==========================================================================

def test_roundtrip_no_edit(tmp_path):
    freqs = [200, 400, 600]
    _, default_txt = _build_project(tmp_path, [[("あ", 200)], [("い", 400)], [("う", 600)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text(default_txt, encoding="utf-8")

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    assert sr == SR
    # 元は 3秒=144000サンプル。クロスフェード分わずかに短いがマージンで概ね同尺
    assert abs(len(audio) - 3 * SR) < SR // 5
    assert _freq_sequence(audio, sr, freqs) == freqs


# ==========================================================================
# §12-2 行削除: 1行消す → 該当セグメントの周波数が出力に無い
# ==========================================================================

def test_delete_line(tmp_path):
    freqs = [200, 400, 600]
    _build_project(tmp_path, [[("あ", 200)], [("い", 400)], [("う", 600)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あ\n[0003] う\n", encoding="utf-8")  # 0002 削除

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    present = _present_freqs(audio, sr, freqs)
    assert 400 not in present
    assert present == {200, 600}
    assert _freq_sequence(audio, sr, freqs) == [200, 600]


# ==========================================================================
# §12-3 行内削除: 数文字消す → 対応単語のみ消え前後が繋がる
# ==========================================================================

def test_intra_line_delete(tmp_path):
    freqs = [200, 400, 600]
    # 1セグメントに3単語
    _build_project(tmp_path, [[("あ", 200), ("い", 400), ("う", 600)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あう\n", encoding="utf-8")  # 中央「い」を削除

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    present = _present_freqs(audio, sr, freqs)
    assert 400 not in present
    assert _freq_sequence(audio, sr, freqs) == [200, 600]
    # 「い」(1秒)が消え約2秒に短縮、前後が繋がる
    assert abs(len(audio) - 2 * SR) < SR // 4


# ==========================================================================
# §11.5 ポーズ区切り: ブロック内1語削除 → 同ブロック全語が出力から消える
# ==========================================================================

def test_block_snap_removes_whole_block(tmp_path):
    freqs = [200, 400, 600]
    # 1セグメント3語。blocks=[0,0,1]（あ・い が同ブロック、う が別ブロック）
    _build_project(
        tmp_path,
        [[("あ", 200), ("い", 400), ("う", 600)]],
        blocks=[[0, 0, 1]],
    )
    txt = tmp_path / "input.sc.txt"
    # block0 の「い」だけ削除 → block0（あ・い）全体が消え「う」のみ残る
    txt.write_text("[0001] あ／う\n", encoding="utf-8")

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    present = _present_freqs(audio, sr, freqs)
    assert 200 not in present  # 巻き込み削除
    assert 400 not in present  # 直接削除
    assert present == {600}
    assert _freq_sequence(audio, sr, freqs) == [600]


# ==========================================================================
# §12-4 並べ替え: 2行入れ替え → 周波数出現順が入れ替わる
# ==========================================================================

def test_reorder_lines(tmp_path):
    freqs = [200, 400, 600]
    _build_project(tmp_path, [[("あ", 200)], [("い", 400)], [("う", 600)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0002] い\n[0001] あ\n[0003] う\n", encoding="utf-8")  # 1と2入替

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    assert _freq_sequence(audio, sr, freqs) == [400, 200, 600]


# ==========================================================================
# §12-5 フィラー: 〔まあ〕残し→消える / 括弧外し→残る
# ==========================================================================

def test_filler_bracket_kept_is_removed(tmp_path):
    freqs = [200, 400, 600]
    # 中央単語「まあ」= 400Hz。前後は「まあ」と文字が衝突しない語にする。
    # まあ[1,2] の両境界に無音を用意し音響安全判定（§4c）を通す。谷スナップは
    # 実波形の RMS 最小点へ寄せるため、宣言した silences と同区間を実際に無音化する
    # （実録音では宣言無音＝実低RMSであることを反映）。
    _build_project(
        tmp_path, [[("か", 200), ("まあ", 400), ("き", 600)]],
        silences=[[0.98, 1.02], [1.98, 2.02]],
        zero_ranges=[[0.98, 1.02], [1.98, 2.02]],
    )
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] か〔まあ〕き\n", encoding="utf-8")  # 括弧残し

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    present = _present_freqs(audio, sr, freqs)
    assert 400 not in present  # フィラーとして削除
    assert _freq_sequence(audio, sr, freqs) == [200, 600]


def test_filler_bracket_removed_is_kept(tmp_path):
    freqs = [200, 400, 600]
    _build_project(tmp_path, [[("か", 200), ("まあ", 400), ("き", 600)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] かまあき\n", encoding="utf-8")  # 括弧を外す→残す

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    assert _freq_sequence(audio, sr, freqs) == [200, 400, 600]


# --------------------------------------------------------------------------
# 複数フィラーの一括削除: 1行に〔〕2つ残す → 両方消え他の語は全部残る
# --------------------------------------------------------------------------

def test_multiple_fillers_deleted_in_one_line(tmp_path):
    freqs = [200, 400, 600, 800, 1000]
    # か・まあ(filler)・き・えっと(filler)・く。まあ[1,2] えっと[3,4] の各境界に
    # 無音を用意し音響安全判定を通す。谷スナップ用に宣言無音と同区間を実無音化する。
    _build_project(
        tmp_path,
        [[("か", 200), ("まあ", 400), ("き", 600), ("えっと", 800), ("く", 1000)]],
        silences=[[0.98, 1.02], [1.98, 2.02], [2.98, 3.02], [3.98, 4.02]],
        zero_ranges=[[0.98, 1.02], [1.98, 2.02], [2.98, 3.02], [3.98, 4.02]],
    )
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] か〔まあ〕き〔えっと〕く\n", encoding="utf-8")

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    present = _present_freqs(audio, sr, freqs)
    assert 400 not in present  # まあ 削除
    assert 800 not in present  # えっと 削除
    assert present == {200, 600, 1000}  # 他の語は全部残る
    assert _freq_sequence(audio, sr, freqs) == [200, 600, 1000]


# --------------------------------------------------------------------------
# 複合編集: フィラー削除 + 行削除 + 行入れ替え が共存できる
# --------------------------------------------------------------------------

def test_filler_delete_with_line_delete_and_reorder(tmp_path):
    freqs = [200, 400, 600, 800, 1000]
    # seg1: あ / seg2: か・まあ(filler)・き / seg3: さ
    _build_project(
        tmp_path,
        [
            [("あ", 200)],
            [("か", 400), ("まあ", 600), ("き", 800)],
            [("さ", 1000)],
        ],
        silences=[[1.98, 2.02], [2.98, 3.02]],  # まあ[2,3] の両境界
        zero_ranges=[[1.98, 2.02], [2.98, 3.02]],  # 谷スナップ用に実無音化
    )
    txt = tmp_path / "input.sc.txt"
    # 0001(あ) を行削除、0003 を 0002 より前へ入れ替え、0002 内のフィラー削除
    txt.write_text("[0003] さ\n[0002] か〔まあ〕き\n", encoding="utf-8")

    out = render.render(str(txt))
    audio, sr = sf.read(out)
    present = _present_freqs(audio, sr, freqs)
    assert 200 not in present  # 行削除
    assert 600 not in present  # フィラー削除
    assert present == {400, 800, 1000}
    # さ → か → き の順（入れ替え反映、フィラー抜け）
    assert _freq_sequence(audio, sr, freqs) == [1000, 400, 800]


# --------------------------------------------------------------------------
# 旧形式 json（silences / suggest / block 無し）でも〔〕フィラー削除が
# 致命的失敗せず動く（音響安全判定は RMS のみへ劣化）
# --------------------------------------------------------------------------

def test_legacy_json_filler_delete_degrades_gracefully(tmp_path):
    freqs = [200, 400, 600]
    # silences/block/suggest を一切書かない旧形式。波形側にだけ無音を作り、
    # RMS のみの安全判定（silences=None）で精密カットが成立することを確認。
    _build_project(
        tmp_path,
        [[("か", 200), ("まあ", 400), ("き", 600)]],
        zero_ranges=[[0.9, 1.1], [1.9, 2.1]],  # まあ[1,2] の両境界を実際に無音化
    )
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] か〔まあ〕き\n", encoding="utf-8")

    out = render.render(str(txt))  # 例外なく完走すること
    audio, sr = sf.read(out)
    assert len(audio) > 0
    present = _present_freqs(audio, sr, freqs)
    assert present == {200, 600}  # 隣接語は残り、フィラーだけ消える
    assert _freq_sequence(audio, sr, freqs) == [200, 600]


# ==========================================================================
# §12-6 異常系: ID行破損・不明IDで具体的エラー
# ==========================================================================

def test_broken_id_line_raises(tmp_path):
    _build_project(tmp_path, [[("あ", 200)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text("あいう\n", encoding="utf-8")  # [ID] が無い
    with pytest.raises(ValueError, match="ID"):
        render.render(str(txt))


def test_unknown_id_raises(tmp_path):
    _build_project(tmp_path, [[("あ", 200)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[9999] なにか\n", encoding="utf-8")
    with pytest.raises(ValueError, match="存在しない"):
        render.render(str(txt))


def test_duplicate_id_raises(tmp_path):
    _build_project(tmp_path, [[("あ", 200)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あ\n[0001] あ\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重複"):
        render.render(str(txt))


# ==========================================================================
# §12-7 音質: samplerate/subtype が元と同一、カット境界に不連続なし
# ==========================================================================

@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24"])
def test_output_format_preserved(tmp_path, subtype):
    _build_project(tmp_path, [[("あ", 200)], [("い", 400)]], subtype=subtype)
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あ\n[0002] い\n", encoding="utf-8")
    out = render.render(str(txt))
    info = sf.info(out)
    assert info.samplerate == SR
    assert info.subtype == subtype


def test_no_click_at_cut_boundary(tmp_path):
    # 3単語のうち中央を削除 → 200Hz と 600Hz がクロスフェード結合される。
    _build_project(tmp_path, [[("あ", 200), ("い", 400), ("う", 600)]])
    txt = tmp_path / "input.sc.txt"
    txt.write_text("[0001] あう\n", encoding="utf-8")  # 中央削除→カット境界発生
    out = render.render(str(txt))
    audio, _ = sf.read(out)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # 位相連続な純 sine の隣接差の理論最大値: 0.3*2π*f/sr（f=600 で約0.024）。
    # カット境界(クロスフェード)でも不連続な跳ねが無いこと。余裕を見て 0.06 未満。
    max_diff = float(np.max(np.abs(np.diff(audio))))
    assert max_diff < 0.06


# ==========================================================================
# 統合テスト(任意/ローカル): say 合成 → transcribe → render
#   `-m slow` 明示時のみ実行。mlx_whisper / say / ffmpeg が必要。
# ==========================================================================

@pytest.mark.slow
def test_integration_say_transcribe_render(tmp_path):
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        pytest.skip("say または ffmpeg が無い")
    try:
        import mlx_whisper  # noqa: F401
    except Exception:
        pytest.skip("mlx_whisper 未導入")

    from stefnceorf import transcribe

    aiff = tmp_path / "speech.aiff"
    subprocess.run(
        ["say", "-o", str(aiff), "結局面倒な作業があるならツールを作ればいい"],
        check=True,
    )
    wav = tmp_path / "speech.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff), str(wav)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    res = transcribe.transcribe(str(wav), lang="ja")
    txt_path = res["txt_path"]
    # 無編集で render → 出力が生成され、元と概ね同尺
    out = render.render(txt_path)
    src, sr = sf.read(str(wav))
    dst, _ = sf.read(out)
    assert len(dst) > 0
    assert abs(len(dst) - len(src)) < sr  # 1秒以内の差
