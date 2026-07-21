from stefnceorf import fillers


def test_load_ja_defaults():
    ja = fillers.load_fillers("ja")
    for w in ["あのー", "そのー", "えー", "えっと", "まあ", "うーん"]:
        assert w in ja
    # 「え」は返事・相槌と紛らわしいため辞書から除外
    # 間投詞でない語も辞書から除外されている
    for w in ["え", "あの", "その", "なんか", "こう"]:
        assert w not in ja


def test_load_en_defaults():
    en = fillers.load_fillers("en")
    for w in ["um", "uh", "uhm", "er", "ah"]:
        assert w in en


def test_none_lang_falls_back_to_ja():
    assert fillers.load_fillers(None) == fillers.load_fillers("ja")


def test_unknown_lang_falls_back_to_ja():
    assert fillers.load_fillers("fr") == fillers.load_fillers("ja")


def test_exact_match_only():
    ja = fillers.load_fillers("ja")
    assert fillers.is_filler("まあ", ja) is True
    # 部分一致はしない
    assert fillers.is_filler("まあまあ", ja) is False
    assert fillers.is_filler("そのウェブサイト", ja) is False


def test_normalize_strips_leading_space():
    ja = fillers.load_fillers("ja")
    # Whisperのwordは先頭スペース付きのことがある
    assert fillers.is_filler(" まあ", ja) is True
    assert fillers.normalize_token(" その") == "その"


def test_non_filler():
    ja = fillers.load_fillers("ja")
    assert fillers.is_filler("結局", ja) is False


def test_sono_not_filler_but_etto_is():
    """「その」は辞書から削除済み→フィラーでない。「えっと」は辞書に残る→フィラー。"""
    ja = fillers.load_fillers("ja")
    assert fillers.is_filler("その", ja) is False
    assert fillers.is_filler("えっと", ja) is True


def test_uun_is_filler():
    """「うーん」が新規追加されフィラーとして検出される。"""
    ja = fillers.load_fillers("ja")
    assert fillers.is_filler("うーん", ja) is True


# ---- verbatim: 句読点除去 / SOFT_FILLERS 読点条件 ----

def test_normalize_strips_punctuation():
    """前後の句読点・記号を除去し、長音「ー」は残す。"""
    assert fillers.normalize_token("まあ、") == "まあ"
    assert fillers.normalize_token("、まあ。") == "まあ"
    assert fillers.normalize_token(" あの、") == "あの"
    # 長音は除去しない
    assert fillers.normalize_token("あのー、") == "あのー"


def test_verbatim_filler_with_comma_matches_dict():
    """読点付き辞書語（verbatim転写）も完全一致する。"""
    ja = fillers.load_fillers("ja")
    assert fillers.is_filler("まあ、", ja) is True
    assert fillers.is_filler("えっと。", ja) is True


def test_soft_filler_requires_trailing_pause():
    """SOFT_FILLERS は読点/句点で終わるときのみフィラー。"""
    ja = fillers.load_fillers("ja")
    assert fillers.is_filler("あの、", ja) is True
    assert fillers.is_filler("あの", ja) is False
    assert fillers.is_filler("その。", ja) is True
    assert fillers.is_filler("その", ja) is False
    # 指示語＋名詞の誤検出はしない
    assert fillers.is_filler("あの本", ja) is False
