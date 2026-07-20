from stefnceorf import fillers


def test_load_ja_defaults():
    ja = fillers.load_fillers("ja")
    for w in ["あの", "あのー", "その", "えー", "まあ", "なんか", "こう"]:
        assert w in ja


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
