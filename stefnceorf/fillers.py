"""フィラー語の辞書ロードと単語トークンの完全一致判定。"""

from __future__ import annotations

from importlib import resources

_DICT_FILES = {
    "ja": "fillers_ja.txt",
    "en": "fillers_en.txt",
}

# 言語未指定・未知言語のときの既定辞書
_DEFAULT_LANG = "ja"

# 読点付き（verbatim転写）でのみフィラー扱いする指示語系（日本語）。
# 「あの本」等の指示語誤検出を避けるため、末尾が読点/句点のときだけ一致とする。
SOFT_FILLERS_JA = {"あの", "その", "なんか", "こう"}

# normalize_token で前後から除去する句読点・記号（長音「ー」は残す）
_STRIP_PUNCT = "、。，．！？!?…・,."

# SOFT_FILLERS 一致の条件となる読点/句点（この文字で終わるときのみ）
_PAUSE_CHARS = "、。，．,."


def load_fillers(lang: str | None) -> set[str]:
    """指定言語のフィラー辞書を集合として返す。

    パッケージ同梱の fillers_<lang>.txt を importlib.resources で読み込む。
    1行1語。前後空白と空行は無視する。
    未対応・未指定の言語は既定 (ja) にフォールバックする。
    """
    key = lang if lang in _DICT_FILES else _DEFAULT_LANG
    filename = _DICT_FILES[key]
    text = resources.files(__package__).joinpath(filename).read_text(encoding="utf-8")
    words: set[str] = set()
    for line in text.splitlines():
        w = line.strip()
        if w:
            words.add(w)
    return words


def normalize_token(word: str) -> str:
    """Whisperのwordトークンを比較用に正規化する。

    Whisperのwordは先頭にスペースが付くことがあるため前後空白を除去し、
    さらに前後の句読点・記号（、。，．！？!?…・,.）を除去する。
    verbatim転写では「まあ、」のように読点付きで出るため辞書完全一致を
    可能にする。長音「ー」は「あのー」等の一部なので除去しない。
    """
    return word.strip().strip(_STRIP_PUNCT)


def _ends_with_pause(word: str) -> bool:
    """元トークン（正規化前）が読点/句点で終わるか。前後空白は無視する。"""
    s = word.strip()
    return bool(s) and s[-1] in _PAUSE_CHARS


def is_filler(word: str, fillers: set[str]) -> bool:
    """単語トークンがフィラーならTrue（部分一致はしない）。

    - 辞書との完全一致（normalize 後）: True
    - SOFT_FILLERS_JA（指示語系）: 元トークンが読点/句点で終わる場合のみ True
      （verbatim転写の「あの、」を拾いつつ「あの本」等の誤検出を避ける）
    """
    norm = normalize_token(word)
    if norm in fillers:
        return True
    if norm in SOFT_FILLERS_JA and _ends_with_pause(word):
        return True
    return False
