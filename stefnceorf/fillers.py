"""フィラー語の辞書ロードと単語トークンの完全一致判定。"""

from __future__ import annotations

from importlib import resources

_DICT_FILES = {
    "ja": "fillers_ja.txt",
    "en": "fillers_en.txt",
}

# 言語未指定・未知言語のときの既定辞書
_DEFAULT_LANG = "ja"


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

    Whisperのwordは先頭にスペースが付くことがあるため前後空白を除去する。
    """
    return word.strip()


def is_filler(word: str, fillers: set[str]) -> bool:
    """単語トークンが辞書と完全一致すればTrue（部分一致はしない）。"""
    return normalize_token(word) in fillers
