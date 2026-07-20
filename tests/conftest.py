"""pytest 共通設定。

`@pytest.mark.slow` を付けたテストは既定で skip し、`-m slow` を明示的に
指定したときのみ実行する（実モデル/say を使うローカル確認用テストのため）。
addopts での deselect は使わず、この収集フックで制御する。
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("-m") or ""
    if "slow" in markexpr:
        # `-m slow` 明示時はそのまま実行（不要分は pytest 側で deselect 済み）
        return
    skip_slow = pytest.mark.skip(reason="slow テスト: `-m slow` を指定すると実行")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
