"""Stefnceorf — テキスト編集で音声を編集するCLIツール。"""

from importlib.metadata import PackageNotFoundError, version

try:
    # インストール済みパッケージからバージョンを取得（pyproject.toml の一元管理）
    __version__ = version("stefnceorf")
except PackageNotFoundError:
    # 未インストールでのソース直実行時
    __version__ = "0.0.0+unknown"
