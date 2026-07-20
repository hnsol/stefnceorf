"""Stefnceorf のコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .transcribe import DEFAULT_MODEL


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stefnceorf",
        description="テキスト編集で音声を編集するCLIツール",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser(
        "transcribe", help="音声を文字起こしして .sc.txt / .sc.json を生成"
    )
    p_tr.add_argument("input", help="入力wavファイル")
    p_tr.add_argument(
        "--lang",
        choices=["ja", "en"],
        default=None,
        help="言語 (未指定なら自動判定)",
    )
    p_tr.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"mlx-whisperモデル (既定: {DEFAULT_MODEL})"
    )
    p_tr.add_argument(
        "--no-filler-suggest",
        action="store_true",
        help="フィラー候補の 〔〕 提案を無効化",
    )

    p_rn = sub.add_parser("render", help="編集後テキストから音声を再構成")
    p_rn.add_argument("input", help="編集後の .sc.txt ファイル")
    p_rn.add_argument("-o", "--output", default=None, help="出力wavファイル")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "transcribe":
        from .transcribe import transcribe

        try:
            res = transcribe(
                args.input,
                lang=args.lang,
                model=args.model,
                filler_suggest=not args.no_filler_suggest,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        print(f"生成: {res['txt_path']}")
        print(f"生成: {res['json_path']}")
        if not args.no_filler_suggest:
            print(f"フィラー候補: {res['filler_count']}箇所")
        return 0

    if args.command == "render":
        from .render import render

        try:
            out = render(args.input, output=args.output)
        except NotImplementedError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        print(f"生成: {out}")
        return 0

    parser.error("不明なコマンドです")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
