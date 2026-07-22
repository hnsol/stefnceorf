"""Stefnceorf のコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .render import GAP_MAX_S, GAP_THRESHOLD_S
from .transcribe import DEFAULT_MODEL, PAUSE_THRESHOLD_S


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
        "transcribe", aliases=["trans"], help="音声を文字起こしして .sc.txt / .sc.json を生成"
    )
    p_tr.add_argument("input", help="入力wavファイル")
    p_tr.add_argument(
        "--lang",
        choices=["ja", "en"],
        default="ja",
        help="言語 (既定: ja)",
    )
    p_tr.add_argument(
        "--model",
        default=None,
        help=f"mlx-whisperモデル (既定: {DEFAULT_MODEL}、--verbatim時は自動で verbatim用モデル)",
    )
    p_tr.add_argument(
        "--filler-suggest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="フィラー候補の 〔〕 提案（既定: 有効。--no-filler-suggest で無効化）",
    )
    p_tr.add_argument(
        "--verbatim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "フィラーや言い淀みも転写する（モデルは whisper-large-v3-mlx、"
            "処理は約2倍遅い。既定: 有効。--no-verbatim で無効化）"
        ),
    )
    p_tr.add_argument(
        "--pause-threshold",
        type=float,
        default=PAUSE_THRESHOLD_S,
        help=(
            f"この秒数以上の無音をカット可能なポーズ区切り（／）とする"
            f"（既定: {PAUSE_THRESHOLD_S}）。0で全単語境界（旧挙動）"
        ),
    )

    p_rn = sub.add_parser("render", help="編集後テキストから音声を再構成")
    p_rn.add_argument("input", help="編集後の .sc.txt ファイル")
    p_rn.add_argument("-o", "--output", default=None, help="出力wavファイル")
    p_rn.add_argument(
        "--gap-threshold",
        type=float,
        default=GAP_THRESHOLD_S,
        help=f"この秒数以下の無音ポーズはそのまま保持 (既定: {GAP_THRESHOLD_S})",
    )
    p_rn.add_argument(
        "--gap-max",
        type=float,
        default=GAP_MAX_S,
        help=f"しきい値超の無音ポーズを切り詰める秒数 (既定: {GAP_MAX_S})",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in ("transcribe", "trans"):
        from .transcribe import transcribe

        try:
            res = transcribe(
                args.input,
                lang=args.lang,
                model=args.model,
                filler_suggest=args.filler_suggest,
                pause_threshold=args.pause_threshold,
                verbatim=args.verbatim,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        print(f"生成: {res['txt_path']}")
        print(f"生成: {res['json_path']}")
        if res.get("hallucination_ranges"):
            from .transcribe import _format_time

            def _fmt(v):
                return _format_time(v) if v is not None else "?"

            ranges = res["hallucination_ranges"]
            rescued = [r for r in ranges if r.get("rescued")]
            failed = [r for r in ranges if not r.get("rescued")]
            for r in rescued:
                n = r.get("rescued_segments", 0)
                print(
                    f"幻覚疑い区間を再認識で復旧: {_fmt(r.get('start'))}-"
                    f"{_fmt(r.get('end'))}（{n} セグメント）"
                )
            if failed:
                print(
                    "警告: 幻覚疑いのセグメントを除去しました。以下の区間は"
                    "再認識でも復旧できず render で音声が削除されるため、必要なら"
                    "聞き直してください:",
                    file=sys.stderr,
                )
                for r in failed:
                    sample = r.get("sample", "")
                    print(
                        f"  {_fmt(r.get('start'))}-{_fmt(r.get('end'))} "
                        f"(「{sample}…」)",
                        file=sys.stderr,
                    )
        if res.get("silence_cut_count"):
            print(
                f"無音切り詰め: {res['silence_cut_count']}箇所 "
                f"-{res['silence_removed_s']:.1f}秒"
            )
        if args.filler_suggest:
            print(f"フィラー候補: {res['filler_count']}箇所")
            if res["filler_count"] == 0 and not args.verbatim:
                print(
                    "ヒント: --verbatim を併用するとフィラーが転写されやすくなります",
                    file=sys.stderr,
                )
        return 0

    if args.command == "render":
        from .render import render

        try:
            out = render(
                args.input,
                output=args.output,
                gap_threshold=args.gap_threshold,
                gap_max=args.gap_max,
            )
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
