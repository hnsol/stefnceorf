"""Stefnceorf のコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .render import GAP_MAX_S, GAP_THRESHOLD_S
from .transcribe import DEFAULT_MODEL, PAUSE_THRESHOLD_S


# 各コマンド開始時に「これから何をするか」を1行宣言する文言（シンプルな英語）。
_ANNOUNCEMENTS = {
    "transcribe": "transcribing audio",
    "render": "rendering edited audio",
    "logic": "exporting FCPXML for Logic Pro",
    "auto": "full pipeline: transcribe -> FCPXML",
}


def _add_transcription_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lang",
        choices=["ja", "en"],
        default="ja",
        help="言語 (既定: ja)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"mlx-whisperモデル (既定: {DEFAULT_MODEL}、--verbatim時は自動で verbatim用モデル)",
    )
    parser.add_argument(
        "--filler-suggest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="フィラー候補の 〔〕 提案（既定: 有効。--no-filler-suggest で無効化）",
    )
    parser.add_argument(
        "--verbatim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "フィラーや言い淀みも転写する（モデルは whisper-large-v3-mlx、"
            "処理は約2倍遅い。既定: 有効。--no-verbatim で無効化）"
        ),
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=PAUSE_THRESHOLD_S,
        help=(
            f"この秒数以上の無音をカット可能なポーズ区切り（／）とする"
            f"（既定: {PAUSE_THRESHOLD_S}）。0で全単語境界（旧挙動）"
        ),
    )


def _add_gap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=GAP_THRESHOLD_S,
        help=f"この秒数以下の無音ポーズはそのまま保持 (既定: {GAP_THRESHOLD_S})",
    )
    parser.add_argument(
        "--gap-max",
        type=float,
        default=GAP_MAX_S,
        help=f"しきい値超の無音ポーズを切り詰める秒数 (既定: {GAP_MAX_S})",
    )


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
    _add_transcription_arguments(p_tr)

    p_rn = sub.add_parser("render", help="編集後テキストから音声を再構成")
    p_rn.add_argument("input", help="編集後の .sc.txt ファイル")
    p_rn.add_argument("-o", "--output", default=None, help="出力wavファイル")
    _add_gap_arguments(p_rn)

    p_logic = sub.add_parser("logic", help="編集後テキストから Logic Pro 用FCPXMLを生成")
    p_logic.add_argument("input", help="編集後の .sc.txt ファイル")
    p_logic.add_argument("-o", "--output", default=None, help="出力fcpxmlファイル")
    _add_gap_arguments(p_logic)

    p_auto = sub.add_parser(
        "auto", help="音声を文字起こしして Logic Pro 用FCPXMLを生成"
    )
    p_auto.add_argument("input", help="入力wavファイル")
    _add_transcription_arguments(p_auto)
    p_auto.add_argument("-o", "--output", default=None, help="出力fcpxmlファイル")
    _add_gap_arguments(p_auto)

    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    if argv and Path(argv[0]).suffix.lower() == ".wav":
        return ["auto", *argv]
    return argv


def _run_transcription(args: argparse.Namespace) -> tuple[int, dict | None]:
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
        return 1, None
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
                "警告: 再認識できない区間を「未認識区間」として保持しました。"
                "聞き直し、不要ならTXTの該当行を削除してください:",
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
    return 0, res


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        parser.print_help()
        return 0
    args = parser.parse_args(_normalize_argv(raw_args))

    if args.command in ("transcribe", "trans"):
        print(f"transcribe: {_ANNOUNCEMENTS['transcribe']}: {args.input}")
        return _run_transcription(args)[0]

    if args.command == "auto":
        print(f"auto: {_ANNOUNCEMENTS['auto']}: {args.input}")
        return_code, res = _run_transcription(args)
        if return_code:
            return return_code
        assert res is not None

        from .fcpxml import export_fcpxml

        stats_lines: list[str] = []
        try:
            out = export_fcpxml(
                res["txt_path"],
                output=args.output,
                gap_threshold=args.gap_threshold,
                gap_max=args.gap_max,
                stats_out=stats_lines,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        print(f"生成: {out}")
        for line in stats_lines:
            print(line)
        return 0

    if args.command == "render":
        from .render import render

        print(f"render: {_ANNOUNCEMENTS['render']}: {args.input}")
        stats_lines: list[str] = []
        try:
            out = render(
                args.input,
                output=args.output,
                gap_threshold=args.gap_threshold,
                gap_max=args.gap_max,
                stats_out=stats_lines,
            )
        except NotImplementedError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        print(f"生成: {out}")
        for line in stats_lines:
            print(line)
        return 0

    if args.command == "logic":
        from .fcpxml import export_fcpxml

        print(f"logic: {_ANNOUNCEMENTS['logic']}: {args.input}")
        stats_lines: list[str] = []
        try:
            out = export_fcpxml(
                args.input,
                output=args.output,
                gap_threshold=args.gap_threshold,
                gap_max=args.gap_max,
                stats_out=stats_lines,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        print(f"生成: {out}")
        for line in stats_lines:
            print(line)
        return 0

    parser.error("不明なコマンドです")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
