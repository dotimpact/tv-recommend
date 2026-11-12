from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

_VENV_DIR = Path(__file__).resolve().parent / ".venv"
_VENV_PYTHON = _VENV_DIR / "bin" / "python"
if _VENV_PYTHON.exists() and Path(sys.prefix).resolve() != _VENV_DIR.resolve():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), *sys.argv])


from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from nasne import (
     DEFAULT_NASNE_IPS,
    JST,
    NasneReservationIndex,
    build_reservation_index,
    parse_display_datetime,
    reserve_program,
)
from recommender import (
    extract_keyword_matches,
    filter_unseen,
    format_weighted_tags,
    is_new_program,
    is_paid_channel,
    recommend_programs,
    score_fallback_local,
    summarize_profile,
)
from scraper import CACHE_DIR, BangumiScraper

CONFIG_PATH = Path(__file__).parent / "config.json"
MAX_SEEN_IDS = 500

DEFAULT_CONFIG: dict[str, Any] = {
    "area_code": "23",
    "broadcast_types": ["terrestrial", "bs"],
    "target_genres": [
        "documentary",
        "movie",
        "hobby",
        "drama",
        "anime",
        "variety",
        "music",
        "news",
        "information",
        "theater",
    ],
    "keywords": [],
    "exclude_keywords": ["WOWOW"],
    "exclude_paid_channels": True,
    "nasne_ips": DEFAULT_NASNE_IPS,
    "nasne_quality": 101,
    "nasne_storage_id": 0,
    "profile_summary": "",
    "favorite_series": [],
    "positive_tags": [],
    "negative_tags": [],
    "liked": [],
    "disliked": [],
    "seen_ids": [],
    "last_checked_date": "",
    "display_limit": 25,
}

console = Console()


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in DEFAULT_CONFIG.items():
        data.setdefault(k, v)
    return data


def save_config(config: dict[str, Any]) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def display_recommendations(
    picks: list[dict[str, Any]], period_label: str = "本日〜明日"
) -> None:
    table = Table(
        title=f"📺 {period_label} のおすすめ番組ピックアップ",
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("No", justify="right", style="bold", width=3)
    table.add_column("判定/スコア", style="magenta", width=12)
    table.add_column("放送日時 / 放送局", style="green", width=22)
    table.add_column("ジャンル", style="yellow", width=12)
    table.add_column("タイトル / あらすじ / 推薦理由", style="white")

    for idx, p in enumerate(picks, start=1):
        if p.get("match_type") == "KEYWORD":
            badge = "[bold yellow]★ KEYWORD[/bold yellow]"
        else:
            score = p.get("score", 3)
            stars = "★" * score + "☆" * max(0, 5 - score)
            badge = f"[cyan]{stars}[/cyan] ({score}/5)"

        new_mark = " [bold bright_green][新][/bold bright_green]" if is_new_program(p) else ""
        rank_mark = " [bold red][注目][/bold red]" if p.get("is_ranked") else ""
        dt_ch = f"{p.get('datetime', '')}\n[bold]{p.get('channel', '')}[/bold]"
        desc = (p.get("description") or "（概要なし）")[:110]
        reason = p.get("reason") or ""

        body = (
            f"[bold white]{p.get('title', '')}[/bold white]{new_mark}{rank_mark}\n"
            f"[dim]{desc}[/dim]\n"
            f"[italic bright_cyan]💡 {reason}[/italic bright_cyan]"
        )
        table.add_row(str(idx), badge, dt_ch, p.get("genre", ""), body)

    console.print(table)


def prompt_tristate_feedback(
    picks: list[dict[str, Any]],
    res_index: NasneReservationIndex | None = None,
) -> dict[str, str] | None:
    """
    1つの一覧画面上で、各番組に対して「〇 選ぶ」「● 録画（選ぶ＋nasne予約）」「× いらない」「ー スルー（初期値）」を
    直感的に切り替えられる対話UI。
    戻り値: {program_id: "like" | "record" | "dislike" | "skip"}、キャンセル時は None。
    """
    states: list[str] = ["skip"] * len(picks)
    cursor = 0
    cycle_next = {
        "skip": "like",
        "like": "dislike",
        "record": "dislike",
        "dislike": "skip",
    }
    cycle_prev = {
        "skip": "dislike",
        "dislike": "like",
        "record": "like",
        "like": "skip",
    }

    reserved_map: dict[int, str] = {}
    if res_index is not None:
        for idx, p in enumerate(picks):
            n_name = res_index.find_reserved_nasne(p)
            if n_name:
                reserved_map[idx] = n_name

    def get_fragments() -> list[tuple[str, str]]:
        frags: list[tuple[str, str]] = [
            (
                "class:header",
                f"📝 フィードバック選択 (全{len(picks)}件) "
                "([↑/↓]移動  [Space/←/→]切替  [o]〇選ぶ  [r]●録画予約  [x]×いらない  [-]ースルー  [Enter]確定  [q]中断)\n\n",
            )
        ]
        # 件数が多い場合もターミナル高さを超えないよう、カーソル周辺（最大16行）をスライド表示
        viewport_size = 16
        total = len(picks)
        if total <= viewport_size:
            start_idx, end_idx = 0, total
        else:
            half = viewport_size // 2
            start_idx = max(0, min(cursor - half, total - viewport_size))
            end_idx = start_idx + viewport_size

        if start_idx > 0:
            frags.append(("class:divider", f"   ▲ 上に {start_idx} 件あります\n"))

        for idx in range(start_idx, end_idx):
            p = picks[idx]
            is_cur = idx == cursor
            pointer = "❯ " if is_cur else "  "
            st = states[idx]
            if st == "record":
                badge_style = "class:record_cur" if is_cur else "class:record"
                badge_text = "[● 録画予約]"
            elif st == "like":
                badge_style = "class:like_cur" if is_cur else "class:like"
                badge_text = "[〇 選ぶ    ]"
            elif st == "dislike":
                badge_style = "class:dislike_cur" if is_cur else "class:dislike"
                badge_text = "[× いらない]"
            else:
                badge_style = "class:skip_cur" if is_cur else "class:skip"
                badge_text = "[ー スルー  ]"

            row_style = "class:row_cur" if is_cur else "class:row"
            new_tag = "[新] " if is_new_program(p) else ""
            res_tag = f"[予約済:{reserved_map[idx]}] " if idx in reserved_map else ""
            line_text = (
                f" [{idx + 1:02d}] {res_tag}{new_tag}{p.get('title', '')} "
                f"({p.get('datetime', '')} / {p.get('channel', '')})\n"
            )
            frags.append((row_style, pointer))
            frags.append((badge_style, badge_text))
            frags.append((row_style, line_text))

        if end_idx < total:
            frags.append(("class:divider", f"   ▼ 下に {total - end_idx} 件あります\n"))

        # 現在カーソルがある番組のあらすじ・推薦理由を下部にリアルタイム表示
        cur_p = picks[cursor]
        desc = (cur_p.get("description") or "（概要なし）")[:140]
        reason = cur_p.get("reason") or ""
        rec_cnt = states.count("record")
        like_cnt = states.count("like") + rec_cnt
        dislike_cnt = states.count("dislike")
        skip_cnt = states.count("skip")

        frags.append(("class:divider", "\n" + "─" * 76 + "\n"))
        frags.append(
            (
                "class:detail_title",
                f"📺 [{cursor + 1:02d}] {cur_p.get('title', '')} [{cur_p.get('genre', '')}]\n",
            )
        )
        frags.append(("class:detail_desc", f"   {desc}\n"))
        frags.append(("class:detail_reason", f"   💡 {reason}\n"))
        frags.append(
            (
                "class:summary",
                f"\n現在の選択状況:  〇 選ぶ: {like_cnt}件 (うち ● 録画予約: {rec_cnt}件)  /  × いらない: {dislike_cnt}件  /  ー スルー: {skip_cnt}件",
            )
        )
        return frags

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(_event: Any) -> None:
        nonlocal cursor
        cursor = (cursor - 1) % len(picks)

    @kb.add("down")
    @kb.add("j")
    def _down(_event: Any) -> None:
        nonlocal cursor
        cursor = (cursor + 1) % len(picks)

    @kb.add("space")
    @kb.add("right")
    @kb.add("l")
    def _cycle_forward(_event: Any) -> None:
        states[cursor] = cycle_next[states[cursor]]

    @kb.add("left")
    @kb.add("h")
    def _cycle_backward(_event: Any) -> None:
        states[cursor] = cycle_prev[states[cursor]]

    @kb.add("o")
    @kb.add("1")
    def _set_like(_event: Any) -> None:
        states[cursor] = "like"

    @kb.add("r")
    def _toggle_record(_event: Any) -> None:
        states[cursor] = "like" if states[cursor] == "record" else "record"

    @kb.add("x")
    @kb.add("2")
    def _set_dislike(_event: Any) -> None:
        states[cursor] = "dislike"

    @kb.add("-")
    @kb.add("s")
    @kb.add("0")
    def _set_skip(_event: Any) -> None:
        states[cursor] = "skip"

    @kb.add("enter")
    def _accept(event: Any) -> None:
        event.app.exit(result={p["id"]: states[i] for i, p in enumerate(picks)})

    @kb.add("c-c")
    @kb.add("q")
    @kb.add("escape")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    style = Style.from_dict(
        {
            "header": "bold cyan",
            "row": "",
            "row_cur": "bold underline",
            "record": "bold ansimagenta",
            "record_cur": "bold ansimagenta reverse",
            "like": "bold ansigreen",
            "like_cur": "bold ansigreen reverse",
            "dislike": "bold ansired",
            "dislike_cur": "bold ansired reverse",
            "skip": "ansibrightblack",
            "skip_cur": "bold reverse",
            "divider": "ansibrightblack",
            "detail_title": "bold white",
            "detail_desc": "ansigray",
            "detail_reason": "italic ansicyan",
            "summary": "bold yellow",
        }
    )

    app: Application[dict[str, str] | None] = Application(
        layout=Layout(HSplit([Window(content=FormattedTextControl(get_fragments))])),
        key_bindings=kb,
        style=style,
        full_screen=False,
    )
    return app.run()


def interactive_feedback(
    picks: list[dict[str, Any]],
    config: dict[str, Any],
    checked_date_str: str | None = None,
) -> tuple[bool, int]:
    """
    フィードバックを保存する。
    戻り値: (完了フラグ: bool, 今回追加された like + dislike の合計件数: int)
    """
    if not picks:
        console.print("[yellow]表示できるおすすめ番組がありませんでした。[/yellow]")
        return False, 0

    nasnes, res_index = build_reservation_index(config)

    result = prompt_tristate_feedback(picks, res_index=res_index)
    if result is None:
        console.print("[yellow]キャンセルされました（設定は変更されていません）。[/yellow]")
        return False, 0

    today_str = dt.date.today().isoformat()
    liked_count = 0
    record_targets: list[tuple[dict[str, Any], dict[str, Any]]] = []
    disliked_count = 0

    for p in picks:
        pid = p["id"]
        choice = result.get(pid, "skip")
        entry = {
            "id": pid,
            "title": p.get("title", ""),
            "genre": p.get("genre", ""),
            "datetime": p.get("datetime", ""),
            "channel": p.get("channel", ""),
            "start_time": p.get("start_time", ""),
            "duration": int(p.get("duration") or 0),
            "description": (p.get("description") or "")[:100],
            "selected_at": today_str,
        }
        if choice in ("like", "record"):
            config["liked"].append(entry)
            liked_count += 1
            if choice == "record":
                record_targets.append((p, entry))
        elif choice == "dislike":
            config["disliked"].append(
                {
                    "title": p.get("title", ""),
                    "genre": p.get("genre", ""),
                    "description": (p.get("description") or "")[:100],
                    "selected_at": today_str,
                }
            )
            disliked_count += 1

        if pid not in config["seen_ids"]:
            config["seen_ids"].append(pid)

    config["seen_ids"] = config["seen_ids"][-MAX_SEEN_IDS:]
    config["last_checked_date"] = checked_date_str or today_str
    save_config(config)

    skip_count = len(picks) - liked_count - disliked_count
    console.print(
        f"\n[bold green]✔ フィードバックを保存しました！[/bold green] "
        f"(〇 選ぶ: {liked_count}件 [うち ● 録画予約: {len(record_targets)}件] / "
        f"× いらない: {disliked_count}件 / ー スルー: {skip_count}件)"
    )

    if record_targets:
        console.print("\n[bold magenta]📹 nasne へ録画予約を登録中...[/bold magenta]")
        failed_count = 0
        for p, entry in record_targets:
            ok, msg = reserve_program(p, config, nasnes=nasnes, res_index=res_index)
            if p.get("start_time"):
                entry["start_time"] = p["start_time"]
            if p.get("duration"):
                entry["duration"] = int(p["duration"])
            if ok:
                entry.pop("record_failed", None)
                entry.pop("record_error", None)
                console.print(f"  [bold green]✔ {msg}[/bold green]")
            else:
                entry["record_failed"] = True
                entry["record_error"] = msg
                failed_count += 1
                console.print(f"  [bold red]✖ {msg}[/bold red]")
        save_config(config)
        if failed_count > 0:
            console.print(
                "[bold yellow]💡 予約失敗した番組も「〇 選ぶ (liked)」に残っています。"
                "競合解消後などに `python main.py record` で再予約できます。[/bold yellow]"
            )
    elif liked_count > 0:
        console.print(
            "[dim]💡 選んだ番組リストからあとで録画予約する場合は `python main.py record` を実行してください。[/dim]"
        )

    return True, (liked_count + disliked_count)


def _resolve_last_checked_date(config: dict[str, Any]) -> dt.date | None:
    raw = (config.get("last_checked_date") or "").strip()
    if not raw:
        # liked / disliked の最新 selected_at から推定
        dates = [
            item.get("selected_at", "")
            for item in (config.get("liked", []) + config.get("disliked", []))
            if item.get("selected_at")
        ]
        if dates:
            raw = max(dates)
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        return None


def prompt_catchup_mode(
    last_date: dt.date, today: dt.date
) -> list[dt.date] | None:
    """
    前回の確認から2日以上経過している場合に、どの期間をチェックするか選択させる。
    戻り値: チェック対象の開始日 (2日単位) のリスト。キャンセル時は None。
    """
    days_gap = (today - last_date).days
    # 過去遡及は bangumi.org のEPG保持期間（最大7日前）まで
    earliest = max(last_date + dt.timedelta(days=1), today - dt.timedelta(days=7))
    chunks: list[dt.date] = []
    cur = earliest
    while cur <= today:
        chunks.append(cur)
        cur += dt.timedelta(days=2)

    yesterday = today - dt.timedelta(days=1)
    chunk_labels = ", ".join(
        f"{c.strftime('%m/%d')}〜{(c + dt.timedelta(days=1)).strftime('%m/%d')}"
        for c in chunks
    )

    options = [
        (
            "catchup",
            f"過去分を2日ごとに順番にチェックする ({chunk_labels})",
        ),
        (
            "yesterday_today",
            f"昨日・今日（{yesterday.strftime('%m/%d')}〜{today.strftime('%m/%d')}）だけチェックする",
        ),
        (
            "today_tomorrow",
            f"今日・明日（{today.strftime('%m/%d')}〜{(today + dt.timedelta(days=1)).strftime('%m/%d')}）だけチェックする",
        ),
    ]
    cursor = 0

    def get_fragments() -> list[tuple[str, str]]:
        frags: list[tuple[str, str]] = [
            (
                "class:header",
                f"🗓 前回の確認（{last_date.isoformat()}）から {days_gap} 日経過しています。チェック方法を選んでください:\n"
                "([↑/↓]移動  [Enter]決定  [q]中断)\n\n",
            )
        ]
        for idx, (_key, label) in enumerate(options):
            is_cur = idx == cursor
            pointer = "❯ " if is_cur else "  "
            style = "class:row_cur" if is_cur else "class:row"
            frags.append((style, f"{pointer}{idx + 1}. {label}\n"))
        return frags

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(_e: Any) -> None:
        nonlocal cursor
        cursor = (cursor - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _down(_e: Any) -> None:
        nonlocal cursor
        cursor = (cursor + 1) % len(options)

    @kb.add("enter")
    def _enter(e: Any) -> None:
        e.app.exit(result=options[cursor][0])

    @kb.add("c-c")
    @kb.add("q")
    @kb.add("escape")
    def _cancel(e: Any) -> None:
        e.app.exit(result=None)

    app: Application[str | None] = Application(
        layout=Layout(HSplit([Window(content=FormattedTextControl(get_fragments))])),
        key_bindings=kb,
        style=Style.from_dict(
            {
                "header": "bold cyan",
                "row": "",
                "row_cur": "bold yellow underline",
            }
        ),
        full_screen=False,
    )
    choice = app.run()
    if choice is None:
        return None
    if choice == "catchup":
        return chunks
    if choice == "yesterday_today":
        return [yesterday]
    return [today]


def prompt_ahead_mode(today: dt.date) -> list[tuple[dt.date, int]] | None:
    """
    1週間先まで（today 〜 today+6日）を2日ごとに先取りチェックするための期間選択UI。
    戻り値: [(start_date, days), ...] のリスト。キャンセル時は None。
    """
    # 7日間 (today + 0..6) を 2日, 2日, 2日, 1日(または2日) に分割
    blocks: list[tuple[dt.date, int]] = [
        (today, 2),
        (today + dt.timedelta(days=2), 2),
        (today + dt.timedelta(days=4), 2),
        (today + dt.timedelta(days=6), 1),
    ]

    def fmt_block(start_d: dt.date, d_cnt: int) -> str:
        if d_cnt == 1:
            return f"{start_d.strftime('%m/%d(%a)')}"
        end_d = start_d + dt.timedelta(days=d_cnt - 1)
        return f"{start_d.strftime('%m/%d')}〜{end_d.strftime('%m/%d')}"

    future_blocks = blocks[1:]
    future_label = ", ".join(fmt_block(s, c) for s, c in future_blocks)
    all_label = ", ".join(fmt_block(s, c) for s, c in blocks)

    options: list[tuple[str, str, list[tuple[dt.date, int]]]] = [
        (
            "future_seq",
            f"明後日〜1週間先までを2日ごとに順番にチェックする ({future_label})",
            future_blocks,
        ),
        (
            "all_seq",
            f"今日〜1週間先までを2日ごとに順番にチェックする ({all_label})",
            blocks,
        ),
    ]
    for s_date, d_cnt in blocks:
        options.append(
            (
                f"single_{s_date.isoformat()}",
                f"期間を指定: {fmt_block(s_date, d_cnt)} だけチェックする",
                [(s_date, d_cnt)],
            )
        )

    cursor = 0

    def get_fragments() -> list[tuple[str, str]]:
        frags: list[tuple[str, str]] = [
            (
                "class:header",
                "🔭 1週間先までの先取りチェック（番組が埋もれないよう最大2日単位で抽出します）:\n"
                "([↑/↓]移動  [Enter]決定  [q]中断)\n\n",
            )
        ]
        for idx, (_key, label, _b) in enumerate(options):
            is_cur = idx == cursor
            pointer = "❯ " if is_cur else "  "
            style = "class:row_cur" if is_cur else "class:row"
            frags.append((style, f"{pointer}{idx + 1}. {label}\n"))
        return frags

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(_e: Any) -> None:
        nonlocal cursor
        cursor = (cursor - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _down(_e: Any) -> None:
        nonlocal cursor
        cursor = (cursor + 1) % len(options)

    @kb.add("enter")
    def _enter(e: Any) -> None:
        e.app.exit(result=options[cursor][2])

    @kb.add("c-c")
    @kb.add("q")
    @kb.add("escape")
    def _cancel(e: Any) -> None:
        e.app.exit(result=None)

    app: Application[list[tuple[dt.date, int]] | None] = Application(
        layout=Layout(HSplit([Window(content=FormattedTextControl(get_fragments))])),
        key_bindings=kb,
        style=Style.from_dict(
            {
                "header": "bold cyan",
                "row": "",
                "row_cur": "bold yellow underline",
            }
        ),
        full_screen=False,
    )
    return app.run()


def _run_single_window(
    start_date: dt.date,
    config: dict[str, Any],
    args: argparse.Namespace,
    days: int = 2,
) -> tuple[bool, int]:
    end_date = start_date + dt.timedelta(days=max(1, days) - 1)
    if days == 1:
        period_label = start_date.strftime("%m/%d")
    else:
        period_label = f"{start_date.strftime('%m/%d')}〜{end_date.strftime('%m/%d')}"

    scraper = BangumiScraper(config)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"番組データ取得準備中 ({period_label})...", total=1
            )

            def _on_progress(cur: int, total: int, desc: str) -> None:
                progress.update(
                    task_id,
                    total=total,
                    completed=cur,
                    description=f"[{period_label}] {desc}",
                )

            programs = scraper.get_programs(
                target_date=start_date,
                force_refresh=args.refresh,
                days=days,
                on_progress=_on_progress,
            )
    finally:
        scraper.close()

    console.print(f"[dim]対象期間 ({period_label}) の取得・重複排除済み番組数: {len(programs)}件[/dim]")

    effective_limit = args.limit or int(config.get("display_limit", 25))
    exclude_paid = bool(config.get("exclude_paid_channels", True))
    evaluated_titles = {
        item.get("title", "")
        for item in (config.get("liked") or []) + (config.get("disliked") or [])
        if item.get("title")
    }
    disliked_titles = {
        item.get("title", "")
        for item in (config.get("disliked") or [])
        if item.get("title")
    }

    if args.fast:
        unseen = filter_unseen(
            programs,
            config.get("seen_ids", []),
            config.get("exclude_keywords", []),
            exclude_paid_channels=exclude_paid,
            evaluated_titles=evaluated_titles,
            disliked_titles=disliked_titles,
        )
        kw_matches, remaining = extract_keyword_matches(unseen, config.get("keywords", []))
        picks = kw_matches + score_fallback_local(remaining, config, top_k=effective_limit)
    else:
        try:
            with console.status(
                f"[bold cyan]ローカルベクトル一次選別＋LLMでおすすめ番組を厳選中 ({period_label})...[/bold cyan]"
            ):
                picks = recommend_programs(programs, config, top_k=effective_limit)
        except Exception as err:
            console.print(f"[bold yellow]⚠ LLM呼び出しスキップ ({type(err).__name__}): {err}[/bold yellow]")
            console.print("[dim]キーワード一致および履歴・ランキングに基づくローカル推薦を表示します。[/dim]")
            unseen = filter_unseen(
                programs,
                config.get("seen_ids", []),
                config.get("exclude_keywords", []),
                exclude_paid_channels=exclude_paid,
                evaluated_titles=evaluated_titles,
                disliked_titles=disliked_titles,
            )
            kw_matches, remaining = extract_keyword_matches(unseen, config.get("keywords", []))
            fallback = score_fallback_local(remaining, config, top_k=effective_limit)
            picks = kw_matches + fallback

    display_recommendations(picks, period_label=period_label)
    if args.no_interactive:
        return True, 0

    checked_marker = min(end_date, dt.date.today()).isoformat()
    return interactive_feedback(picks, config, checked_date_str=checked_marker)


def _print_summary_panel(res: dict[str, Any], config: dict[str, Any]) -> None:
    kw_list = config.get("keywords") or []
    ex_list = config.get("exclude_keywords") or []
    body = (
        f"[bold magenta]■ 手動登録キーワード ({len(kw_list)}件)[/bold magenta]\n"
        f"{', '.join(kw_list) if kw_list else '（なし）'}"
        f"  [dim]/ 除外: {', '.join(ex_list) if ex_list else '（なし）'}[/dim]\n\n"
        f"[bold yellow]■ お気に入り番組・シリーズ枠 ({len(res['favorite_series'])}件 / ★は重要度重み1〜5)[/bold yellow]\n"
        f"{format_weighted_tags(res['favorite_series'])}\n\n"
        f"[bold cyan]■ 蓄積された関心タグ・個別要素 ({len(res['positive_tags'])}件 / ★5:中核 〜 ★1:単発)[/bold cyan]\n"
        f"{format_weighted_tags(res['positive_tags'])}\n\n"
        f"[bold red]■ 蓄積されたNG・除外タグ ({len(res['negative_tags'])}件)[/bold red]\n"
        f"{format_weighted_tags(res['negative_tags'])}\n\n"
        f"[bold green]■ 全体傾向の5箇条 (profile_summary)[/bold green]\n"
        f"{res['profile_summary']}"
    )
    console.print(
        Panel(
            body,
            title="✨ 更新されたプロファイル＆キーワードリスト",
            border_style="green",
        )
    )


def _auto_summarize_after_session(config: dict[str, Any]) -> None:
    """セッション中に新規の〇/×評価があった場合、終了時に自動で重み付きタグとプロファイルを更新して表示する。"""
    try:
        with console.status(
            "[bold cyan]🔄 今回の評価を反映して好みタグ・重みを自動サマライズ中...[/bold cyan]"
        ):
            res = summarize_profile(config)
        config["profile_summary"] = res["profile_summary"]
        config["favorite_series"] = res["favorite_series"]
        config["positive_tags"] = res["positive_tags"]
        config["negative_tags"] = res["negative_tags"]
        save_config(config)
        _print_summary_panel(res, config)
    except Exception as err:
        console.print(f"[dim yellow]※自動サマライズはスキップされました: {err}[/dim yellow]")


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config()
    today = dt.date.today()
    total_evaluated = 0

    if args.ahead:
        blocks = prompt_ahead_mode(today)
        if not blocks:
            console.print("[yellow]キャンセルされました。[/yellow]")
            return
        for idx, (start_d, d_cnt) in enumerate(blocks):
            ok, ev_cnt = _run_single_window(start_d, config, args, days=d_cnt)
            total_evaluated += ev_cnt
            if not ok:
                break
            if idx + 1 < len(blocks):
                next_d = blocks[idx + 1][0]
                console.print(
                    f"\n[bold cyan]続けて次の期間（{next_d.strftime('%m/%d')}〜）へ進みます...[/bold cyan]\n"
                )
        if total_evaluated > 0 and not args.fast:
            _auto_summarize_after_session(config)
        return

    if args.date:
        try:
            target = dt.date.fromisoformat(args.date)
        except ValueError:
            console.print("[red]エラー: --date は YYYY-MM-DD 形式で指定してください。[/red]")
            sys.exit(1)
        _ok, ev_cnt = _run_single_window(target, config, args)
        if ev_cnt > 0 and not args.fast:
            _auto_summarize_after_session(config)
        return

    last_date = _resolve_last_checked_date(config)
    if (
        not args.no_interactive
        and last_date is not None
        and (today - last_date).days >= 2
    ):
        windows = prompt_catchup_mode(last_date, today)
        if not windows:
            console.print("[yellow]キャンセルされました。[/yellow]")
            return
        for idx, start_d in enumerate(windows):
            ok, ev_cnt = _run_single_window(start_d, config, args)
            total_evaluated += ev_cnt
            if not ok:
                break
            if idx + 1 < len(windows):
                console.print(
                    f"\n[bold cyan]続けて次の2日分（{windows[idx + 1].strftime('%m/%d')}〜）へ進みます...[/bold cyan]\n"
                )
        if total_evaluated > 0 and not args.fast:
            _auto_summarize_after_session(config)
        return

    _ok, ev_cnt = _run_single_window(today, config, args)
    if ev_cnt > 0 and not args.fast:
        _auto_summarize_after_session(config)


def _manage_word_list(
    args: argparse.Namespace, config_key: str, label: str, border_style: str
) -> None:
    config = load_config()
    words: list[str] = config.get(config_key, [])

    if args.action == "list":
        if not words:
            console.print(f"[yellow]登録されている{label}はありません。[/yellow]")
        else:
            console.print(
                Panel(
                    "\n".join(f"• {w}" for w in words),
                    title=f"登録{label}一覧 ({len(words)}件)",
                    border_style=border_style,
                )
            )
    elif args.action == "add":
        if not args.word:
            console.print(f"[red]エラー: 追加する{label}を指定してください。[/red]")
            sys.exit(1)
        word = args.word.strip()
        if word in words:
            console.print(f"[yellow]「{word}」は既に登録されています。[/yellow]")
        else:
            words.append(word)
            config[config_key] = words
            save_config(config)
            console.print(f"[green]✔ {label}「{word}」を追加しました。[/green]")
    elif args.action == "rm":
        if not args.word:
            console.print(f"[red]エラー: 削除する{label}を指定してください。[/red]")
            sys.exit(1)
        word = args.word.strip()
        if word not in words:
            console.print(f"[yellow]「{word}」は登録されていません。[/yellow]")
        else:
            words.remove(word)
            config[config_key] = words
            save_config(config)
            console.print(f"[green]✔ {label}「{word}」を削除しました。[/green]")


def cmd_keyword(args: argparse.Namespace) -> None:
    _manage_word_list(args, "keywords", "関心キーワード", "cyan")


def cmd_exclude(args: argparse.Namespace) -> None:
    _manage_word_list(args, "exclude_keywords", "除外キーワード", "red")


def _enrich_and_collect_upcoming_liked(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    config["liked"] のエントリに cache/*.json から id / datetime / channel / start_time / duration を補完し、
    これから放送される（まだ終了していない）番組を放送日時順に返す。
    """
    by_title: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    if CACHE_DIR.exists():
        for cfile in sorted(CACHE_DIR.glob("*.json")):
            try:
                items = json.loads(cfile.read_text(encoding="utf-8"))
                for p in items:
                    if p.get("id"):
                        by_id[p["id"]] = p
                    if p.get("title"):
                        by_title[p["title"]] = p
            except Exception:
                pass

    modified = False
    for entry in config.get("liked", []):
        matched = None
        if entry.get("id") and entry["id"] in by_id:
            matched = by_id[entry["id"]]
        elif entry.get("title") and entry["title"] in by_title:
            matched = by_title[entry["title"]]
        if matched:
            for k in ("id", "datetime", "channel", "start_time", "duration"):
                if not entry.get(k) and matched.get(k):
                    entry[k] = matched[k]
                    modified = True

    if modified:
        save_config(config)

    now = dt.datetime.now(JST)
    exclude_paid = bool(config.get("exclude_paid_channels", True))
    by_dedup_key: dict[str, tuple[dt.datetime, dict[str, Any]]] = {}

    for entry in config.get("liked", []):
        title = entry.get("title", "")
        ch = entry.get("channel", "")
        dt_str = entry.get("datetime", "")
        if not title or not dt_str:
            continue
        if exclude_paid and is_paid_channel(ch):
            continue

        start_dt: dt.datetime | None = None
        if entry.get("start_time"):
            try:
                start_dt = dt.datetime.fromisoformat(str(entry["start_time"]))
            except ValueError:
                start_dt = None
        if start_dt is None:
            start_dt = parse_display_datetime(dt_str)
        if start_dt is None:
            continue

        # 放送開始から15分以上経過している過去番組は除外
        if start_dt < now - dt.timedelta(minutes=15):
            continue

        dedup_key = entry.get("id") or f"{title}_{dt_str}"
        if dedup_key in by_dedup_key:
            prev_entry = by_dedup_key[dedup_key][1]
            if entry.get("record_failed") and not prev_entry.get("record_failed"):
                prev_entry["record_failed"] = True
                prev_entry["record_error"] = entry.get("record_error", "")
            by_dedup_key[dedup_key] = (start_dt, entry)
        else:
            by_dedup_key[dedup_key] = (start_dt, entry)

    upcoming_with_dt = list(by_dedup_key.values())
    upcoming_with_dt.sort(key=lambda x: x[0])
    return [item for _dt, item in upcoming_with_dt]


def prompt_record_from_liked(
    programs: list[dict[str, Any]],
    res_index: NasneReservationIndex,
    nasnes: list[dict[str, str]],
) -> list[dict[str, Any]] | None:
    """
    「〇 選ぶ (liked)」にした未来の番組リストから、nasne に録画予約する番組を選択する TUI。
    前回予約失敗した番組 (record_failed=True) は自動で再予約対象として選択状態にする。
    """
    selected: list[bool] = [False] * len(programs)
    cursor: int | None = None
    reserved_map: dict[int, str] = {}
    unreceivable_set: set[int] = set()
    failed_set: set[int] = set()
    for idx, p in enumerate(programs):
        n_name = res_index.find_reserved_nasne(p)
        if n_name:
            reserved_map[idx] = n_name
            p.pop("record_failed", None)
            p.pop("record_error", None)
        elif not res_index.is_receivable(p):
            unreceivable_set.add(idx)
        elif p.get("record_failed"):
            failed_set.add(idx)
            selected[idx] = True
            if cursor is None:
                cursor = idx
    if cursor is None:
        cursor = 0

    nasne_label = (
        ", ".join(f"{n['name']}({n['ip']})" for n in nasnes)
        if nasnes
        else "未検出"
    )

    def get_fragments() -> list[tuple[str, str]]:
        frags: list[tuple[str, str]] = [
            (
                "class:header",
                f"📹 「〇 選ぶ」済み番組から nasne 録画予約 (全{len(programs)}件 / 接続先: {nasne_label})\n"
                "([↑/↓]移動  [Space/r/o]録画対象トグル  [Enter]予約実行  [q]中断)\n\n",
            )
        ]
        viewport_size = 16
        total = len(programs)
        if total <= viewport_size:
            start_idx, end_idx = 0, total
        else:
            half = viewport_size // 2
            start_idx = max(0, min(cursor - half, total - viewport_size))
            end_idx = start_idx + viewport_size

        if start_idx > 0:
            frags.append(("class:divider", f"   ▲ 上に {start_idx} 件あります\n"))

        for idx in range(start_idx, end_idx):
            p = programs[idx]
            is_cur = idx == cursor
            pointer = "❯ " if is_cur else "  "
            if idx in reserved_map:
                badge_style = "class:reserved_cur" if is_cur else "class:reserved"
                badge_text = f"[予約済:{reserved_map[idx]:<6}]"
            elif idx in unreceivable_set:
                badge_style = "class:skip_cur" if is_cur else "class:skip"
                badge_text = "[受信外:不可 ]"
            elif selected[idx]:
                badge_style = "class:record_cur" if is_cur else "class:record"
                badge_text = (
                    "[● 再予約する]" if idx in failed_set else "[● 録画する  ]"
                )
            elif idx in failed_set:
                badge_style = "class:failed_cur" if is_cur else "class:failed"
                badge_text = "[! 前回失敗  ]"
            else:
                badge_style = "class:skip_cur" if is_cur else "class:skip"
                badge_text = "[ー 未予約   ]"

            row_style = "class:row_cur" if is_cur else "class:row"
            line_text = (
                f" [{idx + 1:02d}] {p.get('title', '')} "
                f"({p.get('datetime', '')} / {p.get('channel', '')})\n"
            )
            frags.append((row_style, pointer))
            frags.append((badge_style, badge_text))
            frags.append((row_style, line_text))

        if end_idx < total:
            frags.append(("class:divider", f"   ▼ 下に {total - end_idx} 件あります\n"))

        cur_p = programs[cursor]
        desc = (cur_p.get("description") or "（概要なし）")[:140]
        rec_cnt = sum(
            1
            for i, sel in enumerate(selected)
            if sel and i not in reserved_map and i not in unreceivable_set
        )

        frags.append(("class:divider", "\n" + "─" * 76 + "\n"))
        frags.append(
            (
                "class:detail_title",
                f"📺 [{cursor + 1:02d}] {cur_p.get('title', '')} [{cur_p.get('genre', '')}]\n",
            )
        )
        frags.append(
            (
                "class:detail_reason",
                f"   🗓 {cur_p.get('datetime', '')}  /  📡 {cur_p.get('channel', '')}\n",
            )
        )
        if cur_p.get("record_error"):
            frags.append(
                (
                    "class:failed",
                    f"   ⚠ 前回の失敗理由: {cur_p.get('record_error')}\n",
                )
            )
        frags.append(("class:detail_desc", f"   {desc}\n"))
        frags.append(
            (
                "class:summary",
                f"\n選択状況:  ● 今回録画予約する: {rec_cnt}件  /  ✔ 予約済み: {len(reserved_map)}件  /  全{total}件",
            )
        )
        return frags

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(_e: Any) -> None:
        nonlocal cursor
        assert cursor is not None
        cursor = (cursor - 1) % len(programs)

    @kb.add("down")
    @kb.add("j")
    def _down(_e: Any) -> None:
        nonlocal cursor
        assert cursor is not None
        cursor = (cursor + 1) % len(programs)

    @kb.add("space")
    @kb.add("r")
    @kb.add("o")
    @kb.add("right")
    @kb.add("left")
    def _toggle(_e: Any) -> None:
        assert cursor is not None
        if cursor not in reserved_map and cursor not in unreceivable_set:
            selected[cursor] = not selected[cursor]

    @kb.add("enter")
    def _accept(e: Any) -> None:
        targets = [
            programs[i]
            for i in range(len(programs))
            if selected[i] and i not in reserved_map and i not in unreceivable_set
        ]
        e.app.exit(result=targets)

    @kb.add("c-c")
    @kb.add("q")
    @kb.add("escape")
    def _cancel(e: Any) -> None:
        e.app.exit(result=None)

    style = Style.from_dict(
        {
            "header": "bold cyan",
            "row": "",
            "row_cur": "bold underline",
            "record": "bold ansimagenta",
            "record_cur": "bold ansimagenta reverse",
            "reserved": "bold ansigreen",
            "reserved_cur": "bold ansigreen reverse",
            "failed": "bold ansired",
            "failed_cur": "bold ansired reverse",
            "skip": "ansibrightblack",
            "skip_cur": "bold reverse",
            "divider": "ansibrightblack",
            "detail_title": "bold white",
            "detail_desc": "ansigray",
            "detail_reason": "italic ansicyan",
            "summary": "bold yellow",
        }
    )

    app: Application[list[dict[str, Any]] | None] = Application(
        layout=Layout(HSplit([Window(content=FormattedTextControl(get_fragments))])),
        key_bindings=kb,
        style=style,
        full_screen=False,
    )
    return app.run()


def cmd_record(args: argparse.Namespace) -> None:
    config = load_config()
    with console.status("[bold cyan]nasne の接続状態と予約リストを確認中...[/bold cyan]"):
        nasnes, res_index = build_reservation_index(config)
        upcoming = _enrich_and_collect_upcoming_liked(config)

    if not upcoming:
        console.print(
            "[yellow]「〇 選ぶ」にした番組のうち、これから放送される予定の番組はありません。[/yellow]"
        )
        return

    if getattr(args, "list_only", False) or getattr(args, "no_interactive", False):
        table = Table(
            title=f"📹 「〇 選ぶ」済み・今後の放送予定リスト ({len(upcoming)}件)",
            show_lines=True,
            header_style="bold cyan",
        )
        table.add_column("No", justify="right", width=3)
        table.add_column("状態", width=16)
        table.add_column("放送日時 / 放送局", style="green", width=22)
        table.add_column("タイトル / 概要", style="white")
        for idx, p in enumerate(upcoming, 1):
            n_name = res_index.find_reserved_nasne(p)
            if n_name:
                st = f"[bold green]✔ 予約済({n_name})[/bold green]"
            elif p.get("record_failed"):
                st = "[bold red]! 前回予約失敗[/bold red]"
            else:
                st = "[dim]ー 未予約[/dim]"
            dt_ch = f"{p.get('datetime', '')}\n[bold]{p.get('channel', '')}[/bold]"
            err_line = (
                f"\n[red]⚠ {p['record_error']}[/red]"
                if (not n_name and p.get("record_error"))
                else ""
            )
            body = f"[bold]{p.get('title', '')}[/bold]\n[dim]{p.get('description', '')}[/dim]{err_line}"
            table.add_row(str(idx), st, dt_ch, body)
        console.print(table)
        return

    targets = prompt_record_from_liked(upcoming, res_index, nasnes)
    if targets is None:
        console.print("[yellow]キャンセルされました。[/yellow]")
        return
    if not targets:
        save_config(config)
        console.print("[dim]新規に録画予約する番組は選択されませんでした。[/dim]")
        return

    console.print(f"\n[bold magenta]📹 {len(targets)}件の番組を nasne に録画予約中...[/bold magenta]")
    for p in targets:
        ok, msg = reserve_program(p, config, nasnes=nasnes, res_index=res_index)
        if ok:
            p.pop("record_failed", None)
            p.pop("record_error", None)
            console.print(f"  [bold green]✔ {msg}[/bold green]")
        else:
            p["record_failed"] = True
            p["record_error"] = msg
            console.print(f"  [bold red]✖ {msg}[/bold red]")
    save_config(config)


def cmd_summarize(_args: argparse.Namespace) -> None:
    config = load_config()
    try:
        with console.status(
            "[bold cyan]liked / disliked 履歴から個別タグ（シリーズ・関心要素・NG要素）と5箇条を抽出・蓄積中...[/bold cyan]"
        ):
            res = summarize_profile(config)
    except (ValueError, RuntimeError) as err:
        console.print(f"[bold red]エラー: {err}[/bold red]")
        sys.exit(1)

    config["profile_summary"] = res["profile_summary"]
    config["favorite_series"] = res["favorite_series"]
    config["positive_tags"] = res["positive_tags"]
    config["negative_tags"] = res["negative_tags"]
    save_config(config)
    _print_summary_panel(res, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="bangumi.org パーソナライズ番組ピックアップ CLI"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="当日キャッシュを無視して bangumi.org から再取得する",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="LLM推薦枠の最大表示件数（デフォルト: config.json の display_limit = 25）",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="一覧表示のみ行い、対話フィードバック入力をスキップする",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="指定した日付 (YYYY-MM-DD) から2日分の番組をチェックする（過去日・未来日どちらも指定可）",
    )
    parser.add_argument(
        "--ahead",
        "-a",
        action="store_true",
        help="1週間先までの番組を2日ごとに先取りチェックするメニューを開く",
    )
    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="LLMを呼ばずローカルのTF-IDFベクトル類似度＋蓄積タグ照合のみで瞬時に選別する（約0.05秒）",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    kw_parser = subparsers.add_parser("keyword", help="関心キーワードの管理 (add / list / rm)")
    kw_parser.add_argument("action", choices=["add", "list", "rm"], help="操作種別")
    kw_parser.add_argument("word", nargs="?", help="追加・削除するキーワード")

    ex_parser = subparsers.add_parser(
        "exclude", help="除外キーワード（放送局・番組名・内容のNGワード）の管理 (add / list / rm)"
    )
    ex_parser.add_argument("action", choices=["add", "list", "rm"], help="操作種別")
    ex_parser.add_argument("word", nargs="?", help="追加・削除する除外キーワード（例: WOWOW）")

    rec_parser = subparsers.add_parser(
        "record",
        help="「〇 選ぶ」にした今後の番組リストから nasne に録画予約する",
    )
    rec_parser.add_argument(
        "--list",
        "-l",
        dest="list_only",
        action="store_true",
        help="対話選択を開かず、今後の「〇 選ぶ」番組と nasne 予約状況の一覧だけを表示する",
    )

    subparsers.add_parser(
        "summarize",
        help="蓄積された liked/disliked から好みの境界線（5箇条）とタグを生成・更新する",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "keyword":
        cmd_keyword(args)
    elif args.subcommand == "exclude":
        cmd_exclude(args)
    elif args.subcommand == "record":
        cmd_record(args)
    elif args.subcommand == "summarize":
        cmd_summarize(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
