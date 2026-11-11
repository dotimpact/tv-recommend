from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://bangumi.org"
CACHE_DIR = Path(__file__).parent / "cache"

GENRE_JA_NAMES: dict[str, str] = {
    "documentary": "ドキュメンタリー",
    "movie": "映画",
    "hobby": "趣味・教育",
    "drama": "ドラマ",
    "anime": "アニメ",
    "variety": "バラエティ",
    "music": "音楽",
    "news": "ニュース",
    "information": "情報",
    "theater": "劇場・公演",
    "sports": "スポーツ",
}

# Minds area code (e.g. "23" for Tokyo) -> bangumi.org ggm_group_id (e.g. "42" for Tokyo)
MINDS_TO_GGM: dict[str, str] = {
    "10": "1", "11": "8", "12": "3", "13": "9", "14": "10",
    "15": "12", "16": "6", "22": "13", "20": "16", "17": "19",
    "18": "22", "19": "25", "21": "28", "23": "42", "24": "45",
    "29": "37", "27": "40", "26": "31", "28": "33", "25": "35",
    "32": "50", "30": "51", "31": "56", "33": "73", "34": "60",
    "35": "67", "36": "62", "37": "58", "38": "76", "39": "64",
    "40": "84", "41": "81", "42": "85", "43": "93", "44": "91",
    "45": "79", "46": "101", "47": "98", "48": "96", "49": "95",
    "50": "105", "51": "112", "52": "110", "53": "109", "54": "116",
    "55": "117", "56": "126", "57": "123", "58": "131", "59": "129",
    "60": "127", "61": "122", "62": "134", "63": "120",
}

BROADCAST_TYPE_TO_SITYPE: dict[str, str] = {
    "terrestrial": "3",
    "bs": "1",
    "cs": "2",
    "bs4k": "20",
}

GENRE_TO_RANKING_ID: dict[str, str] = {
    "movie": "1",
    "drama": "2",
    "sports": "3",
    "music": "4",
    "variety": "5",
    "information": "6",
    "news": "7",
    "anime": "8",
    "documentary": "9",
    "theater": "A",
    "hobby": "B",
}

EPG_GC_TO_GENRE: dict[str, str] = {
    "gc-movie": "映画",
    "gc-drama": "ドラマ",
    "gc-sports": "スポーツ",
    "gc-music": "音楽",
    "gc-anime": "アニメ",
    "no_genre": "教養・情報・バラエティ",
}

WEEKDAY_JA = ("月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜")


def format_epg_datetime(s_attr: str) -> str:
    """EPGの s 属性 ('202611101900') を '11月10日 火曜 19:00' 形式に変換する。"""
    if len(s_attr) < 12 or not s_attr[:12].isdigit():
        return s_attr
    year = int(s_attr[0:4])
    month = int(s_attr[4:6])
    day = int(s_attr[6:8])
    hour = int(s_attr[8:10])
    minute = s_attr[10:12]
    try:
        d = dt.date(year, month, day)
        wd = WEEKDAY_JA[d.weekday()]
        return f"{month}月{day}日 {wd} {hour}:{minute}"
    except ValueError:
        return s_attr


def clean_epg_channel(raw_ch: str) -> str:
    """EPGのチャンネル見出し ('1 NHK総合1..') を読みやすい局名に整形する。"""
    ch = re.sub(r"^\d+\s+", "", raw_ch.strip())
    ch = ch.replace("NHK総合1..", "NHK総合1・東京").replace("NHKEテレ..", "NHKEテレ1東京")
    ch = ch.replace("TOKYO\u3000..", "TOKYO MX")
    return ch.rstrip(".")

# Enclosed ideographs (🈑, 🈞, 🈖, etc.) and bracketed markers ([字], [再], [新], etc.)
SYMBOL_PATTERN = re.compile(
    r"[\u3200-\u32ff\U0001f100-\U0001f2ff]|"
    r"\[(?:字|再|新|終|解|デ|二|多|無|映|生|SS|S|B|HV|5\.1|双|手|声)\]"
)
RERUN_PATTERN = re.compile(r"🈞|\[再\]")
NEW_PATTERN = re.compile(
    r"🈖|\[新\]|(?:第1[話回]|#1(?:\D|$)|＃1(?:\D|$)|[（(]1[)）]|新番組)"
)
EPISODE_SUFFIX_PATTERN = re.compile(
    r"(\s*[#＃♯第]\d+[話回]?|\s*（\d+）|\s*\(\d+\)|\s+[▽▼].*)$"
)


def resolve_ggm_group_id(area_code: str) -> str:
    """config.json の area_code (例: '23'=東京) を bangumi.org の ggm_group_id ('42') に変換する。"""
    code_str = str(area_code).strip()
    if code_str in MINDS_TO_GGM:
        return MINDS_TO_GGM[code_str]
    return code_str or "42"


def clean_title(raw_title: str) -> str:
    """番組タイトルから [字][再][新] や囲み文字（🈑🈞🈖等）を除去して正規化する。"""
    text = SYMBOL_PATTERN.sub("", raw_title)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_series_key(cleaned_title: str) -> str:
    """同一番組・再放送・帯番組・連続ドラマ一挙放送の重複判定用キーを生成する。"""
    base = unicodedata.normalize("NFKC", cleaned_title)
    # 先頭のジャンル飾り（「中国◇」「韓流◇」「時代劇 」など）を除去
    base = re.sub(r"^(?:中国|韓国|韓流|華流|海外|アジア|時代劇|ドラマ|アニメ|映画)[◇▽▼・\s　]+", "", base)
    # 話数表記（#6, 第1話, (1), (字幕版), (全22話) など）以降を除去
    base = re.sub(
        r"(\s*[#♯]\d+.*|\s*第\d+[話回集節章].*|\s*\(\d+\).*|\s*\((?:字幕|吹替|二か国語|全\d+).*|\s+[▽▼◇].*)$",
        "",
        base,
    )
    # メインタイトル部分（スペースやサブタイトル区切り「〜」「【」「「」の前）を抽出
    parts = re.split(r"[\s　▽▼◇【「『〜～]", base, maxsplit=1)
    head = parts[0].strip()
    if len(head) >= 3:
        return head.lower()
    return base.strip().lower()


def parse_supplement(supplement: str) -> tuple[str, str]:
    """'11月11日 水曜 13:40　テレ東' のようなテキストを (datetime, channel) に分割する。"""
    supp = supplement.strip()
    if "\u3000" in supp:
        dt_part, ch_part = supp.rsplit("\u3000", 1)
        return dt_part.strip(), ch_part.strip()
    parts = supp.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return supp, ""


class BangumiScraper:
    def __init__(self, config: dict[str, Any], delay_sec: float = 1.0) -> None:
        self.config = config
        self.delay_sec = max(delay_sec, 1.0)
        self.ggm_group_id = resolve_ggm_group_id(config.get("area_code", "23"))
        self.broadcast_types: list[str] = config.get(
            "broadcast_types", ["terrestrial", "bs"]
        )
        self.target_genres: list[str] = config.get(
            "target_genres", ["documentary", "movie", "hobby"]
        )
        self._last_request_time = 0.0
        self.client = httpx.Client(
            base_url=BASE_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            },
            cookies={"ggm_area_cd": self.ggm_group_id},
            timeout=20.0,
            follow_redirects=True,
        )

    def _get(self, path: str, extra_headers: dict[str, str] | None = None) -> str:
        elapsed = time.time() - self._last_request_time
        if self._last_request_time > 0 and elapsed < self.delay_sec:
            time.sleep(self.delay_sec - elapsed)
        headers = extra_headers or {}
        resp = self.client.get(path, headers=headers)
        self._last_request_time = time.time()
        resp.raise_for_status()
        return resp.text

    def fetch_epg_descriptions_and_programs(
        self,
        date_strs: list[str],
        step_cb: Callable[[str], None] | None = None,
        time_map: dict[str, tuple[str, int]] | None = None,
    ) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
        """
        地上波・BSのEPGページから
        1) {event_id: description} の辞書
        2) {date_str: [program_dict, ...]}（過去日用の番組リスト）
        を一括構築する。
        """
        desc_map: dict[str, str] = {}
        epg_programs_by_date: dict[str, list[dict[str, Any]]] = {
            d: [] for d in date_strs
        }
        include_sports = "sports" in self.target_genres

        for d_str in date_strs:
            fmt_d = f"{d_str[4:6]}/{d_str[6:8]}"
            if "terrestrial" in self.broadcast_types:
                if step_cb:
                    step_cb(f"番組表(地上波 {fmt_d})を取得中...")
                html = self._get(
                    f"/epg/td?broad_cast_date={d_str}&ggm_group_id={self.ggm_group_id}"
                )
                self._extract_epg_data(
                    html, desc_map, epg_programs_by_date[d_str], include_sports, time_map
                )
            if "bs" in self.broadcast_types:
                if step_cb:
                    step_cb(f"番組表(BS {fmt_d})を取得中...")
                html = self._get(f"/epg/bs?broad_cast_date={d_str}")
                self._extract_epg_data(
                    html, desc_map, epg_programs_by_date[d_str], include_sports, time_map
                )
        return desc_map, epg_programs_by_date

    @staticmethod
    def _extract_epg_data(
        html: str,
        desc_map: dict[str, str],
        program_list: list[dict[str, Any]],
        include_sports: bool,
        time_map: dict[str, tuple[str, int]] | None = None,
    ) -> None:
        soup = BeautifulSoup(html, "html.parser")
        ch_els = soup.select("#ch_area li.topmost")
        ch_names = [clean_epg_channel(el.get_text(" ", strip=True)) for el in ch_els]
        jst = dt.timezone(dt.timedelta(hours=9))

        for idx, ul in enumerate(soup.select('ul[id^="program_line_"]')):
            ch_name = ch_names[idx] if idx < len(ch_names) else ""
            for li in ul.select("li[s][e]"):
                a = li.select_one('a.title_link[href^="/tv_events/"]')
                if not a:
                    continue
                href = a.get("href", "")
                ev_id = href.split("?")[0].rstrip("/").split("/")[-1]
                if not ev_id:
                    continue

                det_el = a.select_one("p.program_detail")
                desc_text = det_el.get_text(" ", strip=True) if det_el else ""
                if desc_text:
                    desc_map[ev_id] = desc_text

                # 放送時間（分）が10分未満のミニ枠や通販枠は過去日EPG抽出時にノイズになるため除外
                s_attr = li.get("s", "")
                e_attr = li.get("e", "")
                start_iso = ""
                dur_sec = 0
                if len(s_attr) == 12 and len(e_attr) == 12:
                    try:
                        s_dt = dt.datetime.strptime(s_attr, "%Y%m%d%H%M").replace(tzinfo=jst)
                        e_dt = dt.datetime.strptime(e_attr, "%Y%m%d%H%M").replace(tzinfo=jst)
                        dur_sec = int((e_dt - s_dt).total_seconds())
                        if dur_sec <= 0:
                            dur_sec += 86400
                        start_iso = s_dt.isoformat()
                        if time_map is not None:
                            time_map[ev_id] = (start_iso, dur_sec)
                    except ValueError:
                        pass

                pt_el = li.select_one(".program_time")
                gc_cls = "no_genre"
                if pt_el:
                    for c in pt_el.get("class", []):
                        if c != "program_time":
                            gc_cls = c
                            break
                if gc_cls == "gc-sports" and not include_sports:
                    continue

                title_el = a.select_one("p.program_title")
                raw_title = title_el.get_text(strip=True) if title_el else ""
                if not raw_title:
                    continue
                if any(
                    w in raw_title
                    for w in ("ショッピング", "お買物情報", "ものスタ", "ヒーリングタイム")
                ):
                    continue

                is_rerun = bool(RERUN_PATTERN.search(raw_title))
                is_new = bool(NEW_PATTERN.search(raw_title)) and not is_rerun
                cleaned = clean_title(raw_title)

                program_list.append(
                    {
                        "id": ev_id,
                        "title": cleaned,
                        "raw_title": raw_title,
                        "datetime": format_epg_datetime(s_attr),
                        "start_time": start_iso,
                        "duration": dur_sec,
                        "channel": ch_name,
                        "genre": EPG_GC_TO_GENRE.get(gc_cls, "教養・情報・バラエティ"),
                        "description": desc_text,
                        "is_ranked": False,
                        "is_rerun": is_rerun,
                        "is_new": is_new,
                    }
                )

    def fetch_rankings(
        self, step_cb: Callable[[str], None] | None = None
    ) -> list[dict[str, Any]]:
        """総合ランキングおよび対象ジャンルのランキングを取得する。"""
        programs: list[dict[str, Any]] = []
        ranking_targets: list[tuple[str, str]] = [("総合", "/ranking/")]
        for genre in self.target_genres:
            gid = GENRE_TO_RANKING_ID.get(genre)
            if gid:
                g_name = GENRE_JA_NAMES.get(genre, genre)
                ranking_targets.append((g_name, f"/ranking/?genre_id={gid}"))

        for label, path in ranking_targets:
            if step_cb:
                step_cb(f"ランキング({label})を取得中...")
            html = self._get(path)
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.select('a[href^="/tv_events/"]'):
                item = self._parse_event_anchor(a, is_ranked=True)
                if item:
                    programs.append(item)
        return programs

    def fetch_genre_programs(
        self,
        date_strs: list[str],
        step_cb: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """対象ジャンルの番組一覧を /genres/filter エンドポイントから取得する。"""
        si_types = [
            BROADCAST_TYPE_TO_SITYPE[bt]
            for bt in self.broadcast_types
            if bt in BROADCAST_TYPE_TO_SITYPE
        ]
        if not si_types:
            si_types = ["3", "1"]
        si_param = ",".join(si_types)

        programs: list[dict[str, Any]] = []
        # BS/CSを含む場合、/genres/filter は1リクエストにつき1日分の dates パラメータのみ受け付ける仕様
        for d_str in date_strs:
            fmt_d = f"{d_str[4:6]}/{d_str[6:8]}"
            for genre in self.target_genres:
                g_name = GENRE_JA_NAMES.get(genre, genre)
                if step_cb:
                    step_cb(f"ジャンル一覧({g_name} {fmt_d})を取得中...")
                path = (
                    f"/genres/filter?major={genre}"
                    f"&si_types={si_param}&dates={d_str}"
                )
                html = self._get(
                    path, extra_headers={"X-Requested-With": "XMLHttpRequest"}
                )
                soup = BeautifulSoup(html, "html.parser")
                for li in soup.select("li[data-sitype]"):
                    a = li.select_one('a[href^="/tv_events/"]')
                    if not a:
                        continue
                    item = self._parse_event_anchor(
                        a, is_ranked=False, fallback_genre=genre
                    )
                    if item:
                        programs.append(item)
        return programs

    def _parse_event_anchor(
        self,
        a: Any,
        is_ranked: bool,
        fallback_genre: str = "",
    ) -> dict[str, Any] | None:
        href = a.get("href", "")
        ev_id = href.split("?")[0].rstrip("/").split("/")[-1]
        if not ev_id:
            return None

        title_el = a.select_one(".program_title")
        raw_title = title_el.get_text(strip=True) if title_el else ""
        if not raw_title:
            return None

        is_rerun = bool(RERUN_PATTERN.search(raw_title))
        is_new = bool(NEW_PATTERN.search(raw_title)) and not is_rerun
        cleaned = clean_title(raw_title)

        supp_el = a.select_one(".program_supplement")
        supp_text = supp_el.get_text(strip=True) if supp_el else ""
        dt_str, channel = parse_supplement(supp_text)

        genre_text = fallback_genre
        tag_ps = a.select(".tag p")
        for p in tag_ps:
            classes = p.get("class") or []
            if "rank" not in classes:
                genre_text = p.get_text(strip=True)
                break

        if not ev_id:
            ev_id = hashlib.md5(f"{cleaned}_{dt_str}_{channel}".encode()).hexdigest()[:12]

        return {
            "id": ev_id,
            "title": cleaned,
            "raw_title": raw_title,
            "datetime": dt_str,
            "channel": channel,
            "genre": genre_text,
            "description": "",
            "is_ranked": is_ranked,
            "is_rerun": is_rerun,
            "is_new": is_new,
        }

    def get_programs(
        self,
        target_date: dt.date | None = None,
        force_refresh: bool = False,
        days: int = 2,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """指定開始日から days 日分（デフォルト2日分）の番組一覧を取得・重複排除し、キャッシュに保存して返す。"""
        today = dt.date.today()
        start_date = target_date or today
        date_list = [start_date + dt.timedelta(days=i) for i in range(max(1, days))]
        date_strs = [d.strftime("%Y%m%d") for d in date_list]

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # 当日開始・2日間の場合は既存の cache/YYYY-MM-DD.json 命名を維持
        if start_date == today and days == 2:
            cache_file = CACHE_DIR / f"{start_date.isoformat()}.json"
        else:
            end_date = date_list[-1]
            cache_file = CACHE_DIR / f"{start_date.isoformat()}_{end_date.isoformat()}.json"

        if cache_file.exists() and not force_refresh:
            if on_progress:
                on_progress(1, 1, "キャッシュから番組データを読み込みました")
            with cache_file.open("r", encoding="utf-8") as f:
                return json.load(f)

        # 今日以降の日付と過去日付を分離（/genres/filter は今日以降のみ対応、過去日は EPG から直接抽出）
        future_or_today_strs = [
            d.strftime("%Y%m%d") for d in date_list if d >= today
        ]
        past_strs = [d.strftime("%Y%m%d") for d in date_list if d < today]

        epg_per_day = int("terrestrial" in self.broadcast_types) + int(
            "bs" in self.broadcast_types
        )
        total_steps = len(date_strs) * epg_per_day
        if future_or_today_strs:
            ranking_cnt = 1 + sum(
                1 for g in self.target_genres if g in GENRE_TO_RANKING_ID
            )
            genre_cnt = len(future_or_today_strs) * len(self.target_genres)
            total_steps += ranking_cnt + genre_cnt

        current_step = 0

        def _step_cb(desc: str) -> None:
            nonlocal current_step
            current_step += 1
            if on_progress:
                on_progress(current_step, max(1, total_steps), desc)

        time_map: dict[str, tuple[str, int]] = {}
        epg_descs, epg_programs_by_date = self.fetch_epg_descriptions_and_programs(
            date_strs, step_cb=_step_cb, time_map=time_map
        )

        ranked_items: list[dict[str, Any]] = []
        genre_items: list[dict[str, Any]] = []
        if future_or_today_strs:
            ranked_items = self.fetch_rankings(step_cb=_step_cb)
            genre_items = self.fetch_genre_programs(
                future_or_today_strs, step_cb=_step_cb
            )

        past_items: list[dict[str, Any]] = []
        for p_str in past_strs:
            past_items.extend(epg_programs_by_date.get(p_str, []))

        # IDによるマージ（ランキングフラグの引き継ぎ）
        by_id: dict[str, dict[str, Any]] = {}
        for item in ranked_items + genre_items + past_items:
            ev_id = item["id"]
            if ev_id in by_id:
                if item["is_ranked"]:
                    by_id[ev_id]["is_ranked"] = True
                if item.get("is_new"):
                    by_id[ev_id]["is_new"] = True
            else:
                if not item.get("description"):
                    item["description"] = epg_descs.get(ev_id, "")
                if ev_id in time_map:
                    item["start_time"], item["duration"] = time_map[ev_id]
                by_id[ev_id] = item

        # 再放送・帯番組・同一番組の重複排除（ランキング掲載または初回放送を優先）
        seen_series: set[str] = set()
        deduped: list[dict[str, Any]] = []

        # ランキング掲載番組を先に処理して優先的に残す
        sorted_items = sorted(
            by_id.values(),
            key=lambda x: (not x["is_ranked"], x.get("is_rerun", False)),
        )

        for item in sorted_items:
            # 放送大学の講義やミニニュース帯番組、明らかな再放送の重複を整理
            series_key = normalize_series_key(item["title"])
            if series_key in seen_series:
                continue
            seen_series.add(series_key)

            # 出力スキーマを SPEC.md に合わせる
            deduped.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "datetime": item["datetime"],
                    "start_time": item.get("start_time", ""),
                    "duration": int(item.get("duration") or 0),
                    "channel": item["channel"],
                    "genre": item["genre"],
                    "description": item["description"],
                    "is_ranked": item["is_ranked"],
                    "is_new": bool(item.get("is_new", False)),
                }
            )

        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(deduped, f, ensure_ascii=False, indent=2)

        return deduped

    def close(self) -> None:
        self.client.close()
