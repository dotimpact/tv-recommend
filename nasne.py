from __future__ import annotations

import base64
import datetime as dt
import json
import re
import socket
import urllib.parse
import urllib.request
from typing import Any

import httpx
from bs4 import BeautifulSoup

from scraper import BASE_URL, normalize_series_key

JST = dt.timezone(dt.timedelta(hours=9))

DEFAULT_NASNE_IPS: list[str] = []

# bangumi.org の ID がハッシュ値だった場合のフォールバック用チャンネルマップ (東京エリア・BS)
CHANNEL_FALLBACK_MAP: dict[str, tuple[int, int]] = {
    "NHK総合": (2, 1024),
    "NHKEテレ": (2, 1032),
    "Eテレ": (2, 1032),
    "日テレ": (2, 1040),
    "日本テレビ": (2, 1040),
    "TBS": (2, 1048),
    "フジテレビ": (2, 1056),
    "テレビ朝日": (2, 1064),
    "テレ東": (2, 1072),
    "テレビ東京": (2, 1072),
    "TOKYO MX": (2, 23608),
    "NHK BS": (3, 101),
    "NHKBS": (3, 101),
    "BS日テレ": (3, 141),
    "BS朝日": (3, 151),
    "BS-TBS": (3, 161),
    "BSテレ東": (3, 171),
    "BSフジ": (3, 181),
    "BS10": (3, 200),
    "BS11": (3, 211),
    "BS12": (3, 222),
    "放送大学": (3, 231),
    "BSよしもと": (3, 265),
}


def decode_bangumi_event_id(
    ev_id: str, channel_name: str = ""
) -> tuple[int, int, int] | None:
    """
    bangumi.org の番組ID (例: 'AmTQQAMiEAM') をデコードして
    (broadcastingType, serviceId, eventId) を返す。
    - broadcastingType: 2=地上波, 3=BS, 4=CS
    """
    if ev_id and len(ev_id) == 11:
        try:
            padded = ev_id + "=" * ((4 - len(ev_id) % 4) % 4)
            raw = base64.urlsafe_b64decode(padded)
            if len(raw) == 8:
                h = raw.hex()
                service_id = int(h[5:9], 16)
                event_id = int(h[9:13], 16)
                suffix = h[13:]
                if suffix == "003":
                    btype = 2
                elif suffix == "001":
                    btype = 3
                else:
                    btype = 3 if service_id < 1000 else 2
                if service_id > 0 and event_id > 0:
                    return btype, service_id, event_id
        except Exception:
            pass

    # フォールバック: チャンネル名から推定 (eventId=65535 は時間指定予約)
    norm_ch = channel_name.translate(
        str.maketrans("０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ－　", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ")
    )
    for key, (btype, sid) in CHANNEL_FALLBACK_MAP.items():
        if key in norm_ch:
            return btype, sid, 65535
    return None


def parse_epg_attr_to_iso(
    s_attr: str, e_attr: str
) -> tuple[str, int] | None:
    """
    EPGの s="202611112200", e="202611112230" から
    (startDateTime ISO文字列, duration秒) を返す。
    """
    if len(s_attr) != 12 or len(e_attr) != 12:
        return None
    try:
        s_dt = dt.datetime.strptime(s_attr, "%Y%m%d%H%M").replace(tzinfo=JST)
        e_dt = dt.datetime.strptime(e_attr, "%Y%m%d%H%M").replace(tzinfo=JST)
        dur = int((e_dt - s_dt).total_seconds())
        if dur <= 0:
            dur += 86400
        return s_dt.isoformat(), dur
    except ValueError:
        return None


def parse_display_datetime(
    dt_str: str, ref_date: dt.date | None = None
) -> dt.datetime | None:
    """
    '11月12日 木曜 0:35' のような表示用日時文字列を dt.datetime (JST) に変換する。
    """
    if not dt_str:
        return None
    m = re.search(r"(\d{1,2})月(\d{1,2})日.*?(\d{1,2}):(\d{2})", dt_str)
    if not m:
        return None
    month, day, hour, minute = (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
    )
    base = ref_date or dt.date.today()
    year = base.year
    if base.month == 12 and month == 1:
        year += 1
    elif base.month == 1 and month == 12:
        year -= 1
    extra_days = 0
    if hour >= 24:
        extra_days = hour // 24
        hour = hour % 24
    try:
        d = dt.datetime(year, month, day, hour, minute, tzinfo=JST)
        if extra_days:
            d += dt.timedelta(days=extra_days)
        return d
    except ValueError:
        return None


def fetch_schedule_from_bangumi(
    ev_id: str, fallback_dt_str: str = ""
) -> tuple[str, int] | None:
    """
    bangumi.org の個別番組ページ (/tv_events/<ev_id>) から正確な開始日時と放送時間(秒)を取得する。
    """
    if ev_id and len(ev_id) == 11:
        try:
            url = f"{BASE_URL}/tv_events/{ev_id}"
            resp = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8.0,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                sched_el = soup.select_one(".schedule")
                sched_txt = sched_el.get_text(" ", strip=True) if sched_el else ""
                # 例: "11月11日 水曜 22:00 - 22:30 NHKEテレ１東京"
                m = re.search(
                    r"(\d{1,2})月(\d{1,2})日.*?(\d{1,2}):(\d{2})\s*[-〜～]\s*(\d{1,2}):(\d{2})",
                    sched_txt,
                )
                if m:
                    mo, da, sh, sm, eh, em = (int(x) for x in m.groups())
                    start_dt = parse_display_datetime(f"{mo}月{da}日 {sh}:{sm:02d}")
                    if start_dt:
                        end_dt = start_dt.replace(hour=eh % 24, minute=em)
                        if eh >= 24 or end_dt <= start_dt:
                            end_dt += dt.timedelta(days=1)
                        dur = int((end_dt - start_dt).total_seconds())
                        if 60 <= dur <= 43200:
                            return start_dt.isoformat(), dur
        except Exception:
            pass

    # フォールバック: 表示用 datetime から開始時刻を取得し、デフォルト30分(1800秒)とする
    start_dt = parse_display_datetime(fallback_dt_str)
    if start_dt:
        return start_dt.isoformat(), 1800
    return None


def resolve_program_start_and_duration(
    program: dict[str, Any]
) -> tuple[str, int] | None:
    """番組辞書から (startDateTime ISO文字列, duration秒) を解決する。"""
    if program.get("start_time") and program.get("duration"):
        return str(program["start_time"]), int(program["duration"])
    return fetch_schedule_from_bangumi(
        program.get("id", ""), program.get("datetime", "")
    )


def discover_nasne_ips(timeout: float = 1.8) -> list[str]:
    """UPnP SSDP M-SEARCH で LAN 内の nasne の IP アドレス一覧を自動検出する。"""
    ssdp_msg = (
        "M-SEARCH * HTTP/1.1\r\n"
        "HOST: 239.255.255.250:1900\r\n"
        'MAN: "ssdp:discover"\r\n'
        "MX: 1\r\n"
        "ST: ssdp:all\r\n\r\n"
    ).encode("utf-8")

    found_ips: list[str] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.settimeout(timeout)
        sock.sendto(ssdp_msg, ("239.255.255.250", 1900))
        while True:
            data, addr = sock.recvfrom(4096)
            text = data.decode("utf-8", errors="ignore")
            if "nasne" in text.lower() or ":58888" in text:
                ip = addr[0]
                if ip not in found_ips:
                    found_ips.append(ip)
    except Exception:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return sorted(found_ips)


def get_nasne_box_name(ip: str, timeout: float = 1.5) -> str | None:
    """nasne の本体名 (例: 'nasne2') を取得する。疎通確認も兼ねる。"""
    url = f"http://{ip}:64210/status/boxNameGet"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if data.get("errorcode") == 0:
                return data.get("name") or ip
    except Exception:
        return None
    return None


def get_active_nasnes(config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """
    有効な nasne のリスト [{"ip": "192.168.1.100", "name": "nasne"}, ...] を返す。
    設定ファイルの nasne_ips を優先し、応答がなければ SSDP 自動検出を試みる。
    """
    cfg_ips = (config or {}).get("nasne_ips") or DEFAULT_NASNE_IPS
    active: list[dict[str, str]] = []
    for ip in cfg_ips:
        name = get_nasne_box_name(ip)
        if name:
            active.append({"ip": ip, "name": name})

    if not active:
        for ip in discover_nasne_ips():
            name = get_nasne_box_name(ip)
            if name:
                active.append({"ip": ip, "name": name})
    return active


def clean_nasne_title(title: str) -> str:
    """nasne の EPG タイトルに含まれる ARIB 外字 (PUA: U+E000〜U+F8FF) や全角空白を整理する。"""
    cleaned = "".join(ch for ch in title if not (0xE000 <= ord(ch) <= 0xF8FF))
    cleaned = cleaned.replace("\u3000", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def fetch_nasne_reservations(ip: str, nasne_name: str = "") -> list[dict[str, Any]]:
    """指定した nasne の現在の録画予約一覧を取得する。"""
    url = (
        f"http://{ip}:64220/schedule/reservedListGet"
        "?searchCriteria=0&filter=0&startingIndex=0&requestedCount=200"
        "&sortCriteria=0&withDescriptionLong=0&withUserData=0"
    )
    try:
        with urllib.request.urlopen(url, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            items = data.get("item", [])
            for it in items:
                it["nasne_ip"] = ip
                it["nasne_name"] = nasne_name or ip
                it["clean_title"] = clean_nasne_title(it.get("title", ""))
            return items
    except Exception:
        return []


def fetch_nasne_service_ids(ip: str) -> set[int]:
    """指定した nasne で受信設定されている全チャンネルの serviceId 一覧を取得する。"""
    sids: set[int] = set()
    for btype in (2, 3):
        url = f"http://{ip}:64210/status/channelListGet?broadcastingType={btype}"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                for ch in data.get("channel") or []:
                    sid = int(ch.get("serviceId") or 0)
                    if sid > 0:
                        sids.add(sid)
        except Exception:
            pass
    return sids


class NasneReservationIndex:
    """複数の nasne の予約済み番組および受信可能チャンネルを高速照合するためのインデックス。"""

    def __init__(
        self,
        reservations: list[dict[str, Any]],
        valid_service_ids: set[int] | None = None,
    ) -> None:
        self.reservations = reservations
        self.valid_service_ids: set[int] = valid_service_ids or set()
        self.by_sid_eid: dict[tuple[int, int], str] = {}
        self.by_sid_start: dict[tuple[int, str], str] = {}
        self.by_series: dict[str, str] = {}

        for r in reservations:
            n_name = r.get("nasne_name") or r.get("nasne_ip") or "nasne"
            sid = int(r.get("serviceId") or 0)
            eid = int(r.get("eventId") or 0)
            start_dt = str(r.get("startDateTime") or "")
            if sid and eid and eid < 65535:
                self.by_sid_eid[(sid, eid)] = n_name
            if sid and start_dt:
                self.by_sid_start[(sid, start_dt)] = n_name
            s_key = normalize_series_key(r.get("clean_title") or r.get("title", ""))
            if s_key and len(s_key) >= 3:
                self.by_series[s_key] = n_name

    def is_receivable(self, program: dict[str, Any]) -> bool:
        """番組の放送局が nasne の受信チャンネルリストに含まれているか判定する。"""
        if not self.valid_service_ids:
            return True
        decoded = decode_bangumi_event_id(
            program.get("id", ""), program.get("channel", "")
        )
        if not decoded:
            return False
        _btype, sid, _eid = decoded
        return sid in self.valid_service_ids

    def find_reserved_nasne(self, program: dict[str, Any]) -> str | None:
        """
        番組がすでに nasne で予約済みであれば、予約先の nasne 名 (例: 'nasne2') を返す。
        未予約なら None を返す。
        """
        decoded = decode_bangumi_event_id(
            program.get("id", ""), program.get("channel", "")
        )
        if decoded:
            _btype, sid, eid = decoded
            if (sid, eid) in self.by_sid_eid:
                return self.by_sid_eid[(sid, eid)]
            if program.get("start_time") and (sid, str(program["start_time"])) in self.by_sid_start:
                return self.by_sid_start[(sid, str(program["start_time"]))]

        s_key = normalize_series_key(program.get("title", ""))
        if s_key and s_key in self.by_series:
            return self.by_series[s_key]
        return None


def build_reservation_index(
    config: dict[str, Any] | None = None,
) -> tuple[ list[dict[str, str]], NasneReservationIndex ]:
    """アクティブな全 nasne から予約一覧と受信可能チャンネル一覧を取得してインデックスを構築する。"""
    nasnes = get_active_nasnes(config)
    all_res: list[dict[str, Any]] = []
    valid_sids: set[int] = set()
    for n in nasnes:
        all_res.extend(fetch_nasne_reservations(n["ip"], n["name"]))
        if not valid_sids:
            valid_sids = fetch_nasne_service_ids(n["ip"])
    return nasnes, NasneReservationIndex(all_res, valid_service_ids=valid_sids)


def check_conflict(
    ip: str,
    start_iso: str,
    duration_sec: int,
    broadcasting_type: int,
    service_id: int,
    condition_id: str = "1",
) -> list[dict[str, Any]]:
    """指定した nasne で予約競合があるかチェックし、競合アイテムのリストを返す（空リストなら競合なし）。"""
    params = {
        "startDateTime": start_iso,
        "duration": duration_sec,
        "broadcastingType": broadcasting_type,
        "serviceId": service_id,
        "conditionId": condition_id,
    }
    qs = urllib.parse.urlencode(params)
    url = f"http://{ip}:64220/schedule/conflictListGet?{qs}"
    with urllib.request.urlopen(url, timeout=3.0) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        return data.get("item") or []


def create_reservation_on_nasne(
    ip: str,
    title: str,
    start_iso: str,
    duration_sec: int,
    broadcasting_type: int,
    service_id: int,
    event_id: int,
    quality: int = 101,
    storage_id: int = 0,
    condition_id: str = "1",
) -> dict[str, Any]:
    """指定した nasne に POST /schedule/reservedInfoCreate で録画予約を登録する。"""
    url = f"http://{ip}:64220/schedule/reservedInfoCreate"
    payload = {
        "title": title,
        "startDateTime": start_iso,
        "duration": duration_sec,
        "conditionId": condition_id,
        "quality": quality,
        "broadcastingType": broadcasting_type,
        "serviceId": service_id,
        "eventId": event_id,
        "priorityFlag": 1,
        "storageId": storage_id,
        "creatorId": 2000,
        "forceFlag": 1,
        "parentalId": 0,
        "userData": "{}",
    }
    body = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def reserve_program(
    program: dict[str, Any],
    config: dict[str, Any] | None = None,
    nasnes: list[dict[str, str]] | None = None,
    res_index: NasneReservationIndex | None = None,
) -> tuple[bool, str]:
    """
    番組を nasne に録画予約する。
    複数台の nasne がある場合は優先順に競合チェック (conflictListGet) を行い、
    空きがある nasne に自動で予約を登録する。
    戻り値: (成功フラグ, メッセージ)
    """
    title = program.get("title", "（無題）")
    if res_index is None or nasnes is None:
        nasnes, res_index = build_reservation_index(config)

    if not nasnes:
        return False, f"LAN内に応答する nasne が見つかりませんでした: {title}"

    already = res_index.find_reserved_nasne(program)
    if already:
        return True, f"すでに {already} で予約済みです: {title}"

    decoded = decode_bangumi_event_id(
        program.get("id", ""), program.get("channel", "")
    )
    if not decoded:
        return False, f"チャンネルIDを特定できませんでした ({program.get('channel', '')}): {title}"
    btype, sid, eid = decoded

    if res_index.valid_service_ids and sid not in res_index.valid_service_ids:
        return (
            False,
            f"nasneの受信チャンネルに未登録の放送局（{program.get('channel', '')}）のため録画できません: {title}",
        )

    sched = resolve_program_start_and_duration(program)
    if not sched:
        return False, f"放送開始時刻・尺を取得できませんでした ({program.get('datetime', '')}): {title}"
    start_iso, duration_sec = sched
    program["start_time"] = start_iso
    program["duration"] = duration_sec

    quality = int((config or {}).get("nasne_quality", 101))
    storage_id = int((config or {}).get("nasne_storage_id", 0))

    errors: list[str] = []
    for n in nasnes:
        ip = n["ip"]
        name = n["name"]
        try:
            conflicts = check_conflict(ip, start_iso, duration_sec, btype, sid)
            # 同じ eventId が返ってきた場合は既にその nasne に入っている
            for c in conflicts:
                if int(c.get("serviceId") or 0) == sid and int(c.get("eventId") or 0) == eid:
                    return True, f"すでに {name} で予約済みです: {title}"
            if conflicts:
                c_titles = ", ".join(clean_nasne_title(c.get("title", ""))[:18] for c in conflicts)
                errors.append(f"{name}: 同時間帯に予約競合あり ({c_titles})")
                continue

            resp = create_reservation_on_nasne(
                ip=ip,
                title=title,
                start_iso=start_iso,
                duration_sec=duration_sec,
                broadcasting_type=btype,
                service_id=sid,
                event_id=eid,
                quality=quality,
                storage_id=storage_id,
            )
            err_code = resp.get("errorcode")
            if err_code == 0:
                # インデックスに即時反映
                res_index.by_sid_eid[(sid, eid)] = name
                res_index.by_sid_start[(sid, start_iso)] = name
                s_key = normalize_series_key(title)
                if s_key:
                    res_index.by_series[s_key] = name
                dur_min = duration_sec // 60
                return (
                    True,
                    f"{name} ({ip}) に録画予約しました: {title} [{program.get('datetime', '')} / {dur_min}分]",
                )
            errors.append(f"{name}: errorcode={err_code}")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__} ({e})")

    return False, f"録画予約に失敗しました ({'; '.join(errors)}): {title}"
