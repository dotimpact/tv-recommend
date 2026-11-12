from __future__ import annotations

import json
import math
import os
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Literal

warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"google\.genai.*")

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from scraper import normalize_series_key

ENV_PATH = Path(__file__).parent / ".env"


def _load_dotenv() -> None:
    """同ディレクトリの .env が存在すれば読み込み、環境変数より優先して適用する。"""
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and val:
            os.environ[key] = val


_load_dotenv()
# 選別速度を最優先するため、高速・低遅延な gemini-3.1-flash-lite (約1.5秒) を第一候補にする
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = (
    DEFAULT_MODEL,
    "gemini-3.5-flash",
    "gemini-3.8-flash",
)


def _generate_structured(
    client: genai.Client,
    prompt: str,
    schema: type[BaseModel],
    temperature: float = 0.2,
) -> Any:
    """503 UNAVAILABLE 等の一時エラー時に待たずに即座にフォールバックモデルへ切り替えて呼び出す。"""
    last_err: Exception | None = None
    seen_models: set[str] = set()
    for model_name in FALLBACK_MODELS:
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        try:
            return client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                    http_options=types.HttpOptions(
                        timeout=15000,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                ),
            )
        except Exception as err:
            last_err = err
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("利用可能なLLMモデルがありません。")

GENRE_KEYWORDS_MAP: dict[str, tuple[str, ...]] = {
    "documentary": ("ドキュメンタリー", "教養"),
    "movie": ("映画",),
    "hobby": ("趣味", "教育"),
    "drama": ("ドラマ",),
    "variety": ("バラエティ",),
    "music": ("音楽",),
    "anime": ("アニメ",),
    "sports": ("スポーツ",),
    "news": ("ニュース", "報道"),
    "information": ("情報", "ワイドショー"),
    "theater": ("劇場", "公演"),
}


def is_target_genre(genre_label: str, target_genres: list[str]) -> bool:
    if not genre_label:
        return False
    for tg in target_genres:
        kws = GENRE_KEYWORDS_MAP.get(tg, (tg,))
        if any(k in genre_label for k in kws):
            return True
    return False


class ScoredProgram(BaseModel):
    id: str = Field(description="番組のユニークID")
    score: int = Field(description="おすすめ度スコア (1〜5、5が最高)", ge=1, le=5)
    reason: str = Field(description="なぜユーザーの好みに合うと判断したかの1行解説（40〜60文字程度）")


class RecommendationResponse(BaseModel):
    recommendations: list[ScoredProgram] = Field(
        description="厳選されたおすすめ番組のリスト（指定された件数分）"
    )


class WeightedTag(BaseModel):
    name: str = Field(description="タグ名または番組シリーズ名")
    weight: int = Field(
        description="重要度・出現頻度の重み (1〜5: 5=何度も選んでいる中核の好み、3=明確な関心、1=単発の個別関心)",
        ge=1,
        le=5,
    )


class NegativeWeightedTag(BaseModel):
    name: str = Field(description="除外したい具体的な番組シリーズ名または狭い複合要素名")
    category: Literal["series", "narrow_theme", "subgenre", "broad_genre"] = Field(
        description=(
            "粒度分類: "
            "series=特定の番組名・シリーズ名, "
            "narrow_theme=明確に絞られた複合要素(例:テレビショッピング,幼児向け,韓流時代劇), "
            "subgenre=一般的なサブジャンル(例:料理,旅,釣り,刑事ドラマ,時代劇), "
            "broad_genre=大分類ジャンルや一般名詞(例:アニメ,ドラマ,バラエティ,映画,音楽,スポーツ,歴史,経済)"
        )
    )
    weight: int = Field(
        description="重要度重み (1〜3: series/通販等は最大3、narrow_themeは最大2、subgenreは1)",
        ge=1,
        le=3,
    )


class ProfileSummaryResponse(BaseModel):
    five_rules: list[str] = Field(
        description="ユーザーの好みの全体傾向を表す5箇条", min_length=5, max_length=5
    )
    favorite_series: list[WeightedTag] = Field(
        description="『見たい』履歴から抽出した具体的な番組シリーズ名・放送枠名と重み(1〜5)。※放送大学の個別講義名はここに入れずpositive_tagsに入れること"
    )
    positive_tags: list[WeightedTag] = Field(
        description="『見たい』履歴から抽出した具体的な関心テーマ・固有名詞・出演者・企画要素と重み(1〜5)。複数回選ばれているテーマは4〜5、単発の個別判断は1〜2を付与し、『好みと違う』で否定された要素は除外・減衰させること"
    )
    negative_tags: list[NegativeWeightedTag] = Field(
        description=(
            "『好みと違う』履歴から抽出した具体的な除外番組名または複合的で狭い要素と粒度分類・重み(1〜3)。"
            "※『アニメ』『ドラマ』『バラエティ』『映画』『音楽』『スポーツ』『料理』『旅』のような大分類ジャンルや一般的な広い単語は入れないこと"
        )
    )


# negative_tags への登録を禁止する大分類ジャンル・一般的な広義語
BROAD_NEGATIVE_STOPWORDS: frozenset[str] = frozenset(
    {
        "アニメ",
        "アニメーション",
        "特撮",
        "ドラマ",
        "国内ドラマ",
        "海外ドラマ",
        "連続ドラマ",
        "単発ドラマ",
        "バラエティ",
        "バラエティー",
        "映画",
        "邦画",
        "洋画",
        "ドキュメンタリー",
        "教養",
        "趣味",
        "教育",
        "音楽",
        "音楽番組",
        "ライブ",
        "ニュース",
        "報道",
        "情報",
        "情報番組",
        "劇場",
        "公演",
        "スポーツ",
        "エンタメ",
        "エンターテインメント",
        "番組",
        "テレビ",
        "新番組",
        "再放送",
        "特集",
        "スペシャル",
        "トーク",
        "トーク番組",
        "スタジオ",
        "ランキング",
        "生放送",
        "歴史",
        "科学",
        "文化",
        "社会",
        "経済",
        "政治",
        "芸術",
        "自然",
        "人間",
        "世界",
        "日本",
        "東京",
        "韓国",
        "中国",
        "アメリカ",
    }
)

# やや広いサブジャンル語（negative_tags に入る場合でも重みを最大 2 に制限する）
GENERIC_SUBGENRE_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "料理",
        "料理番組",
        "グルメ",
        "旅",
        "旅番組",
        "旅行",
        "散歩",
        "街ブラ",
        "クイズ",
        "クイズ番組",
        "お笑い",
        "コメディ",
        "恋愛",
        "ラブコメ",
        "青春",
        "学園",
        "ミステリー",
        "サスペンス",
        "刑事",
        "刑事ドラマ",
        "医療ドラマ",
        "時代劇",
        "韓国ドラマ",
        "中国ドラマ",
        "韓流",
        "華流",
        "アイドル",
        "声優",
        "鉄道",
        "ゴルフ",
        "釣り",
        "将棋",
        "囲碁",
        "園芸",
        "体操",
        "健康",
        "美容",
        "ファッション",
        "ペット",
        "動物",
        "キッズ",
        "子ども",
        "幼児",
        "英会話",
        "語学",
        "ワイドショー",
        "経済ニュース",
    }
)


def normalize_weighted_list(raw_list: list[Any] | None) -> list[dict[str, Any]]:
    """
    config.json 内のタグ配列（旧形式の str または新形式の {"name": str, "weight": int}）を
    [{"name": str, "weight": int}, ...] に正規化して重み降順で返す。
    """
    if not raw_list:
        return []
    merged: dict[str, dict[str, Any]] = {}
    for item in raw_list:
        if isinstance(item, str):
            name = item.strip()
            weight = 2
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("tag") or "").strip()
            try:
                weight = max(1, min(5, int(item.get("weight", 2))))
            except (TypeError, ValueError):
                weight = 2
        else:
            continue
        if not name:
            continue
        key = name.lower()
        if key not in merged or weight > merged[key]["weight"]:
            merged[key] = {"name": name, "weight": weight}
    return sorted(merged.values(), key=lambda x: x["weight"], reverse=True)


def _load_cached_corpus_texts() -> list[str]:
    """cache/*.json から全番組の title + description テキスト一覧を読み込む（DF算出用）。"""
    cache_dir = Path(__file__).parent / "cache"
    if not cache_dir.exists():
        return []
    by_id: dict[str, str] = {}
    for cfile in sorted(cache_dir.glob("*.json")):
        try:
            items = json.loads(cfile.read_text(encoding="utf-8"))
            for p in items:
                pid = p.get("id")
                if pid and pid not in by_id:
                    by_id[pid] = f"{p.get('title', '')} {p.get('description', '')}".lower()
        except Exception:
            pass
    return list(by_id.values())


def sanitize_negative_tags(
    raw_list: list[Any] | None,
    liked: list[dict[str, Any]] | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    3つの高度な検知＋固定辞書により negative_tags を精製する：
    1) スキーマ粒度分類 (category) ＆ 固定辞書による大分類・広義語の除外と重み上限
    2) liked（見たい履歴）との衝突検知（過去に選んだ番組にも含まれる一般語は除外）
    3) 番組コーパス全体での出現率 (Document Frequency) 検知（2.5%以上にヒットする広義語は除外、1.2%以上は★1に減衰）
    """
    if not raw_list:
        return []

    # 1. category 属性を保持しつつ正規化
    raw_categories: dict[str, str] = {}
    for item in raw_list:
        if isinstance(item, dict):
            n = str(item.get("name") or item.get("tag") or "").strip().lower()
            cat = str(item.get("category") or "").strip()
            if n and cat:
                raw_categories[n] = cat

    normalized = normalize_weighted_list(raw_list)

    liked_texts: list[str] = [
        f"{x.get('title', '')} {x.get('description', '')}".lower()
        for x in (liked or [])
    ]
    if candidates is not None:
        corpus_texts = [
            f"{c.get('title', '')} {c.get('description', '')}".lower()
            for c in candidates
        ]
    else:
        corpus_texts = _load_cached_corpus_texts()
    n_corpus = len(corpus_texts)

    sanitized: list[dict[str, Any]] = []
    for item in normalized:
        name = unicodedata.normalize("NFKC", item["name"]).strip()
        low = name.lower()
        cat = raw_categories.get(low, "")
        stripped_prog = re.sub(r"番組$", "", low).strip()

        # ③ 粒度分類 & 固定ストップワードによる大分類除外
        if (
            len(low) < 2
            or cat == "broad_genre"
            or low in BROAD_NEGATIVE_STOPWORDS
            or stripped_prog in BROAD_NEGATIVE_STOPWORDS
        ):
            continue

        is_shopping = any(sk.lower() in low for sk in ("ショッピング", "通販", "テレショップ", "買物"))

        # ② liked 履歴との衝突検知（過去に「見たい」と選んだ番組のタイトル・概要に含まれる語はNGタグから除外）
        if liked_texts and not is_shopping:
            liked_hits = sum(1 for lt in liked_texts if low in lt)
            if liked_hits >= 1:
                continue

        weight = item["weight"]
        if (
            cat == "subgenre"
            or low in GENERIC_SUBGENRE_NEGATIVE_WORDS
            or stripped_prog in GENERIC_SUBGENRE_NEGATIVE_WORDS
        ):
            weight = min(weight, 1)
        elif cat == "narrow_theme":
            weight = min(weight, 2)
        else:
            weight = min(weight, 3)

        # ① 番組コーパスにおける出現率 (DF) 検知
        if n_corpus >= 50 and not is_shopping:
            df_count = sum(1 for ct in corpus_texts if low in ct)
            df_ratio = df_count / n_corpus
            if df_ratio >= 0.025:
                # 全体の2.5%以上にヒットするワードは一般的すぎるため除外
                continue
            if df_ratio >= 0.012:
                # 1.2%〜2.5%にヒットするやや広いワードは重みを最大1に制限
                weight = min(weight, 1)

        sanitized.append({"name": name, "weight": weight})
    return sorted(sanitized, key=lambda x: x["weight"], reverse=True)


def format_weighted_tags(items: list[Any] | None) -> str:
    norm = normalize_weighted_list(items)
    return ", ".join(f"{x['name']}(★{x['weight']})" for x in norm)


# BS/CSの代表的な有料チャンネル判定用キーワード（NFKC正規化・小文字化後に channel 名で部分一致判定）
# ※無料の「BS10」は残し、有料の「BS10プレミアム」のみ除外する
PAID_CHANNEL_KEYWORDS: tuple[str, ...] = (
    "wowow",
    "j:com",
    "jcom",
    "bs10プレミアム",
    "スターチャンネル",
    "日本映画専門",
    "ディズニー",
    "アニマックス",
    "釣りビジョン",
    "j sports",
    "jsports",
    "グリーンチャンネル",
    "衛星劇場",
    "東映チャンネル",
    "キッズステーション",
    "フジテレビone",
    "フジテレビtwo",
    "フジテレビnext",
    "tbsチャンネル",
    "日テレプラス",
    "テレ朝チャンネル",
    "スカパー",
)


_NEW_PROG_RE = re.compile(
    r"🈖|\[新\]|(?:第1[話回]|#1(?:\D|$)|＃1(?:\D|$)|[（(]1[)）]|新番組|新シリーズ|初回SP|初回スペシャル)"
)


def is_new_program(prog: dict[str, Any]) -> bool:
    """新番組・初回放送かどうかを判定する（キャッシュ内の is_new フラグ＋タイトル/概要パターン）。"""
    if prog.get("is_new"):
        return True
    text = f"{prog.get('title', '')} {prog.get('description', '')}"
    return bool(_NEW_PROG_RE.search(text))


SHOPPING_KEYWORDS: tuple[str, ...] = (
    "ショッピング",
    "ショップジャパン",
    "通販",
    "テレショップ",
    "お買物情報",
    "ものスタ",
    "ダイレクトテレショップ",
    "ジャパネット",
)


def is_paid_channel(channel_name: str) -> bool:
    if not channel_name:
        return False
    norm_ch = unicodedata.normalize("NFKC", channel_name).strip().lower()
    return any(pk in norm_ch for pk in PAID_CHANNEL_KEYWORDS)


def filter_unseen(
    programs: list[dict[str, Any]],
    seen_ids: list[str],
    exclude_keywords: list[str] | None = None,
    exclude_paid_channels: bool = True,
    evaluated_titles: set[str] | None = None,
    disliked_titles: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    評価済み番組 (seen_ids および evaluated_titles の完全一致別枠放送)、
    過去に Dislike したシリーズの別話数、有料チャンネル、除外キーワード、
    および同一シリーズ複数話の重複（1シリーズ1件に統合）をフィルタする。
    """
    seen_set = set(seen_ids)
    ev_titles = {
        unicodedata.normalize("NFKC", t).strip().lower()
        for t in (evaluated_titles or set())
        if t and t.strip()
    }
    disliked_series_keys = {
        normalize_series_key(t)
        for t in (disliked_titles or set())
        if t and len(normalize_series_key(t)) >= 3
    }
    ex_words = [
        unicodedata.normalize("NFKC", kw).strip().lower()
        for kw in (exclude_keywords or [])
        if kw and kw.strip()
    ]
    filtered: list[dict[str, Any]] = []
    seen_series_in_batch: set[str] = set()

    for p in programs:
        if p.get("id") in seen_set:
            continue
        raw_t = p.get("title", "")
        norm_title = unicodedata.normalize("NFKC", raw_t).strip().lower()
        if norm_title and norm_title in ev_titles:
            continue

        s_key = normalize_series_key(raw_t)
        if s_key and s_key in disliked_series_keys:
            continue
        if s_key and s_key in seen_series_in_batch:
            continue

        ch = p.get("channel", "")
        if exclude_paid_channels and is_paid_channel(ch):
            continue
        title_desc = f"{raw_t} {p.get('description', '')}"
        if any(sk in title_desc for sk in SHOPPING_KEYWORDS):
            continue
        if ex_words:
            haystack = unicodedata.normalize(
                "NFKC",
                f"{ch} {raw_t} {p.get('description', '')}",
            ).lower()
            if any(ew in haystack for ew in ex_words):
                continue

        if s_key:
            seen_series_in_batch.add(s_key)
        filtered.append(p)
    return filtered


def extract_keyword_matches(
    programs: list[dict[str, Any]], keywords: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Stage 1: config.json の keywords が title または description に部分一致する番組を
    最優先枠 ([KEYWORD MATCH]) として抽出する。
    """
    active_keywords = [k.strip() for k in keywords if k and k.strip()]
    if not active_keywords:
        return [], list(programs)

    matched: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []

    for prog in programs:
        haystack = f"{prog.get('title', '')} {prog.get('description', '')}".lower()
        hit_words = [kw for kw in active_keywords if kw.lower() in haystack]
        if hit_words:
            enriched = dict(prog)
            enriched["match_type"] = "KEYWORD"
            enriched["score"] = 5
            enriched["reason"] = f"[KEYWORD MATCH] キーワード「{', '.join(hit_words)}」に一致"
            matched.append(enriched)
        else:
            remaining.append(prog)

    return matched, remaining


def _build_scoring_prompt(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    top_k: int = 25,
) -> str:
    profile_summary = (config.get("profile_summary") or "").strip()
    favorite_series = normalize_weighted_list(config.get("favorite_series"))
    positive_tags = normalize_weighted_list(config.get("positive_tags"))
    negative_tags = sanitize_negative_tags(
        config.get("negative_tags"),
        liked=config.get("liked"),
        candidates=candidates,
    )
    # タグ蓄積がある場合は直近10件のみ、まだない場合は直近20件をFew-shot例として渡す（プロンプト軽量化・高速化）
    recent_limit = 10 if positive_tags else 20
    liked = (config.get("liked") or [])[-recent_limit:]
    disliked = (config.get("disliked") or [])[-15:]
    target_genres = config.get("target_genres") or ["documentary", "movie", "hobby"]

    lines: list[str] = [
        "あなたはテレビ番組のパーソナライズ推薦キュレーターです。",
        f"以下の候補番組リストから、ユーザーにとって見る価値が高い上位 {top_k} 件を厳選し（必ず {top_k} 件近く選出すること）、",
        "score (1〜5) と 1行の推薦理由 (reason) を付与してJSONで出力してください。",
        "※重要: `[新番組/初回]` マークが付いた番組は、新しい良質な番組との出会いを広げるため積極的に上位候補としてピックアップし、スコアを優遇すること。",
        "※注意: 「放送大学」の番組はユーザーの関心テーマに強く合致する場合のみ最大1〜2件までに留め、放送大学ばかりを選ばないこと。",
        "",
        f"## ユーザーの関心ジャンル: {', '.join(target_genres)}",
    ]

    if favorite_series:
        lines.extend(
            [
                "",
                "## 蓄積されたお気に入り番組・シリーズ枠（★1〜5は重要度重み）",
                format_weighted_tags(favorite_series),
            ]
        )

    if positive_tags:
        lines.extend(
            [
                "",
                "## 蓄積された関心キーワード・個別テーマ・人物タグ（★5が最重要、★1は単発関心）",
                format_weighted_tags(positive_tags),
            ]
        )

    if negative_tags:
        lines.extend(
            [
                "",
                "## 蓄積されたマイナス評価要素タグ（★に応じて減点。ただしお気に入り枠や強い関心テーマが上回る場合は推薦可）",
                format_weighted_tags(negative_tags),
            ]
        )

    if profile_summary:
        lines.extend(
            [
                "",
                "## ユーザーの好みの全体傾向（5箇条）",
                profile_summary,
            ]
        )

    if liked:
        lines.append("")
        lines.append("## 過去に「見たい (Positive)」と評価した番組履歴（直近最大30件）")
        for item in liked:
            lines.append(
                f"- [{item.get('genre', '')}] {item.get('title', '')}: {item.get('description', '')}"
            )
    else:
        lines.append("")
        lines.append(
            "※まだ「見たい」履歴が少ないため、対象ジャンル（ドキュメンタリー・映画・趣味/教養など）の中から、"
            "テーマの独自性・知的好奇心・作品性の高い良質な番組を優先して推薦してください。"
            "通販番組・単なる番宣・定時ニュース・講義の単調な羅列は除外してください。"
        )

    if disliked:
        lines.append("")
        lines.append("## 過去に「好みと違う (Negative)」と評価した番組履歴（直近最大30件・除外傾向）")
        for item in disliked:
            lines.append(
                f"- [{item.get('genre', '')}] {item.get('title', '')}: {item.get('description', '')}"
            )

    lines.append("")
    lines.append("## 候補番組リスト")
    for c in candidates:
        new_badge = "[新番組/初回] " if is_new_program(c) else ""
        rank_badge = "[ランキング掲載] " if c.get("is_ranked") else ""
        desc = (c.get("description") or "")[:120]
        lines.append(
            f"- ID: {c['id']} | {new_badge}{rank_badge}[{c.get('genre', '')}] "
            f"{c.get('title', '')} ({c.get('datetime', '')} / {c.get('channel', '')}) "
            f"| 概要: {desc}"
        )

    return "\n".join(lines)


def _char_bigrams(text: str) -> Counter[str]:
    cleaned = "".join(c for c in text.lower() if not c.isspace())
    if len(cleaned) < 2:
        return Counter([cleaned]) if cleaned else Counter()
    return Counter(cleaned[i : i + 2] for i in range(len(cleaned) - 1))


def _apply_channel_cap(
    items: list[dict[str, Any]],
    limit: int,
    ouj_max: int = 2,
    default_ch_max: int = 5,
) -> list[dict[str, Any]]:
    """
    特定チャンネル（特に講義枠が大量にある『放送大学』など）が推薦枠を独占しないよう、
    放送局ごとの上限件数キャップを適用して上位 limit 件を返す。
    """
    selected: list[dict[str, Any]] = []
    ch_counts: Counter[str] = Counter()
    for item in items:
        ch = item.get("channel") or ""
        is_ouj = "放送大学" in ch
        cap = ouj_max if is_ouj else default_ch_max
        key = "放送大学" if is_ouj else ch
        if key and ch_counts[key] >= cap:
            continue
        ch_counts[key] += 1
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def rank_candidates_locally(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    limit: int = 35,
) -> list[dict[str, Any]]:
    """
    文字バイグラム TF-IDF ベクトル類似度（個別Liked番組とのMaxSim）＋ 重み付き蓄積タグ・シリーズ照合により、
    1,000件規模の候補から約0.03秒で上位候補に高速一次選別する（放送大学などの同一局独占も抑制）。
    """
    fav_series = [
        x for x in normalize_weighted_list(config.get("favorite_series")) if len(x["name"]) >= 2
    ]
    pos_tags = [
        x for x in normalize_weighted_list(config.get("positive_tags")) if len(x["name"]) >= 2
    ]
    neg_tags = [
        x
        for x in sanitize_negative_tags(
            config.get("negative_tags"),
            liked=config.get("liked"),
            candidates=candidates,
        )
        if len(x["name"]) >= 2
    ]
    liked: list[dict[str, Any]] = config.get("liked") or []
    disliked: list[dict[str, Any]] = config.get("disliked") or []

    # IDF 計算（全候補＋履歴から高速算出）
    n_docs = max(1, len(candidates) + len(liked))
    df_counts: Counter[str] = Counter()
    doc_vecs: list[Counter[str]] = []
    for c in candidates:
        vec = _char_bigrams(f"{c.get('title', '')} {c.get('description', '')}")
        doc_vecs.append(vec)
        for bg in vec.keys():
            df_counts[bg] += 1

    liked_vecs: list[tuple[str, Counter[str]]] = []
    for item in liked:
        l_title = item.get("title", "")
        vec = _char_bigrams(f"{l_title} {item.get('description', '')}")
        liked_vecs.append((l_title, vec))
        for bg in vec.keys():
            df_counts[bg] += 1

    disliked_vecs: list[Counter[str]] = [
        _char_bigrams(f"{item.get('title', '')} {item.get('description', '')}")
        for item in disliked
    ]

    idf: dict[str, float] = {
        bg: math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        for bg, df in df_counts.items()
    }

    def _to_unit_tfidf(v: Counter[str]) -> dict[str, float]:
        if not v:
            return {}
        weighted = {k: cnt * idf.get(k, 1.0) for k, cnt in v.items()}
        norm = math.sqrt(sum(val * val for val in weighted.values()))
        if norm == 0.0:
            return {}
        return {k: val / norm for k, val in weighted.items()}

    unit_liked = [(title, _to_unit_tfidf(vec)) for title, vec in liked_vecs]
    unit_disliked = [_to_unit_tfidf(vec) for vec in disliked_vecs]

    def _dot(u1: dict[str, float], u2: dict[str, float]) -> float:
        if not u1 or not u2:
            return 0.0
        if len(u1) > len(u2):
            u1, u2 = u2, u1
        return sum(val * u2[k] for k, val in u1.items() if k in u2)

    def _pos_tag_matches(tag_name: str, hay_text: str) -> bool:
        t_low = tag_name.lower().strip()
        if t_low in hay_text:
            return True
        stem = re.sub(r"(番組|講座|企画)$", "", t_low).strip()
        if len(stem) >= 3 and stem in hay_text:
            return True
        return False

    def _neg_tag_matches(tag_name: str, title_desc_text: str) -> bool:
        # NGタグは語幹切り詰め（「韓国ドラマ」→「韓国」など）を行わず、
        # また genre 文字列とも照合しないことで、広範な巻き添え減点を防ぐ
        t_low = tag_name.lower().strip()
        return bool(t_low and t_low in title_desc_text)

    scored_items: list[tuple[float, dict[str, Any]]] = []
    for idx, c in enumerate(candidates):
        title_desc_hay = f"{c.get('title', '')} {c.get('description', '')}".lower()
        hay = f"{title_desc_hay} {c.get('genre', '')}".lower()
        ch = c.get("channel") or ""

        # 学習された negative_tags はタイトル・概要に完全部分一致した場合のみ緩やかに減点し、
        # 減点合計にも上限 (1.0) を設けることでジャンル全体や関心番組が潰れないようにする。
        neg_hits = [nt for nt in neg_tags if _neg_tag_matches(nt["name"], title_desc_hay)]
        neg_tag_penalty = min(1.0, sum(nt["weight"] * 0.25 for nt in neg_hits))

        u_doc = _to_unit_tfidf(doc_vecs[idx])

        # 個別Liked番組との最大類似度 (MaxSim) と最も似た過去番組名を取得
        # ※放送大学同士の定型文（「第○回」「放送大学」）だけで類似度が跳ねないよう、放送大学はタグ一致を重視
        best_sim = 0.0
        best_liked_title = ""
        for l_title, u_l in unit_liked:
            sim = _dot(u_doc, u_l)
            if sim > best_sim:
                best_sim = sim
                best_liked_title = l_title

        max_neg_sim = max((_dot(u_doc, u_d) for u_d in unit_disliked), default=0.0)

        series_hits = [s for s in fav_series if s["name"].lower() in hay]
        tag_hits = [t for t in pos_tags if _pos_tag_matches(t["name"], hay)]

        series_boost = max((s["weight"] * 0.9 for s in series_hits), default=0.0)
        tag_boost = min(5.0, sum(t["weight"] * 0.55 for t in tag_hits))

        # 放送大学は具体的な関心タグにヒットしていない場合、スコアを少し抑える
        ouj_penalty = 0.8 if ("放送大学" in ch and not tag_hits) else 0.0
        is_new = is_new_program(c)
        # 新番組ボーナスは強いNGタグに該当しない番組に対して加算
        new_boost = 0.95 if (is_new and "放送大学" not in ch and neg_tag_penalty < 0.75) else 0.0

        total_score = (
            best_sim * 7.5
            - max_neg_sim * 5.0
            - neg_tag_penalty
            - ouj_penalty
            + series_boost
            + tag_boost
            + new_boost
            + (0.25 if c.get("is_ranked") else 0.0)
            + (0.15 if len(c.get("description") or "") >= 50 else 0.0)
        )

        enriched = dict(c)
        enriched["is_new"] = is_new
        enriched["_local_score"] = round(total_score, 3)
        enriched["_series_hits"] = [s["name"] for s in series_hits]
        enriched["_tag_hits"] = [f"{t['name']}(★{t['weight']})" for t in tag_hits]
        enriched["_best_liked_title"] = best_liked_title if best_sim >= 0.14 else ""
        scored_items.append((total_score, enriched))

    scored_items.sort(key=lambda x: x[0], reverse=True)
    ordered_all = [item for _s, item in scored_items]

    # 放送大学は一次選別35件の中で最大2件までに制限し、他局の番組を広く確保する
    capped = _apply_channel_cap(ordered_all, limit=limit, ouj_max=2, default_ch_max=8)
    return capped


def score_with_llm(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    top_k: int = 25,
) -> list[dict[str, Any]]:
    """
    Stage 2: ローカルTF-IDF＋タグ照合で上位候補に高速一次選別した上で、
    Gemini Structured Outputs で上位 top_k 件をピックアップしてスコアリングする。
    """
    if not candidates:
        return []

    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "環境変数 GEMINI_API_KEY が設定されていません。"
            "`.env` ファイルの `GEMINI_API_KEY=` にキーを記入して再実行してください。"
        )

    # ローカルTF-IDF＋タグ照合で上位候補（表示件数の約1.8倍、最低45件）に一次選別（0.03秒）してからLLMに渡す
    pool_size = max(45, int(top_k * 1.8))
    sorted_candidates = rank_candidates_locally(candidates, config, limit=pool_size)

    prompt = _build_scoring_prompt(sorted_candidates, config, top_k=top_k)

    client = genai.Client(api_key=api_key)
    response = _generate_structured(
        client,
        prompt=prompt,
        schema=RecommendationResponse,
        temperature=0.3,
    )

    parsed = RecommendationResponse.model_validate_json(response.text)
    by_id = {c["id"]: c for c in sorted_candidates}

    results: list[dict[str, Any]] = []
    seen_rec_ids: set[str] = set()
    for rec in parsed.recommendations:
        orig = by_id.get(rec.id)
        if not orig or rec.id in seen_rec_ids:
            continue
        seen_rec_ids.add(rec.id)
        enriched = dict(orig)
        enriched["match_type"] = "LLM"
        enriched["score"] = rec.score
        enriched["reason"] = rec.reason
        results.append(enriched)

    # もしLLMが top_k より少ない件数しか返さなかった場合、ローカル一次選別の上位で補完して件数を満たす
    if len(results) < top_k:
        for fallback_item in score_fallback_local(sorted_candidates, config, top_k=top_k):
            if fallback_item["id"] not in seen_rec_ids:
                seen_rec_ids.add(fallback_item["id"])
                results.append(fallback_item)
                if len(results) >= top_k:
                    break

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return _apply_channel_cap(results, limit=top_k, ouj_max=2, default_ch_max=6)


def score_fallback_local(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """
    ローカルTF-IDFベクトル類似度＋蓄積タグ・シリーズ一致による高速スコアリング（--fast モードやAPI上限時用）。
    """
    ranked_local = rank_candidates_locally(candidates, config, limit=top_k)
    results: list[dict[str, Any]] = []
    for idx, c in enumerate(ranked_local):
        reasons: list[str] = []
        if is_new_program(c):
            reasons.append("新番組・初回放送")
        if c.get("_series_hits"):
            reasons.append(f"お気に入り枠「{', '.join(c['_series_hits'][:2])}」")
        if c.get("_tag_hits"):
            reasons.append(f"関心タグ「{', '.join(c['_tag_hits'][:3])}」")
        if c.get("_best_liked_title"):
            reasons.append(f"履歴「{c['_best_liked_title'][:18]}…」に類似")
        if c.get("is_ranked"):
            reasons.append("ランキング注目")
        if not reasons:
            reasons.append("過去の視聴履歴と内容が類似（TF-IDFベクトル選別）")

        score = 5 if idx < 5 else (4 if idx < 12 else 3)
        enriched = dict(c)
        enriched["match_type"] = "LOCAL_VEC"
        enriched["score"] = score
        enriched["reason"] = " / ".join(reasons)
        results.append(enriched)
    return results


def recommend_programs(
    programs: list[dict[str, Any]],
    config: dict[str, Any],
    top_k: int = 25,
) -> list[dict[str, Any]]:
    """未評価かつ有料局・除外キーワード外の番組から Stage 1 (キーワード一致) + Stage 2 (LLM推薦) を結合して返す。"""
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
    unseen = filter_unseen(
        programs,
        config.get("seen_ids", []),
        config.get("exclude_keywords", []),
        exclude_paid_channels=bool(config.get("exclude_paid_channels", True)),
        evaluated_titles=evaluated_titles,
        disliked_titles=disliked_titles,
    )
    kw_matches, remaining = extract_keyword_matches(unseen, config.get("keywords", []))
    llm_matches = score_with_llm(remaining, config, top_k=top_k)
    return kw_matches + llm_matches


def summarize_profile(config: dict[str, Any]) -> dict[str, Any]:
    """
    蓄積された liked / disliked と現在のタグ群から、
    1) 全体傾向の5箇条 (profile_summary)
    2) 重み付き(1〜5)の具体タグ群 (favorite_series, positive_tags, negative_tags)
    を再評価・統合（頻出要素の重み強化、否定された要素の減衰・整理）して返す。
    """
    liked = config.get("liked") or []
    disliked = config.get("disliked") or []
    if not liked and not disliked:
        raise ValueError("liked / disliked の履歴がまだありません。まずは番組を選択して履歴を蓄積してください。")

    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("`.env` または環境変数に GEMINI_API_KEY が設定されていません。")

    existing_series = normalize_weighted_list(config.get("favorite_series"))
    existing_pos = normalize_weighted_list(config.get("positive_tags"))
    existing_neg = sanitize_negative_tags(config.get("negative_tags"), liked=liked)

    lines = [
        "以下はユーザーが過去に「見たい (Liked)」または「好みと違う (Disliked)」と分類したテレビ番組の履歴と、現在蓄積されているタグ一覧です。",
        "単なる無限追記ではなく、**個別の具体的な判断を残しつつ、重要度の重み付け (weight: 1〜5) と新陳代謝（強化・減衰・整理）** を行って以下の4項目を出力してください：",
        "",
        "1. five_rules: 全体の好みの境界線（5箇条）",
        "   - ※注意: 個別のアニメやドラマ・バラエティを数件『好みと違う』にしただけで「アニメは避ける」「ドラマは避ける」のように大分類ジャンル全体を否定するルールを書かないこと。",
        "2. favorite_series: 『見たい』に含まれる番組のシリーズ名・枠名と重み(1〜5)。",
        "   - 複数回選ばれている定番シリーズは weight=4〜5、1回選ばれたシリーズは weight=2〜3。",
        "   - ※注意: 『放送大学』の個別講義名（例: 『知覚・認知心理学』『現代アメリカの政治と社会』等）はシリーズ枠ではないため favorite_series には入れず、positive_tags 側でテーマとして扱うこと。",
        "   - 『好みと違う (Disliked)』に後から入ったシリーズは削除または減衰させること。",
        "3. positive_tags: 『見たい』の各番組から読み取れる具体的な関心キーワード・テーマ・人物名・監督名・企画要素と重み(1〜5)。※番組名や説明文と部分一致しやすいよう、余計な接尾語（〜番組、〜企画など）を省いた短いキーワードにすること。",
        "   - 複数の番組に共通して現れる中核テーマ（例: 社会心理学、アメリカ政治、博物館、ものづくり、SFアクション等）は weight=4〜5 に強化する。",
        "   - 1回だけ選ばれた個別具体的な固有名詞・トピックは消さずに weight=1〜2 として残す。",
        "   - 『好みと違う (Disliked)』で否定された要素や、広すぎてノイズになる一般語は整理・除外する。",
        "4. negative_tags: 『好みと違う』の各番組から読み取れる具体的な除外対象（特定の番組シリーズ名、または『通販』『幼児向け』『韓流時代劇』『ゴシップワイドショー』など狭い複合カテゴリ）と重み(1〜3)。",
        "   - **絶対禁止**: 『アニメ』『ドラマ』『バラエティ』『映画』『音楽』『スポーツ』『ニュース』『情報』『趣味』『教養』のような大分類ジャンル名や、『料理』『旅』『トーク』『歴史』『経済』のような一般的な広い単語を入れてはいけない。",
        "   - 個別のアニメやドラマをいくつか選ばなかったからといってジャンル全体のNG重みを増やさず、重みは原則 weight=1〜2（明確な通販・幼児枠などでも最大3）に抑えること。",
        "",
        f"※現在の favorite_series: {format_weighted_tags(existing_series) or 'なし'}",
        f"※現在の positive_tags: {format_weighted_tags(existing_pos) or 'なし'}",
        f"※現在の negative_tags: {format_weighted_tags(existing_neg) or 'なし'}",
        "",
        "## 見たい (Liked)",
    ]
    for item in liked[-80:]:
        lines.append(
            f"- [{item.get('genre', '')}] {item.get('title', '')}: {item.get('description', '')}"
        )

    lines.append("")
    lines.append("## 好みと違う (Disliked)")
    for item in disliked[-80:]:
        lines.append(
            f"- [{item.get('genre', '')}] {item.get('title', '')}: {item.get('description', '')}"
        )

    client = genai.Client(api_key=api_key)
    response = _generate_structured(
        client,
        prompt="\n".join(lines),
        schema=ProfileSummaryResponse,
        temperature=0.2,
    )
    parsed = ProfileSummaryResponse.model_validate_json(response.text)
    summary_text = "\n".join(
        f"{idx}. {rule}" for idx, rule in enumerate(parsed.five_rules, start=1)
    )
    return {
        "profile_summary": summary_text,
        "favorite_series": normalize_weighted_list(
            [t.model_dump() for t in parsed.favorite_series]
        ),
        "positive_tags": normalize_weighted_list(
            [t.model_dump() for t in parsed.positive_series if hasattr(parsed, "positive_series")]
            if False
            else [t.model_dump() for t in parsed.positive_tags]
        ),
        "negative_tags": sanitize_negative_tags(
            [t.model_dump() for t in parsed.negative_tags],
            liked=liked,
        ),
    }
