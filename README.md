# tv-recommend

`bangumi.org`（番組表.Gガイド）の番組表・ランキングデータから、あなたの好みに合うテレビ番組（地上波・BS）を毎日ピックアップし、ターミナル上で対話的に選別・学習・**nasne 録画予約**まで完結できるパーソナライズ CLI ツールです。

---

## 主な特徴

1. **ハイブリッド推薦エンジン（ルールベース ＋ ローカルTF-IDF ＋ Gemini API）**
   - **Stage 1（キーワード即時抽出）**: 登録した関心キーワードに一致する番組を最優先で抽出。
   - **Stage 2（ローカル一次選別 ＋ LLM推薦）**: 1,000件規模の番組候補から文字バイグラム TF-IDF 類似度と重み付きタグ照合で約0.03秒で上位候補に絞り込み、Gemini API（Structured Outputs）でおすすめ度（★1〜5）と1行解説を生成。
   - **高速モード (`--fast`)**: LLM API を呼ばず、ローカルのベクトル類似度と学習済みタグのみで約0.05秒で推薦一覧を表示可能。
2. **好みの自動学習・新陳代謝 (`summarize`)**
   - 日々の「〇 選ぶ」「× いらない」の選択履歴から、LLM が**好みの5箇条 (`profile_summary`)**、**お気に入り番組枠 (`favorite_series`)**、**関心タグ (`positive_tags`)**、**NGタグ (`negative_tags`)** を重み付き（★1〜5）で自動抽出・更新。
3. **直感的なターミナル対話UI (TUI)**
   - カーソル移動とワンキー操作で「`〇 選ぶ`」「`● 録画予約`」「`× いらない`」「`ー スルー`」をサクサク切り替え可能。
4. **nasne 録画予約連携（複数台対応）**
   - LAN 内の nasne（UPnP SSDP 自動検出または IP 指定）と連携し、予約済み状態の表示・受信可能チャンネル判定・時間帯の予約競合チェックを自動実行。複数台の nasne がある場合は空きがある筐体へ自動で振り分けて予約登録します。
5. **先取り・遡及チェック対応**
   - デフォルトの「今日・明日（2日間）」に加え、`--ahead` で1週間先までの先取りチェックや、数日空いた際の過去遡及チェックメニューに対応。

---

## 必要環境

- Python 3.11 以上
- Gemini API キー（`GEMINI_API_KEY`）
- （任意）同一 LAN 内の Sony / BUFFALO **nasne**（録画予約機能を使う場合）

---

## セットアップ

```bash
git clone https://github.com/dotimpact/tv-recommend.git
cd tv-recommend

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

プロジェクトルートに `.env` ファイルを作成し、Gemini API キーを設定します。

```env
GEMINI_API_KEY=your_api_key_here
# 任意: 使用するモデルを指定（デフォルト: gemini-3.1-flash-lite）
# GEMINI_MODEL=gemini-3.1-flash-lite
```

---

## 使い方

### 1. 日々の番組チェック＆フィードバック

```bash
python main.py
```

- 初回または当日初回実行時に `bangumi.org` から番組データを取得し、`cache/` にキャッシュします。
- ピックアップ一覧が表示された後、対話UIで各番組を評価できます：
  - `↑` / `↓` (`k` / `j`): 番組移動
  - `Space` / `←` / `→`: 状態切り替え（スルー ⇄ 選ぶ ⇄ いらない）
  - `o` (`1`): **〇 選ぶ**（興味あり履歴に追加）
  - `r`: **● 録画予約**（興味あり履歴に追加 ＋ 確定時に nasne へ録画予約）
  - `x` (`2`): **× いらない**（除外履歴に追加）
  - `-` (`s` / `0`): **ー スルー**
  - `Enter`: 確定して保存（評価後は自動でプロファイル・重み付きタグが更新されます）
  - `q` / `Esc`: キャンセル

#### 主なオプション

```bash
# 1週間先までの番組を2日単位で先取りチェック
python main.py --ahead

# LLMを使わずローカルTF-IDF＋タグ照合のみで瞬時に表示（約0.05秒）
python main.py --fast

# 指定した日付（過去・未来）から2日分をチェック
python main.py --date 2026-11-11

# 表示件数を指定 / キャッシュを無視して再取得 / 一覧表示のみ（対話なし）
python main.py --limit 30 --refresh --no-interactive
```

---

### 2. 「〇 選ぶ」にした番組からあとで nasne 録画予約 (`record`)

対話画面で一旦「`〇 選ぶ`」にしておいた今後の放送予定番組を一覧し、あとからまとめて nasne に録画予約できます。

```bash
# 対話UIで録画予約する番組を選択
python main.py record

# 今後の「〇 選ぶ」番組と nasne 予約状況の一覧表示のみ
python main.py record --list
```

---

### 3. 関心キーワード・除外キーワードの管理 (`keyword` / `exclude`)

```bash
# 関心キーワード（タイトル・概要に部分一致したら無条件で最優先ピックアップ）
python main.py keyword list
python main.py keyword add "SF"
python main.py keyword rm "SF"

# 除外キーワード（放送局・タイトル・概要に含まれていたら候補から除外）
python main.py exclude list
python main.py exclude add "通販"
python main.py exclude rm "通販"
```

---

### 4. 好みプロファイルの手動再集計 (`summarize`)

```bash
python main.py summarize
```

蓄積された `liked` / `disliked` 履歴から、好みの傾向5箇条と重み付きタグ（`favorite_series` / `positive_tags` / `negative_tags`）を再生成します。

---

## ファイル構成

```text
tv-recommend/
├── main.py              # CLI エントリポイント・対話TUI・サブコマンド制御
├── scraper.py           # bangumi.org スクレイピング・EPG解析・重複排除・キャッシュ管理
├── recommender.py       # キーワード抽出・ローカルTF-IDF選別・Gemini API 推薦＆プロファイル要約
├── nasne.py             # nasne 自動検出(SSDP)・予約状況照合・競合チェック・録画予約API連携
├── config.example.json  # 設定ファイルの初期テンプレート（初回起動時に config.json が自動生成されます）
└── requirements.txt     # 依存パッケージ一覧
```

## 設定項目 (`config.json`)

| キー | 説明 | デフォルト例 |
| :--- | :--- | :--- |
| `area_code` | エリアコード（`23` = 東京） | `"23"` |
| `broadcast_types` | 取得対象の放送波（`terrestrial`, `bs`, `cs`, `bs4k`） | `["terrestrial", "bs"]` |
| `target_genres` | 取得対象ジャンル一覧 | `["documentary", "movie", ...]` |
| `display_limit` | 推薦表示件数 | `25` |
| `exclude_paid_channels` | WOWOW 等の有料放送局を自動除外するか | `true` |
| `nasne_ips` | 優先接続する nasne の IP アドレス配列（空または未応答時は SSDP 自動検出） | `[]` |
| `nasne_quality` | nasne 録画画質（`101` = DR、3倍モード等は環境に応じる） | `101` |
| `nasne_storage_id` | nasne 録画先ストレージ ID（`0` = 内蔵HDD、`1` = 外付けHDD） | `0` |
