"""
HYBRID NEWS-FIRST QUANT STOCK PREDICTOR

Educational use only. This is not financial advice.

Core idea:
- News is the main signal, especially fresh catalysts and negative risk headlines.
- Historical price action is still used for trend, pullback, support, resistance,
  relative strength, RSI, MACD, volume, and Monte Carlo risk simulation.
- Company growth/quality is used when yfinance fundamentals are available.
- Macro variables are NOT hardcoded to a specific date. Every run pulls current
  market/economic/geopolitical headlines and converts them into live variables.
"""

from __future__ import annotations

import math
import os
import re
import time
import urllib.parse
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
warnings.filterwarnings("ignore")

try:
    import feedparser
except Exception:
    feedparser = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import requests
except Exception:
    requests = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


APP_NAME = "Hybrid News-First Quant Stock Predictor"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)


DEFAULT_TICKERS = [
    "NVDA", "AVGO", "AMD", "MRVL", "MU", "TSM", "ARM", "ANET", "SMCI", "VRT",
    "MSFT", "GOOGL", "AMZN", "META", "AAPL", "ORCL", "CRM", "NOW",
    "CRWD", "PANW", "NET", "DDOG", "SNOW", "PLTR",
    "LMT", "RTX", "NOC", "GD", "KTOS", "RKLB", "ASTS", "RDW",
    "XOM", "CVX", "COP", "SLB", "CCJ", "CEG", "GEV",
    "LLY", "NVO", "REGN", "VRTX", "AMGN", "MRNA", "UNH",
    "JPM", "V", "MA", "COST", "WMT",
]


OPPORTUNISTIC_TICKERS = [
    # High-beta / smaller-cap / catalyst-sensitive names.
    # These are not recommendations by themselves; they are a broad scan pool.
    "IONQ", "QUBT", "RGTI", "SOUN", "BBAI", "AI", "PATH", "UPST",
    "WOLF", "QS", "ENVX", "ACHR", "JOBY", "LUNR", "KULR", "SERV",
    "ONDS", "HIMS", "SOFI", "HOOD", "RIVN", "LCID", "CHPT", "BLNK",
    "RKLB", "ASTS", "RDW", "IRDM", "SPIR", "PL", "KTOS", "AVAV",
    "UUUU", "UEC", "NXE", "SMR", "OKLO", "LEU", "BE", "FLNC",
    "CRSP", "NTLA", "BEAM", "RXRX", "SDGR", "DNA", "TGTX", "VKTX",
    "CELH", "ELF", "CAVA", "DUOL", "TOST", "AFRM", "COIN", "MARA",
    "RIOT", "CLSK", "IREN", "CIFR",
]


SECTOR_MAP = {
    "NVDA": "ai_semis", "AVGO": "ai_semis", "AMD": "ai_semis", "MRVL": "ai_semis",
    "MU": "ai_semis", "TSM": "ai_semis", "ARM": "ai_semis", "ANET": "ai_semis",
    "SMCI": "ai_semis", "VRT": "data_center",
    "MSFT": "mega_tech", "GOOGL": "mega_tech", "AMZN": "mega_tech", "META": "mega_tech",
    "AAPL": "mega_tech", "ORCL": "mega_tech", "CRM": "software", "NOW": "software",
    "CRWD": "cyber", "PANW": "cyber", "NET": "cyber", "DDOG": "software",
    "SNOW": "software", "PLTR": "defense_ai",
    "LMT": "defense", "RTX": "defense", "NOC": "defense", "GD": "defense",
    "KTOS": "defense", "RKLB": "space_defense", "ASTS": "space", "RDW": "space",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "CCJ": "uranium", "CEG": "nuclear_power", "GEV": "grid_power",
    "LLY": "healthcare", "NVO": "healthcare", "REGN": "biotech", "VRTX": "biotech",
    "AMGN": "biotech", "MRNA": "biotech", "UNH": "managed_care",
    "JPM": "financials", "V": "payments", "MA": "payments",
    "COST": "consumer_quality", "WMT": "consumer_quality",
}


SOURCE_CREDIBILITY = {
    "Reuters": 1.35,
    "Bloomberg": 1.30,
    "Associated Press": 1.25,
    "AP": 1.25,
    "The Wall Street Journal": 1.25,
    "CNBC": 1.15,
    "Barron's": 1.15,
    "MarketWatch": 1.10,
    "Yahoo Finance": 1.05,
    "Investor's Business Daily": 1.00,
    "Investing.com": 0.95,
    "Benzinga": 0.85,
    "TipRanks": 0.85,
    "Seeking Alpha": 0.75,
    "Motley Fool": 0.75,
}


POSITIVE_EVENTS = {
    "earnings_guidance": [
        "beats earnings", "beat earnings", "beats estimates", "earnings beat",
        "raises guidance", "raised guidance", "guidance raised", "record revenue",
        "strong revenue", "strong demand", "profit jumps", "margin expansion",
    ],
    "analyst_upgrade": [
        "upgrade", "upgraded", "price target raised", "buy rating",
        "outperform", "overweight", "top pick", "initiates buy",
    ],
    "contract_partnership": [
        "contract", "deal", "partnership", "collaboration", "agreement",
        "selected by", "supplier", "wins award", "government contract",
    ],
    "ai_data_center": [
        "artificial intelligence", " ai ", "data center", "gpu", "accelerator",
        "semiconductor", "chip demand", "cloud demand", "inference",
    ],
    "defense_geopolitics": [
        "defense spending", "missile", "air defense", "drone", "military",
        "pentagon", "nato", "geopolitical tensions",
    ],
    "energy_oil": [
        "oil rises", "crude rises", "brent rises", "supply disruption",
        "strait of hormuz", "energy prices", "uranium", "nuclear power",
    ],
    "biotech_healthcare": [
        "fda approval", "drug approved", "phase 2", "phase ii", "phase 3",
        "phase iii", "met primary endpoint", "positive trial", "clinical trial",
    ],
    "capital_return": ["buyback", "share repurchase", "dividend hike", "raises dividend"],
}


NEGATIVE_EVENTS = {
    "earnings_miss": [
        "misses earnings", "missed earnings", "misses estimates", "cuts guidance",
        "lowered guidance", "weak revenue", "margin pressure",
    ],
    "analyst_downgrade": [
        "downgrade", "downgraded", "price target cut", "sell rating",
        "underperform", "underweight",
    ],
    "legal_regulatory": [
        "lawsuit", "investigation", "sec probe", "doj probe", "fraud",
        "antitrust", "recall", "ban", "sanctions",
    ],
    "biotech_failure": [
        "failed trial", "missed endpoint", "clinical hold", "fda rejection",
        "safety concern", "adverse event",
    ],
    "market_reaction": [
        "plunges", "tumbles", "falls after", "shares sink", "selloff", "short report",
    ],
    "macro_pressure": [
        "inflation fears", "rate hike", "yields climb", "oil shock",
        "recession risk", "consumer slowdown",
    ],
}


def live_macro_queries() -> List[str]:
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        f"{today} latest US stock market news Fed CPI inflation rates oil war Reuters",
        "latest CNBC market news AI semiconductor chips data center stock market",
        "latest Reuters geopolitical conflict oil prices inflation stock market",
        "latest market sector rotation defense energy healthcare technology Reuters",
        "latest economic data jobs inflation treasury yields Federal Reserve stocks",
    ]


@dataclass
class NewsItem:
    title: str
    source: str = "Unknown"
    url: str = ""
    published: str = ""


@dataclass
class TechnicalSnapshot:
    price: Optional[float] = None
    rsi: Optional[float] = None
    trend_label: str = "unknown"
    pullback_label: str = "unknown"
    support: Optional[float] = None
    resistance: Optional[float] = None
    volatility: Optional[float] = None
    volume_ratio: Optional[float] = None
    return_5d: Optional[float] = None
    return_1w: Optional[float] = None
    return_1m: Optional[float] = None
    above_ma20: Optional[float] = None
    above_ma50: Optional[float] = None
    above_ma200: Optional[float] = None
    macd_above_signal: Optional[bool] = None
    score: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class MonteCarloSnapshot:
    expected_price_20d: Optional[float] = None
    downside_10pct: Optional[float] = None
    upside_90pct: Optional[float] = None
    probability_up_20d: Optional[float] = None
    risk_score: float = 0.0
    paths: Optional[object] = None


@dataclass
class FundamentalsSnapshot:
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    score: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class PredictionSnapshot:
    price_1d: Optional[float] = None
    price_1w: Optional[float] = None
    price_1y: Optional[float] = None
    probability_up_1d: Optional[float] = None
    probability_up_1w: Optional[float] = None
    probability_up_1y: Optional[float] = None
    expected_return_1d: Optional[float] = None
    expected_return_1w: Optional[float] = None
    expected_return_1y: Optional[float] = None
    method: str = "unavailable"


@dataclass
class StockResult:
    ticker: str
    company: str
    sector: str
    action: str
    confidence: float
    final_score: float
    news_score: float
    macro_score: float
    sector_macro_score: float
    technical: TechnicalSnapshot
    monte_carlo: MonteCarloSnapshot
    fundamentals: FundamentalsSnapshot
    prediction: PredictionSnapshot
    ranking_score: float = 0.0
    rising_setup_label: str = "N/A"
    estimated_upside_probability: Optional[float] = None
    small_cap_catalyst_score: float = 0.0
    small_cap_risk_level: str = "N/A"
    small_cap_summary: str = "N/A"
    recommendation_reason: str = "N/A"
    main_risk_warning: str = "N/A"
    suggested_entry_style: str = "N/A"
    risk_management: Dict[str, str] = field(default_factory=dict)
    backtest: Dict[str, object] = field(default_factory=dict)
    positive_hits: Dict[str, int] = field(default_factory=dict)
    negative_hits: Dict[str, int] = field(default_factory=dict)
    headlines: List[NewsItem] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    history: Optional[object] = None


FINBERT_READY = False
TOKENIZER = None
FINBERT = None


def safe_import_warning() -> None:
    missing = []
    for name, module in [
        ("yfinance", yf),
        ("requests", requests),
        ("feedparser", feedparser),
        ("pandas", pd),
        ("numpy", np),
        ("matplotlib", plt),
    ]:
        if module is None:
            missing.append(name)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install example: pip install -r requirements.txt")


def setup_finbert() -> None:
    global FINBERT_READY, TOKENIZER, FINBERT
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        print("FinBERT unavailable. Keyword sentiment backup will be used.")
        return
    try:
        allow_download = os.getenv("FINBERT_DOWNLOAD", "0").strip() == "1"
        print("Loading FinBERT sentiment model...")
        TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=not allow_download)
        FINBERT = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            local_files_only=not allow_download,
        )
        FINBERT.eval()
        FINBERT_READY = True
        print("FinBERT loaded.")
    except Exception as exc:
        print("FinBERT failed. Keyword sentiment backup will be used.")
        print("Tip: set FINBERT_DOWNLOAD=1 to allow model download.")
        print("Reason:", str(exc)[:160])


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_series(data):
    if pd is not None and isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if pd is not None and pd.isna(value):
            return default
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def request_text(url: str, timeout: int = 10) -> str:
    if requests is None:
        return ""
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if response.status_code >= 400:
            return ""
        return response.text
    except Exception:
        return ""


def parse_source(title: str) -> Tuple[str, str]:
    parts = title.rsplit(" - ", 1)
    if len(parts) == 2 and 2 <= len(parts[1]) <= 55:
        return clean_text(parts[0]), clean_text(parts[1])
    return clean_text(title), "Unknown"


def parse_news_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def news_sort_key(item: NewsItem) -> Tuple[float, float]:
    published = parse_news_datetime(item.published)
    timestamp = published.timestamp() if published else 0.0
    return timestamp, credibility_weight(item.source)


def recent_first(items: Iterable[NewsItem], limit: int, max_age_days: int = 3) -> List[NewsItem]:
    unique = dedupe_news(items)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    recent = []
    undated = []
    for item in unique:
        published = parse_news_datetime(item.published)
        if published is None:
            undated.append(item)
        elif published >= cutoff:
            recent.append(item)
    pool = recent if len(recent) >= max(4, min(limit, 10)) else recent + undated
    return sorted(pool, key=news_sort_key, reverse=True)[:limit]


def parse_google_news_feed(url: str, max_items: int) -> List[NewsItem]:
    raw = request_text(url)
    if not raw:
        return []
    try:
        feed = feedparser.parse(raw)
    except Exception:
        return []

    items = []
    for entry in feed.entries[: max_items * 2]:
        title, source = parse_source(getattr(entry, "title", ""))
        if len(title) < 8:
            continue
        items.append(
            NewsItem(
                title=title,
                source=source,
                url=getattr(entry, "link", ""),
                published=getattr(entry, "published", ""),
            )
        )
    return items


def google_news_rss(query: str, max_items: int = 12, recency_days: int = 3) -> List[NewsItem]:
    if feedparser is None:
        return []
    query_variants = [
        f"{query} when:{max(1, recency_days)}d",
        f"{query} when:7d",
        query,
    ]
    collected: List[NewsItem] = []
    for variant in query_variants:
        url = (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(variant)
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        collected.extend(parse_google_news_feed(url, max_items=max_items))
        fresh = recent_first(collected, limit=max_items, max_age_days=max(7, recency_days))
        if fresh:
            return fresh
    return recent_first(collected, limit=max_items, max_age_days=30)


def yahoo_ticker_news(ticker: str, max_items: int = 12) -> List[NewsItem]:
    if yf is None:
        return []
    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items = []
    for row in raw_news[:max_items]:
        title = row.get("title")
        if not isinstance(title, str) or len(title) < 8:
            continue
        source = row.get("publisher") or "Yahoo Finance"
        published = ""
        timestamp = row.get("providerPublishTime") or row.get("pubDate")
        if isinstance(timestamp, (int, float)):
            published = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        elif isinstance(timestamp, str):
            published = timestamp
        items.append(NewsItem(clean_text(title), str(source), row.get("link") or "", published))
    return recent_first(items, limit=max_items, max_age_days=7)


def dedupe_news(items: Iterable[NewsItem]) -> List[NewsItem]:
    unique = []
    seen = set()
    for item in items:
        title = clean_text(item.title)
        key = re.sub(r"[^a-z0-9]+", "", title.lower())[:95]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(NewsItem(title, item.source, item.url, item.published))
    return sorted(unique, key=news_sort_key, reverse=True)


def credibility_weight(source: str) -> float:
    source_lower = source.lower()
    for name, weight in SOURCE_CREDIBILITY.items():
        if name.lower() in source_lower:
            return weight
    return 0.90


def count_event_hits(text: str, dictionary: Dict[str, Sequence[str]]) -> Dict[str, int]:
    hits = {}
    padded = " " + text.lower() + " "
    for event, keywords in dictionary.items():
        count = 0
        for keyword in keywords:
            if keyword.strip().lower() in padded:
                count += 1
        if count:
            hits[event] = count
    return hits


def finbert_score(headlines: Sequence[str]) -> float:
    if not FINBERT_READY or TOKENIZER is None or FINBERT is None or not headlines:
        return 0.0
    try:
        batch = list(headlines[:24])
        inputs = TOKENIZER(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(FINBERT(**inputs).logits, dim=1)
        labels = {label.lower(): idx for idx, label in FINBERT.config.id2label.items()}
        pos_idx = labels.get("positive")
        neg_idx = labels.get("negative")
        if pos_idx is None or neg_idx is None:
            return 0.0
        return float((probs[:, pos_idx] - probs[:, neg_idx]).detach().cpu().numpy().mean())
    except Exception:
        return 0.0


def keyword_sentiment_score(text: str) -> float:
    pos = sum(count_event_hits(text, POSITIVE_EVENTS).values())
    neg = sum(count_event_hits(text, NEGATIVE_EVENTS).values())
    if pos + neg == 0:
        return 0.0
    return max(min((pos - 1.45 * neg) / max(pos + neg, 1), 1.0), -1.0)


def collect_macro_news() -> List[NewsItem]:
    items: List[NewsItem] = []
    for query in live_macro_queries():
        items.extend(google_news_rss(query, max_items=12, recency_days=2))
        time.sleep(0.12)
    return dedupe_news(items)[:60]


def collect_ticker_news(ticker: str, company: str) -> List[NewsItem]:
    queries = [
        f"{ticker} stock latest news",
        f"{company} stock latest Reuters CNBC Bloomberg",
        f"{company} earnings guidance analyst upgrade downgrade",
        f"{company} contract partnership product launch lawsuit investigation",
        f"{company} macro risk AI semiconductor defense energy FDA clinical trial",
    ]
    items: List[NewsItem] = []
    for query in queries:
        items.extend(google_news_rss(query, max_items=9, recency_days=2))
        time.sleep(0.12)
    items.extend(yahoo_ticker_news(ticker, max_items=12))
    return recent_first(items, limit=50, max_age_days=7)


def macro_regime_score(macro_items: Sequence[NewsItem]) -> Tuple[float, float, Dict[str, int], Dict[str, int]]:
    text = clean_text(" ".join(item.title for item in macro_items)).lower()
    positive = {
        "ai_semiconductor_tailwind": [
            "ai", "artificial intelligence", "chip", "semiconductor", "data center",
            "earnings resilience", "tech rebound", "chips rebound",
        ],
        "risk_easing": ["tensions ease", "ceasefire", "diplomatic progress", "peace talks"],
        "growth_resilience": [
            "resilient economy", "jobs increase", "services expand",
            "corporate earnings resilience", "soft landing",
        ],
        "healthcare_rotation": ["healthcare leads", "healthcare outperforms", "biotech rally"],
    }
    risk = {
        "war_geopolitical": [
            "war", "middle east", "iran", "israel", "hormuz", "missile",
            "attack", "geopolitical tensions", "military strikes",
        ],
        "oil_inflation": [
            "oil rises", "crude rises", "brent rises", "energy prices",
            "supply disruption", "inflation", "gasoline",
        ],
        "rates_cpi_fed": [
            "cpi", "rate hike", "fed hike", "higher rates", "yields climb",
            "treasury yields", "inflation pressures",
        ],
        "risk_off": ["selloff", "recession", "consumer slowdown", "profit taking", "stocks fall"],
    }
    pos_hits = count_event_hits(text, positive)
    risk_hits = count_event_hits(text, risk)
    macro_score = sum(pos_hits.values()) * 6.0 - sum(risk_hits.values()) * 4.5
    macro_risk = sum(risk_hits.values()) * 5.0
    return float(macro_score), float(macro_risk), pos_hits, risk_hits


def sector_macro_adjustment(sector: str, macro_pos: Dict[str, int], macro_risk: Dict[str, int]) -> float:
    score = 0.0
    if sector in {"ai_semis", "data_center", "mega_tech", "software", "cyber", "defense_ai"}:
        score += macro_pos.get("ai_semiconductor_tailwind", 0) * 4.0
        score -= macro_risk.get("rates_cpi_fed", 0) * 1.6
        score -= macro_risk.get("risk_off", 0) * 1.2
    if sector in {"defense", "space_defense", "defense_ai"}:
        score += macro_risk.get("war_geopolitical", 0) * 3.0
    if sector in {"energy", "uranium", "nuclear_power", "grid_power"}:
        score += macro_risk.get("oil_inflation", 0) * 2.5
        score += macro_risk.get("war_geopolitical", 0) * 1.0
    if sector in {"healthcare", "biotech", "managed_care"}:
        score += macro_pos.get("healthcare_rotation", 0) * 4.0
        score += 2.0
    if sector in {"consumer_quality", "payments", "financials"}:
        score -= macro_risk.get("rates_cpi_fed", 0) * 2.5
        score -= macro_risk.get("oil_inflation", 0) * 1.5
    if sector == "space":
        score -= macro_risk.get("rates_cpi_fed", 0) * 2.0
        score -= macro_risk.get("risk_off", 0) * 2.0
        score += macro_risk.get("war_geopolitical", 0) * 0.5
    return float(score)


def news_recency_weight(item: NewsItem) -> float:
    published = parse_news_datetime(item.published)
    if published is None:
        return 0.70
    age_hours = (datetime.now(timezone.utc) - published).total_seconds() / 3600.0
    if age_hours <= 6:
        return 1.40
    if age_hours <= 24:
        return 1.25
    if age_hours <= 72:
        return 1.00
    if age_hours <= 168:
        return 0.70
    return 0.40


def score_news(items: Sequence[NewsItem]) -> Tuple[float, Dict[str, int], Dict[str, int], List[str]]:
    if not items:
        return -25.0, {}, {}, ["No fresh news found; confidence reduced."]

    weighted_text_parts = []
    credibility_bonus = 0.0
    recency_bonus = 0.0
    source_names = set()
    dated_items = 0
    fresh_items = 0
    for item in items:
        source_weight = credibility_weight(item.source)
        recency_weight = news_recency_weight(item)
        total_weight = source_weight * recency_weight
        weighted_text_parts.append((item.title + " ") * max(1, int(round(total_weight * 2.4))))
        credibility_bonus += min(source_weight - 0.85, 0.35) * recency_weight
        recency_bonus += max(recency_weight - 0.80, -0.35)
        source_names.add(item.source.lower().strip() or "unknown")
        published = parse_news_datetime(item.published)
        if published is not None:
            dated_items += 1
            if (datetime.now(timezone.utc) - published) <= timedelta(hours=48):
                fresh_items += 1

    text = clean_text(" ".join(weighted_text_parts)).lower()
    positive_hits = count_event_hits(text, POSITIVE_EVENTS)
    negative_hits = count_event_hits(text, NEGATIVE_EVENTS)
    keyword_score = keyword_sentiment_score(text)
    model_score = finbert_score([item.title for item in items])

    positive_points = sum(positive_hits.values()) * 9.0
    negative_points = sum(negative_hits.values()) * 14.0
    sentiment_points = (0.60 * model_score + 0.40 * keyword_score) * 38.0
    breadth_bonus = min(len(items), 30) * 0.7
    diversity_bonus = min(len(source_names), 8) * 1.15
    credibility_bonus = min(credibility_bonus, 11.0)
    recency_bonus = clamp(recency_bonus, -8.0, 10.0)
    score = (
        positive_points
        - negative_points
        + sentiment_points
        + breadth_bonus
        + credibility_bonus
        + diversity_bonus
        + recency_bonus
    )

    notes = []
    if negative_points > positive_points:
        notes.append("Negative headlines dominate positive catalysts.")
    if len(items) < 6:
        notes.append("News sample is thin.")
    if dated_items and fresh_items < max(2, dated_items // 4):
        score -= 7.0
        notes.append("Few headlines are fresh within the last 48 hours.")
    if len(source_names) < 3:
        score -= 4.0
        notes.append("News source diversity is limited.")
    return float(score), positive_hits, negative_hits, notes


def get_company_name_and_info(ticker: str) -> Tuple[str, Dict]:
    if yf is None:
        return ticker, {}
    try:
        info = yf.Ticker(ticker).info or {}
        name = info.get("longName") or info.get("shortName") or ticker
        return str(name), info
    except Exception:
        return ticker, {}


def download_history(ticker: str, period: str = "2y"):
    if yf is None or pd is None:
        return None
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        data = data.copy()
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        return data.dropna()
    except Exception:
        return None


def technical_analysis(history) -> TechnicalSnapshot:
    snap = TechnicalSnapshot()
    if history is None or pd is None or np is None or len(history) < 80:
        snap.notes.append("Not enough price history for technical analysis.")
        return snap

    close = to_series(history["Close"]).astype(float)
    volume = to_series(history["Volume"]).astype(float) if "Volume" in history else close * 0
    price = safe_float(close.iloc[-1])
    snap.price = price

    returns = close.pct_change().dropna()
    volatility = safe_float(returns.tail(60).std() * math.sqrt(252), 0.0)
    snap.volatility = volatility

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = safe_float(rsi_series.iloc[-1], 50.0)
    snap.rsi = rsi

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    x60 = np.arange(min(60, len(close)))
    y60 = close.tail(len(x60)).values
    slope60 = float(np.polyfit(x60, y60, 1)[0] / max(price or 1.0, 1.0))
    x120 = np.arange(min(120, len(close)))
    y120 = close.tail(len(x120)).values
    slope120 = float(np.polyfit(x120, y120, 1)[0] / max(price or 1.0, 1.0))

    support = safe_float(close.tail(60).min())
    resistance = safe_float(close.tail(60).max())
    snap.support = support
    snap.resistance = resistance

    above_ma20 = price / ma20.iloc[-1] - 1 if price and ma20.iloc[-1] else 0.0
    above_ma50 = price / ma50.iloc[-1] - 1 if price and ma50.iloc[-1] else 0.0
    above_ma200 = price / ma200.iloc[-1] - 1 if price and len(ma200.dropna()) else 0.0
    volume_ratio = safe_float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1], 1.0)
    ret_5d = safe_float(close.iloc[-1] / close.iloc[-6] - 1, 0.0) if len(close) > 6 else 0.0
    ret_1m = safe_float(close.iloc[-1] / close.iloc[-22] - 1, 0.0) if len(close) > 22 else 0.0
    snap.volume_ratio = volume_ratio
    snap.return_5d = ret_5d
    snap.return_1w = ret_5d
    snap.return_1m = ret_1m
    snap.above_ma20 = safe_float(above_ma20, 0.0)
    snap.above_ma50 = safe_float(above_ma50, 0.0)
    snap.above_ma200 = safe_float(above_ma200, 0.0)
    snap.macd_above_signal = safe_float(macd.iloc[-1], 0.0) > safe_float(signal.iloc[-1], 0.0)

    score = 0.0
    if slope60 > 0.0007 and slope120 > 0.0003:
        score += 24
        snap.trend_label = "uptrend"
    elif slope60 < -0.0007 and slope120 < -0.0003:
        score -= 28
        snap.trend_label = "downtrend"
    else:
        score += 4
        snap.trend_label = "sideways / base"

    if -0.055 <= above_ma20 <= 0.035 and above_ma50 > -0.08 and snap.trend_label != "downtrend":
        score += 22
        snap.pullback_label = "healthy pullback near moving average"
    elif above_ma20 > 0.12 or ret_5d > 0.16 or rsi > 76:
        score -= 22
        snap.pullback_label = "extended / chased"
    elif above_ma50 < -0.12 and snap.trend_label == "downtrend":
        score -= 18
        snap.pullback_label = "falling trend, not a clean pullback"
    else:
        score += 5
        snap.pullback_label = "neutral setup"

    if 43 <= rsi <= 68:
        score += 13
    elif rsi > 75:
        score -= 15
    elif rsi < 34 and snap.trend_label != "downtrend":
        score += 5

    if snap.macd_above_signal:
        score += 8
    else:
        score -= 4

    if 1.05 <= (volume_ratio or 1.0) <= 3.5:
        score += 8
    elif (volume_ratio or 1.0) > 5.0:
        score -= 10

    if ret_1m and ret_1m > 0.35:
        score -= 16
        snap.notes.append("One-month move is already very extended.")
    if above_ma200 > 0:
        score += 8
    elif above_ma200 < -0.18:
        score -= 10

    snap.score = max(min(score, 85.0), -65.0)
    snap.notes.append(f"RSI {rsi:.1f}, 5d return {ret_5d * 100:.1f}%, 1m return {ret_1m * 100:.1f}%.")
    return snap


def monte_carlo_analysis(history, days: int = 20, sims: int = 600) -> MonteCarloSnapshot:
    snap = MonteCarloSnapshot()
    if history is None or pd is None or np is None or len(history) < 80:
        return snap

    close = to_series(history["Close"]).astype(float)
    returns = close.pct_change().dropna().tail(252)
    if len(returns) < 40:
        return snap

    current_price = float(close.iloc[-1])
    mu = float(returns.mean())
    sigma = float(returns.std())
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        return snap

    rng = np.random.default_rng(42)
    paths = np.zeros((sims, days + 1))
    paths[:, 0] = current_price
    for day in range(1, days + 1):
        shocks = rng.normal(mu, sigma, sims)
        paths[:, day] = paths[:, day - 1] * (1 + shocks)

    final_prices = paths[:, -1]
    expected = float(np.mean(final_prices))
    downside = float(np.percentile(final_prices, 10))
    upside = float(np.percentile(final_prices, 90))
    probability_up = float(np.mean(final_prices > current_price))
    downside_pct = downside / current_price - 1
    upside_pct = upside / current_price - 1

    risk_score = 0.0
    risk_score += (probability_up - 0.50) * 45.0
    risk_score += max(min(upside_pct * 100, 25), -25) * 0.8
    risk_score += max(min(downside_pct * 100, 0), -30) * 1.1

    snap.expected_price_20d = expected
    snap.downside_10pct = downside
    snap.upside_90pct = upside
    snap.probability_up_20d = probability_up
    snap.risk_score = max(min(float(risk_score), 45.0), -45.0)
    snap.paths = paths
    return snap


def fundamentals_analysis(info: Dict) -> FundamentalsSnapshot:
    snap = FundamentalsSnapshot()
    if not info:
        snap.notes.append("Fundamentals unavailable.")
        return snap

    revenue_growth = safe_float(info.get("revenueGrowth"))
    earnings_growth = safe_float(info.get("earningsGrowth"))
    profit_margin = safe_float(info.get("profitMargins"))
    debt_to_equity = safe_float(info.get("debtToEquity"))

    snap.revenue_growth = revenue_growth
    snap.earnings_growth = earnings_growth
    snap.profit_margin = profit_margin
    snap.debt_to_equity = debt_to_equity

    score = 0.0
    if revenue_growth is not None:
        score += max(min(revenue_growth * 85, 22), -12)
    if earnings_growth is not None:
        score += max(min(earnings_growth * 60, 20), -15)
    if profit_margin is not None:
        score += max(min(profit_margin * 45, 18), -12)
    if debt_to_equity is not None:
        if debt_to_equity < 80:
            score += 5
        elif debt_to_equity > 220:
            score -= 8

    snap.score = max(min(float(score), 50.0), -35.0)
    if revenue_growth is not None:
        snap.notes.append(f"Revenue growth {revenue_growth * 100:.1f}%.")
    if earnings_growth is not None:
        snap.notes.append(f"Earnings growth {earnings_growth * 100:.1f}%.")
    return snap


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def probability_from_signal(signal: float) -> float:
    return float(1.0 / (1.0 + math.exp(-signal)))


def signal_alignment_boost(signals: Sequence[float]) -> Tuple[float, float]:
    strong_positive = sum(1 for value in signals if value > 0.18)
    strong_negative = sum(1 for value in signals if value < -0.18)
    weak_or_neutral = len(signals) - strong_positive - strong_negative

    if strong_positive >= 4 and strong_negative == 0:
        return 0.18, 1.12
    if strong_positive >= 3 and strong_negative <= 1:
        return 0.10, 1.06
    if strong_negative >= 3 and strong_positive <= 1:
        return -0.14, 1.08
    if strong_positive >= 2 and strong_negative >= 2:
        return 0.0, 0.78
    if weak_or_neutral >= 3:
        return 0.0, 0.90
    return 0.0, 1.0


def prediction_analysis(
    history,
    news_score: float,
    macro_score: float,
    sector_macro_score: float,
    technical: TechnicalSnapshot,
    fundamentals: FundamentalsSnapshot,
    mc: MonteCarloSnapshot,
    negative_hits: Dict[str, int],
) -> PredictionSnapshot:
    pred = PredictionSnapshot()
    if history is None or pd is None or np is None or technical.price is None or len(history) < 80:
        return pred

    close = to_series(history["Close"]).astype(float)
    returns = close.pct_change().dropna()
    if len(returns) < 40:
        return pred

    current_price = float(technical.price)
    daily_mu = float(returns.tail(252).mean())
    daily_vol = float(returns.tail(252).std())
    if not math.isfinite(daily_mu) or not math.isfinite(daily_vol) or daily_vol <= 0:
        return pred

    news_signal = clamp(news_score / 100.0, -1.25, 1.25)
    macro_signal = clamp((macro_score + sector_macro_score) / 80.0, -0.90, 0.90)
    tech_signal = clamp(technical.score / 85.0, -1.0, 1.0)
    fund_signal = clamp(fundamentals.score / 50.0, -1.0, 1.0)
    mc_signal = clamp(mc.risk_score / 45.0, -1.0, 1.0)
    neg_penalty = clamp(sum(negative_hits.values()) * 0.18, 0.0, 1.25)
    alignment_shift, conviction_multiplier = signal_alignment_boost(
        [news_signal, macro_signal, tech_signal, fund_signal, mc_signal]
    )

    # News gets the strongest directional influence. Fundamentals matter more on the 1y horizon.
    signal_1d = 0.58 * news_signal + 0.20 * tech_signal + 0.12 * macro_signal + 0.10 * mc_signal - neg_penalty
    signal_1w = 0.55 * news_signal + 0.23 * tech_signal + 0.12 * macro_signal + 0.07 * mc_signal + 0.03 * fund_signal - neg_penalty
    signal_1y = 0.38 * news_signal + 0.18 * tech_signal + 0.14 * macro_signal + 0.30 * fund_signal - neg_penalty * 0.75
    signal_1d = signal_1d * conviction_multiplier + alignment_shift
    signal_1w = signal_1w * conviction_multiplier + alignment_shift
    signal_1y = signal_1y * conviction_multiplier + alignment_shift * 0.75

    prob_1d = probability_from_signal(signal_1d)
    prob_1w = probability_from_signal(signal_1w)
    prob_1y = probability_from_signal(signal_1y)
    prob_1d = clamp(prob_1d, 0.35, 0.75)
    prob_1w = clamp(prob_1w, 0.35, 0.75)
    prob_1y = clamp(prob_1y, 0.35, 0.75)

    base_1d = daily_mu
    base_1w = daily_mu * 5.0
    base_1y = daily_mu * 252.0

    ret_1d = base_1d + (prob_1d - 0.50) * daily_vol * 1.55
    ret_1w = base_1w + (prob_1w - 0.50) * daily_vol * math.sqrt(5) * 1.75
    ret_1y = base_1y + (prob_1y - 0.50) * daily_vol * math.sqrt(252) * 1.35

    # Keep projections realistic enough for screening output.
    ret_1d = clamp(ret_1d, -0.12, 0.12)
    ret_1w = clamp(ret_1w, -0.28, 0.32)
    ret_1y = clamp(ret_1y, -0.65, 1.40)

    pred.price_1d = current_price * (1.0 + ret_1d)
    pred.price_1w = current_price * (1.0 + ret_1w)
    pred.price_1y = current_price * (1.0 + ret_1y)
    pred.probability_up_1d = prob_1d
    pred.probability_up_1w = prob_1w
    pred.probability_up_1y = prob_1y
    pred.expected_return_1d = ret_1d
    pred.expected_return_1w = ret_1w
    pred.expected_return_1y = ret_1y
    pred.method = (
        "recency-weighted ensemble forecast using news catalysts, source quality, "
        "technical trend, fundamentals, macro regime, Monte Carlo risk, and signal alignment"
    )
    return pred


def bullish_bearish_signal_counts(
    news_score: float,
    positive_hits: Dict[str, int],
    negative_hits: Dict[str, int],
    technical: TechnicalSnapshot,
    fundamentals: FundamentalsSnapshot,
    mc: MonteCarloSnapshot,
    sector_macro_score: float,
    macro_risk_score: float,
) -> Tuple[int, int, List[str], List[str]]:
    """Count independent bullish/bearish signals so one headline cannot dominate ranking."""
    bullish: List[str] = []
    bearish: List[str] = []
    neg_count = sum(negative_hits.values())

    if news_score > 28 or sum(positive_hits.values()) >= 2:
        bullish.append("positive catalyst news")
    if technical.trend_label == "uptrend":
        bullish.append("uptrend")
    if (technical.above_ma50 or 0.0) > -0.01 or (technical.above_ma200 or 0.0) > 0:
        bullish.append("price above key moving average")
    if technical.rsi is not None and 45 <= technical.rsi <= 70:
        bullish.append("RSI in constructive range")
    if (technical.volume_ratio or 0.0) >= 1.15:
        bullish.append("above-average volume")
    if (mc.probability_up_20d or 0.0) >= 0.55:
        bullish.append("Monte Carlo chance up above 55%")
    if fundamentals.score > 5:
        bullish.append("positive fundamentals")
    if sector_macro_score > 0:
        bullish.append("sector macro tailwind")
    if neg_count <= 1:
        bullish.append("low negative headline count")

    if technical.trend_label == "downtrend":
        bearish.append("downtrend")
    if neg_count >= 2:
        bearish.append("negative catalyst news")
    if any(key in negative_hits for key in ("biotech_failure", "legal_regulatory", "analyst_downgrade")):
        bearish.append("regulatory/legal/downgrade risk")
    if technical.rsi is not None and technical.rsi >= 76:
        bearish.append("extremely overbought RSI")
    if (technical.return_1w or 0.0) >= 0.18 or (technical.return_1m or 0.0) >= 0.40:
        bearish.append("large recent spike already happened")
    if (mc.probability_up_20d or 0.50) < 0.45:
        bearish.append("Monte Carlo chance up below 45%")
    if fundamentals.score < -5:
        bearish.append("weak fundamentals")
    if macro_risk_score >= 25:
        bearish.append("high macro risk")

    return len(bullish), len(bearish), bullish, bearish


def volume_activity_score(technical: TechnicalSnapshot) -> float:
    """Score unusual but not reckless volume expansion."""
    ratio = technical.volume_ratio
    if ratio is None:
        return 0.0
    if 1.2 <= ratio <= 3.5:
        return min(10.0, 4.0 + (ratio - 1.2) * 2.6)
    if 3.5 < ratio <= 6.0:
        return 4.0
    if ratio > 6.0:
        return -4.0
    return max(0.0, (ratio - 0.8) * 5.0)


def trend_quality_score(technical: TechnicalSnapshot) -> float:
    """Separate trend quality from the broader technical score."""
    score = 0.0
    if technical.trend_label == "uptrend":
        score += 9.0
    elif technical.trend_label == "sideways / base":
        score += 4.0
    elif technical.trend_label == "downtrend":
        score -= 10.0
    if (technical.above_ma50 or 0.0) > 0:
        score += 3.0
    if (technical.above_ma200 or 0.0) > 0:
        score += 3.0
    if technical.pullback_label == "healthy pullback near moving average":
        score += 2.0
    if technical.pullback_label == "extended / chased":
        score -= 5.0
    return clamp(score, -15.0, 15.0)


def risk_keyword_count(positive_hits: Dict[str, int], negative_hits: Dict[str, int], news: Sequence[NewsItem]) -> int:
    """Detect severe small-cap/biotech style risk phrases beyond the regular dictionaries."""
    text = clean_text(" ".join(item.title for item in news)).lower()
    extra_risks = [
        "dilution", "offering", "share offering", "bankruptcy", "going concern",
        "reverse split", "short report", "clinical hold", "failed trial", "fda rejection",
    ]
    return sum(negative_hits.values()) + sum(1 for phrase in extra_risks if phrase in text)


def compute_probability_ranking_score(
    news_score: float,
    positive_hits: Dict[str, int],
    negative_hits: Dict[str, int],
    technical: TechnicalSnapshot,
    fundamentals: FundamentalsSnapshot,
    mc: MonteCarloSnapshot,
    macro_score: float,
    macro_risk_score: float,
    sector_macro_score: float,
) -> Tuple[float, float, List[str], List[str]]:
    """Blend independent signals into a probability-based ranking score."""
    bullish_count, bearish_count, bullish_reasons, bearish_reasons = bullish_bearish_signal_counts(
        news_score,
        positive_hits,
        negative_hits,
        technical,
        fundamentals,
        mc,
        sector_macro_score,
        macro_risk_score,
    )
    news_component = clamp((news_score + 30.0) / 140.0, 0.0, 1.0) * 35.0
    technical_component = clamp((technical.score + 35.0) / 120.0, 0.0, 1.0) * 18.0
    trend_component = max(0.0, trend_quality_score(technical)) / 15.0 * 15.0
    volume_component = max(0.0, volume_activity_score(technical))
    fundamentals_component = clamp((fundamentals.score + 10.0) / 60.0, 0.0, 1.0) * 10.0
    mc_prob = mc.probability_up_20d if mc.probability_up_20d is not None else 0.50
    mc_component = clamp((mc_prob - 0.35) / 0.40, 0.0, 1.0) * 10.0
    macro_component = clamp((macro_score * 0.35 + sector_macro_score + 15.0) / 45.0, 0.0, 1.0) * 10.0

    risk_penalty = 0.0
    risk_penalty += sum(negative_hits.values()) * 6.5
    risk_penalty += max(0.0, macro_risk_score - 15.0) * 0.45
    if technical.pullback_label == "extended / chased":
        risk_penalty += 12.0
    if technical.trend_label == "downtrend":
        risk_penalty += 14.0
    if (technical.return_1w or 0.0) > 0.20:
        risk_penalty += 8.0
    if (technical.volatility or 0.0) > 0.75:
        risk_penalty += 6.0

    alignment_boost = 0.0
    if bullish_count >= 6 and bearish_count <= 1:
        alignment_boost = 24.0
    elif bullish_count >= 5 and bearish_count <= 2:
        alignment_boost = 17.0
    elif bullish_count >= 4 and bearish_count <= 2:
        alignment_boost = 9.0
    if bearish_count >= 3:
        risk_penalty += 22.0 + (bearish_count - 3) * 5.0

    final_score = (
        news_component
        + technical_component
        + trend_component
        + volume_component
        + fundamentals_component
        + mc_component
        + macro_component
        + alignment_boost
        - risk_penalty
    )
    return clamp(final_score, 0.0, 100.0), risk_penalty, bullish_reasons, bearish_reasons


def detect_rising_stock_setup(
    technical: TechnicalSnapshot,
    news_score: float,
    negative_hits: Dict[str, int],
    mc: MonteCarloSnapshot,
    sector_macro_score: float,
) -> str:
    """Classify the current setup without promising an outcome."""
    try:
        neg_count = sum(negative_hits.values())
        rsi = technical.rsi or 50.0
        volume_ratio = technical.volume_ratio or 1.0
        mc_up = mc.probability_up_20d or 0.50
        near_ma = -0.06 <= (technical.above_ma20 or 0.0) <= 0.04 or -0.07 <= (technical.above_ma50 or 0.0) <= 0.04

        if technical.trend_label == "downtrend" or neg_count >= 4 or mc_up < 0.42:
            return "Bearish / Avoid"
        if technical.pullback_label == "extended / chased" or rsi > 74 or (technical.return_1w or 0.0) > 0.18:
            return "Overextended / Wait"
        if near_ma and volume_ratio >= 1.05 and 45 <= rsi <= 68 and news_score > 15 and neg_count <= 2:
            return "Healthy Pullback Buy Setup"
        if technical.trend_label == "uptrend" and volume_ratio >= 1.45 and rsi <= 72 and mc_up >= 0.55:
            return "Momentum Breakout Setup"
        if 45 <= rsi <= 70 and (technical.macd_above_signal or volume_ratio >= 1.2) and news_score > 5 and mc_up >= 0.52 and sector_macro_score >= 0:
            return "Early Bullish Setup"
    except Exception:
        return "N/A"
    return "Neutral / Wait"


def small_cap_catalyst_score(
    info: Dict,
    sector: str,
    technical: TechnicalSnapshot,
    positive_hits: Dict[str, int],
    negative_hits: Dict[str, int],
    news: Sequence[NewsItem],
) -> Tuple[float, str, str]:
    """Flag high-risk/high-reward catalyst setups, especially small-cap and biotech names."""
    try:
        market_cap = safe_float(info.get("marketCap"))
        text = clean_text(" ".join(item.title for item in news)).lower()
        catalyst_terms = [
            "fda approval", "phase 2", "phase ii", "phase 3", "phase iii",
            "positive trial", "met primary endpoint", "clinical trial", "partnership",
            "patent", "government contract", "contract award", "selected by",
        ]
        severe_risk_terms = [
            "dilution", "offering", "fda rejection", "failed trial", "clinical hold",
            "bankruptcy", "short report", "reverse split",
        ]
        score = 0.0
        if market_cap is not None and market_cap < 2_000_000_000:
            score += 15.0
        elif market_cap is not None and market_cap < 10_000_000_000:
            score += 8.0
        if sector in {"biotech", "space", "space_defense", "defense_ai"}:
            score += 5.0
        if (technical.volume_ratio or 0.0) >= 2.0:
            score += 14.0
        score += sum(1 for term in catalyst_terms if term in text) * 7.0
        score += positive_hits.get("biotech_healthcare", 0) * 6.0
        score += positive_hits.get("contract_partnership", 0) * 4.0
        if technical.pullback_label != "extended / chased":
            score += 5.0

        severe_risk_count = sum(1 for term in severe_risk_terms if term in text)
        risk_points = severe_risk_count * 18.0 + sum(negative_hits.values()) * 5.5
        score -= min(risk_points * 0.45, 25.0)
        score = clamp(score, 0.0, 100.0)

        if severe_risk_count >= 2 or sum(negative_hits.values()) >= 5:
            risk_level = "Extreme"
        elif severe_risk_count >= 1 or (technical.volatility or 0.0) > 0.95:
            risk_level = "High"
        elif (technical.volatility or 0.0) > 0.60 or score >= 45:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        summary = "Small-cap catalyst profile requires smaller position sizing and tighter risk control."
        if score >= 55:
            summary = "High-risk/high-reward catalyst setup; position size should be smaller than normal."
        if risk_level in {"High", "Extreme"}:
            summary += " Severe catalyst or financing risk is present."
        return score, risk_level, summary
    except Exception as exc:
        return 0.0, "N/A", f"Small-cap catalyst analysis unavailable: {str(exc)[:80]}"


def suggested_entry_style(setup_label: str, technical: TechnicalSnapshot) -> str:
    if setup_label == "Healthy Pullback Buy Setup":
        return "Healthy pullback setup"
    if setup_label == "Momentum Breakout Setup":
        return "Breakout candidate"
    if setup_label == "Early Bullish Setup":
        return "Wait for pullback"
    if setup_label == "Overextended / Wait":
        return "Avoid chasing"
    if setup_label == "Bearish / Avoid":
        return "Avoid"
    if technical.pullback_label == "extended / chased":
        return "Avoid chasing"
    return "Wait for better entry"


def compute_ranking_score(
    final_score: float,
    confidence: float,
    setup_label: str,
    mc: MonteCarloSnapshot,
    technical: TechnicalSnapshot,
    negative_hits: Dict[str, int],
) -> float:
    setup_bonus = {
        "Healthy Pullback Buy Setup": 14.0,
        "Momentum Breakout Setup": 12.0,
        "Early Bullish Setup": 8.0,
        "Overextended / Wait": -12.0,
        "Bearish / Avoid": -25.0,
    }.get(setup_label, 0.0)
    mc_bonus = ((mc.probability_up_20d or 0.50) - 0.50) * 40.0
    volume_bonus = 5.0 if 1.2 <= (technical.volume_ratio or 0.0) <= 3.5 else 0.0
    risk_penalty = sum(negative_hits.values()) * 5.0
    if technical.pullback_label == "extended / chased":
        risk_penalty += 10.0
    return float(final_score + confidence * 0.30 + setup_bonus + mc_bonus + volume_bonus - risk_penalty)


def main_risk_warning_from_components(result: StockResult) -> str:
    if sum(result.negative_hits.values()) >= 3:
        return "Negative catalyst count is elevated; wait for risk to clear."
    if result.technical.pullback_label == "extended / chased":
        return "Price looks extended; avoid chasing without a new base."
    if (result.technical.volatility or 0.0) > 0.85:
        return "Volatility is high; position size should be reduced."
    if result.small_cap_risk_level in {"High", "Extreme"}:
        return "Small-cap or clinical catalyst risk is high; use very small sizing."
    if result.technical.trend_label == "downtrend":
        return "Trend is down; positive news needs confirmation from price action."
    return "No single dominant risk, but news and market conditions can change quickly."


def generate_recommendation_reason(result: StockResult) -> str:
    """Create a concise 'why this stock' explanation for ranking transparency."""
    reasons = []
    if result.news_score > 25:
        reasons.append("positive catalyst news")
    if result.technical.trend_label == "uptrend":
        reasons.append("strong uptrend")
    if result.technical.rsi is not None and 45 <= result.technical.rsi <= 70:
        reasons.append("constructive RSI")
    if result.sector_macro_score > 0:
        reasons.append("positive sector macro tailwind")
    if (result.monte_carlo.probability_up_20d or 0.0) >= 0.55:
        reasons.append("Monte Carlo upside probability above 55%")
    if (result.technical.volume_ratio or 0.0) >= 1.2:
        reasons.append("above-average volume")
    if not reasons:
        reasons.append("the setup is mixed and needs confirmation")
    risk = "medium"
    if result.small_cap_risk_level in {"High", "Extreme"} or (result.technical.volatility or 0.0) > 0.85:
        risk = "high"
    elif sum(result.negative_hits.values()) <= 1 and (result.technical.volatility or 0.0) < 0.45:
        risk = "lower"
    return (
        f"{result.ticker} ranks here because " + ", ".join(reasons[:5])
        + f". Risk is {risk}; this is a probability-based candidate, not financial advice."
    )


def risk_management_suggestion(result: StockResult) -> Dict[str, str]:
    """Produce simple educational risk-management levels from support/resistance and volatility."""
    price = result.technical.price
    support = result.technical.support
    resistance = result.technical.resistance
    volatility = result.technical.volatility or 0.0
    if price is None:
        return {"summary": "N/A"}
    stop = support * 0.98 if support else price * (0.92 if volatility < 0.55 else 0.88)
    take_profit_low = resistance if resistance and resistance > price else price * 1.08
    take_profit_high = price * (1.18 if volatility < 0.65 else 1.25)
    if result.small_cap_risk_level in {"High", "Extreme"} or volatility > 0.90:
        size = "Very Small"
    elif volatility > 0.55 or sum(result.negative_hits.values()) >= 2:
        size = "Small"
    else:
        size = "Normal"
    warnings = []
    if volatility > 0.65:
        warnings.append("Volatility elevated")
    if sum(result.negative_hits.values()) >= 2:
        warnings.append("Headline risk elevated")
    if any(key in result.negative_hits for key in ("biotech_failure", "legal_regulatory")):
        warnings.append("FDA/legal event risk")
    return {
        "stopLossArea": f"${stop:,.2f}",
        "takeProfitZone": f"${take_profit_low:,.2f} - ${take_profit_high:,.2f}",
        "positionSize": size,
        "warnings": "; ".join(warnings) if warnings else "Use normal risk controls; no guarantee of outcome.",
    }


def backtest_strategy(ticker: str, history) -> Dict[str, object]:
    """Educational 20-day forward test using price/volume rules similar to the live setup logic."""
    if history is None or pd is None or np is None or len(history) < 260:
        return {"available": False, "reason": "Not enough history for backtest."}
    try:
        data = history.copy()
        close = to_series(data["Close"]).astype(float)
        volume = to_series(data["Volume"]).astype(float) if "Volume" in data else close * 0
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        volume_ratio = volume / volume.rolling(20).mean()
        momentum_20d = close / close.shift(20) - 1

        entries = (
            (close > ma50)
            & (close > ma200)
            & (rsi.between(45, 70))
            & (volume_ratio > 1.10)
            & (momentum_20d > -0.03)
            & (momentum_20d < 0.22)
        )
        forward_return = close.shift(-20) / close - 1
        trades = forward_return[entries].dropna()
        if trades.empty:
            return {"available": False, "reason": "No historical rule-based entries found."}
        equity = (1 + trades).cumprod()
        drawdown = equity / equity.cummax() - 1
        buy_hold = close.iloc[-1] / close.iloc[0] - 1
        return {
            "available": True,
            "ticker": ticker,
            "trades": int(len(trades)),
            "winRate": float((trades > 0).mean()),
            "averageReturn20d": float(trades.mean()),
            "medianReturn20d": float(trades.median()),
            "maxDrawdown": float(drawdown.min()),
            "buyAndHoldReturn": float(buy_hold),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:120]}


def action_from_score(
    final_score: float,
    confidence: float,
    negative_hits: Dict[str, int],
    technical: TechnicalSnapshot,
    news_score: float,
    mc: MonteCarloSnapshot,
) -> str:
    neg_count = sum(negative_hits.values())
    bullish_core = 0
    if news_score > 25:
        bullish_core += 1
    if technical.trend_label == "uptrend":
        bullish_core += 1
    if (mc.probability_up_20d or 0.0) >= 0.55:
        bullish_core += 1
    if final_score >= 85 and confidence >= 75 and neg_count <= 1 and technical.trend_label != "downtrend" and bullish_core >= 2:
        return "HIGH CONVICTION BUY CANDIDATE"
    if final_score >= 70 and technical.trend_label != "downtrend" and neg_count <= 2:
        return "STRONG BUY CANDIDATE"
    if final_score >= 55:
        return "WATCHLIST / POSSIBLE BUY"
    if final_score >= 40:
        return "NEUTRAL / WAIT FOR BETTER ENTRY"
    return "AVOID / HIGH RISK"


def component_quality_adjustment(
    news_score: float,
    technical: TechnicalSnapshot,
    fundamentals: FundamentalsSnapshot,
    mc: MonteCarloSnapshot,
    macro_score: float,
    sector_macro_score: float,
) -> Tuple[float, List[str]]:
    signals = [
        clamp(news_score / 100.0, -1.25, 1.25),
        clamp(technical.score / 85.0, -1.0, 1.0),
        clamp(fundamentals.score / 50.0, -1.0, 1.0),
        clamp(mc.risk_score / 45.0, -1.0, 1.0),
        clamp((macro_score + sector_macro_score) / 80.0, -0.90, 0.90),
    ]
    positive = sum(1 for value in signals if value > 0.18)
    negative = sum(1 for value in signals if value < -0.18)
    notes: List[str] = []
    if positive >= 4 and negative == 0:
        notes.append("Independent signals broadly agree to the upside.")
        return 9.0, notes
    if positive >= 3 and negative <= 1:
        notes.append("Most independent signals agree to the upside.")
        return 5.0, notes
    if positive >= 2 and negative >= 2:
        notes.append("Signals conflict, so confidence is reduced.")
        return -8.0, notes
    if negative >= 3:
        notes.append("Multiple independent signals point to elevated downside risk.")
        return -10.0, notes
    return 0.0, notes


def confidence_from_components(
    news_count: int,
    final_score: float,
    negative_count: int,
    has_history: bool,
    quality_adjustment: float = 0.0,
) -> float:
    confidence = 42.0 + min(news_count, 25) * 1.0 + max(min(final_score, 90), -40) * 0.35
    confidence -= negative_count * 3.8
    if has_history:
        confidence += 8.0
    confidence += quality_adjustment * 0.55
    return max(5.0, min(93.0, confidence))


def analyze_ticker(
    ticker: str,
    macro: Tuple[float, float, Dict[str, int], Dict[str, int]],
) -> StockResult:
    ticker = ticker.strip().upper()
    company, info = get_company_name_and_info(ticker)
    sector = SECTOR_MAP.get(ticker, "general")

    news = collect_ticker_news(ticker, company)
    news_score, positive_hits, negative_hits, notes = score_news(news)

    macro_score, macro_risk_score, macro_pos, macro_risk = macro
    sector_macro_score = sector_macro_adjustment(sector, macro_pos, macro_risk)

    history = download_history(ticker, period="2y")
    technical = technical_analysis(history)
    mc = monte_carlo_analysis(history)
    fundamentals = fundamentals_analysis(info)
    prediction = prediction_analysis(
        history,
        news_score,
        macro_score,
        sector_macro_score,
        technical,
        fundamentals,
        mc,
        negative_hits,
    )

    final_score, risk_penalty, bullish_reasons, bearish_reasons = compute_probability_ranking_score(
        news_score,
        positive_hits,
        negative_hits,
        technical,
        fundamentals,
        mc,
        macro_score,
        macro_risk_score,
        sector_macro_score,
    )
    quality_adjustment, quality_notes = component_quality_adjustment(
        news_score,
        technical,
        fundamentals,
        mc,
        macro_score,
        sector_macro_score,
    )
    notes.extend(quality_notes)
    if bullish_reasons:
        notes.append("Bullish alignment: " + ", ".join(bullish_reasons[:5]) + ".")
    if bearish_reasons:
        notes.append("Risk alignment: " + ", ".join(bearish_reasons[:5]) + ".")
    if risk_penalty > 25:
        notes.append("Risk penalty is elevated, so ranking is conservative.")

    if technical.trend_label == "downtrend" and news_score < 45:
        final_score -= 14
        notes.append("Downtrend requires very strong positive news; extra penalty applied.")
    if technical.pullback_label == "extended / chased":
        final_score -= 8
        notes.append("Price looks extended, so chasing risk is high.")
    if technical.pullback_label == "healthy pullback near moving average" and news_score > 20:
        final_score += 5
        notes.append("Positive news plus healthy pullback setup.")
    final_score = clamp(final_score, 0.0, 100.0)

    confidence = confidence_from_components(
        len(news),
        final_score,
        sum(negative_hits.values()),
        history is not None,
        quality_adjustment,
    )
    setup_label = detect_rising_stock_setup(technical, news_score, negative_hits, mc, sector_macro_score)
    small_score, small_risk, small_summary = small_cap_catalyst_score(
        info,
        sector,
        technical,
        positive_hits,
        negative_hits,
        news,
    )
    ranking_score = compute_ranking_score(final_score, confidence, setup_label, mc, technical, negative_hits)
    estimated_upside_probability = prediction.probability_up_1w or prediction.probability_up_1d or mc.probability_up_20d
    action = action_from_score(final_score, confidence, negative_hits, technical, news_score, mc)

    result = StockResult(
        ticker=ticker,
        company=company,
        sector=sector,
        action=action,
        confidence=confidence,
        final_score=float(final_score),
        news_score=float(news_score),
        macro_score=float(macro_score),
        sector_macro_score=float(sector_macro_score),
        technical=technical,
        monte_carlo=mc,
        fundamentals=fundamentals,
        prediction=prediction,
        ranking_score=ranking_score,
        rising_setup_label=setup_label,
        estimated_upside_probability=estimated_upside_probability,
        small_cap_catalyst_score=small_score,
        small_cap_risk_level=small_risk,
        small_cap_summary=small_summary,
        suggested_entry_style=suggested_entry_style(setup_label, technical),
        backtest=backtest_strategy(ticker, history),
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        headlines=sorted(
            news,
            key=lambda item: (parse_news_datetime(item.published) or datetime.min.replace(tzinfo=timezone.utc), credibility_weight(item.source)),
            reverse=True,
        )[:8],
        notes=notes,
        history=history,
    )
    result.recommendation_reason = generate_recommendation_reason(result)
    result.main_risk_warning = main_risk_warning_from_components(result)
    result.risk_management = risk_management_suggestion(result)
    return result


def discover_tickers_from_news(max_tickers: int = 80) -> List[str]:
    patterns = [
        r"\(([A-Z]{1,5})\)",
        r"\$([A-Z]{1,5})\b",
        r"\bNASDAQ:\s*([A-Z]{1,5})\b",
        r"\bNYSE:\s*([A-Z]{1,5})\b",
    ]
    blocked = {
        "CEO", "CFO", "USA", "FDA", "SEC", "IPO", "ETF", "AI", "GDP", "EPS",
        "USD", "THE", "AND", "FOR", "NEW", "TOP", "BUY", "SELL", "CPI", "FED",
    }
    queries = [
        "latest stocks to watch today catalyst upgrade earnings contract",
        "latest Reuters stocks moving AI chips defense oil healthcare",
        "latest CNBC premarket stocks moving analyst upgrade",
        "latest stock market biggest movers positive news",
    ]
    found: List[str] = []
    for query in queries:
        for item in google_news_rss(query, max_items=25, recency_days=2):
            for pattern in patterns:
                for match in re.findall(pattern, item.title):
                    ticker = match.upper()
                    if ticker not in blocked and ticker not in found:
                        found.append(ticker)
        time.sleep(0.12)
    return found[:max_tickers]


def parse_user_tickers(raw: str) -> List[str]:
    if not raw.strip():
        discovered = discover_tickers_from_news(max_tickers=80)
        pool = list(dict.fromkeys(discovered + DEFAULT_TICKERS))
        print("\nNews-discovered tickers:", discovered)
        return pool
    return [x.strip().upper() for x in re.split(r"[,\s]+", raw) if x.strip()]


def print_main_menu() -> str:
    print("\n" + "=" * 78)
    print("MODE SELECT")
    print("=" * 78)
    print("1. Analyze specific stock ticker(s)")
    print("2. Recommend stocks from a broad live-news scan")
    print("=" * 78)
    choice = input("Choose 1 or 2: ").strip()
    if choice not in {"1", "2"}:
        print("Invalid choice. Defaulting to mode 1.")
        return "1"
    return choice


def get_specific_tickers() -> List[str]:
    raw = input("Enter ticker(s), separated by comma or space. Example: NVDA, RDW, RKLB: ")
    tickers = [x.strip().upper() for x in re.split(r"[,\s]+", raw) if x.strip()]
    return list(dict.fromkeys(tickers))


def get_recommendation_tickers() -> List[str]:
    print("\nBuilding broad recommendation pool...")
    print("This includes live news-discovered tickers plus large-cap and small-cap watch pools.")
    discovered = discover_tickers_from_news(max_tickers=100)
    tickers = list(dict.fromkeys(discovered + DEFAULT_TICKERS + OPPORTUNISTIC_TICKERS))
    print("\nNews-discovered tickers:", discovered if discovered else "none")
    print(f"Broad scan pool size: {len(tickers)}")
    return tickers


def print_macro_summary(macro_items: Sequence[NewsItem], macro: Tuple[float, float, Dict[str, int], Dict[str, int]]) -> None:
    macro_score, macro_risk_score, macro_pos, macro_risk = macro
    print("\n" + "=" * 78)
    print("LIVE MACRO / ECONOMIC / WAR NEWS VARIABLES")
    print("=" * 78)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Macro score: {macro_score:.1f} | Macro risk score: {macro_risk_score:.1f}")
    print("Positive macro variables:", macro_pos or "none")
    print("Risk macro variables:", macro_risk or "none")
    print("\nFresh macro headlines used:")
    for item in macro_items[:8]:
        print(f"- {item.title} [{item.source}]")


def print_result(result: StockResult, rank: int) -> None:
    price = "N/A" if result.technical.price is None else f"${result.technical.price:,.2f}"
    mc = result.monte_carlo
    expected = "N/A" if mc.expected_price_20d is None else f"${mc.expected_price_20d:,.2f}"
    low = "N/A" if mc.downside_10pct is None else f"${mc.downside_10pct:,.2f}"
    high = "N/A" if mc.upside_90pct is None else f"${mc.upside_90pct:,.2f}"
    prob_up = "N/A" if mc.probability_up_20d is None else f"{mc.probability_up_20d * 100:.1f}%"
    pred = result.prediction

    def fmt_price(value: Optional[float]) -> str:
        return "N/A" if value is None else f"${value:,.2f}"

    def fmt_prob(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value * 100:.1f}%"

    def fmt_return(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value * 100:+.2f}%"

    print("\n" + "-" * 78)
    print(f"{rank}. {result.ticker} | {result.company}")
    print("-" * 78)
    print("SUMMARY")
    print(f"  Action      : {result.action}")
    print(f"  Price       : {price}")
    print(f"  Sector      : {result.sector}")
    print(f"  Confidence  : {result.confidence:.1f}%")
    print(f"  Final score : {result.final_score:.1f}")
    print(f"  Ranking score: {result.ranking_score:.1f}")
    print(f"  Rising setup : {result.rising_setup_label}")
    est_prob = "N/A" if result.estimated_upside_probability is None else f"{result.estimated_upside_probability * 100:.1f}%"
    print(f"  Est. upside probability: {est_prob}")
    print(f"  Entry style  : {result.suggested_entry_style}")

    print("\nSCORE BREAKDOWN")
    print(f"  News catalyst         : {result.news_score:.1f}")
    print(f"  Technical / pullback  : {result.technical.score:.1f}")
    print(f"  Trend quality         : {trend_quality_score(result.technical):.1f}")
    print(f"  Volume activity       : {volume_activity_score(result.technical):.1f}")
    print(f"  Fundamentals / growth : {result.fundamentals.score:.1f}")
    print(f"  Monte Carlo risk      : {result.monte_carlo.risk_score:.1f}")
    print(f"  Live macro            : {result.macro_score:.1f}")
    print(f"  Sector macro          : {result.sector_macro_score:.1f}")
    print(f"  Small-cap catalyst    : {result.small_cap_catalyst_score:.1f}")
    print(f"  Small-cap risk level  : {result.small_cap_risk_level}")

    print("\nFORECAST")
    print("  Horizon | Predicted price | Expected return | Chance up")
    print("  --------|-----------------|-----------------|----------")
    print(f"  1 day   | {fmt_price(pred.price_1d):>15} | {fmt_return(pred.expected_return_1d):>15} | {fmt_prob(pred.probability_up_1d):>8}")
    print(f"  1 week  | {fmt_price(pred.price_1w):>15} | {fmt_return(pred.expected_return_1w):>15} | {fmt_prob(pred.probability_up_1w):>8}")
    print(f"  1 year  | {fmt_price(pred.price_1y):>15} | {fmt_return(pred.expected_return_1y):>15} | {fmt_prob(pred.probability_up_1y):>8}")

    print("\nTECHNICAL SETUP")
    print(f"  Trend      : {result.technical.trend_label}")
    print(f"  Pullback   : {result.technical.pullback_label}")
    print(f"  RSI        : {result.technical.rsi if result.technical.rsi is not None else 'N/A'}")
    print(f"  Support    : {result.technical.support or 'N/A'}")
    print(f"  Resistance : {result.technical.resistance or 'N/A'}")

    print("\nMONTE CARLO 20D RISK")
    print(f"  Expected price : {expected}")
    print(f"  10% downside   : {low}")
    print(f"  90% upside     : {high}")
    print(f"  Chance up      : {prob_up}")

    print("\nNEWS EVENTS")
    print("  Positive:", result.positive_hits or "none")
    print("  Negative:", result.negative_hits or "none")
    print("\nWHY THIS STOCK?")
    print(f"  {result.recommendation_reason}")
    print("\nMAIN RISK WARNING")
    print(f"  {result.main_risk_warning}")
    if result.small_cap_summary != "N/A":
        print("\nSMALL-CAP / CATALYST NOTE")
        print(f"  {result.small_cap_summary}")
    if result.risk_management:
        print("\nRISK MANAGEMENT")
        print(f"  Suggested stop-loss area : {result.risk_management.get('stopLossArea', 'N/A')}")
        print(f"  Take-profit zone         : {result.risk_management.get('takeProfitZone', 'N/A')}")
        print(f"  Position size level      : {result.risk_management.get('positionSize', 'N/A')}")
        print(f"  Risk notes               : {result.risk_management.get('warnings', 'N/A')}")
    if result.backtest.get("available"):
        print("\nEDUCATIONAL BACKTEST")
        print(f"  Trades          : {result.backtest.get('trades', 'N/A')}")
        print(f"  Win rate        : {result.backtest.get('winRate', 0.0) * 100:.1f}%")
        print(f"  Avg 20d return  : {result.backtest.get('averageReturn20d', 0.0) * 100:+.2f}%")
        print(f"  Max drawdown    : {result.backtest.get('maxDrawdown', 0.0) * 100:.1f}%")
        print(f"  Buy/hold return : {result.backtest.get('buyAndHoldReturn', 0.0) * 100:+.2f}%")
    if result.notes:
        print("\nNOTES")
        for note in result.notes[:4]:
            print(f"  - {note}")
    print("\nTOP HEADLINES")
    for item in result.headlines[:5]:
        print(f"  - {item.title} [{item.source}]")


def show_graphs(best: StockResult) -> None:
    if plt is None or pd is None or np is None or best.history is None:
        print("\nGraph skipped because matplotlib/pandas/numpy or price history is unavailable.")
        return

    history = best.history.copy()
    close = to_series(history["Close"]).astype(float)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    plt.figure(figsize=(11, 5))
    plt.plot(close.tail(252), label="Close", linewidth=1.8)
    plt.plot(ma20.tail(252), label="MA20", linewidth=1.0)
    plt.plot(ma50.tail(252), label="MA50", linewidth=1.0)
    if len(ma200.dropna()) > 0:
        plt.plot(ma200.tail(252), label="MA200", linewidth=1.0)
    plt.title(f"{best.ticker} price trend / pullback analysis")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if best.monte_carlo.paths is not None:
        paths = best.monte_carlo.paths
        plt.figure(figsize=(11, 5))
        for i in range(min(90, len(paths))):
            plt.plot(paths[i], alpha=0.12)
        plt.title(f"{best.ticker} Monte Carlo 20-day risk simulation")
        plt.xlabel("Days forward")
        plt.ylabel("Simulated price")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()


def run() -> None:
    print("\n" + APP_NAME)
    print("News is the main signal, but trend, pullback, growth, and Monte Carlo risk are included.")
    print("Macro variables are refreshed from live news every run. Educational only, not financial advice.\n")
    safe_import_warning()
    setup_finbert()

    mode = print_main_menu()
    if mode == "1":
        tickers = get_specific_tickers()
        mode_label = "Specific stock analysis"
    else:
        tickers = get_recommendation_tickers()
        mode_label = "Broad recommendation scan"

    if not tickers:
        print("No tickers found.")
        return

    print(f"\nSelected mode: {mode_label}")
    print(f"\nTicker pool size: {len(tickers)}")
    print("Collecting live macro/economic/geopolitical news...")
    macro_items = collect_macro_news()
    macro = macro_regime_score(macro_items)
    print_macro_summary(macro_items, macro)

    results: List[StockResult] = []
    print("\nAnalyzing stocks...")
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] {ticker}")
        try:
            results.append(analyze_ticker(ticker, macro))
        except Exception as exc:
            print(f"{ticker} skipped due to error: {str(exc)[:180]}")

    if not results:
        print("No valid results.")
        return

    results.sort(key=lambda row: (row.ranking_score, row.final_score, row.confidence), reverse=True)

    print("\n" + "=" * 78)
    if mode == "1":
        print("SPECIFIC STOCK ANALYSIS RESULT")
    else:
        print("FINAL RECOMMENDATION RANKING")
    print("=" * 78)
    display_count = len(results) if mode == "1" else min(15, len(results))
    for rank, result in enumerate(results[:display_count], start=1):
        print_result(result, rank)

    best = results[0]
    print("\n" + "=" * 78)
    print("BEST CANDIDATE GRAPH CHECK")
    print("=" * 78)
    show_graphs(best)

    print("\nScoring weights: news 55%, technical/pullback 20%, fundamentals 12%, Monte Carlo 13%,")
    print("plus live macro/sector adjustments, 1d/1w/1y forecasts, and strong penalties for negative news or downtrends.")


if __name__ == "__main__":
    run()
