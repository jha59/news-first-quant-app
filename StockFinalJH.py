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
from datetime import datetime
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


def google_news_rss(query: str, max_items: int = 12) -> List[NewsItem]:
    if feedparser is None:
        return []
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    raw = request_text(url)
    if not raw:
        return []
    try:
        feed = feedparser.parse(raw)
    except Exception:
        return []

    items = []
    for entry in feed.entries[:max_items]:
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
        items.append(NewsItem(clean_text(title), str(source), row.get("link") or "", ""))
    return items


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
    return unique


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
        items.extend(google_news_rss(query, max_items=12))
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
        items.extend(google_news_rss(query, max_items=9))
        time.sleep(0.12)
    items.extend(yahoo_ticker_news(ticker, max_items=12))
    return dedupe_news(items)[:50]


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


def score_news(items: Sequence[NewsItem]) -> Tuple[float, Dict[str, int], Dict[str, int], List[str]]:
    if not items:
        return -25.0, {}, {}, ["No fresh news found; confidence reduced."]

    weighted_text_parts = []
    credibility_bonus = 0.0
    for item in items:
        weight = credibility_weight(item.source)
        weighted_text_parts.append((item.title + " ") * max(1, int(round(weight * 2))))
        credibility_bonus += min(weight - 0.85, 0.35)

    text = clean_text(" ".join(weighted_text_parts)).lower()
    positive_hits = count_event_hits(text, POSITIVE_EVENTS)
    negative_hits = count_event_hits(text, NEGATIVE_EVENTS)
    keyword_score = keyword_sentiment_score(text)
    model_score = finbert_score([item.title for item in items])

    positive_points = sum(positive_hits.values()) * 9.0
    negative_points = sum(negative_hits.values()) * 14.0
    sentiment_points = (0.60 * model_score + 0.40 * keyword_score) * 38.0
    breadth_bonus = min(len(items), 30) * 0.7
    credibility_bonus = min(credibility_bonus, 11.0)
    score = positive_points - negative_points + sentiment_points + breadth_bonus + credibility_bonus

    notes = []
    if negative_points > positive_points:
        notes.append("Negative headlines dominate positive catalysts.")
    if len(items) < 6:
        notes.append("News sample is thin.")
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

    if safe_float(macd.iloc[-1], 0.0) > safe_float(signal.iloc[-1], 0.0):
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

    # News gets the strongest directional influence. Fundamentals matter more on the 1y horizon.
    signal_1d = 0.58 * news_signal + 0.20 * tech_signal + 0.12 * macro_signal + 0.10 * mc_signal - neg_penalty
    signal_1w = 0.55 * news_signal + 0.23 * tech_signal + 0.12 * macro_signal + 0.07 * mc_signal + 0.03 * fund_signal - neg_penalty
    signal_1y = 0.38 * news_signal + 0.18 * tech_signal + 0.14 * macro_signal + 0.30 * fund_signal - neg_penalty * 0.75

    prob_1d = probability_from_signal(signal_1d)
    prob_1w = probability_from_signal(signal_1w)
    prob_1y = probability_from_signal(signal_1y)

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
    pred.method = "news-weighted hybrid forecast using recent returns, volatility, trend, fundamentals, macro, and Monte Carlo"
    return pred


def action_from_score(final_score: float, negative_hits: Dict[str, int], technical: TechnicalSnapshot) -> str:
    neg_count = sum(negative_hits.values())
    if final_score >= 78 and neg_count <= 1 and technical.trend_label != "downtrend":
        return "STRONG WATCH / BUY-CANDIDATE"
    if final_score >= 55 and neg_count <= 3:
        return "WATCHLIST"
    if final_score >= 30:
        return "NEUTRAL / WAIT"
    return "AVOID / HIGH RISK"


def confidence_from_components(news_count: int, final_score: float, negative_count: int, has_history: bool) -> float:
    confidence = 42.0 + min(news_count, 25) * 1.0 + max(min(final_score, 90), -40) * 0.35
    confidence -= negative_count * 3.8
    if has_history:
        confidence += 8.0
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

    direct_negative_risk = sum(negative_hits.values()) * 7.0

    final_score = (
        news_score * 0.55
        + technical.score * 0.20
        + fundamentals.score * 0.12
        + mc.risk_score * 0.13
        + sector_macro_score
        + macro_score * 0.22
        - macro_risk_score * 0.08
        - direct_negative_risk
    )

    if technical.trend_label == "downtrend" and news_score < 45:
        final_score -= 14
        notes.append("Downtrend requires very strong positive news; extra penalty applied.")
    if technical.pullback_label == "extended / chased":
        notes.append("Price looks extended, so chasing risk is high.")
    if technical.pullback_label == "healthy pullback near moving average" and news_score > 20:
        final_score += 8
        notes.append("Positive news plus healthy pullback setup.")

    action = action_from_score(final_score, negative_hits, technical)
    confidence = confidence_from_components(len(news), final_score, sum(negative_hits.values()), history is not None)

    return StockResult(
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
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        headlines=sorted(news, key=lambda item: credibility_weight(item.source), reverse=True)[:8],
        notes=notes,
        history=history,
    )


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
        for item in google_news_rss(query, max_items=25):
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

    print("\nSCORE BREAKDOWN")
    print(f"  News / catalyst       : {result.news_score:.1f}  (highest weight)")
    print(f"  Technical / pullback  : {result.technical.score:.1f}")
    print(f"  Fundamentals / growth : {result.fundamentals.score:.1f}")
    print(f"  Monte Carlo risk      : {result.monte_carlo.risk_score:.1f}")
    print(f"  Live macro            : {result.macro_score:.1f}")
    print(f"  Sector macro          : {result.sector_macro_score:.1f}")

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

    results.sort(key=lambda row: (row.final_score, row.confidence), reverse=True)

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
