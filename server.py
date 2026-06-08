from __future__ import annotations

import json
import os
import re
import socket
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
STATIC_ROOT = ROOT / "static"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import StockFinalJH as engine  # noqa: E402


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _headline_payload(item: Any) -> dict[str, Any]:
    return {
        "title": getattr(item, "title", ""),
        "source": getattr(item, "source", "Unknown"),
        "url": getattr(item, "url", ""),
        "published": getattr(item, "published", ""),
    }


def _result_payload(result: Any, rank: int) -> dict[str, Any]:
    technical = result.technical
    mc = result.monte_carlo
    fundamentals = result.fundamentals
    prediction = result.prediction
    return {
        "rank": rank,
        "ticker": result.ticker,
        "company": result.company,
        "sector": result.sector,
        "action": result.action,
        "confidence": result.confidence,
        "finalScore": result.final_score,
        "newsScore": result.news_score,
        "macroScore": result.macro_score,
        "sectorMacroScore": result.sector_macro_score,
        "positiveHits": result.positive_hits,
        "negativeHits": result.negative_hits,
        "notes": result.notes[:5],
        "headlines": [_headline_payload(item) for item in result.headlines[:8]],
        "technical": {
            "price": technical.price,
            "rsi": technical.rsi,
            "trendLabel": technical.trend_label,
            "pullbackLabel": technical.pullback_label,
            "support": technical.support,
            "resistance": technical.resistance,
            "volatility": technical.volatility,
            "score": technical.score,
            "notes": technical.notes[:4],
        },
        "monteCarlo": {
            "expectedPrice20d": mc.expected_price_20d,
            "downside10pct": mc.downside_10pct,
            "upside90pct": mc.upside_90pct,
            "probabilityUp20d": mc.probability_up_20d,
            "riskScore": mc.risk_score,
        },
        "fundamentals": {
            "revenueGrowth": fundamentals.revenue_growth,
            "earningsGrowth": fundamentals.earnings_growth,
            "profitMargin": fundamentals.profit_margin,
            "debtToEquity": fundamentals.debt_to_equity,
            "score": fundamentals.score,
            "notes": fundamentals.notes[:4],
        },
        "prediction": {
            "price1d": prediction.price_1d,
            "price1w": prediction.price_1w,
            "price1y": prediction.price_1y,
            "probabilityUp1d": prediction.probability_up_1d,
            "probabilityUp1w": prediction.probability_up_1w,
            "probabilityUp1y": prediction.probability_up_1y,
            "expectedReturn1d": prediction.expected_return_1d,
            "expectedReturn1w": prediction.expected_return_1w,
            "expectedReturn1y": prediction.expected_return_1y,
            "method": prediction.method,
        },
    }


def _parse_tickers(raw: str) -> list[str]:
    return list(dict.fromkeys(x.strip().upper() for x in re.split(r"[,\s]+", raw or "") if x.strip()))


def _analyze(payload: dict[str, Any]) -> dict[str, Any]:
    mode = payload.get("mode", "specific")
    limit = int(payload.get("limit") or 8)
    limit = max(1, min(limit, 20))

    engine.safe_import_warning()
    if not engine.FINBERT_READY:
        engine.setup_finbert()

    if mode == "scan":
        discovered = engine.discover_tickers_from_news(max_tickers=30)
        tickers = list(dict.fromkeys(discovered + engine.DEFAULT_TICKERS + engine.OPPORTUNISTIC_TICKERS))[:limit]
        mode_label = "Broad live-news scan"
    else:
        tickers = _parse_tickers(str(payload.get("tickers", "")))[:limit]
        discovered = []
        mode_label = "Specific stock analysis"

    if not tickers:
        raise ValueError("분석할 티커를 입력해 주세요. 예: NVDA, AAPL, RKLB")

    macro_items = engine.collect_macro_news()
    macro = engine.macro_regime_score(macro_items)
    results = []

    for ticker in tickers:
        try:
            results.append(engine.analyze_ticker(ticker, macro))
        except Exception as exc:
            results.append(
                {
                    "ticker": ticker,
                    "error": str(exc)[:220],
                }
            )

    valid = [row for row in results if not isinstance(row, dict)]
    errors = [row for row in results if isinstance(row, dict)]
    valid.sort(key=lambda row: (row.final_score, row.confidence), reverse=True)

    macro_score, macro_risk_score, macro_pos, macro_risk = macro
    return {
        "appName": engine.APP_NAME,
        "mode": mode,
        "modeLabel": mode_label,
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers": tickers,
        "discoveredTickers": discovered,
        "macro": {
            "score": macro_score,
            "riskScore": macro_risk_score,
            "positiveVariables": macro_pos,
            "riskVariables": macro_risk,
            "headlines": [_headline_payload(item) for item in macro_items[:8]],
        },
        "results": [_result_payload(row, rank) for rank, row in enumerate(valid, start=1)],
        "errors": _json_safe(errors),
        "disclaimer": "Educational use only. This is not financial advice.",
    }


class MobileAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
            response = _analyze(payload)
            self._send_json(response)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True, "app": engine.APP_NAME})
            return
        if self.path in {"/", "/index.html"}:
            self.path = "/index.html"
        super().do_GET()

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", os.getenv("CORS_ORIGIN", "*"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        try:
            sock.close()
        except Exception:
            pass


def main() -> None:
    port = int(os.getenv("PORT", "8765"))
    server = ThreadingHTTPServer(("0.0.0.0", port), MobileAppHandler)
    print(f"Mobile app: http://127.0.0.1:{port}")
    print(f"iPhone on same Wi-Fi: http://{local_ip()}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
