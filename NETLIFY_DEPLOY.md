# Netlify Deploy Notes

Netlify can host the iPhone-facing web app, so the phone does not need to be on
the same Wi-Fi as this Mac.

The current stock analyzer still needs the Python backend because it imports
`StockFinalJH.py`, `yfinance`, `feedparser`, FinBERT, and the existing scoring
logic. Netlify's normal static hosting cannot run that Python process.

Use this setup:

1. Host `mobile_stock_app/server.py` on a Python-capable service such as Render,
   Fly.io, Railway, or a VPS. The included `render.yaml` is ready for Render.
2. Copy the public Python backend URL, for example:

```text
https://news-first-quant-api.onrender.com
```

3. Deploy this project on Netlify using the repository root. The included
   `netlify.toml` publishes `mobile_stock_app/static` and installs the
   `netlify/functions/analyze.js` proxy.
4. In Netlify, set this environment variable:

```text
BACKEND_URL=https://your-public-python-backend-url
```

5. The iPhone can then use the Netlify site URL from any internet connection.

For local testing, keep `NEWS_QUANT_API_BASE_URL` as an empty string so the app
uses the same local server at `/api/analyze`.
