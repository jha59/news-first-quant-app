const translations = {
  en: {
    eyebrow: "Live news-first quant predictor",
    subcopy: "Uses the same news-first logic: live headlines, technical setup, fundamentals, and Monte Carlo risk.",
    specificMode: "Ticker Analysis",
    scanMode: "News Scan",
    tickerLabel: "Tickers",
    limitLabel: "Max stocks",
    analyzeButton: "Analyze",
    loadingButton: "Analyzing news and price data",
    loadingStatus: "Loading live macro news, stock headlines, and price data.",
    resultCount: "results",
    macroTitle: "Live Macro Variables",
    run: "Run",
    macro: "Macro",
    risk: "Risk",
    finalScore: "Final Score",
    rankingScore: "Ranking Score",
    setup: "Rising Setup",
    upsideProb: "Upside Probability",
    entryStyle: "Entry Style",
    confidence: "Confidence",
    price: "Price",
    newsScore: "News Score",
    oneDay: "1 Day",
    oneWeek: "1 Week",
    oneYear: "1 Year",
    up: "Up",
    trend: "Trend",
    pullback: "Pullback",
    monteCarlo: "20D Monte Carlo",
    sector: "Sector",
    tech: "Tech",
    fund: "Fund",
    mc: "MC",
    smallCap: "Small-cap Catalyst",
    smallCapRisk: "Small-cap Risk",
    newsQuality: "News Quality",
    verifiedCatalyst: "Verified Catalyst",
    bestCandidate: "Best match",
    relatedContext: "Related Context",
    adaptivePolicy: "Adaptive Ensemble",
    learningLoop: "Learning Loop",
    whyThis: "Why This Stock?",
    riskWarning: "Main Risk Warning",
    riskManagement: "Risk Management",
    summary: "Summary",
    errorFallback: "Analysis failed",
    apiHint: "Set window.NEWS_QUANT_API_BASE_URL in config.js when this static app is hosted separately.",
  },
  ko: {
    eyebrow: "실시간 뉴스 우선 퀀트 예측기",
    subcopy: "기존 코드와 같은 방식으로 실시간 뉴스, 기술적 흐름, 펀더멘털, 몬테카를로 리스크를 분석합니다.",
    specificMode: "티커 분석",
    scanMode: "뉴스 스캔",
    tickerLabel: "티커",
    limitLabel: "최대 종목 수",
    analyzeButton: "분석 시작",
    loadingButton: "뉴스와 가격 데이터 분석 중",
    loadingStatus: "실시간 매크로 뉴스, 종목 뉴스, 가격 데이터를 불러오는 중입니다.",
    resultCount: "개 결과",
    macroTitle: "실시간 매크로 변수",
    run: "실행",
    macro: "매크로",
    risk: "리스크",
    finalScore: "최종 점수",
    rankingScore: "랭킹 점수",
    setup: "상승 셋업",
    upsideProb: "상승 확률",
    entryStyle: "진입 스타일",
    confidence: "신뢰도",
    price: "가격",
    newsScore: "뉴스 점수",
    oneDay: "1일",
    oneWeek: "1주",
    oneYear: "1년",
    up: "상승",
    trend: "추세",
    pullback: "눌림목",
    monteCarlo: "20일 몬테카를로",
    sector: "섹터",
    tech: "기술",
    fund: "펀더멘털",
    mc: "MC",
    smallCap: "소형주 촉매",
    smallCapRisk: "소형주 리스크",
    newsQuality: "뉴스 품질",
    verifiedCatalyst: "검증된 촉매",
    bestCandidate: "최우선 후보",
    relatedContext: "관련 배경",
    adaptivePolicy: "적응형 앙상블",
    learningLoop: "학습 루프",
    whyThis: "왜 이 종목인가?",
    riskWarning: "주요 리스크",
    riskManagement: "리스크 관리",
    summary: "요약",
    errorFallback: "분석 실패",
    apiHint: "정적 앱을 따로 호스팅할 때 config.js에 window.NEWS_QUANT_API_BASE_URL을 설정하세요.",
  },
};

const phraseTranslations = {
  ko: {
    "Specific stock analysis": "개별 종목 분석",
    "Broad live-news scan": "실시간 뉴스 광역 스캔",
    "STRONG WATCH / BUY-CANDIDATE": "강한 관심 / 매수 후보",
    WATCHLIST: "관심 목록",
    "NEUTRAL / WAIT": "중립 / 대기",
    "HIGH CONVICTION BUY CANDIDATE": "고확신 매수 후보",
    "STRONG BUY CANDIDATE": "강한 매수 후보",
    "WATCHLIST / POSSIBLE BUY": "관심 목록 / 매수 가능 후보",
    "NEUTRAL / WAIT FOR BETTER ENTRY": "중립 / 더 좋은 진입 대기",
    "AVOID / HIGH RISK": "회피 / 고위험",
    "Early Bullish Setup": "초기 상승 셋업",
    "Momentum Breakout Setup": "모멘텀 돌파 셋업",
    "Healthy Pullback Buy Setup": "건강한 눌림목 매수 셋업",
    "Overextended / Wait": "과열 / 대기",
    "Bearish / Avoid": "약세 / 회피",
    uptrend: "상승 추세",
    downtrend: "하락 추세",
    "sideways / base": "횡보 / 베이스",
    "healthy pullback near moving average": "이동평균 근처 건강한 눌림목",
    "extended / chased": "과열 / 추격 위험",
    "falling trend, not a clean pullback": "하락 추세, 깨끗한 눌림목 아님",
    "neutral setup": "중립 셋업",
  },
};

const state = {
  mode: "specific",
  lang: localStorage.getItem("newsQuantLang") || "en",
};

const $ = (selector) => document.querySelector(selector);
const resultsEl = $("#results");
const macroEl = $("#macro");
const statusEl = $("#status");
const analyzeButton = $("#analyze");
const tickerField = $("#tickerField");
let latestData = null;

function t(key) {
  return translations[state.lang][key] || translations.en[key] || key;
}

function trPhrase(value) {
  return phraseTranslations[state.lang]?.[value] || value;
}

function apiUrl(path) {
  const base = window.NEWS_QUANT_API_BASE_URL || "";
  return `${base}${path}`;
}

function money(value) {
  if (!isFiniteNumber(value)) return "Not enough data";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
}

function pct(value, signed = false) {
  if (!isFiniteNumber(value)) return "Not enough data";
  const body = `${(value * 100).toFixed(1)}%`;
  return signed && value > 0 ? `+${body}` : body;
}

function num(value, digits = 1) {
  if (!isFiniteNumber(value)) return "Not enough data";
  return Number(value).toFixed(digits);
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function safeNumber(...values) {
  for (const value of values) {
    if (isFiniteNumber(value)) return value;
  }
  return 0;
}

function hasRealText(value) {
  return typeof value === "string" && value.trim() && value.trim() !== "N/A";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function scoreClass(value) {
  if (value >= 55) return "";
  if (value >= 30) return "warn";
  return "bad";
}

function fallbackSetup(row) {
  const trend = row.technical?.trendLabel || "unknown";
  const rsi = safeNumber(row.technical?.rsi, 50);
  const newsScore = safeNumber(row.newsScore);
  const negCount = Object.values(row.negativeHits || {}).reduce((sum, value) => sum + safeNumber(value), 0);
  const mcUp = safeNumber(row.monteCarlo?.probabilityUp20d, row.prediction?.probabilityUp1w, 0.5);
  const volumeRatio = safeNumber(row.technical?.volumeRatio, 1);
  if (trend === "downtrend" || negCount >= 4 || mcUp < 0.42) return "Bearish / Avoid";
  if (rsi > 74 || safeNumber(row.technical?.return1w) > 0.18) return "Overextended / Wait";
  if (trend === "uptrend" && volumeRatio >= 1.35 && rsi <= 72 && mcUp >= 0.55) return "Momentum Breakout Setup";
  if (trend === "uptrend" && rsi >= 45 && rsi <= 70 && newsScore > 5 && mcUp >= 0.52) return "Early Bullish Setup";
  return "Neutral / Wait";
}

function fallbackEntryStyle(setup) {
  if (setup === "Momentum Breakout Setup") return "Breakout candidate";
  if (setup === "Healthy Pullback Buy Setup") return "Healthy pullback setup";
  if (setup === "Early Bullish Setup") return "Wait for pullback";
  if (setup === "Overextended / Wait") return "Avoid chasing";
  if (setup === "Bearish / Avoid") return "Avoid";
  return "Wait for better entry";
}

function fallbackRankingScore(row, setup) {
  const finalScore = safeNumber(row.finalScore);
  const confidence = safeNumber(row.confidence);
  const mcUp = safeNumber(row.monteCarlo?.probabilityUp20d, row.prediction?.probabilityUp1w, 0.5);
  const probUp = safeNumber(row.prediction?.probabilityUp1w, row.monteCarlo?.probabilityUp20d, 0.5);
  const upside1w = Math.max(0, Math.min(0.3, safeNumber(row.prediction?.expectedReturn1w)));
  const upside1y = Math.max(0, Math.min(1.2, safeNumber(row.prediction?.expectedReturn1y)));
  const volumeRatio = safeNumber(row.technical?.volumeRatio, 1);
  const negCount = Object.values(row.negativeHits || {}).reduce((sum, value) => sum + safeNumber(value), 0);
  const setupBonus = {
    "Healthy Pullback Buy Setup": 18,
    "Momentum Breakout Setup": 17,
    "Early Bullish Setup": 11,
    "Overextended / Wait": -20,
    "Bearish / Avoid": -35,
  }[setup] || 0;
  let alignment = 0;
  if (safeNumber(row.newsScore) > 28) alignment += 4;
  if (row.technical?.trendLabel === "uptrend") alignment += 5;
  if (safeNumber(row.technical?.rsi, 50) >= 45 && safeNumber(row.technical?.rsi, 50) <= 70) alignment += 3;
  if (volumeRatio >= 1.2 && volumeRatio <= 3.5) alignment += 7;
  if (mcUp >= 0.55) alignment += 5;
  if (safeNumber(row.sectorMacroScore) > 0) alignment += 3;
  let riskPenalty = negCount * 6;
  if (row.technical?.trendLabel === "downtrend") riskPenalty += 18;
  if (safeNumber(row.technical?.return1w) >= 0.2 || safeNumber(row.technical?.return1m) >= 0.45) riskPenalty += 12;
  return finalScore + confidence * 0.28 + setupBonus + (mcUp - 0.5) * 55 + (probUp - 0.5) * 70 + upside1w * 45 + upside1y * 10 + alignment - riskPenalty;
}

function fallbackSmallCapRisk(row) {
  const volatility = safeNumber(row.technical?.volatility);
  const negCount = Object.values(row.negativeHits || {}).reduce((sum, value) => sum + safeNumber(value), 0);
  if (negCount >= 5 || volatility > 0.95) return "Extreme";
  if (negCount >= 2 || volatility > 0.7) return "High";
  if (volatility > 0.45 || safeNumber(row.smallCapCatalystScore) > 35) return "Medium";
  return "Low";
}

function fallbackNewsQuality(row) {
  const sources = (row.headlines || []).map((item) => String(item.source || "").toLowerCase());
  if (!sources.length) return 40;
  const trusted = sources.filter((source) =>
    ["reuters", "bloomberg", "associated press", "ap", "wall street journal", "cnbc", "marketwatch", "yahoo finance", "sec"].some((name) => source.includes(name))
  ).length;
  return Math.min(100, 45 + (trusted / sources.length) * 45 + Math.min(sources.length, 6) * 2);
}

function fallbackVerifiedCatalyst(row) {
  if (row.verifiedCatalyst && row.verifiedCatalyst !== "N/A") return row.verifiedCatalyst;
  const trustedHeadlines = (row.headlines || []).filter((item) => {
    const source = String(item.source || "").toLowerCase();
    return ["reuters", "bloomberg", "associated press", "ap", "wall street journal", "cnbc", "marketwatch", "yahoo finance", "sec"].some((name) => source.includes(name));
  });
  if (trustedHeadlines.length >= 2 && safeNumber(row.newsScore) > 20) return "High-trust catalyst coverage";
  return "None";
}

function fallbackReason(row, setup) {
  const reasons = [];
  if (safeNumber(row.newsScore) > 15) reasons.push("recent catalyst news");
  if (row.technical?.trendLabel === "uptrend") reasons.push("an uptrend");
  if (safeNumber(row.technical?.rsi, 50) >= 45 && safeNumber(row.technical?.rsi, 50) <= 70) reasons.push("constructive RSI");
  if (safeNumber(row.sectorMacroScore) > 0) reasons.push("sector macro support");
  if (safeNumber(row.monteCarlo?.probabilityUp20d, 0.5) >= 0.55) reasons.push("Monte Carlo upside probability above 55%");
  const reasonText = reasons.length ? reasons.join(", ") : "mixed signals that require confirmation";
  return `${row.ticker} ranks here because ${reasonText}. Current setup: ${setup}. This is a probability-based candidate, not financial advice.`;
}

function fallbackRiskWarning(row, smallCapRisk) {
  const negCount = Object.values(row.negativeHits || {}).reduce((sum, value) => sum + safeNumber(value), 0);
  if (negCount >= 3) return "Negative headline risk is elevated; wait for confirmation.";
  if (row.technical?.trendLabel === "downtrend") return "Trend is down, so positive news needs price confirmation.";
  if (smallCapRisk === "High" || smallCapRisk === "Extreme") return "Catalyst and volatility risk are high; use smaller sizing.";
  if (safeNumber(row.technical?.volatility) > 0.7) return "Volatility is elevated; risk controls matter more than usual.";
  return "No single dominant risk, but news and market conditions can change quickly.";
}

function fallbackRiskManagement(row) {
  const price = safeNumber(row.technical?.price);
  const support = safeNumber(row.technical?.support);
  const resistance = safeNumber(row.technical?.resistance);
  const volatility = safeNumber(row.technical?.volatility);
  const stop = support > 0 ? support * 0.98 : price * (volatility > 0.65 ? 0.88 : 0.92);
  const targetLow = resistance > price ? resistance : price * 1.08;
  const targetHigh = price * (volatility > 0.65 ? 1.25 : 1.18);
  const size = volatility > 0.85 ? "Very Small" : volatility > 0.55 ? "Small" : "Normal";
  return {
    stopLossArea: money(stop),
    takeProfitZone: `${money(targetLow)} - ${money(targetHigh)}`,
    positionSize: size,
  };
}

function hydrateResult(row) {
  const setup = hasRealText(row.risingSetupLabel) ? row.risingSetupLabel : fallbackSetup(row);
  const estimatedUpsideProbability = safeNumber(
    row.estimatedUpsideProbability,
    row.prediction?.probabilityUp1w,
    row.prediction?.probabilityUp1d,
    row.monteCarlo?.probabilityUp20d,
    0.5
  );
  const smallCapScore = safeNumber(row.smallCapCatalystScore);
  const smallCapRisk = hasRealText(row.smallCapRiskLevel) ? row.smallCapRiskLevel : fallbackSmallCapRisk({ ...row, smallCapCatalystScore: smallCapScore });
  const fallbackRisk = fallbackRiskManagement(row);
  const riskManagement = row.riskManagement && Object.keys(row.riskManagement).length
    ? {
        stopLossArea: hasRealText(row.riskManagement.stopLossArea) ? row.riskManagement.stopLossArea : fallbackRisk.stopLossArea,
        takeProfitZone: hasRealText(row.riskManagement.takeProfitZone) ? row.riskManagement.takeProfitZone : fallbackRisk.takeProfitZone,
        positionSize: hasRealText(row.riskManagement.positionSize) ? row.riskManagement.positionSize : fallbackRisk.positionSize,
      }
    : fallbackRisk;
  return {
    ...row,
    rankingScore: isFiniteNumber(row.rankingScore) ? row.rankingScore : fallbackRankingScore(row, setup),
    risingSetupLabel: setup,
    estimatedUpsideProbability,
    smallCapCatalystScore: smallCapScore,
    smallCapRiskLevel: smallCapRisk,
    newsQualityScore: isFiniteNumber(row.newsQualityScore) ? row.newsQualityScore : fallbackNewsQuality(row),
    verifiedCatalyst: fallbackVerifiedCatalyst(row),
    recommendationReason: hasRealText(row.recommendationReason) ? row.recommendationReason : fallbackReason(row, setup),
    mainRiskWarning: hasRealText(row.mainRiskWarning) ? row.mainRiskWarning : fallbackRiskWarning(row, smallCapRisk),
    suggestedEntryStyle: hasRealText(row.suggestedEntryStyle) ? row.suggestedEntryStyle : fallbackEntryStyle(setup),
    riskManagement,
    technical: {
      ...row.technical,
      volumeRatio: isFiniteNumber(row.technical?.volumeRatio) ? row.technical.volumeRatio : 1,
    },
  };
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  analyzeButton.classList.toggle("loading", isLoading);
  $(".button-label").textContent = isLoading ? t("loadingButton") : t("analyzeButton");
}

function applyLanguage() {
  document.documentElement.lang = state.lang === "ko" ? "ko" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll(".lang").forEach((button) => {
    button.classList.toggle("active", button.dataset.lang === state.lang);
  });
  if (latestData) {
    statusEl.textContent = statusText(latestData);
    renderMacro(latestData);
    resultsEl.innerHTML = latestData.results.map(renderResult).join("");
  }
}

function statusText(data) {
  const modeLabel = trPhrase(data.modeLabel);
  return state.lang === "ko"
    ? `${modeLabel}: ${data.results.length}${t("resultCount")}`
    : `${modeLabel}: ${data.results.length} ${t("resultCount")}`;
}

function renderMacro(data) {
  const macro = data.macro;
  macroEl.innerHTML = `
    <article class="summary">
      <h2>${t("macroTitle")}</h2>
      <div class="meta">
        <span class="pill">${t("run")} ${escapeHtml(data.generatedAt)}</span>
        <span class="pill">${t("macro")} ${num(macro.score)}</span>
        <span class="pill">${t("risk")} ${num(macro.riskScore)}</span>
      </div>
      <ul class="headlines">
        ${macro.headlines.map((item) => `
          <li>
            ${item.url ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>` : `<span>${escapeHtml(item.title)}</span>`}
            ${item.summary ? `<p class="headline-summary">${escapeHtml(item.summary)}</p>` : ""}
            <div class="source">${escapeHtml(item.source)}</div>
          </li>
        `).join("")}
      </ul>
    </article>
  `;
}

function renderResult(rawRow) {
  const row = hydrateResult(rawRow);
  const actionClass = scoreClass(row.finalScore);
  const showBestBadge = latestData?.mode === "scan" && row.rank === 1;
  const bestBadge = showBestBadge ? `<span class="best-badge">${t("bestCandidate")}</span>` : "";
  return `
    <article class="stock-card">
      <div class="stock-head">
        <div>
          ${bestBadge}
          <div class="ticker">${escapeHtml(row.ticker)}</div>
          <div class="company">${escapeHtml(row.company)}</div>
        </div>
        <div class="action ${actionClass}">${escapeHtml(trPhrase(row.action))}</div>
      </div>

      <div class="score-grid">
        <div class="metric"><span>${t("finalScore")}</span><strong class="${actionClass}">${num(row.finalScore)}</strong></div>
        <div class="metric"><span>${t("rankingScore")}</span><strong>${num(row.rankingScore)}</strong></div>
        <div class="metric"><span>${t("confidence")}</span><strong>${num(row.confidence)}%</strong></div>
        <div class="metric"><span>${t("upsideProb")}</span><strong>${pct(row.estimatedUpsideProbability)}</strong></div>
      </div>

      <div class="forecast-grid">
        <div class="metric"><span>${t("oneDay")}</span><strong>${money(row.prediction.price1d)}</strong><span>${pct(row.prediction.expectedReturn1d, true)} / ${t("up")} ${pct(row.prediction.probabilityUp1d)}</span></div>
        <div class="metric"><span>${t("oneWeek")}</span><strong>${money(row.prediction.price1w)}</strong><span>${pct(row.prediction.expectedReturn1w, true)} / ${t("up")} ${pct(row.prediction.probabilityUp1w)}</span></div>
        <div class="metric"><span>${t("oneYear")}</span><strong>${money(row.prediction.price1y)}</strong><span>${pct(row.prediction.expectedReturn1y, true)} / ${t("up")} ${pct(row.prediction.probabilityUp1y)}</span></div>
      </div>

      <div class="score-grid">
        <div class="metric"><span>${t("setup")}</span><strong>${escapeHtml(trPhrase(row.risingSetupLabel))}</strong></div>
        <div class="metric"><span>${t("entryStyle")}</span><strong>${escapeHtml(row.suggestedEntryStyle || "Wait for better entry")}</strong></div>
        <div class="metric"><span>${t("trend")}</span><strong>${escapeHtml(trPhrase(row.technical.trendLabel))}</strong></div>
        <div class="metric"><span>RSI</span><strong>${num(row.technical.rsi)}</strong></div>
      </div>

      <div class="score-grid">
        <div class="metric"><span>${t("price")}</span><strong>${money(row.technical.price)}</strong></div>
        <div class="metric"><span>${t("newsScore")}</span><strong>${num(row.newsScore)}</strong></div>
        <div class="metric"><span>${t("newsQuality")}</span><strong>${num(row.newsQualityScore)}</strong></div>
        <div class="metric"><span>${t("verifiedCatalyst")}</span><strong>${escapeHtml(row.verifiedCatalyst || "None")}</strong></div>
      </div>

      <div class="score-grid">
        <div class="metric"><span>${t("smallCap")}</span><strong>${num(row.smallCapCatalystScore)}</strong></div>
        <div class="metric"><span>${t("smallCapRisk")}</span><strong>${escapeHtml(row.smallCapRiskLevel || "Low")}</strong></div>
        <div class="metric"><span>${t("pullback")}</span><strong>${escapeHtml(trPhrase(row.technical.pullbackLabel))}</strong></div>
        <div class="metric"><span>${t("monteCarlo")}</span><strong>${pct(row.monteCarlo.probabilityUp20d)}</strong></div>
      </div>

      <div class="meta">
        <span class="pill">${t("sector")} ${escapeHtml(row.sector)}</span>
        <span class="pill">${t("tech")} ${num(row.technical.score)}</span>
        <span class="pill">${t("fund")} ${num(row.fundamentals.score)}</span>
        <span class="pill">${t("mc")} ${num(row.monteCarlo.riskScore)}</span>
        <span class="pill">Vol ${num(row.technical.volumeRatio)}x</span>
      </div>

      <div class="metric narrative">
        <span>${t("whyThis")}</span>
        <p>${escapeHtml(row.recommendationReason || fallbackReason(row, row.risingSetupLabel))}</p>
      </div>

      <div class="metric narrative">
        <span>${t("relatedContext")}</span>
        <p>${escapeHtml(row.backgroundSummary || "Related background context unavailable.")}</p>
      </div>

      <div class="metric narrative">
        <span>${t("adaptivePolicy")}</span>
        <p>${escapeHtml(row.adaptiveEnsemble?.policy || "Maintain current model weights")} · Sim prob ${pct(row.adaptiveEnsemble?.probability)} · Stability ${num(row.adaptiveEnsemble?.stability, 2)}</p>
      </div>

      <div class="metric narrative">
        <span>${t("learningLoop")}</span>
        <p>Evaluated now: ${escapeHtml(row.learningState?.evaluatedNow ?? 0)} · Avg error: ${row.learningState?.lastAverageError == null ? "Waiting for 7-day results" : pct(row.learningState.lastAverageError)}</p>
      </div>

      <div class="metric narrative">
        <span>${t("riskWarning")}</span>
        <p>${escapeHtml(row.mainRiskWarning || fallbackRiskWarning(row, row.smallCapRiskLevel))}</p>
      </div>

      <div class="metric narrative">
        <span>${t("riskManagement")}</span>
        <p>Stop: ${escapeHtml(row.riskManagement?.stopLossArea || "Not enough data")} · Target: ${escapeHtml(row.riskManagement?.takeProfitZone || "Not enough data")} · Size: ${escapeHtml(row.riskManagement?.positionSize || "Not enough data")}</p>
      </div>

      <ul class="headlines">
        ${row.headlines.slice(0, 5).map((item) => `
          <li>
            ${item.url ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>` : `<span>${escapeHtml(item.title)}</span>`}
            ${item.summary ? `<p class="headline-summary">${escapeHtml(item.summary)}</p>` : ""}
            <div class="source">${escapeHtml(item.source)}</div>
          </li>
        `).join("")}
      </ul>
    </article>
  `;
}

async function analyze() {
  const body = {
    mode: state.mode,
    tickers: $("#tickers").value,
    limit: Number($("#limit").value || 5),
  };

  setLoading(true);
  statusEl.textContent = t("loadingStatus");
  macroEl.innerHTML = "";
  resultsEl.innerHTML = "";

  try {
    const response = await fetch(apiUrl("/api/analyze"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || t("errorFallback"));
    latestData = data;
    statusEl.textContent = statusText(data);
    renderMacro(data);
    resultsEl.innerHTML = data.results.map(renderResult).join("");
  } catch (error) {
    statusEl.textContent = `${error.message}. ${t("apiHint")}`;
  } finally {
    setLoading(false);
  }
}

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.mode = button.dataset.mode;
    tickerField.style.display = state.mode === "scan" ? "none" : "grid";
  });
});

document.querySelectorAll(".lang").forEach((button) => {
  button.addEventListener("click", () => {
    state.lang = button.dataset.lang;
    localStorage.setItem("newsQuantLang", state.lang);
    applyLanguage();
  });
});

analyzeButton.addEventListener("click", analyze);
applyLanguage();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/service-worker.js").catch(() => {});
}
