import os
import time
import logging
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

import spacy

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────
MAX_HEADLINES = 30
BUY_CONFIDENCE_THRESHOLD   = 0.80   # multi-news aggregated avg
SINGLE_BUY_THRESHOLD       = 0.85   # single-headline BUY
HOLD_LOWER_THRESHOLD       = 0.65   # single-headline HOLD
BEARISH_MARKET_RATIO       = 0.50   # >50 % negative → bearish

NSE_WHITELIST = {
    "reliance", "tcs", "tata consultancy", "infosys", "hdfc bank",
    "icici bank", "sbi", "state bank", "axis bank", "l&t",
    "larsen", "toubro", "itc", "bharti airtel", "airtel",
    "hul", "hindustan unilever", "wipro", "tata motors",
    "maruti", "sun pharma", "ntpc", "power grid",
    "adani enterprises", "adani ports", "bajaj finance",
}

# Canonical display names mapped from possible NLP extractions
COMPANY_CANONICAL = {
    "tata consultancy": "TCS",
    "tcs": "TCS",
    "reliance": "Reliance Industries",
    "infosys": "Infosys",
    "hdfc bank": "HDFC Bank",
    "icici bank": "ICICI Bank",
    "state bank": "SBI",
    "sbi": "SBI",
    "axis bank": "Axis Bank",
    "larsen": "L&T",
    "toubro": "L&T",
    "l&t": "L&T",
    "itc": "ITC",
    "bharti airtel": "Bharti Airtel",
    "airtel": "Bharti Airtel",
    "hindustan unilever": "HUL",
    "hul": "HUL",
    "wipro": "Wipro",
    "tata motors": "Tata Motors",
    "maruti": "Maruti Suzuki",
    "sun pharma": "Sun Pharma",
    "ntpc": "NTPC",
    "power grid": "Power Grid",
    "adani enterprises": "Adani Enterprises",
    "adani ports": "Adani Ports",
    "bajaj finance": "Bajaj Finance",
}

NOISE_ENTITIES = {
    "india", "rbi", "government", "market", "economy", "ministry",
    "sebi", "nse", "bse", "sensex", "nifty", "rupee", "gdp",
    "fed", "us", "china", "europe", "budget", "parliament",
}

NEWS_SOURCES = [
    {
        "name": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/stocks/news",
        "tag": "a",
        "class_": "eachStory",
        "headline_tag": "h3",
    },
    {
        "name": "Moneycontrol Markets",
        "url": "https://www.moneycontrol.com/news/business/markets/",
        "tag": "li",
        "class_": "clearfix",
        "headline_tag": "h2",
    },
    {
        "name": "LiveMint Markets",
        "url": "https://www.livemint.com/market/stock-market-news",
        "tag": "div",
        "class_": "listingNew",
        "headline_tag": "h2",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")


# ─────────────────────────────────────────────
# A. NEWS COLLECTION
# ─────────────────────────────────────────────
def fetch_headlines() -> list[str]:
    """Scrape headlines from multiple Indian financial news sources."""
    all_headlines: list[str] = []

    for source in NEWS_SOURCES:
        try:
            response = requests.get(source["url"], headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            containers = soup.find_all(source["tag"], class_=source["class_"])
            for container in containers:
                tag = container.find(source["headline_tag"])
                if tag and tag.get_text(strip=True):
                    all_headlines.append(tag.get_text(strip=True))

            log.info(f"✅ Fetched from {source['name']}")

        except requests.exceptions.Timeout:
            log.warning(f"⏱ Timeout fetching {source['name']}")
        except requests.exceptions.RequestException as exc:
            log.warning(f"⚠ Error fetching {source['name']}: {exc}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for h in all_headlines:
        if h not in seen:
            seen.add(h)
            unique.append(h)

    headlines = unique[:MAX_HEADLINES]
    log.info(f"📰 Total unique headlines collected: {len(headlines)}")
    return headlines


# ─────────────────────────────────────────────
# B. FINBERT SENTIMENT ENGINE
# ─────────────────────────────────────────────
class SentimentEngine:
    """Loads FinBERT once and exposes per-headline inference."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"🔧 Loading FinBERT on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.labels = ["positive", "negative", "neutral"]   # FinBERT label order
        log.info("✅ FinBERT loaded.")

    @torch.no_grad()
    def analyse(self, headline: str) -> dict:
        """Return sentiment label, confidence, and raw action."""
        inputs = self.tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs  = F.softmax(logits, dim=-1).squeeze().cpu().tolist()

        scores = dict(zip(self.labels, probs))
        top_label = max(scores, key=scores.get)
        confidence = scores[top_label]

        action = self._decide(top_label, confidence)
        return {"sentiment": top_label, "confidence": round(confidence, 4), "action": action}

    @staticmethod
    def _decide(sentiment: str, confidence: float) -> str:
        if sentiment == "positive":
            if confidence >= SINGLE_BUY_THRESHOLD:
                return "BUY"
            if confidence >= HOLD_LOWER_THRESHOLD:
                return "HOLD"
        if sentiment == "negative":
            return "SELL"
        return "HOLD"


# ─────────────────────────────────────────────
# C. SMART COMPANY DETECTION
# ─────────────────────────────────────────────
def load_nlp() -> spacy.Language:
    try:
        nlp = spacy.load("en_core_web_sm")
        log.info("✅ spaCy model loaded.")
        return nlp
    except OSError:
        log.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise


def extract_company(headline: str, nlp: spacy.Language) -> str | None:
    """Extract the first whitelisted NSE company from a headline."""
    doc = nlp(headline)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            name_lower = ent.text.strip().lower()
            if name_lower in NOISE_ENTITIES:
                continue
            for key in NSE_WHITELIST:
                if key in name_lower or name_lower in key:
                    return COMPANY_CANONICAL.get(key, ent.text.strip())
    return None


# ─────────────────────────────────────────────
# D. MULTI-NEWS AGGREGATION
# ─────────────────────────────────────────────
def aggregate_signals(
    company_data: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    For each company with multiple headlines, compute:
      - avg_confidence
      - majority sentiment
      - final action
    """
    aggregated: dict[str, dict] = {}

    for company, records in company_data.items():
        positives = [r for r in records if r["sentiment"] == "positive"]
        negatives = [r for r in records if r["sentiment"] == "negative"]
        avg_conf  = round(sum(r["confidence"] for r in records) / len(records), 4)

        if len(positives) > len(negatives) and avg_conf >= BUY_CONFIDENCE_THRESHOLD:
            final_action = "BUY"
        elif len(negatives) >= len(positives):
            final_action = "SELL"
        else:
            final_action = "HOLD"

        # Best BUY headline as "top reason"
        best = max(records, key=lambda r: r["confidence"] if r["sentiment"] == "positive" else 0)

        aggregated[company] = {
            "avg_confidence": avg_conf,
            "final_action": final_action,
            "news_count": len(records),
            "positive_count": len(positives),
            "top_headline": best["headline"],
        }

    return aggregated


# ─────────────────────────────────────────────
# E. MARKET CONTEXT FILTER
# ─────────────────────────────────────────────
def market_mood(all_results: list[dict]) -> str:
    """Return 'Bearish', 'Bullish', or 'Neutral' based on overall headlines."""
    if not all_results:
        return "Neutral"
    neg_count = sum(1 for r in all_results if r["sentiment"] == "negative")
    ratio = neg_count / len(all_results)
    if ratio > BEARISH_MARKET_RATIO:
        return "Bearish"
    pos_count = sum(1 for r in all_results if r["sentiment"] == "positive")
    if pos_count / len(all_results) > BEARISH_MARKET_RATIO:
        return "Bullish"
    return "Neutral"


# ─────────────────────────────────────────────
# G. TELEGRAM ALERTS
# ─────────────────────────────────────────────
def send_telegram(message: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        log.warning("⚠ BOT_TOKEN or CHAT_ID not set. Skipping Telegram alert.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        log.info("📨 Telegram alert sent.")
    except requests.exceptions.RequestException as exc:
        log.error(f"Telegram send failed: {exc}")


def format_buy_alert(company: str, data: dict, mood: str) -> str:
    return (
        f"📈 <b>AI STOCK BUY SIGNAL</b>\n\n"
        f"🏢 <b>Company:</b> {company}\n"
        f"📊 <b>Confidence:</b> {data['avg_confidence']:.0%}\n"
        f"📰 <b>News Count:</b> {data['news_count']} headline(s) "
        f"({data['positive_count']} positive)\n"
        f"💡 <b>Top Reason:</b> {data['top_headline']}\n"
        f"🌐 <b>Market Mood:</b> {mood}"
    )


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main() -> None:
    start = time.time()
    log.info("=" * 60)
    log.info("  AI PRE-MARKET STOCK INTELLIGENCE SYSTEM — STARTING")
    log.info("=" * 60)

    # ── Load models ──────────────────────────
    engine = SentimentEngine()
    nlp    = load_nlp()

    # ── Fetch news ───────────────────────────
    headlines = fetch_headlines()
    if not headlines:
        log.warning("No headlines fetched. Exiting.")
        send_telegram("⚠ No headlines fetched today. System could not run analysis.")
        return

    # ── Analyse each headline ─────────────────
    all_results:    list[dict]                    = []
    company_data:   dict[str, list[dict]]         = defaultdict(list)
    alerted_set:    set[str]                      = set()   # F. duplicate protection

    for headline in headlines:
        result = engine.analyse(headline)
        result["headline"] = headline
        all_results.append(result)

        company = extract_company(headline, nlp)
        if company:
            company_data[company].append(result)

        log.info(
            f"  [{result['sentiment'].upper():8s}] conf={result['confidence']:.2f} | {headline[:80]}"
        )

    # ── Market context ────────────────────────
    mood = market_mood(all_results)
    log.info(f"\n🌐 Market Mood: {mood}")

    # ── Aggregate & decide ────────────────────
    aggregated = aggregate_signals(dict(company_data))

    log.info("\n── COMPANY SIGNAL SUMMARY ──")
    buy_alerts: list[str] = []

    for company, data in aggregated.items():
        action = data["final_action"]

        # E. Bearish market → suppress BUY
        if mood == "Bearish" and action == "BUY":
            action = "HOLD"
            log.info(f"  {company:30s} | conf={data['avg_confidence']:.2f} | BUY → HOLD (bearish market)")
        else:
            log.info(f"  {company:30s} | conf={data['avg_confidence']:.2f} | {action}")

        # F. Duplicate guard + send alert
        if action == "BUY" and company not in alerted_set:
            alerted_set.add(company)
            buy_alerts.append(format_buy_alert(company, data, mood))

    # ── Send Telegram ─────────────────────────
if buy_alerts:
    for alert in buy_alerts:
        send_telegram(alert)
else:
    send_telegram(
        "🔕 <b>No strong BUY signals today.</b>\n"
        f"Market sentiment: <b>{mood}</b>. Staying cautious."
    )

# ─────────────────────────────────────────
# 🔥 NEW: SAVE DATA FOR DASHBOARD
# ─────────────────────────────────────────
import json

dashboard_data = {
    "date": time.strftime("%d %b %Y"),
    "market_mood": mood,
    "buy_signals": len(buy_alerts),
    "avg_confidence": round(
        sum(d["avg_confidence"] for d in aggregated.values()) / len(aggregated)
        if aggregated else 0, 2
    ),
    "signals": []
}

for company, data in aggregated.items():
    dashboard_data["signals"].append({
        "name": company,
        "sector": "N/A",
        "confidence": int(data["avg_confidence"] * 100),
        "action": data["final_action"].lower(),
        "reason": data["top_headline"]
    })

with open("data.json", "w") as f:
    json.dump(dashboard_data, f, indent=4)

log.info("📊 data.json updated for dashboard")

# ── END ───────────────────────────────────
elapsed = round(time.time() - start, 2)
log.info(f"\n✅ Pipeline complete in {elapsed}s. Alerts sent: {len(buy_alerts)}")
log.info("=" * 60)
