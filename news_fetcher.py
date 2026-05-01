"""
AI Pre-Market Stock Intelligence System — v2
=============================================
Author  : ML Engineer Portfolio Project
Changes : NewsAPI → dynamic company extraction → Rise/Fall prediction
          No fixed whitelist — analyses ANY company found in the news.
"""

import os
import re
import json
import time
import logging
import requests
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import spacy

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NEWS_API_KEY   = os.getenv("NEWS_API_KEY")       # from GitHub Secret
BOT_TOKEN      = os.getenv("BOT_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")
ALERTS_LOG     = Path("docs/alerts.json")

MAX_ARTICLES            = 50    # fetch up to 50 articles from NewsAPI
MIN_HEADLINES_FOR_SIGNAL = 2    # need at least 2 headlines per company to signal
BUY_CONFIDENCE_THRESHOLD = 0.75 # aggregated avg confidence for RISE signal
SELL_CONFIDENCE_THRESHOLD= 0.75 # aggregated avg confidence for FALL signal
BEARISH_MARKET_RATIO     = 0.50 # >50% negative headlines → suppress RISE signals

# Generic noise words to ignore as company names
NOISE_WORDS = {
    "india", "rbi", "government", "market", "economy", "ministry",
    "sebi", "nse", "bse", "sensex", "nifty", "rupee", "gdp",
    "fed", "us", "china", "europe", "budget", "parliament",
    "court", "supreme", "finance", "banking", "sector", "investors",
    "traders", "analyst", "analysts", "sources", "report", "reuters",
    "bloomberg", "press", "trust", "india", "bharat", "ltd", "limited",
    "corp", "corporation", "group", "holdings", "ventures", "capital",
    "markets", "exchange", "board", "authority", "committee", "fund",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────
# A. NEWS COLLECTION — NewsAPI
# ─────────────────────────────────────────────
def fetch_news() -> list[dict]:
    """
    Fetch Indian business/stock news from NewsAPI.
    Returns list of {title, description, url, source, published_at}
    """
    if not NEWS_API_KEY:
        log.error("NEWS_API_KEY not set. Cannot fetch news.")
        return []

    # Date range: last 24 hours
    from_date = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

    queries = [
        "India stock market NSE BSE shares",
        "India company earnings profit quarterly results",
        "India business acquisition merger deal",
    ]

    all_articles: list[dict] = []
    seen_titles:  set[str]   = set()

    for q in queries:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":          q,
                    "from":       from_date,
                    "language":   "en",
                    "sortBy":     "relevancy",
                    "pageSize":   20,
                    "apiKey":     NEWS_API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            for article in data.get("articles", []):
                title = (article.get("title") or "").strip()
                desc  = (article.get("description") or "").strip()

                if not title or title in seen_titles:
                    continue
                if "[Removed]" in title:
                    continue

                seen_titles.add(title)
                all_articles.append({
                    "title":        title,
                    "description":  desc,
                    "url":          article.get("url", ""),
                    "source":       article.get("source", {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt", ""),
                    "full_text":    f"{title}. {desc}".strip(),
                })

            log.info(f"✅ NewsAPI query '{q[:40]}' → {len(data.get('articles',[]))} articles")

        except requests.exceptions.RequestException as exc:
            log.warning(f"⚠ NewsAPI error for query '{q}': {exc}")

    articles = all_articles[:MAX_ARTICLES]
    log.info(f"📰 Total unique articles: {len(articles)}")
    return articles


# ─────────────────────────────────────────────
# B. FINBERT SENTIMENT ENGINE
# ─────────────────────────────────────────────
class SentimentEngine:
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"🔧 Loading FinBERT on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.labels = ["positive", "negative", "neutral"]
        log.info("✅ FinBERT loaded.")

    @torch.no_grad()
    def analyse(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs  = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
        scores = dict(zip(self.labels, probs))

        top_label  = max(scores, key=scores.get)
        confidence = round(scores[top_label], 4)

        return {
            "sentiment":        top_label,
            "confidence":       confidence,
            "positive_score":   round(scores["positive"], 4),
            "negative_score":   round(scores["negative"], 4),
            "neutral_score":    round(scores["neutral"],  4),
        }


# ─────────────────────────────────────────────
# C. DYNAMIC COMPANY EXTRACTION (no whitelist)
# ─────────────────────────────────────────────
def load_nlp() -> spacy.Language:
    try:
        nlp = spacy.load("en_core_web_sm")
        log.info("✅ spaCy model loaded.")
        return nlp
    except OSError:
        log.error("Run: python -m spacy download en_core_web_sm")
        raise


def is_valid_company(name: str) -> bool:
    """
    Filter out generic/noise entities.
    Accept proper named companies (2+ chars, not a noise word, not all digits).
    """
    clean = name.strip().lower()

    if len(clean) < 3:
        return False
    if clean in NOISE_WORDS:
        return False
    if any(noise in clean for noise in ["government", "ministry", "court", "reserve bank"]):
        return False
    if re.fullmatch(r"[\d\s\W]+", clean):
        return False

    return True


def extract_companies(text: str, nlp: spacy.Language) -> list[str]:
    """Extract all valid ORG entities from text. Returns list of canonical names."""
    doc = nlp(text)
    companies = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT"):
            name = ent.text.strip()
            # Clean up suffixes for canonical name
            canonical = re.sub(
                r"\b(Ltd\.?|Limited|Corp\.?|Corporation|Inc\.?|Pvt\.?|Private|Group|Holdings?|Ventures?|Capital)\b",
                "", name, flags=re.IGNORECASE
            ).strip().rstrip(".,")

            if canonical and is_valid_company(canonical) and canonical.lower() not in seen:
                seen.add(canonical.lower())
                companies.append(canonical)

    return companies


# ─────────────────────────────────────────────
# D. RISE / FALL PREDICTION ENGINE
# ─────────────────────────────────────────────
def predict_stock_movement(records: list[dict]) -> dict:
    """
    Aggregate multiple headline sentiments for one company.

    Returns:
        prediction: RISE | FALL | NEUTRAL
        confidence: float (0–1)
        reasoning:  str
    """
    if not records:
        return {"prediction": "NEUTRAL", "confidence": 0.0, "reasoning": "No data"}

    total     = len(records)
    positives = [r for r in records if r["sentiment"] == "positive"]
    negatives = [r for r in records if r["sentiment"] == "negative"]

    # Weighted score: +positive_score - negative_score for each article
    weighted_scores = [
        r["positive_score"] - r["negative_score"]
        for r in records
    ]
    avg_weighted = sum(weighted_scores) / total

    # Avg confidence of the dominant sentiment
    pos_avg_conf = sum(r["positive_score"] for r in records) / total
    neg_avg_conf = sum(r["negative_score"] for r in records) / total

    pos_ratio = len(positives) / total
    neg_ratio = len(negatives) / total

    # Decision logic
    if avg_weighted > 0.15 and pos_ratio >= 0.6 and pos_avg_conf >= BUY_CONFIDENCE_THRESHOLD:
        prediction = "RISE"
        confidence = round(pos_avg_conf, 4)
        reasoning  = (
            f"{len(positives)}/{total} articles positive · "
            f"avg sentiment score +{avg_weighted:.2f}"
        )

    elif avg_weighted < -0.15 and neg_ratio >= 0.6 and neg_avg_conf >= SELL_CONFIDENCE_THRESHOLD:
        prediction = "FALL"
        confidence = round(neg_avg_conf, 4)
        reasoning  = (
            f"{len(negatives)}/{total} articles negative · "
            f"avg sentiment score {avg_weighted:.2f}"
        )

    else:
        prediction = "NEUTRAL"
        confidence = round(max(pos_avg_conf, neg_avg_conf), 4)
        reasoning  = (
            f"Mixed signals ({len(positives)} pos / {len(negatives)} neg / "
            f"{total - len(positives) - len(negatives)} neu)"
        )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "reasoning":  reasoning,
        "pos_ratio":  round(pos_ratio, 2),
        "neg_ratio":  round(neg_ratio, 2),
        "article_count": total,
    }


# ─────────────────────────────────────────────
# E. MARKET MOOD
# ─────────────────────────────────────────────
def overall_market_mood(all_results: list[dict]) -> str:
    if not all_results:
        return "Neutral"
    neg = sum(1 for r in all_results if r["sentiment"] == "negative")
    pos = sum(1 for r in all_results if r["sentiment"] == "positive")
    n   = len(all_results)
    if neg / n > BEARISH_MARKET_RATIO:
        return "Bearish"
    if pos / n > BEARISH_MARKET_RATIO:
        return "Bullish"
    return "Neutral"


# ─────────────────────────────────────────────
# F. SAVE TO docs/alerts.json
# ─────────────────────────────────────────────
def save_alert(data: dict) -> None:
    try:
        ALERTS_LOG.parent.mkdir(exist_ok=True)
        alerts = []
        if ALERTS_LOG.exists():
            with open(ALERTS_LOG, "r") as f:
                alerts = json.load(f)
        alerts.insert(0, data)
        with open(ALERTS_LOG, "w") as f:
            json.dump(alerts, f, indent=2)
        log.info(f"💾 Saved to {ALERTS_LOG}")
    except Exception as exc:
        log.error(f"Failed to save: {exc}")


# ─────────────────────────────────────────────
# G. TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(message: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        log.warning("⚠ BOT_TOKEN / CHAT_ID not set.")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("📨 Telegram sent.")
    except requests.exceptions.RequestException as exc:
        log.error(f"Telegram failed: {exc}")


def format_rise_alert(company: str, pred: dict, top_headline: str, mood: str) -> str:
    conf_pct = round(pred["confidence"] * 100)
    bar      = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)
    return (
        f"📈 <b>STOCK RISE PREDICTION</b>\n\n"
        f"🏢 <b>Company:</b> {company}\n"
        f"🔮 <b>Prediction:</b> <b>RISE ▲</b>\n"
        f"📊 <b>Confidence:</b> {conf_pct}%\n"
        f"     <code>{bar}</code>\n"
        f"📰 <b>Articles analysed:</b> {pred['article_count']} "
        f"({round(pred['pos_ratio']*100)}% positive)\n"
        f"🧠 <b>Reasoning:</b> {pred['reasoning']}\n"
        f"💡 <b>Top Headline:</b> {top_headline}\n"
        f"🌐 <b>Market Mood:</b> {mood}\n"
        f"⏰ <b>Signal Time:</b> {datetime.now().strftime('%d %b %Y %H:%M')} IST"
    )


def format_fall_alert(company: str, pred: dict, top_headline: str, mood: str) -> str:
    conf_pct = round(pred["confidence"] * 100)
    bar      = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)
    return (
        f"📉 <b>STOCK FALL PREDICTION</b>\n\n"
        f"🏢 <b>Company:</b> {company}\n"
        f"🔮 <b>Prediction:</b> <b>FALL ▼</b>\n"
        f"📊 <b>Confidence:</b> {conf_pct}%\n"
        f"     <code>{bar}</code>\n"
        f"📰 <b>Articles analysed:</b> {pred['article_count']} "
        f"({round(pred['neg_ratio']*100)}% negative)\n"
        f"🧠 <b>Reasoning:</b> {pred['reasoning']}\n"
        f"💡 <b>Top Headline:</b> {top_headline}\n"
        f"🌐 <b>Market Mood:</b> {mood}\n"
        f"⏰ <b>Signal Time:</b> {datetime.now().strftime('%d %b %Y %H:%M')} IST"
    )


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main() -> None:
    start = time.time()
    now   = datetime.now()

    log.info("=" * 65)
    log.info("  AI PRE-MARKET STOCK INTELLIGENCE SYSTEM v2 — STARTING")
    log.info("=" * 65)

    # ── Load models ──────────────────────────────────────────────────
    engine = SentimentEngine()
    nlp    = load_nlp()

    # ── Fetch news ───────────────────────────────────────────────────
    articles = fetch_news()
    if not articles:
        log.warning("No articles fetched.")
        send_telegram("⚠ No news articles could be fetched today.")
        save_alert({"type": "no_signal", "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M"), "market_mood": "Unknown"})
        return

    # ── Analyse each article + extract companies ──────────────────────
    company_articles: dict[str, list[dict]] = defaultdict(list)
    all_sentiments:   list[dict]            = []

    for article in articles:
        text   = article["full_text"]
        result = engine.analyse(text)
        result.update(article)       # merge sentiment + article metadata
        all_sentiments.append(result)

        companies = extract_companies(text, nlp)
        for company in companies:
            company_articles[company].append(result)

        log.info(
            f"  [{result['sentiment'].upper():8s}] {result['confidence']:.2f} | "
            f"{article['title'][:70]}"
        )

    log.info(f"\n📊 Companies detected: {len(company_articles)}")
    for c, arts in company_articles.items():
        log.info(f"    {c:35s} → {len(arts)} article(s)")

    # ── Market mood ───────────────────────────────────────────────────
    mood = overall_market_mood(all_sentiments)
    log.info(f"\n🌐 Overall Market Mood: {mood}")

    # ── Predict per company ───────────────────────────────────────────
    log.info("\n── PREDICTIONS ──")
    signals_sent = 0
    alerted      = set()

    for company, records in company_articles.items():

        # Need at least MIN_HEADLINES_FOR_SIGNAL articles to make a call
        if len(records) < MIN_HEADLINES_FOR_SIGNAL:
            log.info(f"  {company:35s} → SKIP (only {len(records)} article)")
            continue

        pred = predict_stock_movement(records)
        log.info(
            f"  {company:35s} → {pred['prediction']:7s} "
            f"conf={pred['confidence']:.2f} | {pred['reasoning']}"
        )

        if company in alerted:
            continue

        # Top headline = highest confidence article for this company
        top = max(records, key=lambda r: r["confidence"])
        top_headline = top["title"]

        # ── RISE signal ───────────────────────────────────────────────
        if pred["prediction"] == "RISE":
            # Suppress if market is Bearish
            if mood == "Bearish":
                log.info(f"    ↳ RISE suppressed (Bearish market)")
                continue

            msg = format_rise_alert(company, pred, top_headline, mood)
            send_telegram(msg)
            save_alert({
                "type":          "rise",
                "date":          now.strftime("%Y-%m-%d"),
                "time":          now.strftime("%H:%M"),
                "company":       company,
                "prediction":    "RISE",
                "confidence":    str(round(pred["confidence"] * 100)),
                "article_count": str(pred["article_count"]),
                "pos_ratio":     str(round(pred["pos_ratio"] * 100)),
                "reasoning":     pred["reasoning"],
                "top_headline":  top_headline,
                "top_source":    top.get("source", ""),
                "market_mood":   mood,
            })
            alerted.add(company)
            signals_sent += 1

        # ── FALL signal ───────────────────────────────────────────────
        elif pred["prediction"] == "FALL":
            msg = format_fall_alert(company, pred, top_headline, mood)
            send_telegram(msg)
            save_alert({
                "type":          "fall",
                "date":          now.strftime("%Y-%m-%d"),
                "time":          now.strftime("%H:%M"),
                "company":       company,
                "prediction":    "FALL",
                "confidence":    str(round(pred["confidence"] * 100)),
                "article_count": str(pred["article_count"]),
                "neg_ratio":     str(round(pred["neg_ratio"] * 100)),
                "reasoning":     pred["reasoning"],
                "top_headline":  top_headline,
                "top_source":    top.get("source", ""),
                "market_mood":   mood,
            })
            alerted.add(company)
            signals_sent += 1

    # ── No signals ────────────────────────────────────────────────────
    if signals_sent == 0:
        send_telegram(
            f"🔕 <b>No strong signals today.</b>\n"
            f"Market mood: <b>{mood}</b>\n"
            f"Articles analysed: {len(articles)}\n"
            f"Companies found: {len(company_articles)}\n"
            f"Staying cautious — no high-confidence RISE/FALL predictions."
        )
        save_alert({
            "type":        "no_signal",
            "date":        now.strftime("%Y-%m-%d"),
            "time":        now.strftime("%H:%M"),
            "market_mood": mood,
            "articles_analysed": len(articles),
            "companies_found":   len(company_articles),
        })

    elapsed = round(time.time() - start, 2)
    log.info(f"\n✅ Done in {elapsed}s · Signals sent: {signals_sent}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
