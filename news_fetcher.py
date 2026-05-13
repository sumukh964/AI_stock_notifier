import os
import re
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime
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
# CONFIG — TUNED THRESHOLDS
# ─────────────────────────────────────────────
NEWS_API_KEY  = os.getenv("NEWS_API_KEY")
BOT_TOKEN     = os.getenv("BOT_TOKEN")
CHAT_ID       = os.getenv("CHAT_ID")
ALERTS_LOG    = Path("docs/alerts.json")

MAX_ARTICLES   = 60
MAX_SIGNALS    = 5          # max alerts per run to avoid spam

# Single article thresholds
SINGLE_RISE_CONF  = 0.80    # 1 article: needs very high confidence to signal RISE
SINGLE_FALL_CONF  = 0.80    # 1 article: needs very high confidence to signal FALL

# Multi article thresholds (2+ articles) — more relaxed
MULTI_RISE_CONF   = 0.65    # avg positive score across articles
MULTI_FALL_CONF   = 0.65    # avg negative score across articles
MULTI_RISE_RATIO  = 0.55    # at least 55% of articles must be positive
MULTI_FALL_RATIO  = 0.55    # at least 55% of articles must be negative

BEARISH_SUPPRESS  = 0.60    # >60% negative headlines → suppress RISE signals

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

NOISE_WORDS = {
    "india", "rbi", "government", "market", "economy", "ministry",
    "sebi", "nse", "bse", "sensex", "nifty", "rupee", "gdp",
    "fed", "us", "china", "europe", "budget", "parliament",
    "court", "supreme", "finance", "banking", "sector", "investors",
    "traders", "analyst", "analysts", "sources", "report", "reuters",
    "bloomberg", "press", "trust", "bharat", "corp", "corporation",
    "markets", "exchange", "board", "authority", "committee", "fund",
    "quarter", "fiscal", "year", "rate", "policy", "inflation",
    "stock", "share", "shares", "equity", "index", "indices",
}

SCRAPE_SOURCES = [
    {
        "name": "Economic Times",
        "url":  "https://economictimes.indiatimes.com/markets/stocks/news",
        "tag":  "a",   "class_": "eachStory",       "headline_tag": "h3",
    },
    {
        "name": "Moneycontrol",
        "url":  "https://www.moneycontrol.com/news/business/markets/",
        "tag":  "li",  "class_": "clearfix",        "headline_tag": "h2",
    },
    {
        "name": "LiveMint",
        "url":  "https://www.livemint.com/market/stock-market-news",
        "tag":  "div", "class_": "listingNew",      "headline_tag": "h2",
    },
    {
        "name": "Business Standard",
        "url":  "https://www.business-standard.com/markets",
        "tag":  "h2",  "class_": "headline",        "headline_tag": None,
    },
    {
        "name": "NDTV Profit",
        "url":  "https://www.ndtvprofit.com/markets",
        "tag":  "h2",  "class_": "story__headline", "headline_tag": None,
    },
]


# ─────────────────────────────────────────────
# A. SCRAPING (PRIMARY)
# ─────────────────────────────────────────────
def scrape_headlines() -> list[dict]:
    articles: list[dict] = []
    seen:     set[str]   = set()

    for src in SCRAPE_SOURCES:
        try:
            resp = requests.get(src["url"], headers=HEADERS, timeout=12)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            containers = soup.find_all(src["tag"], class_=src["class_"])
            count = 0
            for container in containers:
                if src["headline_tag"]:
                    tag  = container.find(src["headline_tag"])
                    text = tag.get_text(strip=True) if tag else ""
                else:
                    text = container.get_text(strip=True)

                text = text.strip()
                if not text or text in seen or len(text) < 25:
                    continue

                seen.add(text)
                articles.append({
                    "title":     text,
                    "full_text": text,
                    "source":    src["name"],
                })
                count += 1

            log.info(f"✅ {src['name']} → {count} headlines")

        except requests.exceptions.Timeout:
            log.warning(f"⏱ Timeout: {src['name']}")
        except Exception as exc:
            log.warning(f"⚠ {src['name']}: {exc}")

    log.info(f"📰 Scraped: {len(articles)} total headlines")
    return articles


# ─────────────────────────────────────────────
# B. NEWSAPI (BONUS — free tier endpoint)
# ─────────────────────────────────────────────
def fetch_newsapi() -> list[dict]:
    if not NEWS_API_KEY:
        return []

    articles: list[dict] = []
    seen:     set[str]   = set()

    try:
        resp = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "country":  "in",
                "category": "business",
                "pageSize": 30,
                "apiKey":   NEWS_API_KEY,
            },
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "ok":
            for art in data.get("articles", []):
                title = (art.get("title") or "").strip()
                desc  = (art.get("description") or "").strip()
                if not title or title in seen or "[Removed]" in title:
                    continue
                seen.add(title)
                articles.append({
                    "title":     title,
                    "full_text": f"{title}. {desc}".strip(),
                    "source":    art.get("source", {}).get("name", "NewsAPI"),
                })
            log.info(f"✅ NewsAPI → {len(articles)} articles")
        else:
            log.warning(f"NewsAPI: {data.get('message','error')}")

    except Exception as exc:
        log.warning(f"⚠ NewsAPI: {exc}")

    return articles


# ─────────────────────────────────────────────
# C. MERGE ALL
# ─────────────────────────────────────────────
def collect_news() -> list[dict]:
    scraped  = scrape_headlines()
    api_news = fetch_newsapi()

    seen:   set[str]   = set()
    merged: list[dict] = []

    for art in scraped + api_news:
        key = art["title"].lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            merged.append(art)

    final = merged[:MAX_ARTICLES]
    log.info(f"📦 Final pool: {len(final)} articles")
    return final


# ─────────────────────────────────────────────
# D. FINBERT ENGINE
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
        top    = max(scores, key=scores.get)
        return {
            "sentiment":      top,
            "confidence":     round(scores[top], 4),
            "positive_score": round(scores["positive"], 4),
            "negative_score": round(scores["negative"], 4),
            "neutral_score":  round(scores["neutral"],  4),
        }


# ─────────────────────────────────────────────
# E. COMPANY EXTRACTION
# ─────────────────────────────────────────────
def load_nlp() -> spacy.Language:
    try:
        nlp = spacy.load("en_core_web_sm")
        log.info("✅ spaCy loaded.")
        return nlp
    except OSError:
        log.error("Run: python -m spacy download en_core_web_sm")
        raise


def is_valid_company(name: str) -> bool:
    clean = name.strip().lower()
    if len(clean) < 3:                    return False
    if clean in NOISE_WORDS:              return False
    if re.fullmatch(r"[\d\s\W]+", clean): return False
    for noise in ["government", "ministry", "court", "reserve bank",
                  "prime minister", "lok sabha", "rajya sabha"]:
        if noise in clean:                return False
    return True


def extract_companies(text: str, nlp: spacy.Language) -> list[str]:
    doc  = nlp(text)
    seen = set()
    out  = []
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT"):
            canonical = re.sub(
                r"\b(Ltd\.?|Limited|Corp\.?|Corporation|Inc\.?|"
                r"Pvt\.?|Private|Group|Holdings?|Ventures?|Capital)\b",
                "", ent.text, flags=re.IGNORECASE
            ).strip().rstrip(".,")
            if canonical and is_valid_company(canonical) and canonical.lower() not in seen:
                seen.add(canonical.lower())
                out.append(canonical)
    return out


# ─────────────────────────────────────────────
# F. PREDICTION ENGINE — RELAXED THRESHOLDS
# ─────────────────────────────────────────────
def predict(records: list[dict]) -> dict:
    total     = len(records)
    positives = [r for r in records if r["sentiment"] == "positive"]
    negatives = [r for r in records if r["sentiment"] == "negative"]
    pos_ratio = len(positives) / total
    neg_ratio = len(negatives) / total
    pos_avg   = sum(r["positive_score"] for r in records) / total
    neg_avg   = sum(r["negative_score"] for r in records) / total
    weighted  = sum(r["positive_score"] - r["negative_score"] for r in records) / total

    # ── Single article: needs high confidence ──────────────────────
    if total == 1:
        r = records[0]
        if r["sentiment"] == "positive" and r["positive_score"] >= SINGLE_RISE_CONF:
            return {
                "prediction":    "RISE",
                "confidence":    round(r["positive_score"], 4),
                "reasoning":     f"Strong positive signal · confidence {round(r['positive_score']*100)}%",
                "pos_ratio":     1.0, "neg_ratio": 0.0, "article_count": 1,
            }
        if r["sentiment"] == "negative" and r["negative_score"] >= SINGLE_FALL_CONF:
            return {
                "prediction":    "FALL",
                "confidence":    round(r["negative_score"], 4),
                "reasoning":     f"Strong negative signal · confidence {round(r['negative_score']*100)}%",
                "pos_ratio":     0.0, "neg_ratio": 1.0, "article_count": 1,
            }
        return {
            "prediction": "NEUTRAL", "confidence": round(max(pos_avg, neg_avg), 4),
            "reasoning": f"Single article below threshold",
            "pos_ratio": pos_ratio, "neg_ratio": neg_ratio, "article_count": total,
        }

    # ── Multiple articles: relaxed thresholds ──────────────────────
    if pos_ratio >= MULTI_RISE_RATIO and pos_avg >= MULTI_RISE_CONF and weighted > 0.10:
        return {
            "prediction":    "RISE",
            "confidence":    round(pos_avg, 4),
            "reasoning":     f"{len(positives)}/{total} positive · avg score +{weighted:.2f}",
            "pos_ratio":     round(pos_ratio, 2),
            "neg_ratio":     round(neg_ratio, 2),
            "article_count": total,
        }
    if neg_ratio >= MULTI_FALL_RATIO and neg_avg >= MULTI_FALL_CONF and weighted < -0.10:
        return {
            "prediction":    "FALL",
            "confidence":    round(neg_avg, 4),
            "reasoning":     f"{len(negatives)}/{total} negative · avg score {weighted:.2f}",
            "pos_ratio":     round(pos_ratio, 2),
            "neg_ratio":     round(neg_ratio, 2),
            "article_count": total,
        }

    return {
        "prediction":    "NEUTRAL",
        "confidence":    round(max(pos_avg, neg_avg), 4),
        "reasoning":     f"Mixed: {len(positives)} pos / {len(negatives)} neg / {total - len(positives) - len(negatives)} neu",
        "pos_ratio":     round(pos_ratio, 2),
        "neg_ratio":     round(neg_ratio, 2),
        "article_count": total,
    }


# ─────────────────────────────────────────────
# G. MARKET MOOD
# ─────────────────────────────────────────────
def market_mood(results: list[dict]) -> str:
    if not results: return "Neutral"
    n   = len(results)
    neg = sum(1 for r in results if r["sentiment"] == "negative") / n
    pos = sum(1 for r in results if r["sentiment"] == "positive") / n
    if neg > BEARISH_SUPPRESS:  return "Bearish"
    if pos > 0.55:              return "Bullish"
    return "Neutral"


# ─────────────────────────────────────────────
# H. SAVE ALERT
# ─────────────────────────────────────────────
def save_alert(data: dict) -> None:
    try:
        ALERTS_LOG.parent.mkdir(exist_ok=True)
        alerts = []
        if ALERTS_LOG.exists():
            with open(ALERTS_LOG) as f:
                alerts = json.load(f)
        alerts.insert(0, data)
        with open(ALERTS_LOG, "w") as f:
            json.dump(alerts, f, indent=2)
        log.info(f"💾 Saved → {ALERTS_LOG}")
    except Exception as exc:
        log.error(f"Save failed: {exc}")


# ─────────────────────────────────────────────
# I. TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(msg: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        log.warning("⚠ BOT_TOKEN/CHAT_ID not set")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        log.info("📨 Telegram sent.")
    except Exception as exc:
        log.error(f"Telegram failed: {exc}")


def rise_msg(company, pred, headline, source, mood) -> str:
    c   = round(pred["confidence"] * 100)
    bar = "█" * (c // 10) + "░" * (10 - c // 10)
    arts = pred["article_count"]
    return (
        f"📈 <b>AI STOCK BUY SIGNAL</b>\n\n"
        f"🏢 <b>Company:</b> {company}\n"
        f"🔮 <b>Prediction:</b> RISE ▲\n"
        f"📊 <b>Confidence:</b> {c}%\n"
        f"     <code>{bar}</code>\n"
        f"📰 <b>Articles:</b> {arts} ({round(pred['pos_ratio']*100)}% positive)\n"
        f"🧠 <b>Reasoning:</b> {pred['reasoning']}\n"
        f"💡 <b>Top Headline:</b> {headline}\n"
        f"📡 <b>Source:</b> {source}\n"
        f"🌐 <b>Market Mood:</b> {mood}\n"
        f"⏰ {datetime.now().strftime('%d %b %Y %H:%M')} IST"
    )


def fall_msg(company, pred, headline, source, mood) -> str:
    c   = round(pred["confidence"] * 100)
    bar = "█" * (c // 10) + "░" * (10 - c // 10)
    arts = pred["article_count"]
    return (
        f"📉 <b>AI STOCK FALL SIGNAL</b>\n\n"
        f"🏢 <b>Company:</b> {company}\n"
        f"🔮 <b>Prediction:</b> FALL ▼\n"
        f"📊 <b>Confidence:</b> {c}%\n"
        f"     <code>{bar}</code>\n"
        f"📰 <b>Articles:</b> {arts} ({round(pred['neg_ratio']*100)}% negative)\n"
        f"🧠 <b>Reasoning:</b> {pred['reasoning']}\n"
        f"💡 <b>Top Headline:</b> {headline}\n"
        f"📡 <b>Source:</b> {source}\n"
        f"🌐 <b>Market Mood:</b> {mood}\n"
        f"⏰ {datetime.now().strftime('%d %b %Y %H:%M')} IST"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    start = time.time()
    now   = datetime.now()

    log.info("=" * 65)
    log.info("  AI STOCK INTELLIGENCE v4 — RELAXED THRESHOLDS")
    log.info("=" * 65)

    engine = SentimentEngine()
    nlp    = load_nlp()

    articles = collect_news()
    if not articles:
        msg = "⚠ <b>System Error</b>\nCould not fetch any news today. Check GitHub Actions log."
        send_telegram(msg)
        save_alert({"type":"no_signal","date":now.strftime("%Y-%m-%d"),
                    "time":now.strftime("%H:%M"),"market_mood":"Unknown"})
        return

    # ── Analyse ──────────────────────────────────────────────────────
    company_articles: dict[str, list[dict]] = defaultdict(list)
    all_sentiments:   list[dict]            = []

    for art in articles:
        result = engine.analyse(art["full_text"])
        result.update(art)
        all_sentiments.append(result)

        for company in extract_companies(art["full_text"], nlp):
            company_articles[company].append(result)

        log.info(
            f"  [{result['sentiment'].upper():8s}] {result['confidence']:.2f} | "
            f"{art['title'][:72]}"
        )

    log.info(f"\n📊 Companies detected: {len(company_articles)}")
    mood = market_mood(all_sentiments)
    log.info(f"🌐 Market Mood: {mood}")

    # ── Predict + rank by confidence ────────────────────────────────
    predictions = []
    for company, records in company_articles.items():
        pred = predict(records)
        if pred["prediction"] in ("RISE", "FALL"):
            top      = max(records, key=lambda r: r["confidence"])
            predictions.append((company, pred, top))
            log.info(
                f"  ✦ {company:35s} → {pred['prediction']:5s} "
                f"conf={round(pred['confidence']*100)}% | {pred['reasoning']}"
            )
        else:
            log.info(
                f"  · {company:35s} → NEUTRAL conf={round(pred['confidence']*100)}%"
            )

    # Sort by confidence, highest first
    predictions.sort(key=lambda x: x[1]["confidence"], reverse=True)

    # ── Send top N signals ───────────────────────────────────────────
    signals_sent = 0
    alerted      = set()

    for company, pred, top in predictions:
        if signals_sent >= MAX_SIGNALS:
            break
        if company in alerted:
            continue

        headline = top["title"]
        source   = top.get("source", "")

        if pred["prediction"] == "RISE":
            if mood == "Bearish":
                log.info(f"  ↳ {company} RISE suppressed (Bearish market)")
                continue
            send_telegram(rise_msg(company, pred, headline, source, mood))
            save_alert({
                "type":          "rise",
                "date":          now.strftime("%Y-%m-%d"),
                "time":          now.strftime("%H:%M"),
                "company":       company,
                "confidence":    str(round(pred["confidence"] * 100)),
                "article_count": str(pred["article_count"]),
                "pos_ratio":     str(round(pred["pos_ratio"] * 100)),
                "reasoning":     pred["reasoning"],
                "top_headline":  headline,
                "top_source":    source,
                "market_mood":   mood,
            })

        elif pred["prediction"] == "FALL":
            send_telegram(fall_msg(company, pred, headline, source, mood))
            save_alert({
                "type":          "fall",
                "date":          now.strftime("%Y-%m-%d"),
                "time":          now.strftime("%H:%M"),
                "company":       company,
                "confidence":    str(round(pred["confidence"] * 100)),
                "article_count": str(pred["article_count"]),
                "neg_ratio":     str(round(pred["neg_ratio"] * 100)),
                "reasoning":     pred["reasoning"],
                "top_headline":  headline,
                "top_source":    source,
                "market_mood":   mood,
            })

        alerted.add(company)
        signals_sent += 1

    # ── No signals ───────────────────────────────────────────────────
    if signals_sent == 0:
        send_telegram(
            f"🔕 <b>No strong signals today.</b>\n"
            f"Market Mood: <b>{mood}</b>\n"
            f"Articles analysed: {len(articles)}\n"
            f"Companies found: {len(company_articles)}\n"
            f"All predictions below confidence threshold."
        )
        save_alert({
            "type":              "no_signal",
            "date":              now.strftime("%Y-%m-%d"),
            "time":              now.strftime("%H:%M"),
            "market_mood":       mood,
            "articles_analysed": len(articles),
            "companies_found":   len(company_articles),
        })

    elapsed = round(time.time() - start, 2)
    log.info(f"\n✅ Done in {elapsed}s · Signals sent: {signals_sent}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
