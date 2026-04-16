import os
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# VADER runs locally — loaded once, reused across all sentiment calls
_vader = SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> dict:
    """
    Scores a piece of text as bullish, bearish, or neutral using VADER.

    VADER returns a compound score from -1.0 (most negative) to +1.0 (most positive).
    We apply standard financial thresholds:
      compound > 0.05  → bullish
      compound < -0.05 → bearish
      otherwise        → neutral

    Why VADER and not a more sophisticated model?
    - It runs locally with no API cost
    - It was designed for short, informal text (news headlines are ideal)
    - Fast enough to score dozens of articles instantly
    - Good enough for headline-level signal; we're not trading on this alone
    """
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "bullish"
        emoji = "🟢"
    elif compound <= -0.05:
        label = "bearish"
        emoji = "🔴"
    else:
        label = "neutral"
        emoji = "⚪"

    return {
        "label": label,
        "emoji": emoji,
        "compound": round(compound, 3),
    }

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Load once at module level — reused across calls
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast, good quality
    return _model


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


def _ensure_table():
    """Creates the news_articles table if it doesn't exist."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id          SERIAL PRIMARY KEY,
                    ticker      TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    description TEXT,
                    url         TEXT,
                    published_at TIMESTAMPTZ,
                    embedding   vector(384),
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(ticker, url)
                );
            """)
            # Index for fast cosine similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS news_embedding_idx
                ON news_articles
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 10);
            """)
        conn.commit()


def fetch_and_store_news(ticker: str) -> int:
    """
    Fetches recent news headlines for a ticker from NewsAPI,
    embeds them with sentence-transformers, and upserts into pgvector.

    Returns the number of new articles stored.
    """
    _ensure_table()

    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # Search for articles mentioning the ticker in the last 7 days.
    #
    # Why exclude_domains?
    # - yahoo.com / finance.yahoo.com: redirects through consent.yahoo.com
    #   (GDPR cookie wall) — links open to "please try again later"
    # - msn.com: aggregates content behind a Microsoft login redirect
    # These domains consistently produce broken links, so we exclude them
    # and let NewsAPI surface articles from the original publishers instead.
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    response = newsapi.get_everything(
        q=ticker,
        from_param=from_date,
        language="en",
        sort_by="relevancy",
        page_size=20,
        exclude_domains="yahoo.com,finance.yahoo.com,msn.com",
    )

    articles = response.get("articles", [])
    if not articles:
        print(f"No news found for {ticker}")
        return 0

    model = _get_model()

    # Build text to embed: title + description gives richer signal than title alone
    texts = []
    valid_articles = []
    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        if not title or title == "[Removed]":
            continue
        texts.append(f"{title}. {description}".strip())
        valid_articles.append(article)

    if not texts:
        return 0

    embeddings = model.encode(texts, normalize_embeddings=True)  # cosine = dot product on normalized

    stored = 0
    with _get_conn() as conn:
        with conn.cursor() as cur:
            for article, embedding, text in zip(valid_articles, embeddings, texts):
                url = article.get("url", "")
                published_raw = article.get("publishedAt")
                published_at = None
                if published_raw:
                    try:
                        published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                    except ValueError:
                        pass

                # ON CONFLICT DO NOTHING — safe to re-run without duplicates
                cur.execute("""
                    INSERT INTO news_articles (ticker, title, description, url, published_at, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (ticker, url) DO NOTHING
                """, (
                    ticker,
                    article.get("title", ""),
                    article.get("description", ""),
                    url,
                    published_at,
                    embedding.tolist()
                ))
                if cur.rowcount > 0:
                    stored += 1
        conn.commit()

    print(f"Stored {stored} new articles for {ticker} (skipped {len(valid_articles) - stored} duplicates)")
    return stored


def retrieve_relevant_news(ticker: str, query: str, top_k: int = 5) -> list:
    """
    Embeds the query and retrieves the top_k most relevant news articles
    for the given ticker using cosine similarity in pgvector.

    Returns a list of dicts: {title, description, url, published_at, relevance_score}
    """
    _ensure_table()

    model = _get_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    title,
                    description,
                    url,
                    published_at,
                    1 - (embedding <=> %s::vector) AS relevance_score
                FROM news_articles
                WHERE ticker = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                query_embedding.tolist(),
                ticker,
                query_embedding.tolist(),
                top_k
            ))
            rows = cur.fetchall()

    results = []
    for row in rows:
        title, description, url, published_at, relevance_score = row
        # Score sentiment on title + description combined — more signal than title alone
        sentiment = score_sentiment(f"{title}. {description or ''}")
        results.append({
            "title": title,
            "description": description or "",
            "url": url or "",
            "published_at": str(published_at.date()) if published_at else "unknown",
            "relevance_score": round(float(relevance_score), 4),
            "sentiment": sentiment,
        })

    return results


if __name__ == "__main__":
    ticker = "AAPL"

    print(f"\n--- Fetching news for {ticker} ---")
    count = fetch_and_store_news(ticker)
    print(f"Total new articles stored: {count}")

    print(f"\n--- Retrieving relevant news ---")
    query = f"{ticker} stock price movement earnings forecast"
    articles = retrieve_relevant_news(ticker, query, top_k=5)

    for i, a in enumerate(articles, 1):
        print(f"\n{i}. [{a['published_at']}] {a['title']}")
        print(f"   Score: {a['relevance_score']}")
        print(f"   {a['description'][:120]}..." if len(a['description']) > 120 else f"   {a['description']}")
