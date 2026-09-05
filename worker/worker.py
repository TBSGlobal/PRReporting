"""
worker.py — always-on background worker on Render.com (free tier).

New in this version:
  - Sentiment + country inference via Gemini (once per new article, cached in Supabase)
  - /share endpoint — POST article IDs + recipient emails → branded HTML email via Resend

Environment variables (set in Render dashboard):
  SUPABASE_URL          — base project URL, no trailing slash, no /rest/v1/
  SUPABASE_SERVICE_KEY  — service role key (write access)
  GEMINI_API_KEY        — Google AI Studio key (free tier)
  RESEND_API_KEY        — Resend.com API key (free tier, 100 emails/day)
  RESEND_FROM_EMAIL     — sender address, e.g. digest@thecommsboardroom.com
                          or onboarding@resend.dev for sandbox
  VAPID_PRIVATE_KEY     — for Web Push notifications
  VAPID_PUBLIC_KEY      — for Web Push notifications
  VAPID_EMAIL           — e.g. mailto:samuel@thecommsboardroom.com
  GOOGLE_ALERT_RSS_URL  — your Google Alert RSS feed URL (optional)
"""

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import feedparser
import requests
from pywebpush import webpush, WebPushException

from feeds import FEEDS, is_relevant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

GEMINI_MODEL    = "gemini-3.5-flash-lite"
GEMINI_FALLBACK = "gemini-3.6-flash"
GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE_CODES = {429, 500, 503}

RESEND_API_KEY   = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM      = os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")


# ── Supabase helpers ──────────────────────────────────────────────────────────

def get_seen_links() -> set:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/news_items",
        headers=SUPABASE_HEADERS,
        params={"select": "link", "fetched_at": f"gte.{cutoff}"},
        timeout=15,
    )
    if resp.status_code != 200:
        log.warning(f"Could not fetch seen links: {resp.text[:200]}")
        return set()
    return {row["link"] for row in resp.json()}


def insert_items(items: list) -> int:
    if not items:
        return 0
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/news_items",
        headers={**SUPABASE_HEADERS, "Prefer": "resolution=ignore-duplicates,return=minimal"},
        json=items,
        timeout=20,
    )
    if resp.status_code not in (200, 201):
        log.warning(f"Insert failed ({resp.status_code}): {resp.text[:300]}")
        return 0
    return len(items)


def update_item_sentiment(item_id: str, sentiment: str, reason: str, country: str):
    """Patch a single news_item row with sentiment and country after Gemini analysis."""
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/news_items",
        headers={**SUPABASE_HEADERS, "Prefer": "return=minimal"},
        params={"id": f"eq.{item_id}"},
        json={"sentiment": sentiment, "sentiment_reason": reason, "country": country},
        timeout=10,
    )
    if resp.status_code not in (200, 201, 204):
        log.warning(f"Sentiment update failed for {item_id}: {resp.text[:200]}")


def fetch_items_by_ids(ids: list) -> list:
    """Fetch full news_item rows by a list of IDs for the /share endpoint."""
    if not ids:
        return []
    id_filter = f"in.({','.join(str(i) for i in ids)})"
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/news_items",
        headers=SUPABASE_HEADERS,
        params={"select": "id,title,link,summary,source,category", "id": id_filter},
        timeout=15,
    )
    if resp.status_code != 200:
        log.warning(f"Could not fetch items by IDs: {resp.text[:200]}")
        return []
    return resp.json()


def log_email_share(recipients: list, item_ids: list, subject: str,
                    status: str, error: str = None):
    requests.post(
        f"{SUPABASE_URL}/rest/v1/email_shares",
        headers={**SUPABASE_HEADERS, "Prefer": "return=minimal"},
        json={
            "recipients": recipients,
            "item_ids":   item_ids,
            "item_count": len(item_ids),
            "subject":    subject,
            "status":     status,
            "error":      error,
        },
        timeout=10,
    )


def get_push_subscriptions() -> list:
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/push_subscriptions",
        headers=SUPABASE_HEADERS,
        params={"select": "endpoint,p256dh,auth"},
        timeout=15,
    )
    if resp.status_code != 200:
        return []
    return resp.json()


def delete_subscription(endpoint: str):
    requests.delete(
        f"{SUPABASE_URL}/rest/v1/push_subscriptions",
        headers=SUPABASE_HEADERS,
        params={"endpoint": f"eq.{endpoint}"},
        timeout=10,
    )


# ── Gemini helpers ────────────────────────────────────────────────────────────

def _call_gemini(model: str, prompt: str, schema: dict,
                 max_retries: int = 3) -> tuple:
    """Returns (parsed_json, error_string). One or the other is None."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY not set"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema":   schema,
            "temperature":      0.2,
            "maxOutputTokens":  256,
        },
    }
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{GEMINI_BASE.format(model=model)}?key={api_key}",
                json=payload, timeout=30,
            )
            if resp.status_code in RETRYABLE_CODES:
                wait = 2 ** attempt
                log.warning(f"Gemini {model} {resp.status_code} — retry in {wait}s")
                time.sleep(wait)
                last_error = f"{resp.status_code} retryable"
                continue
            if resp.status_code >= 400:
                return None, f"Gemini error {resp.status_code}: {resp.text[:200]}"
            candidates = resp.json().get("candidates", [])
            if not candidates:
                return None, "No candidates in Gemini response"
            return json.loads(candidates[0]["content"]["parts"][0]["text"]), None
        except Exception as e:
            last_error = str(e)
            time.sleep(2 ** attempt)
    return None, last_error


SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment":        {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "sentiment_reason": {"type": "string"},
        "country":          {"type": "string"},
    },
    "required": ["sentiment", "sentiment_reason", "country"],
}

SENTIMENT_PROMPT = """Analyse this PR/comms/martech news headline and summary.

Title: {title}
Summary: {summary}

Return JSON with:
- sentiment: "positive" (good news for the industry), "negative" (bad news, job cuts, failures),
  or "neutral" (informational, product launches, reports)
- sentiment_reason: one short sentence (max 12 words) explaining the sentiment
- country: the PRIMARY country this news originates from or is most relevant to.
  Use full English country names (e.g. "United Kingdom", "Nigeria", "United States").
  If it's global or unclear, use "Global".

Be conservative — most PRTech news is neutral."""


def infer_sentiment_and_country(item_id: str, title: str, summary: str):
    """Call Gemini to infer sentiment and country for one article. Updates Supabase directly."""
    prompt = SENTIMENT_PROMPT.format(title=title, summary=summary or "")
    result, err = _call_gemini(GEMINI_MODEL, prompt, SENTIMENT_SCHEMA)
    if result is None:
        log.warning(f"Primary model failed for {item_id}: {err} — trying fallback")
        result, err = _call_gemini(GEMINI_FALLBACK, prompt, SENTIMENT_SCHEMA)
    if result is None:
        log.warning(f"Sentiment inference failed for item {item_id}: {err}")
        return
    update_item_sentiment(
        item_id,
        result.get("sentiment", "neutral"),
        result.get("sentiment_reason", ""),
        result.get("country", "Global"),
    )


# ── Push notifications ────────────────────────────────────────────────────────

def send_push_notifications(new_items: list):
    if not new_items:
        return
    subs = get_push_subscriptions()
    if not subs:
        log.info("No push subscribers yet — skipping notification send.")
        return

    vapid_private = os.environ.get("VAPID_PRIVATE_KEY")
    vapid_email   = os.environ.get("VAPID_EMAIL", "mailto:hello@thecommsboardroom.com")
    if not vapid_private:
        log.warning("VAPID_PRIVATE_KEY not set — push notifications disabled.")
        return

    count    = len(new_items)
    top_item = new_items[0]
    payload  = json.dumps({
        "title": f"PRTech Digest — {count} new item{'s' if count > 1 else ''}",
        "body":  top_item["title"][:80],
        "icon":  "/icon-192.png",
        "badge": "/badge-72.png",
        "url":   top_item["link"],
        "tag":   "prtech-digest",
    })

    dead = []
    for sub in subs:
        try:
            webpush(
                subscription_info={"endpoint": sub["endpoint"],
                                   "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]}},
                data=payload,
                vapid_private_key=vapid_private,
                vapid_claims={"sub": vapid_email},
            )
        except WebPushException as e:
            if "410" in str(e) or "404" in str(e):
                dead.append(sub["endpoint"])
            else:
                log.warning(f"Push send failed: {e}")

    for ep in dead:
        delete_subscription(ep)
        log.info(f"Removed expired subscription: {ep[:40]}...")

    log.info(f"Push sent to {len(subs) - len(dead)} subscriber(s).")


# ── Email share builder ───────────────────────────────────────────────────────

def build_share_email_html(items: list, note: str = "") -> str:
    """Build a branded HTML email for the /share endpoint."""
    from html import escape

    items_html = ""
    for item in items:
        cat   = item.get("category") or ""
        src   = item.get("source") or ""
        title = item.get("title") or "Untitled"
        link  = item.get("link") or "#"
        summ  = item.get("summary") or ""
        items_html += f"""
        <div style="margin-bottom:20px;padding:16px;background:#f8f9ff;
                    border-left:3px solid #5379F6;border-radius:6px;">
          <div style="margin-bottom:6px;">
            <span style="background:#5379F6;color:#fff;font-size:10px;font-weight:700;
                         padding:2px 8px;border-radius:20px;text-transform:uppercase;
                         letter-spacing:0.05em;">{escape(cat)}</span>
            <span style="color:#888;font-size:11px;margin-left:8px;">{escape(src)}</span>
          </div>
          <h3 style="margin:0 0 8px;font-size:15px;font-family:Montserrat,sans-serif;">
            <a href="{escape(link)}" style="color:#131836;text-decoration:none;">{escape(title)}</a>
          </h3>
          <p style="margin:0;font-size:13px;color:#555;line-height:1.5;">
            {escape(summ[:250])}{"…" if len(summ) > 250 else ""}
          </p>
          <a href="{escape(link)}" style="display:inline-block;margin-top:10px;font-size:12px;
             color:#5379F6;font-weight:600;">Read full article →</a>
        </div>"""

    note_html = (f'<p style="background:#fffbea;border-left:3px solid #C9A84C;'
                 f'padding:12px 16px;border-radius:4px;font-size:13px;color:#555;'
                 f'margin-bottom:24px;">{escape(note)}</p>') if note else ""

    return f"""
    <div style="font-family:Inter,Arial,sans-serif;max-width:620px;margin:0 auto;color:#131836;">
      <div style="background:#131836;padding:24px;border-radius:10px 10px 0 0;">
        <h1 style="color:#fff;margin:0;font-size:20px;font-family:Montserrat,sans-serif;">
          PRTech Digest
        </h1>
        <p style="color:#5379F6;margin:4px 0 0;font-size:12px;">
          by The Comms Boardroom — curated PRTech &amp; CommsTech intelligence
        </p>
      </div>
      <div style="border:1px solid #e0e4f0;border-top:none;padding:24px;
                  border-radius:0 0 10px 10px;">
        <p style="font-size:14px;color:#444;margin:0 0 20px;">
          Here are <strong>{len(items)} PRTech &amp; CommsTech stories</strong> selected for you:
        </p>
        {note_html}
        {items_html}
        <hr style="border:none;border-top:1px solid #e0e4f0;margin:24px 0;">
        <p style="font-size:11px;color:#999;margin:0;">
          Sent via PRTech Digest by
          <a href="https://thecommsboardroom.com" style="color:#5379F6;">
            The Comms Boardroom</a>.
          Africa's PRTech &amp; CommsTech intelligence feed.
        </p>
      </div>
    </div>"""


def send_share_email(recipients: list, item_ids: list,
                     subject: str, note: str) -> tuple:
    """Fetch items, build email, send via Resend. Returns (success_bool, error_str)."""
    if not RESEND_API_KEY:
        return False, "RESEND_API_KEY not configured"
    if not recipients:
        return False, "No recipients provided"
    if not item_ids:
        return False, "No item IDs provided"

    items = fetch_items_by_ids(item_ids)
    if not items:
        return False, "Could not fetch selected articles from Supabase"

    html = build_share_email_html(items, note)
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}",
                 "Content-Type": "application/json"},
        json={"from":    RESEND_FROM,
              "to":      recipients,
              "subject": subject or f"PRTech Digest — {len(items)} stories for you",
              "html":    html},
        timeout=30,
    )
    if resp.status_code >= 300:
        return False, f"Resend error {resp.status_code}: {resp.text[:300]}"
    return True, None


# ── Main feed fetch ───────────────────────────────────────────────────────────

def run_fetch():
    log.info("=== Feed fetch run starting ===")
    seen   = get_seen_links()
    log.info(f"Loaded {len(seen)} known links for dedup.")

    all_feeds  = list(FEEDS)
    google_url = os.environ.get("GOOGLE_ALERT_RSS_URL")
    if google_url:
        all_feeds.append(("Google Alert (PRTech)", google_url, "prtech"))

    new_items = []
    cutoff    = datetime.now(timezone.utc) - timedelta(hours=25)

    for source_name, url, base_category in all_feeds:
        try:
            parsed = feedparser.parse(url)
            if not parsed.entries:
                log.info(f"  {source_name}: empty or unreachable")
                continue

            count = 0
            for entry in parsed.entries:
                link = entry.get("link", "").strip()
                if not link or link in seen:
                    continue

                title   = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", ""))[:500]

                pub_date = None
                for key in ("published_parsed", "updated_parsed"):
                    val = entry.get(key)
                    if val:
                        pub_date = datetime(*val[:6], tzinfo=timezone.utc)
                        break

                if pub_date and pub_date < cutoff:
                    continue

                relevant, category, score = is_relevant(title, summary, source_name)
                if not relevant:
                    continue

                new_items.append({
                    "source":         source_name,
                    "title":          title,
                    "link":           link,
                    "summary":        summary,
                    "published_at":   pub_date.isoformat() if pub_date else None,
                    "category":       category or base_category,
                    "relevance_score": score,
                })
                seen.add(link)
                count += 1

            log.info(f"  {source_name}: {count} new relevant items")

        except Exception as e:
            log.warning(f"  {source_name}: error — {e}")

    inserted = insert_items(new_items)
    log.info(f"Inserted {inserted} new items into Supabase.")

    # Infer sentiment + country for each new item immediately after insert.
    # We fetch the inserted rows back to get their Supabase-assigned IDs.
    if new_items:
        try:
            recent_cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/news_items",
                headers=SUPABASE_HEADERS,
                params={
                    "select":     "id,title,summary",
                    "fetched_at": f"gte.{recent_cutoff}",
                    "sentiment":  "is.null",   # only items not yet analysed
                    "order":      "fetched_at.desc",
                    "limit":      str(len(new_items) + 5),
                },
                timeout=15,
            )
            if resp.status_code == 200:
                rows = resp.json()
                log.info(f"Running sentiment inference on {len(rows)} new items...")
                for row in rows:
                    infer_sentiment_and_country(
                        str(row["id"]), row["title"], row.get("summary") or ""
                    )
                    time.sleep(0.5)  # gentle rate-limit on Gemini free tier
            else:
                log.warning(f"Could not fetch new items for sentiment: {resp.text[:200]}")
        except Exception as e:
            log.warning(f"Sentiment inference batch failed: {e}")

        send_push_notifications(new_items)

    log.info("=== Run complete ===")
    return len(new_items)



# ── Sentiment backfill ────────────────────────────────────────────────────────

def run_sentiment_backfill(batch_size: int = 50) -> int:
    """
    Process all existing news_items rows that have NULL sentiment.
    Called via GET /backfill — safe to run multiple times.
    Processes in batches of batch_size with a short sleep between items
    to stay within Gemini free-tier rate limits.
    """
    log.info("=== Sentiment backfill starting ===")
    total = 0

    while True:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/news_items",
            headers=SUPABASE_HEADERS,
            params={
                "select":    "id,title,summary",
                "sentiment": "is.null",
                "order":     "fetched_at.desc",
                "limit":     str(batch_size),
            },
            timeout=15,
        )
        if resp.status_code != 200:
            log.warning(f"Backfill fetch failed: {resp.text[:200]}")
            break

        rows = resp.json()
        if not rows:
            break

        log.info(f"  Backfilling {len(rows)} rows...")
        for row in rows:
            infer_sentiment_and_country(
                str(row["id"]), row["title"], row.get("summary") or ""
            )
            time.sleep(0.8)  # gentle rate limit
            total += 1

        if len(rows) < batch_size:
            break  # last batch

    log.info(f"=== Backfill complete — {total} items processed ===")
    return total

# ── HTTP server ───────────────────────────────────────────────────────────────

def _read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if not length:
        return {}
    try:
        return json.loads(handler.rfile.read(length))
    except Exception:
        return {}


def _json_response(handler, status: int, data: dict):
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress default access log noise

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/run":
            count = run_fetch()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"Done. {count} new items.".encode())
        elif self.path == "/backfill":
            count = run_sentiment_backfill()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"Backfill done. {count} items processed.".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        # ── /subscribe — PWA push subscription registration ──────────────────
        if self.path == "/subscribe":
            body = _read_body(self)
            sub  = body.get("subscription", {})
            ep   = sub.get("endpoint")
            keys = sub.get("keys", {})
            if ep and keys.get("p256dh") and keys.get("auth"):
                resp = requests.post(
                    f"{SUPABASE_URL}/rest/v1/push_subscriptions",
                    headers={**SUPABASE_HEADERS,
                             "Prefer": "resolution=ignore-duplicates,return=minimal"},
                    json={"endpoint": ep, "p256dh": keys["p256dh"], "auth": keys["auth"]},
                    timeout=10,
                )
                _json_response(self, 200 if resp.status_code in (200, 201) else 500,
                               {"ok": resp.status_code in (200, 201)})
            else:
                _json_response(self, 400, {"error": "Missing subscription fields"})

        # ── /share — email selected articles to a list of recipients ─────────
        elif self.path == "/share":
            body       = _read_body(self)
            recipients = body.get("recipients", [])  # list of email strings
            item_ids   = body.get("item_ids", [])    # list of news_item IDs
            subject    = body.get("subject", "")
            note       = body.get("note", "")        # optional personal note from Samuel

            if not recipients or not item_ids:
                _json_response(self, 400,
                               {"error": "recipients and item_ids are required"})
                return

            success, error = send_share_email(recipients, item_ids, subject, note)
            log_email_share(recipients, item_ids, subject,
                            "sent" if success else "failed", error)

            if success:
                log.info(f"Share email sent to {len(recipients)} recipient(s), "
                         f"{len(item_ids)} items.")
                _json_response(self, 200, {"ok": True, "sent_to": len(recipients)})
            else:
                log.warning(f"Share email failed: {error}")
                _json_response(self, 500, {"ok": False, "error": error})

        else:
            self.send_response(404)
            self.end_headers()


def run_server():
    port   = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    log.info(f"HTTP server listening on port {port}")
    server.serve_forever()


def run_loop():
    """Background thread: fetch immediately, then every hour."""
    run_fetch()
    while True:
        time.sleep(3600)
        run_fetch()


if __name__ == "__main__":
    # Start HTTP server FIRST so Render's health check passes immediately.
    # Feed fetching (including Gemini sentiment calls) runs in the background.
    Thread(target=run_loop, daemon=True).start()
    run_server()  # blocks forever — keeps the process alive
