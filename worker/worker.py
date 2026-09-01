"""
worker.py — the always-on background worker that runs on Render.com (free tier).

Render keeps this alive as a web service with a simple health endpoint.
A separate cron (configured in Render dashboard OR via the /run endpoint below)
triggers the actual fetch every 60 minutes.

Environment variables needed (set in Render dashboard):
  SUPABASE_URL          — your Supabase project URL
  SUPABASE_SERVICE_KEY  — service role key (not anon key — needs write access)
  VAPID_PRIVATE_KEY     — for sending Web Push notifications
  VAPID_PUBLIC_KEY      — also needed for push
  VAPID_EMAIL           — e.g. mailto:samuel@thecommsboardroom.com
  GOOGLE_ALERT_RSS_URL  — your Google Alert RSS feed (PRTech/CommsTech alert)
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
from py_vapid import Vapid
from pywebpush import webpush, WebPushException

from feeds import FEEDS, is_relevant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}


# ── Supabase helpers ──────────────────────────────────────────────────────────

def get_seen_links() -> set[str]:
    """Fetch all known links from the last 7 days to use as dedup set."""
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


def insert_items(items: list[dict]) -> int:
    """Bulk insert new items. Returns count actually inserted."""
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


def get_push_subscriptions() -> list[dict]:
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


# ── Push notification sender ──────────────────────────────────────────────────

def send_push_notifications(new_items: list[dict]):
    """Send a push notification to all subscribed devices for each batch of new items."""
    if not new_items:
        return

    subs = get_push_subscriptions()
    if not subs:
        log.info("No push subscribers yet — skipping notification send.")
        return

    vapid_private = os.environ.get("VAPID_PRIVATE_KEY")
    vapid_email = os.environ.get("VAPID_EMAIL", "mailto:hello@thecommsboardroom.com")

    if not vapid_private:
        log.warning("VAPID_PRIVATE_KEY not set — push notifications disabled.")
        return

    # Build a single notification payload for the batch
    count = len(new_items)
    top_item = new_items[0]
    payload = json.dumps({
        "title": f"PRTech Digest — {count} new item{'s' if count > 1 else ''}",
        "body": top_item["title"][:80],
        "icon": "/icon-192.png",
        "badge": "/badge-72.png",
        "url": top_item["link"],
        "tag": "prtech-digest",  # collapses multiple notifications into one
    })

    dead_endpoints = []
    for sub in subs:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
                },
                data=payload,
                vapid_private_key=vapid_private,
                vapid_claims={"sub": vapid_email},
            )
        except WebPushException as e:
            if "410" in str(e) or "404" in str(e):
                # Subscription expired/revoked — clean it up
                dead_endpoints.append(sub["endpoint"])
            else:
                log.warning(f"Push send failed: {e}")

    for ep in dead_endpoints:
        delete_subscription(ep)
        log.info(f"Removed expired subscription: {ep[:40]}...")

    log.info(f"Push sent to {len(subs) - len(dead_endpoints)} subscriber(s).")


# ── Main fetch-and-store logic ────────────────────────────────────────────────

def run_fetch():
    log.info("=== Feed fetch run starting ===")
    seen = get_seen_links()
    log.info(f"Loaded {len(seen)} known links for dedup.")

    all_feeds = list(FEEDS)
    google_alert = os.environ.get("GOOGLE_ALERT_RSS_URL")
    if google_alert:
        all_feeds.append(("Google Alert (PRTech)", google_alert, "prtech"))

    new_items = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=25)

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

                title = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", ""))[:500]

                # Parse date
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
                    "source": source_name,
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published_at": pub_date.isoformat() if pub_date else None,
                    "category": category or base_category,
                    "relevance_score": score,
                })
                seen.add(link)
                count += 1

            log.info(f"  {source_name}: {count} new relevant items")

        except Exception as e:
            log.warning(f"  {source_name}: error — {e}")

    inserted = insert_items(new_items)
    log.info(f"Inserted {inserted} new items into Supabase.")

    if new_items:
        send_push_notifications(new_items)

    log.info("=== Run complete ===")
    return len(new_items)


# ── Simple HTTP server (keeps Render.com happy + provides health endpoint) ────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress default access logging

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/run":
            # Manual trigger endpoint — also called by Render's cron
            count = run_fetch()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"Done. {count} new items.".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        # Handle push subscription registration from the PWA
        if self.path == "/subscribe":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            sub = body.get("subscription", {})
            endpoint = sub.get("endpoint")
            keys = sub.get("keys", {})

            if endpoint and keys.get("p256dh") and keys.get("auth"):
                resp = requests.post(
                    f"{SUPABASE_URL}/rest/v1/push_subscriptions",
                    headers={**SUPABASE_HEADERS, "Prefer": "resolution=ignore-duplicates,return=minimal"},
                    json={
                        "endpoint": endpoint,
                        "p256dh": keys["p256dh"],
                        "auth": keys["auth"],
                    },
                    timeout=10,
                )
                self.send_response(200 if resp.status_code in (200, 201) else 500)
            else:
                self.send_response(400)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def run_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    log.info(f"HTTP server listening on port {port}")
    server.serve_forever()


if __name__ == "__main__":
    # Run an immediate fetch on startup
    run_fetch()

    # Start HTTP server in background
    Thread(target=run_server, daemon=False).start()

    # Then loop — fetch every 60 minutes
    while True:
        time.sleep(3600)
        run_fetch()
