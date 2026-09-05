"""
feeds.py — RSS sources for PRTech / CommsTech digest.
Cast wide as requested — Samuel filters in the app/email.

Feed verification status (June 2026):
  ✅ = confirmed live by direct fetch or authoritative source
  ⚠️ = plausible URL, unconfirmed — logs an empty result if dead, never crashes
"""

# Each entry: (display_name, rss_url, primary_category)
# category values: prtech | commstech | ai-comms | measurement | martech | africa-tech
FEEDS = [
    # --- Samuel's key sources (explicitly requested) ---
    ("TechCabal",              "https://techcabal.com/feed/",                              "africa-tech"),   # ✅ WordPress standard
    ("PRovoke Media (Latest)", "https://www.provokemedia.com/newsfeed/provoke-media-latest","prtech"),        # ✅ confirmed from their RSS page
    ("PRovoke Media (Reads)",  "https://www.provokemedia.com/newsfeed/provoke-media-longreads","prtech"),     # ✅ confirmed from their RSS page
    ("PRWeek (UK)",            "https://www.prweek.com/rss/uk/blog",                       "prtech"),        # ✅ confirmed live
    ("PRWeek (US)",            "https://www.prweek.com/rss/us/blogs",                      "prtech"),        # ✅ confirmed live
    ("Axios",                  "https://www.axios.com/feeds/feed.rss",                     "commstech"),     # ⚠️ main feed only — /technology and /business pages are JS-rendered, no native section RSS
    ("Wadds Inc. (Waddington)","https://wadds.co.uk/?format=rss",                          "prtech"),        # ✅ confirmed — Waddington published this URL himself

    # --- Core PR Tech / Comms Tech press ---
    ("PR Daily",               "https://www.prdaily.com/feed/",                            "prtech"),        # ⚠️ unconfirmed
    ("Ragan Communications",   "https://www.ragan.com/feed/",                              "commstech"),     # ⚠️ unconfirmed
    ("Agility PR",             "https://www.agilitypr.com/pr-news/feed/",                  "prtech"),        # ⚠️ unconfirmed
    ("Muck Rack Blog",         "https://muckrack.com/blog/feed",                           "prtech"),        # ⚠️ unconfirmed
    ("Prowly Magazine",        "https://prowly.com/magazine/feed/",                        "prtech"),        # ⚠️ unconfirmed
    ("SpinSucks",              "https://spinsucks.com/feed/",                               "prtech"),        # ⚠️ unconfirmed
    ("Cision Blog",            "https://www.cision.com/blog/feed/",                        "prtech"),        # ⚠️ unconfirmed
    ("The Drum",               "https://feeds.thedrum.com/rss/latest.rss",                 "commstech"),     # ⚠️ unconfirmed

    # --- AI in comms / martech ---
    ("MarTech",                "https://martech.org/feed/",                                "martech"),       # ⚠️ unconfirmed
    ("Marketing AI Institute", "https://www.marketingaiinstitute.com/blog/rss.xml",        "ai-comms"),      # ⚠️ unconfirmed
    ("Adweek",                 "https://www.adweek.com/feed/",                             "martech"),       # ⚠️ unconfirmed

    # --- Measurement / analytics ---
    ("AMEC",                   "https://amecorg.com/feed/",                                "measurement"),   # ⚠️ unconfirmed
    ("CommsPRO",               "https://www.commspro.com/feed/",                           "commstech"),     # ⚠️ unconfirmed
]

# ── Relevance filtering ───────────────────────────────────────────────────────

# Items from Samuel's key sources bypass relevance filtering entirely —
# everything TechCabal, PRovoke, PRWeek, Axios, and Waddington publish
# is considered in-scope by definition. This also prevents the Africa-tech
# angle of TechCabal from being filtered out by comms-specific keyword logic.
KEY_SOURCE_NAMES = {
    "TechCabal",
    "PRovoke Media (Latest)",
    "PRovoke Media (Reads)",
    "PRWeek (UK)",
    "PRWeek (US)",
    "Axios",
    "Wadds Inc. (Waddington)",
}

# Terms that make an item STRONGLY relevant regardless of source
STRONG_TERMS = [
    "pr tech", "prtech", "comms tech", "commstech",
    "media monitoring", "media intelligence", "press release software",
    "pr software", "pr tool", "pr platform", "pr analytics",
    "communications technology", "pr measurement", "earned media",
    "journalist outreach tool", "media database", "pr automation",
    "ai in pr", "ai in communications", "generative ai pr",
    "reputation management software", "crisis communications tool",
    "muck rack", "cision", "prowly", "meltwater", "brandwatch",
    "talkwalker", "coverage book", "prezly", "agility pr",
    "social listening", "sentiment analysis tool",
    # Africa tech angle (relevant to Samuel's positioning)
    "african tech", "nigeria tech", "africa startup", "african startup",
    "lagos tech", "nairobi tech", "african media", "nigerian media",
]

# Terms that, combined with a PR/comms anchor, make an item relevant
CONTEXT_TERMS = [
    "artificial intelligence", "machine learning", "automation", "saas",
    "platform launch", "api", "integration", "analytics dashboard",
    "measurement framework", "reporting tool", "martech", "adtech",
    "agency technology", "communications software", "newsroom technology",
]

PR_CONTEXT_ANCHORS = [
    "pr", "public relations", "communications", "comms",
    "media", "journalist", "press", "newsroom", "agency",
    "brand", "reputation", "campaign",
]

TECH_SIGNAL_TERMS = [
    "launches", "raises", "acquires", "integrates", "partners",
    "announces", "releases", "unveils", "powered by", "built for",
    "designed for", "new feature", "update", "version",
]


def is_relevant(title: str, summary: str, source_name: str = "") -> tuple[bool, str, int]:
    """
    Returns (is_relevant, category, relevance_score 1-3).
    Score 3 = strong signal, 2 = moderate, 1 = weak but included.

    Items from KEY_SOURCE_NAMES pass through unconditionally at score 2.
    """
    # Key sources: everything passes through
    if source_name in KEY_SOURCE_NAMES:
        text = f"{title} {summary}".lower()
        # Bump to score 3 if a strong term also appears
        if any(t in text for t in STRONG_TERMS):
            return True, "prtech", 3
        return True, "commstech", 2

    text = f"{title} {summary}".lower()

    # Strong match — definitely in
    if any(t in text for t in STRONG_TERMS):
        return True, "prtech", 3

    # Moderate match — PR/comms context + tech signal + context term
    has_pr      = any(t in text for t in PR_CONTEXT_ANCHORS)
    has_tech    = any(t in text for t in TECH_SIGNAL_TERMS)
    has_context = any(t in text for t in CONTEXT_TERMS)

    if has_pr and has_tech and has_context:
        return True, "commstech", 2

    if has_pr and has_context:
        return True, "ai-comms", 1

    return False, "", 0
