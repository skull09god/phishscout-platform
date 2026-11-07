#!/usr/bin/env python3
"""PhishScout v2 - URL feature extractor.

Extends the v1 35-feature set with signals that directly attack the
documented v1 limitations:
  - bare trusted-domain phishing (brand-in-subdomain, brand-on-free-host)
  - homoglyph / IDN homograph attacks
  - typosquatting (edit-distance to brands)
  - TLD tricks (suspicious free TLDs, .com-typos like .con/.cm)
  - free-hosting abuse (ngrok, pages.dev, vercel.app, ...)
  - open-redirect / encoded-URL tricks
  - multilingual & regional brand coverage

All features are computed from the URL string alone -> keeps the core model
fully on-device and offline. The feature set is intentionally compact and
interpretable so the resulting LightGBM stays well under 1 MB.
"""
from __future__ import annotations

import re
import math
from collections import Counter
from urllib.parse import urlparse

try:
    from Levenshtein import distance as _lev
except Exception:  # pragma: no cover
    def _lev(a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                cur = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = cur
        return dp[n]


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "banking", "password", "webscr",
    "delivery", "invoice", "wallet", "authenticate", "unlock", "recover",
    "suspended", "limited", "alert", "dispute", "refund", "billing", "bonus",
    "prize", "lottery", "free", "support", "help", "centre", "center", "service",
    "official", "auth", "otp", "token", "reset", "activation", "validate",
    "credential", "session", "expire", "reactivate", "blocked", "kyc",
]

BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "live",
    "google", "gmail", "yahoo", "amazon", "netflix", "facebook", "fb", "instagram",
    "ig", "whatsapp", "messenger", "chase", "wellsfargo", "citibank",
    "bankofamerica", "bofa", "hsbc", "barclays", "amex", "americanexpress", "visa",
    "mastercard", "coinbase", "binance", "steam", "spotify", "dropbox", "dhl",
    "fedex", "usps", "ups", "payoneer", "samsung", "twitter", "x", "linkedin",
    "tiktok", "telegram", "discord", "roblox", "epicgames", "nvidia", "adobe",
    "skype", "ebay", "booking", "airbnb", "uber", "lyft", "t-mobile", "tmobile",
    "verizon", "att", "spectrum", "docuSign", "docussign", "oracle", "ibm",
    "salesforce", "shopify", "stripe", "webroot", "norton", "mcafee", "axa",
    "santander", "natwest", "lloyds", "halifax", "monzo", "revolut", "n26",
    "bpi", "caixabank", "bbva", "ing", "bancosantander", "posteitaliane",
    "creditagricole", "bnp", "societegenerale", "sparkasse", "commerzbank",
    "deutschebank", "rabobank", "abn", "ingdirect", "sberbank", "tinkoff",
    "sbi", "hdfcbank", "icicibank", "axisbank", "paytm", "phonepe", "gpay",
    "bhim", "yuibank", "zenith", "gtbank", "accessbank", "firstbank", "uob",
    "dbs", "ocbc", "maybank", "bca", "bri", "mandiri", "tokopedia", "shopee",
    "grab", "gojek", "line", "rakuten", "softbank", "docomo", "momo", "zalo",
    "vietcombank", "techcombank", "mbbank", "naver", "kakao", "kbank", "scb",
    "truewallet", "btc", "eth", "metamask", "trustwallet", "blockchain",
    "cryptocom", "kucoin", "okx", "bybit", "pancakeswap", "opensea", "uniswap",
    "robinhood", "webull", "etrade", "fidelity", "schwab", "vanguard",
]

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "click", "country", "link",
    "ru", "cn", "su", "pw", "cc", "ws", "nu", "st", "rest", "racing", "loan",
    "trade", "stream", "download", "science", "gdn", "host", "press", "fit",
    "review", "icu", "buzz", "monster", "cyou", "shop", "vip", "work", "live",
    "online", "site", "website", "fun", "games", "space",
}

FREE_HOSTS = {
    "ngrok.io", "ngrok-free.app", "herokuapp.com", "github.io", "pages.dev",
    "vercel.app", "netlify.app", "netlify.com", "repl.co", "repl.it",
    "web.app", "appspot.com", "r.appspot.com", "duckdns.org", "eu.org",
    "glitch.me", "blogspot.com", "weebly.com", "wixsite.com", "square.site",
    "firebaseapp.com", "page.link", "tilda.ws", "wordpress.com", "github.io",
    "neocities.org", "000webhostapp.com", "infinityfree.net", "epizy.com",
    "comn.io", "ddns.net", "hopto.org", "servehttp.com", "zapto.org",
    "sytes.net", "no-ip.org", "myftp.biz", "ddnsking.com", "freeddns.org",
}

COM_TYPO_TLDS = {"con", "cm", "co", "comn", "comm", "om", "cp", "cem", "c0m", "cn"}

REDIRECT_PARAM_KEYS = {
    "redirect", "redirecturl", "redirect_uri", "redir", "url", "next", "goto",
    "return", "returnurl", "return_to", "returnurl", "continue", "dest",
    "destination", "forward", "rurl", "target", "u", "link", "to", "out",
}

NONHTTP_SCHEMES = {"javascript", "data", "vbscript", "file", "ftp", "ftps"}

HOMOGLYPH_RE = re.compile(
    "[\u0400-\u04ff\u0370-\u03ff\u0500-\u052f\u1d00-\u1d7f\u0250-\u02af"
    "\uff21-\uff5a]"
)

TRUSTED_TLDS = {
    ".com", ".org", ".net", ".edu", ".gov", ".int", ".mil", ".co", ".io",
    ".ai", ".dev", ".app", ".xyz",
}

# URL shorteners heavily abused for phishing (obfuscate the final destination).
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "rb.gy", "cutt.ly",
    "cut.ly", "is.gd", "buff.ly", "adf.ly", "shorturl.at", "rebrand.ly",
    "tiny.cc", "snip.ly", "clck.ru", "qr.ae", "tr.im", "mcaf.ee", "short.io",
    "bitly.com", "t.ly", "s.id", "soo.gd", "v.gd", "po.st", "bc.vc", "tinyarrows.com",
}

# Character n-gram hashing (stateless; JS-portable via FNV-1a 32-bit).
NGRAM_BUCKETS = 512
NGRAM_MIN = 3
NGRAM_MAX = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [c / len(s) for c in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs)


def _fnv1a(s: str, seed: int = 0x811C9DC5) -> int:
    h = seed
    for ch in s.encode("utf-8", "ignore"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def char_ngram_vector(url: str, n_min: int = NGRAM_MIN, n_max: int = NGRAM_MAX,
                      buckets: int = NGRAM_BUCKETS):
    """L2-normalized hashed character n-gram counts over the lowercased URL.

    Stateless and dependency-free: the same FNV-1a hashing can be reproduced in
    JavaScript so browser-side feature extraction matches training exactly.
    """
    vec = [0.0] * buckets
    s = url.lower()
    total = 0
    for n in range(n_min, n_max + 1):
        for i in range(len(s) - n + 1):
            b = _fnv1a(s[i:i + n]) % buckets
            vec[b] += 1.0
            total += 1
    if total > 0:
        norm = 1.0 / math.sqrt(total)
        for i in range(buckets):
            vec[i] *= norm
    return vec


def _brand_edit_distance(host: str) -> int:
    best = 99
    for label in host.split("."):
        if not label:
            continue
        for b in BRANDS:
            d = _lev(label, b)
            if d < best:
                best = d
    return best


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

def extract_features(url_raw) -> dict:
    url_raw = "" if url_raw is None else str(url_raw)
    url = url_raw.lower().strip()
    has_proto = url.startswith("http")
    full = url if has_proto else "http://" + url
    parsed = urlparse(full)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    netloc = parsed.netloc or ""
    scheme = parsed.scheme or ""

    f = {}
    # ---- base 16 (kept byte-compatible with v1) ----
    f["url_length"] = len(full)
    f["hyphen_count"] = url.count("-")
    f["digit_count"] = sum(1 for c in url if c.isdigit())
    f["subdomain_count"] = max(0, len(host.split(".")) - 2)
    f["protocol_exists"] = int(bool(re.match(r"^https?://", url)))
    f["special_char_count"] = sum(1 for c in url if c in "@-_.,;:#~!$&'()*+/:=?")
    f["entropy"] = compute_entropy(full)
    f["path_depth"] = len([p for p in path.split("/") if p])
    f["domain_length"] = len(host)
    f["is_domain_ip"] = int(bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host)))
    f["has_at_symbol"] = int("@" in url)
    f["has_double_slash_redirect"] = int(url.count("//") > 1)
    tld_match = re.search(r"\.([a-z]{2,24})$", host)
    tld = tld_match.group(1) if tld_match else ""
    f["tld_length"] = len(tld)
    f["trusted_tld"] = int("." + tld in TRUSTED_TLDS)
    f["query_param_count"] = len(query.split("&")) if query else 0
    f["path_length"] = len(path)

    # ---- expanded 19 (kept compatible with v1) ----
    f["keyword_count"] = sum(url.count(k) for k in SUSPICIOUS_KEYWORDS)
    host_brands = sum(1 for b in BRANDS if b in host)
    path_brands = sum(1 for b in BRANDS if b in path)
    f["brand_in_host"] = int(host_brands > 0)
    f["brand_in_path"] = int(path_brands > 0)
    f["brand_count"] = host_brands + path_brands
    f["has_punycode"] = int("xn--" in host)
    f["has_nonstandard_port"] = int(bool(re.search(r":(?!80|443)(\d+)", netloc)))
    f["percent_encoding_count"] = url.count("%")
    f["has_hex_ip"] = int(bool(re.search(r"0x[0-9a-f]{1,8}", url)))
    f["uppercase_ratio"] = round(sum(1 for c in url_raw if c.isupper()) / max(1, len(url_raw)), 4)
    f["www_count"] = url.count("www.")
    runs = re.findall(r"\d+", url)
    f["max_digit_run"] = max((len(r) for r in runs), default=0)
    labels = host.split(".")
    f["max_label_len"] = max((len(l) for l in labels), default=0)
    f["host_label_count"] = len(labels)
    f["has_domain_in_path"] = int(bool(re.search(r"/[a-z0-9-]+\.[a-z]{2,}/", path)))
    f["query_length"] = len(query)
    f["equals_count"] = query.count("=")
    f["sensitive_extension"] = int(bool(re.search(r"\.(exe|scr|apk|zip|jar|bat|cmd|js|docm|iso|dmg|msi)$", url)))
    f["has_multiple_tlds"] = int(len(re.findall(r"\.[a-z]{2,}/", url)) > 1)
    f["consecutive_punct"] = int(bool(re.search(r"[@_.:;,]{3,}", url)))

    # ---- v2 additions ----
    homo = HOMOGLYPH_RE.findall(host)
    f["homoglyph_count"] = len(homo)
    f["has_homoglyph"] = int(len(homo) > 0)
    nonascii = [c for c in host if ord(c) > 127]
    f["host_nonascii_count"] = len(nonascii)
    f["has_mixed_script"] = int(len(nonascii) > 0 and len(host) > len(nonascii))

    bd = _brand_edit_distance(host)
    f["brand_min_editdist"] = bd
    f["brand_typo_in_subdomain"] = int(bd <= 2 and bd > 0)
    apex = labels[-2] if len(labels) >= 2 else (labels[0] if labels else "")
    f["brand_apex_match"] = int(apex in BRANDS)
    f["brand_in_subdomain_not_apex"] = int(
        any(b in ".".join(labels[:-2]) for b in BRANDS) and apex not in BRANDS
    )

    f["tld_is_suspicious"] = int(tld in SUSPICIOUS_TLDS)
    f["tld_looks_like_com"] = int(tld in COM_TYPO_TLDS)
    f["tld_is_cctld"] = int(len(tld) == 2)

    f["num_dots"] = host.count(".")
    f["num_labels_gt3"] = int(len(labels) > 3)
    reg_dom = ".".join(labels[-2:]) if len(labels) >= 2 else host
    f["registrable_domain_length"] = len(reg_dom)

    left = labels[0] if labels else ""
    f["leftmost_subdomain_len"] = len(left)
    f["leftmost_subdomain_entropy"] = compute_entropy(left)
    vowels = sum(1 for c in left if c in "aeiou")
    f["leftmost_subdomain_vowel_ratio"] = round(vowels / max(1, len(left)), 4)
    f["has_long_random_subdomain"] = int(len(left) > 18 and compute_entropy(left) > 3.5)

    dom_alnum = [c for c in host if c.isalnum()]
    digits = sum(1 for c in host if c.isdigit())
    f["domain_digit_ratio"] = round(digits / max(1, len(dom_alnum)), 4)

    f["has_octal_ip"] = int(bool(re.search(r"0[0-7]{2,}", url)))
    f["has_decimal_ip"] = int(bool(re.search(r"\b(\d{9,12})\b", url)))
    f["has_dword_ip"] = int(bool(re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", host)))

    f["scheme_is_nonthttp"] = int(scheme in NONHTTP_SCHEMES)
    f["has_userinfo"] = int(bool(re.search(r"//[^/]*@[^/]*/", full)))

    qkeys = {k.split("=")[0] for k in query.split("&") if k}
    f["open_redirect_param"] = int(bool(qkeys & REDIRECT_PARAM_KEYS))
    f["encoded_url_in_path"] = int(bool(re.search(r"%68%74%74%70|%2f%2f", url)))

    f["host_is_free_host"] = int(any(host == fh or host.endswith("." + fh) for fh in FREE_HOSTS))
    f["brand_on_free_host"] = int(
        f["host_is_free_host"] and (host_brands > 0 or f["brand_in_subdomain_not_apex"])
    )

    f["consecutive_identical_chars"] = int(bool(re.search(r"(.)\1{3,}", url)))
    f["emoji_in_url"] = int(any(ord(c) > 0x1F000 for c in url_raw))
    f["query_has_suspicious_keyword"] = int(
        any(k in query for k in ("password", "token", "otp", "login", "verify"))
    )

    # URL shortener obfuscation
    f["host_is_shortener"] = int(any(host == sh or host.endswith("." + sh) for sh in URL_SHORTENERS))

    # ---- hashed character n-gram lexical features (stateless, JS-portable) ----
    ng = char_ngram_vector(url)
    for i, v in enumerate(ng):
        f[f"cng{i}"] = round(v, 6)

    return f


FEATURE_COLUMNS = list(extract_features("http://example.com/a?b=c").keys())


if __name__ == "__main__":
    import json
    print("n_features:", len(FEATURE_COLUMNS))
    print(json.dumps(extract_features("http://paypa1-verify.ngrok.io/login?token=x"), indent=2))
