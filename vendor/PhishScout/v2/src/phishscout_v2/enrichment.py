#!/usr/bin/env python3
"""PhishScout v2 - optional page/TLS enrichment stage.

This is the SECOND stage of v2. It is fully optional and network-dependent: when
the caller has network access it fetches the target page and TLS certificate and
extracts signals that the URL-only core model cannot see (login forms, brand/
domain mismatch, redirect-to-other-domain, invalid certs, etc.), then combines
them with the on-device core probability.

Design rules (see models/AGENTS.md and root AGENTS.md):
  - Never required for core inference. If the fetch fails or there is no
    network, callers must fall back to the core model alone.
  - Stdlib-only so it stays dependency-light and cleanly separable from core.
  - The combiner uses a small meta-model (logistic) when
    enrichment_model.json exists, otherwise a transparent heuristic.
"""
from __future__ import annotations

import json
import math
import ssl
import socket
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import numpy as np

from .features import BRANDS, SUSPICIOUS_KEYWORDS
from .onnx_model import PhishScoutV2

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

UA = "PhishScout-v2-enrichment/1.0 (+https://huggingface.co/saidutta69/PhishScout)"
LETSENCRYPT_OIDS = ("lets encrypt", "let's encrypt", "letsencrypt", "r3", "r11", "r10")


class _FormPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.n_password_fields = 0
        self.n_input_fields = 0
        self.n_iframes = 0
        self.form_actions = []
        self.links = []
        self.title = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        t = tag.lower()
        if t == "input":
            self.n_input_fields += 1
            if a.get("type", "").lower() in ("password",):
                self.n_password_fields += 1
        elif t == "iframe":
            self.n_iframes += 1
        elif t == "form":
            act = a.get("action", "")
            if act:
                self.form_actions.append(act)
        elif t == "a":
            href = a.get("href", "")
            if href:
                self.links.append(href)
        elif t == "title":
            self._in_title = True

    def handle_data(self, data):
        if self._in_title:
            self.title.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False


def _domain_of(url: str) -> str:
    try:
        h = urllib.parse.urlparse(url if url.startswith("http") else "http://" + url).hostname or ""
        return h.lower()
    except Exception:
        return ""


def _brand_in_text(text: str) -> bool:
    t = text.lower()
    return any(b in t for b in BRANDS)


def _suspicious_in_text(text: str) -> int:
    t = text.lower()
    return sum(1 for k in SUSPICIOUS_KEYWORDS if k in t)


def _brand_count_in_text(text: str) -> int:
    t = text.lower()
    return sum(1 for b in BRANDS if b in t)


def _parse_iso(s: str):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _rdap_age(domain: str):
    """Best-effort domain registration age (days) via RDAP. Returns (age, known)."""
    try:
        req = urllib.request.Request(
            f"https://rdap.org/domain/{domain}", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.loads(r.read().decode())
        for ev in data.get("events", []):
            if ev.get("eventAction") in ("registration", "created", "expiration"):
                d = ev.get("eventDate")
                if d:
                    dt = _parse_iso(d)
                    if dt:
                        age = (datetime.now(dt.tzinfo or timezone.utc) - dt).days
                        return max(0, age), True
    except Exception:
        pass
    return -1, False


class PageTlsEnricher:
    """Fetches a URL and extracts page/TLS signals. Stdlib only."""

    def __init__(self, timeout: float = 6.0, max_redirects: int = 5, verify_tls: bool = False):
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.verify_tls = verify_tls  # default off so we can flag invalid certs

    def fetch(self, url: str) -> dict:
        orig = url if url.startswith("http") else "http://" + url
        ctx = ssl.create_default_context()
        if not self.verify_tls:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        result = {
            "fetch_error": True, "http_status": None, "scheme_https": orig.startswith("https"),
            "final_url": orig, "redirect_count": 0, "redirects_to_other_domain": False,
            "tls_valid": None, "tls_issuer": "", "tls_issuer_letsencrypt": False,
            "content_length": 0, "n_password_fields": 0, "n_input_fields": 0, "n_iframes": 0,
            "n_external_links": 0, "external_form_action": False, "title": "",
            "page_brand_mismatch": False, "title_brand_count": 0, "page_suspicious_kw": 0,
            "domain_age_days": -1, "age_known": False, "error": "",
        }
        try:
            cur = orig
            seen_domains = {_domain_of(cur)}
            for _ in range(self.max_redirects + 1):
                req = urllib.request.Request(cur, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                    result["http_status"] = getattr(resp, "status", None)
                    result["final_url"] = resp.geturl()
                    body = resp.read(1_000_000)
                # TLS info from the socket
                host = _domain_of(cur)
                try:
                    with socket.create_connection((host, 443), timeout=self.timeout) as sock:
                        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                            cert = ssock.getpeercert()
                            result["tls_valid"] = True
                            issuer = json.dumps(cert.get("issuer", "")).lower()
                            result["tls_issuer"] = issuer
                            result["tls_issuer_letsencrypt"] = any(
                                o in issuer for o in LETSENCRYPT_OIDS)
                except Exception:
                    result["tls_valid"] = False
                # redirect handling
                if resp.geturl() != cur and _domain_of(resp.geturl()) not in seen_domains:
                    seen_domains.add(_domain_of(resp.geturl()))
                    result["redirect_count"] += 1
                    cur = resp.geturl()
                    continue
                break
            result["fetch_error"] = False
            # parse body
            parser = _FormPageParser()
            try:
                parser.feed(body.decode("utf-8", "ignore"))
            except Exception:
                pass
            result["content_length"] = len(body)
            result["n_password_fields"] = parser.n_password_fields
            result["n_input_fields"] = parser.n_input_fields
            result["n_iframes"] = parser.n_iframes
            title = "".join(parser.title).strip()
            result["title"] = title
            orig_domain = _domain_of(orig)
            for act in parser.form_actions:
                if _domain_of(act) and _domain_of(act) != orig_domain:
                    result["external_form_action"] = True
                    break
            page_text = title + " " + body.decode("utf-8", "ignore")[:4000]
            result["page_brand_mismatch"] = (
                _brand_in_text(page_text) and not _brand_in_text(orig_domain)
            )
            result["title_brand_count"] = _brand_count_in_text(title)
            result["page_suspicious_kw"] = _suspicious_in_text(page_text)
            n_ext = 0
            for href in parser.links:
                d = _domain_of(href)
                if d and d != orig_domain:
                    n_ext += 1
            result["n_external_links"] = n_ext
            if _domain_of(result["final_url"]) not in {orig_domain} and result["redirect_count"] > 0:
                result["redirects_to_other_domain"] = True
            age, known = _rdap_age(orig_domain)
            result["domain_age_days"] = age
            result["age_known"] = known
        except (urllib.error.URLError, socket.timeout, ConnectionError, ssl.SSLError,
                ssl.CertificateError, OSError) as e:
            result["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        except Exception as e:  # noqa: BLE001 - never let enrichment crash the caller
            result["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return result

    def signals_for(self, url: str) -> dict:
        return self.fetch(url)


class EnrichedScorer:
    """Combines the on-device core model with optional page/TLS signals."""

    SIGNAL_FEATURES = [
        "n_password_fields", "n_input_fields", "n_iframes", "n_external_links",
        "external_form_action", "page_brand_mismatch", "title_brand_count",
        "page_suspicious_kw", "redirects_to_other_domain", "redirect_count",
        "tls_valid_neg", "tls_issuer_letsencrypt", "domain_age_days", "age_known",
    ]

    def __init__(self, core: PhishScoutV2 | None = None, enricher: PageTlsEnricher | None = None,
                 meta_path: str | Path = MODELS_DIR / "enrichment_model.json", use_meta: bool = True):
        self.core = core or PhishScoutV2()
        self.enricher = enricher or PageTlsEnricher()
        self.use_meta = use_meta
        self.meta = None
        if use_meta:
            mp = Path(meta_path)
            if mp.exists():
                self.meta = json.loads(mp.read_text())

    def _signal_vector(self, sig: dict) -> list[float]:
        tls_valid = sig.get("tls_valid")
        return [
            float(sig.get("n_password_fields", 0)),
            float(sig.get("n_input_fields", 0)),
            float(sig.get("n_iframes", 0)),
            float(sig.get("n_external_links", 0)),
            float(sig.get("external_form_action", 0)),
            float(sig.get("page_brand_mismatch", 0)),
            float(sig.get("title_brand_count", 0)),
            float(sig.get("page_suspicious_kw", 0)),
            float(sig.get("redirects_to_other_domain", 0)),
            float(sig.get("redirect_count", 0)),
            float(0.0 if tls_valid else 1.0),
            float(sig.get("tls_issuer_letsencrypt", 0)),
            float(sig.get("domain_age_days", -1)),
            float(sig.get("age_known", 0)),
        ]

    def score(self, url: str, threshold: float | None = None) -> dict:
        core = self.core.predict_url(url, threshold)
        sig = self.enricher.signals_for(url)
        if sig.get("fetch_error"):
            return {
                **core, "enriched": False,
                "enrichment_error": sig.get("error", ""),
                "probability": core["probability"],
            }
        vec = self._signal_vector(sig)
        p = core["probability"]
        if self.use_meta and self.meta:
            x = np.array([p] + vec, dtype=np.float64)
            mean = self.meta.get("scaler_mean")
            scale = self.meta.get("scaler_scale")
            if mean is not None and scale is not None:
                x = (x - np.array(mean, dtype=np.float64)) / np.array(scale, dtype=np.float64)
            z = self.meta["b"] + float(np.dot(self.meta["w"], x))
            p_enr = 1.0 / (1.0 + math.exp(-z))
        else:
            p_enr = self._heuristic(p, sig)
        t = self.core.threshold if threshold is None else threshold
        return {
            **core, "enriched": True, "probability": float(p_enr),
            "label": int(p_enr > t), "is_phishing": bool(p_enr > t),
            "signals": {k: sig.get(k) for k in self.SIGNAL_FEATURES},
        }

    @staticmethod
    def _heuristic(core_p: float, sig: dict) -> float:
        boost = 0.0
        if sig.get("n_password_fields", 0) > 0 and sig.get("page_brand_mismatch"):
            boost += 0.25
        if sig.get("redirects_to_other_domain"):
            boost += 0.15
        if sig.get("external_form_action"):
            boost += 0.15
        if sig.get("tls_valid") is False:
            boost += 0.10
        if sig.get("tls_issuer_letsencrypt") and sig.get("page_brand_mismatch"):
            boost += 0.05
        boost += min(sig.get("page_suspicious_kw", 0), 5) * 0.01
        boost += min(sig.get("n_external_links", 0), 10) * 0.005
        if boost > 0:
            return min(0.99, core_p + boost)
        return core_p


if __name__ == "__main__":
    import math
    s = EnrichedScorer()
    for u in ["http://paypa1-verify.ngrok.io/login?token=x", "https://www.google.com"]:
        print(u, "->", s.score(u))
