"""PhishScout web platform: FastAPI backend + single-page UI around detector.py.

Run:  python -m uvicorn app:app --reload
Open: http://127.0.0.1:8000
"""
import time
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import detector
from detector import categorize_url

BASE = Path(__file__).parent
MAX_URL_LEN = 2048
MAX_BATCH = 50

app = FastAPI(title="PhishScout URL Scanner", version="1.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


class ScanRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=MAX_URL_LEN)


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


def normalize(url: str) -> str:
    """Trim and add http:// when the user typed a bare domain (google.com)."""
    url = url.strip()
    if not url:
        raise ValueError("empty URL")
    if len(url) > MAX_URL_LEN:
        raise ValueError("URL too long")
    if "://" not in url:
        url = "http://" + url
    return url


def _signals(url: str, prob: float) -> list[dict]:
    """Human-readable reasons, derived from the model's own URL features."""
    from phishscout_v2.features import extract_features

    f = extract_features(url)
    out = []

    def add(flag, label, detail, severity="warn"):
        if flag:
            out.append({"label": label, "detail": detail, "severity": severity})

    add(f["is_domain_ip"] or f["has_decimal_ip"] or f["has_hex_ip"] or f["has_octal_ip"] or f["has_dword_ip"],
        "IP address as host", "Legitimate sites almost always use a domain name.", "bad")
    add(f["has_nonstandard_port"], "Non-standard port", "Unusual port in the URL, common for payload servers.")
    add(f["has_userinfo"] or f["has_at_symbol"], "Credentials / @ in URL",
        "Text before '@' can disguise the real destination.", "bad")
    add(f["has_punycode"] or f["has_homoglyph"] or f["has_mixed_script"],
        "Lookalike characters", "Punycode or mixed-script characters can imitate a real brand.", "bad")
    add(f["brand_typo_in_subdomain"] and prob >= 0.4, "Possible brand typo", "Hostname is a near-miss of a well-known brand.", "bad")
    add(f["brand_in_subdomain_not_apex"], "Brand in subdomain only",
        "A brand name appears in a subdomain, not in the real registrable domain.", "bad")
    add(f["brand_on_free_host"], "Brand on free hosting", "A brand name on a free hosting service.", "bad")
    add(f["host_is_free_host"], "Free hosting / tunnel service", "Often abused to host throwaway phishing pages.")
    add(f["host_is_shortener"], "URL shortener", "The final destination is hidden.")
    add(f["tld_is_suspicious"] or f["tld_looks_like_com"], "Suspicious TLD",
        "TLD is frequently used for abuse or imitates .com.")
    add(f["sensitive_extension"], "Executable / script file", "Path ends in a script or payload-type file.", "bad")
    add(f["keyword_count"] >= 2, "Phishing keywords",
        f"{f['keyword_count']} words like login/verify/secure in the URL.")
    add(f["open_redirect_param"] or f["encoded_url_in_path"], "Redirect / embedded URL",
        "Another URL is embedded in this one.")
    add(f["has_long_random_subdomain"], "Random-looking subdomain", "Looks machine-generated.")
    add(f["subdomain_count"] >= 3, "Many subdomains", f"{f['subdomain_count']} subdomain levels.")
    add(f["url_length"] > 100, "Very long URL", f"{f['url_length']} characters.")
    add(not f["protocol_exists"] or url.lower().startswith("http://"),
        "Not HTTPS", "Connection is unencrypted.", "info")
    return out


def scan_one(raw: str) -> dict:
    t0 = time.perf_counter()
    try:
        url = normalize(raw)
    except ValueError as e:
        return {"input": raw[:200], "error": str(e)}
    host = (urlparse(url).hostname or "").lower()
    result = categorize_url(url)
    whitelisted = "note" in result["raw_result"]
    return {
        "input": raw.strip(),
        "url": url,
        "host": host,
        "verdict": result["verdict"],
        "probability": round(result["probability"], 4),
        "whitelisted": whitelisted,
        "signals": [] if whitelisted else _signals(url, result["probability"]),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/health")
def health():
    detector._get_model()
    return {"status": "ok", "model": "PhishScout v2 (LightGBM/ONNX)"}


@app.post("/api/scan")
def scan(req: ScanRequest):
    res = scan_one(req.url)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@app.post("/api/scan/batch")
def scan_batch(req: BatchRequest):
    return {"results": [scan_one(u) for u in req.urls if u.strip()]}
