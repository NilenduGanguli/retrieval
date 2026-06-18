#!/usr/bin/env python3
"""Fetch real SEC EDGAR filings for major US public companies into sample_docs/.

Respects SEC fair-access policy: declares a User-Agent and throttles requests
to well under 10 req/sec.
"""
import json
import time
import sys
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "sample_docs"
OUT.mkdir(exist_ok=True)

# SEC requires a descriptive User-Agent with contact info.
HEADERS = {
    "User-Agent": "Retrieval Sample Dataset neelu indrajha5314@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# Major US public companies by ticker. CIKs are resolved from EDGAR.
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK-B",
    "JPM", "JNJ", "V", "WMT", "XOM", "PG", "KO", "NFLX", "DIS", "INTC",
    "CSCO", "PFE", "BAC", "HD", "MA", "CVX", "ABBV",
]

# Filing types we want, and how many of each (most recent first) per company.
WANT_FORMS = {"10-K": 1, "10-Q": 1, "8-K": 1}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def throttle():
    time.sleep(0.25)  # ~4 req/sec, comfortably under SEC's 10/sec cap


def get(url):
    throttle()
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    return r


def load_cik_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    data = get(url).json()
    by_ticker = {}
    for row in data.values():
        by_ticker[row["ticker"].upper()] = (
            str(row["cik_str"]).zfill(10),
            row["title"],
        )
    return by_ticker


def fetch_company(ticker, cik, title, manifest):
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        sub = get(sub_url).json()
    except Exception as e:
        print(f"  ! {ticker}: could not load submissions ({e})")
        return

    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])
    descs = recent.get("primaryDocDescription", [])

    taken = {f: 0 for f in WANT_FORMS}
    cik_int = str(int(cik))  # no leading zeros for Archives path

    for i, form in enumerate(forms):
        if form not in WANT_FORMS:
            continue
        if taken[form] >= WANT_FORMS[form]:
            continue
        primary = docs[i]
        if not primary or not primary.lower().endswith((".htm", ".html")):
            continue
        acc_nodash = accs[i].replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{acc_nodash}/{primary}"
        )
        date = dates[i]
        safe_form = form.replace("/", "-")
        fname = f"{ticker}_{safe_form}_{date}_{acc_nodash}.html"
        dest = OUT / fname
        try:
            content = get(url).content
        except Exception as e:
            print(f"  ! {ticker} {form}: download failed ({e})")
            continue
        dest.write_bytes(content)
        taken[form] += 1
        size_kb = len(content) // 1024
        print(f"  + {fname}  ({size_kb} KB)")
        manifest.append({
            "ticker": ticker,
            "company": title,
            "cik": cik,
            "form": form,
            "filing_date": date,
            "accession": accs[i],
            "description": descs[i] if i < len(descs) else "",
            "source_url": url,
            "file": fname,
            "bytes": len(content),
        })


def main():
    print("Resolving CIKs from EDGAR...")
    cik_map = load_cik_map()
    manifest = []
    for ticker in TICKERS:
        key = ticker.upper()
        # EDGAR uses BRK-B style; ticker file uses BRK-B too, but some use no dash
        entry = cik_map.get(key) or cik_map.get(key.replace("-", ""))
        if not entry:
            print(f"- {ticker}: CIK not found, skipping")
            continue
        cik, title = entry
        print(f"{ticker}  ({title})  CIK={cik}")
        fetch_company(ticker, cik, title, manifest)
        if len(manifest) >= 40:
            break

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} documents saved to {OUT}")
    print(f"Manifest: {OUT/'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
