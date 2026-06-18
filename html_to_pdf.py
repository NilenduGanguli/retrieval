#!/usr/bin/env python3
"""Convert the SEC EDGAR HTML filings in sample_docs/ to PDF via headless Chromium.

Uses Playwright's Chromium (the same engine Chrome uses) for faithful rendering
of the filings' tables and layout.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "sample_docs"


def convert(only=None):
    html_files = sorted(OUT.glob("*.html"))
    if only:
        html_files = [f for f in html_files if f.name in only]
    if not html_files:
        print("No HTML files found.")
        return 0

    done, failed = 0, []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        for html in html_files:
            pdf_path = html.with_suffix(".pdf")
            try:
                page.goto(html.resolve().as_uri(),
                          wait_until="load", timeout=120_000)
                page.pdf(
                    path=str(pdf_path),
                    format="Letter",
                    print_background=True,
                    margin={"top": "0.5in", "bottom": "0.5in",
                            "left": "0.5in", "right": "0.5in"},
                )
                size_kb = pdf_path.stat().st_size // 1024
                print(f"  + {pdf_path.name}  ({size_kb} KB)")
                done += 1
            except Exception as e:
                print(f"  ! {html.name}: FAILED ({e})")
                failed.append(html.name)
        browser.close()

    print(f"\nConverted {done} file(s).")
    if failed:
        print(f"Failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(convert(only=set(args) if args else None))
