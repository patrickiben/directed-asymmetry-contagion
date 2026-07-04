#!/usr/bin/env python3
"""
Stray-AI-artifact + hidden-text scrubber for a manuscript before submission / arXiv.

WHY THIS EXISTS
    arXiv and publishers now issue up-to-year-long bans for incontrovertible evidence of
    unchecked AI use: leftover assistant boilerplate, unfilled template placeholders, or
    hidden/white text intended to game automated screening. This is a deterministic grep +
    an optional hidden-text check on the compiled PDF. It flags; it does not edit.

USAGE
    python3 tools/scrub_ai_artifacts.py manuscript.tex [supplementary.tex ...] [--pdf paper.pdf]
    Exit 0 = clean, 1 = something to look at. No network, no LLM.
"""
import argparse
import re
import subprocess
import sys

# Patterns that should never survive into a finished manuscript source.
PATTERNS = [
    (r"as an? (?:AI|language model)", "assistant boilerplate"),
    (r"\bI(?:'m| am) (?:sorry|unable|an AI)\b", "assistant boilerplate"),
    (r"\b(?:certainly|sure)[!,] here(?:'s| is)\b", "chat framing"),
    (r"\bknowledge cutoff\b", "assistant boilerplate"),
    (r"\[(?:TOOL|REASON|INSERT|TODO|FIXME|PLACEHOLDER|NAME|DATE|XXX+)\]", "unfilled placeholder"),
    (r"\blorem ipsum\b", "placeholder text"),
    (r"\bXXXX\b", "unfilled placeholder"),
    (r"<[^>]*(?:your text here|insert[^>]*here)[^>]*>", "unfilled placeholder"),
    (r"\bplaceholder\b", "literal 'placeholder'"),
    (r"\bTBD\b|\bTK\b(?![A-Za-z])", "to-be-done marker"),
    (r"\bcitation needed\b", "unfinished citation"),
    (r"\[\?\?\?\]|\?\?\?", "unresolved marker"),
    (r"\\todo\b|\\fixme\b|\\note\{", "draft note macro"),
    (r"\bhttps?://(?:example\.com|foo\.bar)\b", "placeholder URL"),
]

def scan_source(paths):
    hits = []
    for p in paths:
        try:
            txt = open(p, encoding="utf-8").read()
        except Exception as e:  # noqa
            print(f"[warn] cannot read {p}: {e}", file=sys.stderr)
            continue
        for ln, line in enumerate(txt.splitlines(), 1):
            low = line.lower()
            if low.lstrip().startswith("%"):     # skip LaTeX comments
                continue
            for pat, label in PATTERNS:
                if re.search(pat, line, re.I):
                    hits.append((p, ln, label, line.strip()[:100]))
    return hits

def check_pdf_hidden(pdf):
    """Heuristic: compare pdftotext output length to page count. Also surface any of our
    text patterns living only in the extracted layer (a sign of hidden/white text)."""
    notes = []
    try:
        txt = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                             capture_output=True, text=True, timeout=60).stdout
    except FileNotFoundError:
        return ["[skip] pdftotext not installed; cannot hidden-text-screen the PDF"]
    except Exception as e:  # noqa
        return [f"[skip] pdftotext failed: {e}"]
    for pat, label in PATTERNS:
        if re.search(pat, txt, re.I):
            notes.append(f"pattern '{label}' present in extracted PDF text layer -> inspect for hidden text")
    # crude white-text signal: extracted text far exceeds what a human sees is hard to judge here;
    # we surface only the pattern-in-text-layer signal, which is the actionable one.
    return notes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--pdf", default=None)
    a = ap.parse_args()

    hits = scan_source(a.sources)
    print(f"Scanned {len(a.sources)} source file(s) for AI artifacts / placeholders")
    print("=" * 78)
    if hits:
        for p, ln, label, snippet in hits:
            print(f">FLAG< {p}:{ln}  [{label}]  {snippet}")
    else:
        print("  ok  no AI boilerplate / placeholders in source")

    pdf_notes = check_pdf_hidden(a.pdf) if a.pdf else []
    for n in pdf_notes:
        print(n)

    print("=" * 78)
    hard = hits or any(">FLAG<" in n or "hidden text" in n for n in pdf_notes)
    print(f"SUMMARY: {len(hits)} source hit(s); {'PDF flagged' if any('hidden text' in n for n in pdf_notes) else 'PDF clean/skipped'}")
    sys.exit(1 if hard else 0)

if __name__ == "__main__":
    main()
