#!/usr/bin/env python3
"""
Deterministic citation-integrity gate for a manuscript's bibliography.

WHY THIS EXISTS
    A fabricated / hallucinated reference is the single most damaging (and cheapest to
    catch) integrity failure in an AI-assisted manuscript. This tool resolves every
    reference against authoritative bibliographic APIs and DIFFS the returned metadata
    against what the .tex claims. It is INTENTIONALLY not an LLM: existence and metadata
    are decided by CrossRef / OpenAlex / arXiv, never by a language model.

WHAT IT CATCHES
    1. Non-existent references          -> no confident match anywhere.
    2. "Identifier hijacking"           -> a real DOI that resolves to a DIFFERENT paper
                                           than the one cited (a naive existence check passes).
    3. Year / title drift               -> the cited year or title disagrees with the record.
    4. Retractions (best effort)        -> the DOI record carries a retraction notice.

USAGE
    python3 tools/check_citations.py path/to/manuscript.tex [--mailto you@example.com]
                                     [--json report.json] [--strict]
    Parses either an inline \\begin{thebibliography} block (\\bibitem entries) or a .bib file.
    Exit code 0 = clean; 1 = at least one hard FLAG (fabrication / hijack / dead DOI / retraction).
    --strict also fails on soft REVIEW items (weak matches). Network required.

STDLIB ONLY (urllib) so it runs unchanged in CI with no extra install.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from html import unescape

CROSSREF = "https://api.crossref.org/works"
ARXIV = "http://export.arxiv.org/api/query"
TIMEOUT = 20
UA = "citation-check/1.0 (deposit gate; mailto:{mailto})"

# ------------------------------------------------------------------ HTTP
def _get(url, mailto, accept="application/json", retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": UA.format(mailto=mailto), "Accept": accept})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read().decode("utf-8", "replace"), r.status
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, 404
            last = e
            time.sleep(1.5 * (attempt + 1))  # be polite on 429/5xx
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5 * (attempt + 1))
    if last:
        print(f"    [warn] request failed after {retries} tries: {last}", file=sys.stderr)
    return None, None

# ------------------------------------------------------------------ LaTeX -> text
def delatex(s):
    # accented letters first: {\"u} / \"{u} / \'e -> u / e (keep the base letter, drop the accent)
    s = re.sub(r'\{\\[\'"`^~=.c]\{?([a-zA-Z])\}?\}', r"\1", s)
    s = re.sub(r'\\[\'"`^~=.c]\{?([a-zA-Z])\}?', r"\1", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = s.replace("~", " ").replace("\\&", "&").replace("--", "-").replace("\\ ", " ")
    s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r"\1", s)      # any other \cmd{arg}
    s = re.sub(r'\\[a-zA-Z]+', "", s)                    # bare \cmd (accents etc.)
    s = re.sub(r'[{}]', "", s)
    s = re.sub(r"``|''|\"", "", s)
    return re.sub(r"\s+", " ", s).strip()

# ------------------------------------------------------------------ parse bib entries
def parse_thebibliography(tex):
    m = re.search(r"\\begin\{thebibliography\}.*?\n(.*?)\\end\{thebibliography\}", tex, re.S)
    body = m.group(1) if m else tex
    parts = re.split(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", body)
    entries = []
    for i in range(1, len(parts), 2):
        entries.append((parts[i].strip(), parts[i + 1].strip()))
    return entries

def reference_parity(tex):
    r"""Cross-check that every \bibitem is \cite'd and every \cite'd key is listed
    (false-floor lesson 3: prose edits can silently drop a listed-or-cited reference)."""
    listed = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", tex))
    cited = set()
    for m in re.finditer(r"\\(?:cite|citep|citet|citealt|citealp|citeauthor|citeyear|"
                         r"textcite|parencite|autocite)\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    return sorted(listed - cited), sorted(cited - listed)

def _bib_field(body, name):
    """Extract a bibtex field value with balanced-brace / quoted / bare handling, NOT requiring a
    trailing newline (false-floor lesson 2a: fields are often all on one line)."""
    m = re.search(r"\b" + name + r"\s*=\s*", body, re.I)
    if not m or m.end() >= len(body):
        return ""
    i = m.end(); c = body[i]
    if c == "{":
        depth, j = 0, i
        while j < len(body):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        val = body[i + 1:j]
    elif c == '"':
        j = body.find('"', i + 1)
        val = body[i + 1:j] if j != -1 else body[i + 1:]
    else:                                   # bare value (a number) up to the next comma / brace
        j = i
        while j < len(body) and body[j] not in ",}\n":
            j += 1
        val = body[i:j]
    return delatex(val.strip())

def parse_bibfile(bib):
    entries = []
    # entry-boundary matched so one-entry-per-line .bib files (closing brace NOT on its own line)
    # are not silently skipped (false-floor lesson 2a).
    for m in re.finditer(r"@\w+\s*\{\s*([^,]+),(.*?)\}\s*(?=@|\Z)", bib, re.S):
        key, body = m.group(1).strip(), m.group(2)
        parts = [_bib_field(body, f) for f in ("author", "title", "journal", "booktitle",
                 "volume", "number", "pages", "year", "doi", "eprint")]
        entries.append((key, " ".join(x for x in parts if x)))
    return entries

# ------------------------------------------------------------------ metadata extraction
def extract(raw):
    txt = delatex(raw)
    doi = None
    dm = re.search(r"10\.\d{4,9}/[^\s}\"]+", txt)
    if dm:
        doi = dm.group(0).rstrip(".,;")
    arx = None
    am = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", raw, re.I) or re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", txt, re.I)
    if am:
        arx = am.group(1)
    # Harvest the year from text with the DOI / arXiv id REMOVED, so a DOI suffix such as
    # 10.1098/rspb.2001.1812 is not mis-read as the year 1812 (false-floor lesson 2b).
    # Prefer a parenthesised year (the publication year in Nature style), else the last bare year.
    yr_src = txt
    if doi:
        yr_src = yr_src.replace(doi, " ")
    if arx:
        yr_src = re.sub(r"arXiv:\s*\d{4}\.\d{4,5}", " ", yr_src, flags=re.I)
    paren = re.findall(r"\((1[89]\d{2}|20\d{2})[a-z]?\)", yr_src)
    year = paren[-1] if paren else None
    if year is None:
        for y in re.findall(r"\b(1[89]\d{2}|20\d{2})\b", yr_src):
            year = y  # last plausible bare year wins
    # title heuristic: text between the author block (ends at first '. ') and the journal/venue
    # authors end at the first period that follows an initial or name run.
    body = txt
    tm = re.match(r"^(.*?[.\?])\s+(.*)$", body)
    title = ""
    if tm:
        after_authors = tm.group(2)
        # title runs until the italic journal (already de-italicised) -> stop at next '. ' or ' vol'
        title = re.split(r"\.\s|\barXiv\b|\bSSRN\b|\bPreprint\b|\bIn Proc", after_authors)[0]
    title = title.strip().rstrip(".")
    is_book = bool(re.search(r"\b(Press|Wiley|Springer|Elsevier|MIT Press|Cambridge Univ|Oxford Univ)\b", txt))
    # DOI-less conference/proceedings venues resolve poorly on CrossRef; treat as REVIEW, not FLAG
    # (false-floor lesson 2c).
    non_journal = bool(arx or re.search(
        r"\b(SSRN|OpenReview|Preprint|working paper|mimeo|PMLR|ICML|NeurIPS|NIPS|ICLR|MLSys|TMLR|"
        r"UAI|AAAI|AISTATS|Proc\.|Proceedings|Conference|Symposium|Workshop|Adv\.\s*Neural)\b", txt, re.I))
    return dict(doi=doi, arxiv=arx, year=year, title=title, text=txt, is_book=is_book, non_journal=non_journal)

# ------------------------------------------------------------------ similarity
_STOP = set("a an the of on in for to and or with from into via using their its is are be as by at".split())
def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2)
def jaccard(a, b):
    A, B = toks(a), toks(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
def containment(sub, whole):
    """Fraction of `sub`'s meaningful tokens present in `whole`.
    Robust to author-initial noise: the full citation text always CONTAINS the real
    title, so a correctly-resolved record's title tokens are (almost) all present."""
    S, W = toks(sub), toks(whole)
    if len(S) < 3:            # too short to score reliably
        return 0.0
    return len(S & W) / len(S)

# ------------------------------------------------------------------ resolvers
def cr_by_doi(doi, mailto):
    body, status = _get(f"{CROSSREF}/{urllib.parse.quote(doi)}?mailto={mailto}", mailto)
    if status == 404 or not body:
        return None
    try:
        return json.loads(body)["message"]
    except Exception:
        return None

def cr_by_query(text, mailto, rows=5):
    q = urllib.parse.quote(text[:400])
    body, _ = _get(f"{CROSSREF}?query.bibliographic={q}&rows={rows}&mailto={mailto}", mailto)
    if not body:
        return []
    try:
        return json.loads(body)["message"]["items"]
    except Exception:
        return []

def arxiv_lookup(aid, mailto):
    body, _ = _get(f"{ARXIV}?id_list={aid}&max_results=1", mailto, accept="application/atom+xml")
    if not body:
        return None
    tm = re.search(r"<entry>.*?<title>(.*?)</title>", body, re.S)
    if not tm or "Error" in body[:200]:
        return None
    return unescape(re.sub(r"\s+", " ", tm.group(1)).strip())

def cr_retracted(msg):
    """Best-effort: does the CrossRef record carry a retraction notice?"""
    for key in ("updated-by", "update-to"):
        for u in (msg.get(key) or []):
            if "retract" in json.dumps(u).lower():
                return True
    for a in (msg.get("assertion") or []):
        if "retract" in (a.get("name", "") + a.get("value", "")).lower():
            return True
    if "retract" in (msg.get("type", "") or "").lower():
        return True
    return False

def cr_title_year(msg):
    t = (msg.get("title") or [""])
    t = t[0] if t else ""
    y = None
    for k in ("published-print", "published-online", "issued", "created"):
        dp = (msg.get(k) or {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            y = str(dp[0][0])
            break
    return t, y

# ------------------------------------------------------------------ per-entry check
def check_entry(key, raw, mailto):
    e = extract(raw)
    disp = e["title"] if len(toks(e["title"])) >= 3 else e["text"][:90]
    r = dict(key=key, doi=e["doi"], arxiv=e["arxiv"], year=e["year"], title=disp,
             verdict="OK", detail="", matched_title="", score=None)

    # (1) DOI present -> resolve + diff (catches hijacking, dead DOI, retraction)
    if e["doi"]:
        msg = cr_by_doi(e["doi"], mailto)
        if msg is None:
            r.update(verdict="FLAG", detail=f"DOI does not resolve on CrossRef: {e['doi']}")
            return r
        mt, my = cr_title_year(msg)
        r["matched_title"] = mt
        sc = containment(mt, e["text"]) if mt else None
        r["score"] = round(sc, 2) if sc is not None else None
        if cr_retracted(msg):
            r.update(verdict="FLAG", detail=f"DOI resolves but record carries a RETRACTION notice: {e['doi']}")
            return r
        if sc is not None and sc < 0.40 and mt:
            r.update(verdict="FLAG",
                     detail=f"identifier-hijack risk: DOI resolves to a DIFFERENT title -> \"{mt}\"")
            return r
        if e["year"] and my and abs(int(e["year"]) - int(my)) > 1:
            r.update(verdict="REVIEW", detail=f"year drift: cited {e['year']} vs record {my}")
            return r
        r.update(verdict="OK", detail=f"DOI resolves; title match={r['score']}, year={my}")
        return r

    # (2) arXiv present -> verify via arXiv API
    if e["arxiv"]:
        at = arxiv_lookup(e["arxiv"], mailto)
        if not at:
            r.update(verdict="FLAG", detail=f"arXiv id not found: {e['arxiv']}")
            return r
        sc = containment(at, e["text"])
        r.update(matched_title=at, score=round(sc, 2))
        if sc < 0.40:
            r.update(verdict="REVIEW", detail=f"arXiv id exists but title differs -> \"{at}\"")
        else:
            r.update(verdict="OK", detail=f"arXiv verified (title containment={r['score']})")
        return r

    # (3) no identifier -> bibliographic search, scored by title-in-citation containment
    items = cr_by_query(e["text"], mailto)
    best, best_sc = None, 0.0
    for it in items:
        mt, _my = cr_title_year(it)
        sc = containment(mt, e["text"])
        if sc > best_sc:
            best, best_sc = it, sc
    r["score"] = round(best_sc, 2)
    my = None
    if best is not None:
        mt, my = cr_title_year(best)
        r["matched_title"] = mt
    # a right-title / wrong-year citation passes a pure containment check; catch it deterministically
    # (false-floor lesson 2: the deterministic gate must still verify year, not only existence).
    year_mismatch = bool(e["year"] and my and abs(int(e["year"]) - int(my)) > 1)
    if best_sc >= 0.70 and not year_mismatch:
        r.update(verdict="OK", detail=f"CrossRef match (title containment={best_sc:.2f})")
    elif best_sc >= 0.70 and year_mismatch:
        r.update(verdict="REVIEW",
                 detail=f"title matches but YEAR differs (cited {e['year']} vs record {my}); check volume/issue")
    elif best_sc >= 0.45:
        r.update(verdict="REVIEW", detail=f"partial CrossRef match ({best_sc:.2f}); verify by hand")
    else:
        if e["is_book"] or e["non_journal"]:
            r.update(verdict="REVIEW",
                     detail="book/preprint/working-paper: not reliably indexed by CrossRef; verify by hand")
        else:
            r.update(verdict="FLAG",
                     detail=f"NO confident match on CrossRef (best score={best_sc:.2f}) -> possible fabrication")
    return r

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--mailto", default="p.iben@saeny.net")
    ap.add_argument("--json", default=None)
    ap.add_argument("--strict", action="store_true", help="also fail on REVIEW items")
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds between requests (politeness)")
    ap.add_argument("--dump-entries", action="store_true",
                    help="print the parsed bibliography as JSON [{key,text,doi,arxiv,year}] and exit "
                         "(feeds the adversarial citation-metadata workflow; no network)")
    a = ap.parse_args()

    tex = open(a.path, encoding="utf-8").read()
    entries = parse_bibfile(tex) if a.path.endswith(".bib") else parse_thebibliography(tex)
    if not entries:
        print("No bibliography entries found.", file=sys.stderr)
        sys.exit(2)

    if a.dump_entries:
        dump = [dict(key=k, **{f: extract(raw)[f] for f in ("text", "doi", "arxiv", "year")})
                for k, raw in entries]
        print(json.dumps(dump))
        sys.exit(0)

    unlisted = []
    if not a.path.endswith(".bib"):
        uncited, unlisted = reference_parity(tex)
        print(f"Reference parity: {len(uncited)} listed-but-uncited, {len(unlisted)} cited-but-unlisted")
        if unlisted:
            print("  >FLAG< cited but NOT in the bibliography (prints as ??):", ", ".join(unlisted))
        if uncited:
            print("  note: listed but never cited:", ", ".join(uncited[:20]) + (" ..." if len(uncited) > 20 else ""))
        print("=" * 78)

    print(f"Checking {len(entries)} references in {a.path}\n" + "=" * 78)
    results = []
    for key, raw in entries:
        res = check_entry(key, raw, a.mailto)
        results.append(res)
        mark = {"OK": "  ok ", "REVIEW": "REVIEW", "FLAG": ">FLAG<"}[res["verdict"]]
        print(f"[{mark}] {key:22.22s} {res['detail']}")
        if res["verdict"] != "OK" and res["matched_title"]:
            print(f"            cited : {res['title'][:88]}")
            print(f"            record: {res['matched_title'][:88]}")
        time.sleep(a.sleep)

    flags = [r for r in results if r["verdict"] == "FLAG"]
    review = [r for r in results if r["verdict"] == "REVIEW"]
    print("=" * 78)
    print(f"SUMMARY: {len(results)-len(flags)-len(review)} OK | {len(review)} REVIEW | {len(flags)} FLAG")
    if flags:
        print("HARD FLAGS (fabrication / hijack / dead-DOI / retraction):")
        for r in flags:
            print(f"  - {r['key']}: {r['detail']}")
    if a.json:
        json.dump(results, open(a.json, "w"), indent=2)
        print(f"[wrote {a.json}]")

    fail = bool(flags) or bool(unlisted) or (a.strict and bool(review))
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()
