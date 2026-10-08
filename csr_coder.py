"""
CSR Coding Workbench — local edition
====================================
The same coding logic as the hosted page, run on your own machine against your
own Anthropic API key. The rules, the codebook vocabulary, the worked
precedents, the exclusion list and the partner register all come from
`csr_rules.json`, which is exported from the hosted tool — so the two cannot
drift apart.

Run it with:    streamlit run csr_coder.py

See README_for_researchers.md for the step-by-step, including how to get a key.
"""

import io
import json
import os
import re
import unicodedata
from collections import defaultdict

import streamlit as st

# ----------------------------------------------------------------- the rules
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "csr_rules.json"), encoding="utf-8") as fh:
    R = json.load(fh)

VOCAB = R["vocab"]
RULES = R["rules"]
PRECEDENTS = R["precedents"]
REGISTER = R["register"]
EXCLUSIONS = R["exclusions"]
COLS = R["columns"]
HEADERS = R["headers"]
PURPOSES = ["acc", "adv", "cap", "dig", "drug", "rnd", "edu", "fund"]
PURPOSE_LABEL = {"acc": "Access to care", "adv": "Advocacy", "cap": "Capacity building",
                 "dig": "Digitalization", "drug": "Drug availability", "rnd": "R&D",
                 "edu": "Education", "fund": "Funding"}
CONTROLLED = {"report": "Report type", "status": "Initiative status", "loc": "Initiative Location",
              "ptn": "Initiative Partners", "target": "Initiative Target"}

PASS_CHARS = 24000   # about nine pages a pass. Smaller passes repeat the
                     # one-page overlap and the protocol far more often: at
                     # 10,000 a 128-page report cost 65 passes and 834,000
                     # input tokens, against 18 passes and 369,000 here.
LANES = 4


# ------------------------------------------------------------------ text work
def norm(s):
    s = unicodedata.normalize("NFKC", str(s or ""))
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("‑", "-"), (" ", " "),
                 ("­", "")]:
        s = s.replace(a, b)
    s = re.sub(r"-\s*\n\s*", "", s)
    return re.sub(r"\s+", " ", s).strip()


def squash(s):
    return re.sub(r"[^a-z0-9]", "", norm(s).lower())


def page_lines(page):
    """The PDF's own reading order, which keeps columns apart. Sorting words by
    position instead runs two columns into each other, and that is how a
    description ends up stitched from the wrong column - the error that took
    three rows out of the first Lilly 2023 build."""
    return [ln.strip() for ln in (page.extract_text(use_text_flow=True) or "").split("\n")]


def strip_furniture(all_lines):
    """Running headers, footers and page numbers sit on their own line, so once
    the newlines are collapsed they land in the middle of a sentence and the
    description stops being verbatim. A line that repeats across a quarter of
    the report is furniture, not content."""
    from collections import Counter
    seen = Counter()
    for lines in all_lines:
        for ln in {l for l in lines if l and len(l) <= 90}:
            seen[re.sub(r"\d+", "#", ln)] += 1
    floor = max(3, int(len(all_lines) * 0.25))
    boiler = {k for k, v in seen.items() if v >= floor}
    out = []
    for lines in all_lines:
        out.append([l for l in lines if l
                    and re.sub(r"\d+", "#", l) not in boiler
                    and not re.fullmatch(r"[\d\s|/.\-ivxlcIVXLC]{1,14}", l)])
    return out


def assemble(lines):
    text = ""
    for ln in lines:
        if not text:
            text = ln
        elif re.search(r"[-‐-―]$", text) and re.match(r"^[a-z]", ln):
            text = re.sub(r"[-‐-―]$", "", text) + ln
        else:
            text += " " + ln
    return norm(text)


def sorted_text(page):
    """A second reading of the page, words sorted by position. Used only as a
    fallback when verifying a description, never for the text sent to Claude."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return ""
    left = min(w["x0"] for w in words)
    right = max(w["x1"] for w in words)
    width = right - left
    split = None
    if width > 200:
        bins, cov = 72, [0] * 72
        for w in words:
            a = max(0, int((w["x0"] - left) / width * bins))
            b = min(bins - 1, int((w["x1"] - left) / width * bins))
            for i in range(a, b + 1):
                cov[i] += 1
        i = int(bins * 0.33)
        while i <= int(bins * 0.67):
            if cov[i] == 0:
                j = i
                while j < bins and cov[j] == 0:
                    j += 1
                if j - i >= 3:
                    split = left + ((i + j) / 2) / bins * width
                    break
                i = j
            i += 1
    groups = [words] if split is None else [
        [w for w in words if (w["x0"] + w["x1"]) / 2 < split],
        [w for w in words if (w["x0"] + w["x1"]) / 2 >= split]]
    out = []
    for g in groups:
        if not g:
            continue
        g.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
        line, last_top = [], None
        for w in g:
            if last_top is None or abs(w["top"] - last_top) <= 3:
                line.append(w)
                last_top = w["top"] if last_top is None else last_top
            else:
                out.append(" ".join(x["text"] for x in sorted(line, key=lambda x: x["x0"])))
                line, last_top = [w], w["top"]
        if line:
            out.append(" ".join(x["text"] for x in sorted(line, key=lambda x: x["x0"])))
    return assemble(out)


def read_pdf(upload, progress=None):
    """Returns (pages, alt): `pages` is what Claude reads and what a description
    is checked against; `alt` is the second reading, consulted only when the
    first one fails a check."""
    import pdfplumber
    raw, alt = [], []
    with pdfplumber.open(upload) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            raw.append(page_lines(page))
            alt.append(sorted_text(page))
            if progress and (i % 5 == 0 or i == total):
                progress(i / total, "Reading the report — page %d of %d" % (i, total))
    return [assemble(l) for l in strip_furniture(raw)], alt


class Book(list):
    """The report's pages, plus the second reading used only for verification."""

    def __init__(self, pages, alt=None):
        super().__init__(pages)
        self.alt = list(alt or [])


def verify_desc(desc, pages, pdf_page):
    """The description must sit inside ONE page's text, not a concatenation of
    two: a concatenation is exactly what lets a two-column stitch pass
    unnoticed."""
    if not desc or len(norm(desc)) < 20:
        return False
    d = norm(desc)
    forms = (d, d.replace("-", ""), squash(d))
    def holds(t):
        return forms[0] in t or forms[1] in t.replace("-", "") or forms[2] in squash(t)

    for source in (pages, getattr(pages, "alt", [])):
        for p in (pdf_page, pdf_page + 1, pdf_page - 1):
            if 1 <= p <= len(source) and holds(source[p - 1]):
                return True
        # a passage may run over a page break, so two CONSECUTIVE pages count as
        # one slice - but never two arbitrary pages, and never two columns
        for p in (pdf_page, pdf_page - 1):
            if 1 <= p < len(source) and holds(source[p - 1] + " " + source[p]):
                return True
    return False


# -------------------------------------------------------------- the prompting
def rules_text():
    return "\n".join("- %s: %s" % (a.replace("&#8211;", "-").replace("&#8212;", "-").replace("&amp;", "&"), b)
                     for a, b in RULES)


def precedent_text():
    return "\n\n".join("CASE: %s\n%s" % (p["case"], json.dumps(p["row"]).replace("&amp;", "&"))
                       for p in PRECEDENTS)


def ref_block(refs, company, chunk_text):
    hits = []
    t = squash(chunk_text)
    for r in refs:
        if not same_company(r.get("company", ""), company):
            continue
        pn, gn = squash(r.get("pname", "")), squash(r.get("group", ""))
        if (len(pn) >= 6 and pn in t) or (len(gn) >= 8 and gn in t):
            hits.append(r)
        if len(hits) >= 6:
            break
    if not hits:
        return ""
    rows = "\n".join("%s · %s" % (r.get("year", ""), json.dumps(
        {k: r.get(k, "") for k in ["group", "status", "loc", "cty", "ptn", "pname"] + PURPOSES + ["target"]}))
        for r in hits)
    return ("ROWS THIS TEAM HAS ALREADY CODED AND REVIEWED FOR %s, for partners or initiatives named in the "
            "passage below. Where the passage describes the same initiative or the same partner, REPRODUCE "
            "THESE VALUES EXACTLY - same Partners category, same Partner name spelling, same Initiative "
            "Target, same purpose keywords - and take only the description, the page and the status from this "
            "year's text:\n%s\n" % (company.upper(), rows))


def protocol_block():
    """The part of the prompt that is identical on every pass of every report.
    It is sent as the system prompt with a cache marker, so after the first pass
    it is charged at a twentieth of the input price instead of in full. Nothing
    report-specific may go in here, or the cache never hits."""
    return "\n".join([
        "You are coding corporate social initiatives into an academic firm-year-initiative panel dataset.",
        "Follow these rules exactly; they are the project's coding protocol.",
        rules_text(), "",
        "ALLOWED VALUES (use these strings exactly; several comma-separated keywords per purpose column are "
        "allowed; leave a column as an empty string when it does not apply):",
        json.dumps(VOCAB), "",
        "WORKED ROWS FROM THIS PROJECT'S OWN FILES. Match them: they settle how the rules are applied in "
        "practice, and a row that contradicts them is wrong even if it looks defensible.",
        precedent_text(), "",
        "Find EVERY distinct social initiative in the text you are given. Completeness matters more than "
        "brevity: a missed initiative is the worst outcome.",
        "One page often carries several distinct initiatives - a paragraph, a bullet, a table row, a sidebar, a "
        "pull quote and a caption can each be one. Code each separately rather than merging them, and code each "
        "clause of a list separately where the clauses name different programmes, partners or countries.",
        "A LIST IS NOT ONE ROW. A bulleted or run-on list of grants, donations, awards or responses - "
        "\"$500,000 to A ... $200,000 to B\" - is ONE ROW PER ITEM, each with its own recipient in Partner name, "
        "its own Initiative Target and its own purpose.",
        "Apply the firm-action rule before you fill the purpose columns: ask what THIS COMPANY did. Where it "
        "gave money or product and a partner procured, installed, distributed, screened, trained or ran the "
        "programme, the purposes are the funding keyword alone.",
        "Where you cannot tell whether something qualifies, INCLUDE it as a row and begin its notes with FLAG.",
        EXCLUSIONS, "",
        'Reply with ONLY a JSON array of row objects, each exactly: {"initiative":string,"page":number,'
        '"status":string,"loc":string,"cty":string,"ptn":string,"pname":string,"acc":string,"adv":string,'
        '"cap":string,"dig":string,"drug":string,"rnd":string,"edu":string,"fund":string,"target":string,'
        '"otherdesc":string,"desc":string,"notes":string}',
        '"desc" must be copied VERBATIM from the report text you are given: character for character, no '
        'paraphrase, no ellipsis. Make it the COMPLETE passage that carries the initiative - normally two to '
        'six consecutive sentences, including the heading or lead-in and the sentence carrying the figures, '
        'the partner name and the countries.',
        '"page" is the printed page the description sits on. "initiative" is a short name. At least one purpose '
        'column must be non-empty.',
    ])


def build_prompt(meta, chunk, refs, extra=""):
    """The report-specific half: what changes from pass to pass."""
    text = "\n".join("<<<PAGE %d>>>\n%s\n" % (p, t) for p, t in chunk)
    parts = [
        ref_block(refs, meta["company"], text), "",
        "COMPANY: %s   REPORT YEAR: %s   REPORT TYPE: %s" % (meta["company"], meta["year"], meta["report"]),
        "REPORT TITLE for the Notes line: %s" % meta["title"],
        "Printed page = PDF page + %d (the text below is marked with PDF page numbers; cite the PRINTED page)."
        % meta["offset"], "",
        "REPORT TEXT:", text, "",
        extra,
    ]
    return "\n".join(x for x in parts if x)


# ------------------------------------------------------------------ the model
# Prices are USD per million tokens, as published 2026-10-08. Keep them here
# rather than in the UI text, so the running cost and the label cannot disagree.
MODELS = {
    "claude-sonnet-5-5": {"label": "Sonnet 5.5 - the default: best accuracy per dollar",
                          "in": 2.0, "out": 10.0},
    "claude-haiku-5-5": {"label": "Haiku 5.5 - about 20x cheaper input; check a report you have already coded "
                                  "before trusting it", "in": 0.10, "out": 0.50},
    "claude-opus-5-5": {"label": "Opus 5.5 - the most capable, twice the price of Sonnet",
                        "in": 4.0, "out": 20.0},
}
CACHE_READ = 0.05        # a cache hit costs this share of the input price
CACHE_WRITE = 1.25       # writing the cache costs this multiple, once
MAX_TOKENS = 16000


def spend(model):
    u = st.session_state.usage
    p = MODELS[model]
    return (u["in"] * p["in"] + u["cache_read"] * p["in"] * CACHE_READ
            + u["cache_write"] * p["in"] * CACHE_WRITE + u["out"] * p["out"]) / 1e6


def call_claude(client, model, prompt, max_tokens=MAX_TOKENS):
    """One pass. The protocol goes in the system prompt behind a cache marker,
    so it is charged in full once and at a twentieth of that on every later
    pass. Returns (rows, truncated)."""
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=0,
        system=[{"type": "text", "text": protocol_block(),
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}])
    u = getattr(msg, "usage", None)
    if u is not None and "usage" in st.session_state:
        acc = st.session_state.usage
        acc["in"] += getattr(u, "input_tokens", 0) or 0
        acc["out"] += getattr(u, "output_tokens", 0) or 0
        acc["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        acc["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    truncated = getattr(msg, "stop_reason", "") == "max_tokens"
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return [], truncated
    try:
        out = json.loads(m.group(0))
        return (out if isinstance(out, list) else []), truncated
    except json.JSONDecodeError:
        return [], truncated


def draft_chunk(client, model, meta, chunk, refs, extra=""):
    """A pass, and if the answer was cut off at the token limit, the same pages
    again in two halves - otherwise a long page's last initiatives are lost
    silently, which is the one real risk of sending more pages at a time."""
    rows, truncated = call_claude(client, model, build_prompt(meta, chunk, refs, extra))
    if truncated and len(chunk) > 1:
        half = len(chunk) // 2
        rows = []
        for part in (chunk[:half], chunk[half:]):
            more, _ = call_claude(client, model, build_prompt(meta, part, refs, extra))
            rows += more
    return rows


# ------------------------------------------------------ coercion and alignment
def pick(v, allowed, fallback):
    v = norm(v)
    if v in allowed:
        return v
    for a in allowed:
        if squash(a) == squash(v):
            return a
    return fallback


def clean_list(v, allowed):
    out = []
    for part in [p.strip() for p in norm(v).split(",") if p.strip()]:
        hit = next((a for a in allowed if squash(a) == squash(part)), None) or \
              next((a for a in allowed if squash(a).startswith(squash(part))), None)
        if hit and hit not in out:
            out.append(hit)
    return ", ".join(out)


CO_STOP = {"the", "and", "inc", "incorporated", "plc", "ltd", "limited", "llc", "lp", "sa", "nv", "ag",
           "gmbh", "co", "company", "corp", "corporation", "group", "holdings", "holding", "pharma",
           "pharmaceutical", "pharmaceuticals", "laboratories", "labs", "international", "global", "usa"}
CO_ALIAS = [("gsk", "glaxosmithkline"), ("jnj", "johnsonjohnson"), ("bms", "bristolmyerssquibb"),
            ("msd", "merck"), ("bi", "boehringeringelheim"), ("novo", "novonordisk")]


def co_tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", norm(s).lower()) if len(t) >= 2 and t not in CO_STOP]


def same_company(a, b):
    """“Lilly” in the master and “Eli Lilly and Company” typed into the form are the same
    firm. A prefix test alone misses that, and a missed match means a reviewed row
    silently fails to carry — the quietest way for two runs to disagree."""
    x, y = squash(a), squash(b)
    if not x or not y:
        return True
    if x == y:
        return True
    for p, q in CO_ALIAS:
        if (p in x and q in y) or (q in x and p in y):
            return True
    ta, tb = co_tokens(a), co_tokens(b)
    if ta and tb and any(t in tb for t in ta if len(t) >= 4):
        return True
    short, long_ = (x, y) if len(x) < len(y) else (y, x)
    return len(short) >= 3 and long_.startswith(short)


def align(row, refs, pages):
    """Deterministic: no model involved, so the same input gives the same output."""
    if not refs:
        return row
    notes = []
    same = None
    g, d = squash(row["group"]), squash(row["desc"])[:70]
    for r in refs:
        if not same_company(r.get("company"), row["company"]):
            continue
        rg, rd = squash(r.get("group", "")), squash(r.get("desc", ""))[:70]
        if g and rg and (g == rg or (len(g) > 8 and len(rg) > 8 and (g in rg or rg in g))):
            same = r
            break
        if d and rd and len(d) > 30 and d[:45] == rd[:45] and same is None:
            same = r
    if same:
        kept = []
        for key, label in (("ptn", "Partners"), ("target", "Target")):
            if same.get(key) and same[key] != row[key]:
                kept.append("%s %s -> %s" % (label, row[key], same[key]))
                row[key] = same[key]
        if same.get("pname"):
            if squash(same["pname"]) != squash(row["pname"]):
                kept.append("Partner name -> " + same["pname"])
            row["pname"] = same["pname"]
        for k in PURPOSES:
            now = clean_list(same.get(k, ""), VOCAB[k])
            if squash(row[k]) != squash(now):
                kept.append("%s “%s” -> “%s”" % (PURPOSE_LABEL[k], row[k] or "blank", now or "blank"))
                row[k] = now
        if same.get("group"):
            row["group"] = same["group"]
        if same.get("desc") and squash(same["desc"]) != squash(row["desc"]) and \
                verify_desc(same["desc"], pages, int(row["_pdfpage"] or 0)):
            kept.append("Description replaced with the reviewed wording, which the report still carries verbatim")
            row["desc"] = same["desc"]
            row["verified"] = True
        notes.append("Carried from the reviewed %s %s row for the same initiative: %s." %
                     (same.get("company"), same.get("year"), "; ".join(kept)) if kept else
                     "Matches the reviewed %s %s row for the same initiative." % (same.get("company"), same.get("year")))
    else:
        pn = squash(row["pname"])
        if len(pn) >= 5:
            cand = [r for r in refs if squash(r.get("pname", "")) and
                    (squash(r["pname"]) == pn or pn in squash(r["pname"]) or squash(r["pname"]) in pn)]
            if cand:
                counts = defaultdict(int)
                for r in cand:
                    counts[r.get("ptn", "")] += 1
                want = max(counts, key=counts.get)
                row["pname"] = cand[0]["pname"]
                if want and want != row["ptn"]:
                    notes.append("Partners aligned to the reviewed rows for %s: %s -> %s."
                                 % (cand[0]["pname"], row["ptn"], want))
                    row["ptn"] = want
    if notes:
        row["notes"] = (row["notes"] + " " + " ".join(notes)).strip()
    return row


def make_row(r, meta, pages):
    page = int(r.get("page") or 0)
    pdf_page = page - meta["offset"]
    row = {
        "company": meta["company"], "year": int(meta["year"]), "report": meta["report"],
        "status": pick(r.get("status"), VOCAB["status"], "Ongoing"),
        "loc": pick(r.get("loc"), VOCAB["loc"], "Unspecified"),
        "cty": norm(r.get("cty")) or "Unspecified",
        "ptn": pick(r.get("ptn"), VOCAB["ptn"], "Go alone"),
        "pname": norm(r.get("pname")),
        "target": pick(r.get("target"), VOCAB["target"], "Unspecified target"),
        "otherdesc": norm(r.get("otherdesc")), "desc": norm(r.get("desc")),
        "group": norm(r.get("initiative")) or "(unnamed)", "page": page,
        "_pdfpage": pdf_page,
    }
    for k in PURPOSES:
        row[k] = clean_list(r.get(k, ""), VOCAB[k])
    row["verified"] = verify_desc(row["desc"], pages, pdf_page)
    row["notes"] = ("Initiative: %s. Source: %s, p.%s. %s"
                    % (row["group"], meta["title"], page or "?", norm(r.get("notes")))).strip()
    if not any(row[k] for k in PURPOSES):
        row["notes"] += " FLAG: no purpose keyword proposed."
    return row


# --------------------------------------------------------------- audit + score
def audit_row(r):
    f = []
    for k, label in CONTROLLED.items():
        v = norm(r.get(k))
        if not v:
            f.append(label + " is blank")
        elif v not in VOCAB[k]:
            f.append("%s is not a codebook value: “%s”" % (label, v))
    if not norm(r.get("cty")):
        f.append("Countries list is blank")
    if not norm(r.get("desc")):
        f.append("Description is blank")
    for k in PURPOSES:
        for part in [p.strip() for p in norm(r.get(k)).split(",") if p.strip()]:
            if part not in VOCAB[k]:
                f.append("%s keyword is not in the codebook: “%s”" % (PURPOSE_LABEL[k], part))
    if not any(norm(r.get(k)) for k in PURPOSES):
        f.append("no purpose keyword, so CHECK is FALSE")
    if not r.get("page"):
        f.append("no page number")
    nt = norm(r.get("notes"))
    if not nt.startswith("Initiative:") or ", p." not in nt:
        f.append("Notes do not open with the initiative name and the source page")
    if r.get("verified") is False:
        f.append("description was not found verbatim on the page it cites")
    pk = squash(r.get("pname"))
    for key, want in REGISTER:
        if pk and key in pk and r.get("ptn") != want:
            f.append("the panel register fixes this collaboration as " + want)
    return f


def confidence(r, refs):
    why = []
    if r.get("verified") is False:
        why.append("the description was not found verbatim on the page it cites")
    issues = [x for x in audit_row(r) if "verbatim" not in x]
    why += issues
    if re.search(r"\bFLAG\b", norm(r.get("notes"))):
        why.append("Claude flagged a judgement call in the Notes")
    hard = r.get("verified") is False or any(
        re.search(r"blank|codebook|CHECK|register", x) for x in issues)
    level = "Check" if hard else ("Ambiguous" if why else "Clean")
    if refs and level == "Clean":
        pn = squash(r.get("pname"))
        known = any(squash(x.get("pname", "")) and (pn == squash(x["pname"]) or
                    (len(pn) >= 5 and (pn in squash(x["pname"]) or squash(x["pname"]) in pn))) for x in refs)
        if not known:
            level = "Ambiguous"
            why.append("no counterpart among your reference rows - new to this year, or a new initiative")
    return level, why


# ------------------------------------------------------------------- the app
st.set_page_config(page_title="CSR Coding Workbench", layout="wide")
st.title("CSR Coding Workbench")
st.caption("Local edition — the same rules, precedents and register as the hosted tool, "
           "read from csr_rules.json so the two cannot drift apart.")

if "library" not in st.session_state:
    st.session_state.library = []
if "refs" not in st.session_state:
    st.session_state.refs = []
if "drafts" not in st.session_state:
    st.session_state.drafts = []
if "usage" not in st.session_state:
    st.session_state.usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}

with st.sidebar:
    st.header("Setup")
    key = st.text_input("Anthropic API key", type="password",
                        value=os.environ.get("ANTHROPIC_API_KEY", ""),
                        help="Starts with sk-ant-. It is held in memory for this session only.")
    model = st.selectbox("Model", list(MODELS), index=0,
                         format_func=lambda m: MODELS[m]["label"])
    st.divider()
    st.header("Cost")
    u = st.session_state.usage
    if u["in"] or u["cache_read"]:
        st.metric("Spent this session", "$%.2f" % spend(model))
        st.caption("%s input tokens, %s of them read from the cache at a twentieth of the price; "
                   "%s output." % ("{:,}".format(u["in"]), "{:,}".format(u["cache_read"]),
                                   "{:,}".format(u["out"])))
    else:
        st.caption("Nothing spent yet. A 130-page report costs roughly $0.70–$0.90 on Sonnet 5.5, "
                   "or a few cents on Haiku 5.5.")
    st.divider()
    st.header("Reference rows")
    st.caption("Rows you have already coded and reviewed. They are used as precedents in the prompt and to "
               "align drafted rows deterministically.")
    ref_file = st.file_uploader("Reviewed rows (.xlsx or .csv)", type=["xlsx", "xls", "csv"], key="ref")
    if ref_file is not None and st.button("Load reference rows"):
        import pandas as pd
        df = pd.read_excel(ref_file) if ref_file.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(ref_file)
        colmap = {}
        for c in df.columns:
            t = str(c).lower().replace("_", " ").strip()
            for pat, key_ in [(r"^company|^firm", "company"), (r"year", "year"), (r"report.?type|^type$", "report"),
                              (r"status", "status"), (r"location", "loc"), (r"countr", "cty"),
                              (r"partner.?name", "pname"), (r"initiative.?partner|^partners?$", "ptn"),
                              (r"access", "acc"), (r"advoca", "adv"), (r"capacit", "cap"), (r"digital", "dig"),
                              (r"drug", "drug"), (r"r.?&?.?d", "rnd"), (r"educat", "edu"), (r"funding|^fund", "fund"),
                              (r"other.?target", "otherdesc"), (r"initiative.?target|^target", "target"),
                              (r"description", "desc"), (r"note", "notes"),
                              (r"initiative.?group|^group", "group")]:
                if re.search(pat, t) and key_ not in colmap.values():
                    colmap[c] = key_
                    break
        rows = []
        for _, rec in df.iterrows():
            r = {v: norm(rec[k]) for k, v in colmap.items()}
            if not r.get("company") or not r.get("desc") or len(r["desc"]) < 15:
                continue
            m = re.search(r"(19|20)\d{2}", str(r.get("year", "")))
            if not m:
                continue
            r["year"] = int(m.group(0))
            rows.append(r)
        st.session_state.refs = rows
        st.success("%d reference rows loaded, covering %s." %
                   (len(rows), ", ".join(sorted({"%s %s" % (r["company"], r["year"]) for r in rows}))))
    if st.session_state.refs:
        st.info("%d reference rows in memory." % len(st.session_state.refs))

tab_code, tab_lib, tab_rules = st.tabs(["Code a report", "Library", "Rules in force"])

with tab_code:
    c1, c2, c3, c4 = st.columns([2, 1, 1.4, 1])
    company = c1.text_input("Company", placeholder="Eli Lilly and Company")
    year = c2.text_input("Report year", placeholder="2023")
    report = c3.selectbox("Report type", VOCAB["report"], index=1)
    offset = c4.number_input("Printed page = PDF page +", value=0, step=1)
    title = st.text_input("Report title for the Notes line",
                          value=("%s %s %s" % (company, report, year)).strip() if company and year else "")
    pdf = st.file_uploader("The report (PDF)", type=["pdf"])

    st.caption("Every page with readable text is coded. There is no page filter: measured against 115 "
               "reviewed rows, filtering by initiative language kept 80% of the text and lost 6 "
               "initiatives. If a run costs more than you want, change the model, not the page count.")

    if st.button("Generate coded data", type="primary", disabled=not (pdf and company and year and key)):
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        meta = {"company": company.strip(), "year": year.strip(), "report": report,
                "title": title.strip() or "%s %s %s" % (company, report, year), "offset": int(offset)}
        bar0 = st.progress(0.0, text="Reading the report")
        body, alt = read_pdf(pdf, lambda f, t: bar0.progress(f, text=t))
        pages = Book(body, alt)
        bar0.empty()
        st.write("%d pages read." % len(pages))

        # EVERY page with readable text. No filter, and no option for one.
        live = [i for i, t in enumerate(pages, 1) if len(squash(t)) >= 150]

        chunks, cur, size = [], [], 0
        for i in live:
            t = pages[i - 1]
            piece = len(t) + 20
            if size + piece > PASS_CHARS and cur:
                chunks.append(cur)
                cur, size = [cur[-1]], len(cur[-1][1])      # one page of overlap
            cur.append((i, t))
            size += piece
        if cur:
            chunks.append(cur)

        drafts = []
        bar = st.progress(0.0, text="Drafting")
        for n, ch in enumerate(chunks, 1):
            bar.progress(n / len(chunks), text="Pass %d of %d — %d rows so far · $%.2f"
                         % (n, len(chunks), len(drafts), spend(model)))
            for raw in draft_chunk(client, model, meta, ch, st.session_state.refs):
                row = make_row(raw, meta, pages)
                key_ = squash(row["desc"])[:80]
                if key_ and any(squash(d["desc"])[:80] == key_ for d in drafts):
                    continue
                drafts.append(align(row, st.session_state.refs, pages))
        bar.empty()

        # the itemised-giving sweep: a list of grants is one row per item
        give = re.compile(r"\$\s?\d[\d.,]*\s*(?:million|billion)?\s+to\s+(?:the\s+)?"
                          r"([A-Z][A-Za-z0-9&.,'’ -]{3,70}?)(?=\s+(?:to|for|as|related|in|that|which)\s|[.,;])")
        have = " ".join(squash(d["pname"] + d["desc"][:200]) for d in drafts)
        gaps = []
        for i, t in enumerate(pages, 1):
            miss = [m.group(0).strip() for m in give.finditer(t)
                    if squash(m.group(1))[:18] and squash(m.group(1))[:18] not in have]
            if miss:
                gaps.append((i, t, miss))
        if gaps:
            st.write("Itemised giving: %d clauses with no row of their own." % sum(len(g[2]) for g in gaps))
            for i, t, miss in gaps:
                extra = ("THIS IS AN ITEMISED-GIVING PASS. Return ONE ROW PER ITEM; a list of grants is never a "
                         "single row. The items with no row yet:\n" + "\n".join("- " + m for m in miss))
                for raw in draft_chunk(client, model, meta, [(i, t)], st.session_state.refs, extra):
                    row = make_row(raw, meta, pages)
                    key_ = squash(row["desc"])[:80]
                    if key_ and any(squash(d["desc"])[:80] == key_ for d in drafts):
                        continue
                    drafts.append(align(row, st.session_state.refs, pages))

        st.session_state.drafts = drafts
        st.success("%d rows drafted from %d pages — $%.2f spent on this report."
                   % (len(drafts), len(live), spend(model)))

    if st.session_state.drafts:
        import pandas as pd
        rows = st.session_state.drafts
        view = []
        for r in rows:
            level, why = confidence(r, st.session_state.refs)
            view.append({"Confidence": level, "Why": "; ".join(why), **{HEADERS[i]: r[c] for i, c in enumerate(COLS)},
                         "Initiative group": r["group"], "Printed page": r["page"]})
        st.subheader("Draft rows — review before saving")
        st.dataframe(pd.DataFrame(view), use_container_width=True, height=400)
        if st.button("Save to the library (replaces this company-year)"):
            co, yr = rows[0]["company"], rows[0]["year"]
            before = len(st.session_state.library)
            st.session_state.library = [r for r in st.session_state.library
                                        if not (squash(r["company"]) == squash(co) and r["year"] == yr)]
            removed = before - len(st.session_state.library)
            st.session_state.library += rows
            st.session_state.drafts = []
            st.success("Saved %d rows%s." % (len(rows),
                       ", replacing the %d this company-year held before" % removed if removed else ""))
            st.rerun()

with tab_lib:
    if not st.session_state.library:
        st.info("Nothing saved yet.")
    else:
        import pandas as pd
        lib = st.session_state.library
        st.write("%d rows · %s" % (len(lib), ", ".join(sorted({"%s %s" % (r["company"], r["year"]) for r in lib}))))
        counts = defaultdict(int)
        for r in lib:
            counts[confidence(r, st.session_state.refs)[0]] += 1
        st.write("Confidence: %d clean · %d to check · %d ambiguous"
                 % (counts["Clean"], counts["Check"], counts["Ambiguous"]))

        groups = defaultdict(list)
        for r in lib:
            groups[(squash(r["company"]), squash(r["group"]) or squash(r["pname"]))].append(r)
        ordered = []
        multi = [g for g in groups.values() if len({x["year"] for x in g}) > 1]
        single = [g for g in groups.values() if len({x["year"] for x in g}) == 1]
        for g in multi + single:
            for r in sorted(g, key=lambda x: x["year"]):
                r["_v"] = (r["group"] if g in multi else "[single year] " + r["group"])
                ordered.append(r)

        out = []
        for i, r in enumerate(ordered, 2):
            issues = audit_row(r)
            notes = r["notes"] + ((" FLAG: " + "; ".join(issues) + ".") if issues else "")
            rec = {HEADERS[j]: (r["year"] if c == "year" else (notes if c == "notes" else r[c]))
                   for j, c in enumerate(COLS)}
            rec["CHECK"] = ('=IF((COUNTA(C{0})+COUNTA(D{0})+COUNTA(E{0})+COUNTA(G{0})+COUNTA(Q{0})+COUNTA(S{0}))=6,'
                            'IF(COUNTA(I{0}:P{0})>=1,"TRUE","FALSE"),"FALSE")').format(i)
            rec["Initiative group"] = r.get("_v", r["group"])
            out.append(rec)
        df = pd.DataFrame(out)
        st.dataframe(df, use_container_width=True, height=380)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xl:
            df.to_excel(xl, index=False, sheet_name="Rows")
        st.download_button("Download .xlsx", buf.getvalue(), "csr_rows.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download .csv", df.to_csv(index=False).encode("utf-8"), "csr_rows.csv", "text/csv")
        st.text_area("Paste-ready block — select all, copy, paste into the master",
                     df.to_csv(index=False, sep="\t", header=False), height=140)

with tab_rules:
    st.caption("Sent to Claude with every pass, so a row can always be traced to the rule that produced it.")
    for name, body in RULES:
        st.markdown("**%s**" % name.replace("&#8211;", "–").replace("&#8212;", "—").replace("&amp;", "&"))
        st.write(body.replace("&amp;", "&"))
    st.markdown("**Exclusions carried into every sweep**")
    st.write(EXCLUSIONS)
    st.markdown("**Named collaborations fixed panel-wide**")
    st.table([{"Collaboration": k, "Partners": v} for k, v in REGISTER])
