"""Statistics of an artist's lyrics across albums.

Unit of evidence is the song: a chorus repeated many times is one song's worth of evidence.
Rates that could be inflated by repetition are computed on each song's lines with in-song
repeated lines removed ("unique lines"). Every test lands in ``tests`` with its effect size,
p (or BH-corrected q), and a plain verdict, including the ones that fail.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
from scipy import stats as ss

from .. import lyrics as lyr

RNG_SEED = 0
CONTRACTIONS = [
    (r"\b(i)'m\b", r"\1 am"), (r"\b(\w+)'re\b", r"\1 are"), (r"\b(\w+)'ve\b", r"\1 have"),
    (r"\b(\w+)'ll\b", r"\1 will"), (r"\b(\w+)'d\b", r"\1 would"), (r"\bcan't\b", "can not"),
    (r"\bwon't\b", "will not"), (r"\b(\w+)n't\b", r"\1 not"), (r"\b(it|that|what|there|he|she|who)'s\b", r"\1 is"),
    (r"\b(\w+)in'\b", r"\1ing"),
]
PRONOUNS = {
    "I": {"i", "me", "my", "mine", "myself"},
    "we": {"we", "us", "our", "ours", "ourselves"},
    "you": {"you", "your", "yours", "yourself", "yourselves"},
}
# Generic theme lexicon (lemmas). A choice, not ground truth: other lists give other rates.
THEMES = {
    "water": "water sea ocean river rain wave tide drown flood stream lake deep shore swim whale fish ice",
    "sky & cosmos": "sky star sun moon heaven space planet cosmos universe cloud wind air fly wing orbit galaxy",
    "light": "light shine bright glow sun ray dawn spark gleam flame radiant",
    "darkness": "dark darkness shadow night black gloom shade dusk void",
    "fire": "fire burn flame ash smoke heat blaze ember inferno",
    "earth & nature": "earth ground land soil stone rock mountain tree forest root seed leaf grass dust nature field",
    "life & growth": "life live alive born birth grow breathe breath awake rise heal",
    "death & pain": "death die dead kill blood pain bleed grave wound hurt suffer corpse murder",
    "mind & soul": "mind soul spirit dream think thought conscious believe memory god pray faith",
    "love": "love heart kiss baby darling lover desire hold touch embrace",
    "time": "time forever moment day year age eternal past future yesterday tomorrow",
    "conflict": "war fight battle enemy weapon gun army soldier strike hate rage anger destroy",
}
PAIRS = [
    ("Light vs dark", {"light", "bright", "shine", "day"}, {"dark", "darkness", "shadow", "night"}),
    ("Life vs death", {"life", "live", "living", "alive"}, {"death", "die", "dead", "dying"}),
    ("Love vs hate", {"love"}, {"hate"}),
    ("Rise vs fall", {"rise", "rising", "up"}, {"fall", "falling", "down"}),
    ("Heaven vs hell", {"heaven"}, {"hell"}),
    ("Hope vs fear", {"hope"}, {"fear"}),
]


# ----------------------------------------------------------------------------- text prep

def expand(text: str) -> str:
    t = text.lower().replace("’", "'")
    for pat, rep in CONTRACTIONS:
        t = re.sub(pat, rep, t)
    return t


def song_lines(lyrics: str) -> List[str]:
    return [expand(ln) for ln in lyr.clean_lines(lyrics)]


def unique_lines(lines: List[str]) -> List[str]:
    seen, out = set(), []
    for ln in lines:
        k = re.sub(r"\W+", " ", ln.lower()).strip()
        if k and k not in seen:
            seen.add(k)
            out.append(ln)
    return out


_lemma_cache: Dict[str, str] = {}


# Fillers that the lemmatiser maps onto real words ("yeah" -> "yes"), so they are caught by
# form *and* by lemma. Found on the 20-artist run, where "yes" topped three artists.
FILLER = frozenset({"yes", "yeah", "well", "okay", "ok", "gonna", "wanna", "gotta", "oh", "ooh", "na", "la"})


def lemma(w: str) -> str:
    """simplemma lemma, except stop words and fillers stay as typed (so they stay filtered)."""
    if w in lyr.STOPWORDS or w in FILLER:
        return w
    if w not in _lemma_cache:
        import simplemma
        _lemma_cache[w] = simplemma.lemmatize(w, lang="en").lower()
    return _lemma_cache[w]


def toks(lines: List[str]) -> List[str]:
    return [w.replace("'", "") for ln in lines for w in lyr.tokenize(ln)]


# ----------------------------------------------------------------------------- statistics helpers

def bootstrap_ci(x, n=2000, seed=RNG_SEED):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def spearman_exact(means: List[float], max_enum: int = 8, n_mc: int = 20000, seed=RNG_SEED):
    """Spearman rho between album order and album means; two-sided p over all album orders
    (exact enumeration up to 8 albums, Monte Carlo above)."""
    y = np.asarray(means, float)
    k = len(y)
    if k < 3 or np.allclose(y, y[0]):
        return float("nan"), float("nan"), False
    order = np.arange(k)
    rho = ss.spearmanr(order, y).statistic
    ry = ss.rankdata(y)
    ro = ss.rankdata(order)

    def r(perm):
        return np.corrcoef(ro[list(perm)], ry)[0, 1]

    if k <= max_enum:
        perms = itertools.permutations(range(k))
        vals = np.fromiter((r(p) for p in perms), float)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        vals = np.array([r(rng.permutation(k)) for _ in range(n_mc)])
        exact = False
    p = float(np.mean(np.abs(vals) >= abs(rho) - 1e-12))
    return float(rho), p, exact


def bh(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q.tolist()


def score(tests: List[Dict]) -> List[Dict]:
    """Benjamini-Hochberg within each family of related tests, then a verdict per test.

    Held: q < 0.05. Partly: p < 0.05 but q >= 0.05 (does not survive the correction).
    Failed: p >= 0.05. Not testable: no variation to test. Fit: a description, not a test.
    """
    out = [t for t in tests]
    fams = defaultdict(list)
    for t in out:
        if t.get("family") != "fit" and t["p"] is not None and np.isfinite(t["p"]):
            fams[t["family"]].append(t)
    for fam, ts in fams.items():
        for t, q in zip(ts, bh([t["p"] for t in ts])):
            t["q"] = float(q)
    for t in out:
        if t.get("family") == "fit":
            t["result"] = "Fit"
        elif "q" not in t:
            t["result"], t["q"] = "Not testable", float("nan")
        else:
            t["result"] = "Held" if t["q"] < 0.05 else ("Partly" if t["p"] < 0.05 else "Failed")
    return out


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------- main

def analyze(corpus, min_album_songs: int = 1) -> Dict:
    songs = [s for s in corpus.with_lyrics if s.order >= 0]
    songs.sort(key=lambda s: (s.order, s.position))
    albums = [a for a in corpus.albums]
    A = len(albums)
    alb_name = [a["title"] for a in albums]
    alb_year = [a["year"] for a in albums]
    tests: List[Dict] = []
    facts: Dict[str, object] = {}

    # per-song preparation
    S = []
    for s in songs:
        lines = song_lines(s.lyrics)
        ul = unique_lines(lines)
        t_all, t_u = toks(lines), toks(ul)
        lem_all = [lemma(w) for w in t_all]
        lem_u = [lemma(w) for w in t_u]
        S.append(dict(song=s, lines=lines, ulines=ul, t=t_all, tu=t_u, l=lem_all, lu=lem_u))
    n = len(S)
    if n < 3:
        return {"ok": False, "reason": f"only {n} songs with lyrics", "artist": corpus.artist}

    all_tokens = [w for d in S for w in d["t"]]
    form_counts = Counter(all_tokens)
    facts.update(songs=n, albums=A, lines=sum(len(d["lines"]) for d in S), tokens=len(all_tokens),
                 types=len(form_counts))
    hapax = sum(1 for c in form_counts.values() if c == 1)
    facts.update(hapax=hapax, hapax_pct=hapax / len(form_counts))

    # per-song measures
    vader = None
    from afinn import Afinn
    afinn = Afinn()
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    theme_sets = {k: set(v.split()) for k, v in THEMES.items()}
    per_song = []
    for d in S:
        tu, lu = d["tu"], d["lu"]
        nu = max(len(tu), 1)
        row = {
            "title": d["song"].title, "album": d["song"].album, "year": d["song"].year, "order": d["song"].order,
            "words": len(d["t"]), "words_unique_lines": len(tu),
            "repetition": 1 - len(tu) / max(len(d["t"]), 1),
            "mattr": lyr.mattr(d["t"]), "mattr_u": lyr.mattr(tu),
            "vader": float(np.mean([vader.polarity_scores(ln)["compound"] for ln in d["ulines"]])) if d["ulines"] else 0.0,
            "afinn": 100 * sum(afinn.score(ln) for ln in d["ulines"]) / nu,
        }
        for g, ws in PRONOUNS.items():
            row[f"pron_{g}"] = 100 * sum(w in ws for w in tu) / nu
        for th, ws in theme_sets.items():
            row[f"theme_{th}"] = 1000 * sum(w in ws for w in lu) / nu
        per_song.append(row)
    for r in per_song:
        r["repetition_pct"] = 100 * r["repetition"]

    def by_album(key):
        out = []
        for i in range(A):
            vals = [r[key] for r in per_song if r["order"] == i and r[key] is not None]
            m, lo, hi = bootstrap_ci(vals)
            out.append({"album": alb_name[i], "year": alb_year[i], "order": i, "mean": m, "lo": lo, "hi": hi,
                        "n": len(vals)})
        return out

    def trend(key, label, unit=""):
        rows = by_album(key)
        means = [r["mean"] for r in rows if r["n"]]
        rho, p, exact = spearman_exact(means)
        xs = [r["order"] for r in per_song if r[key] is not None]
        ys = [r[key] for r in per_song if r[key] is not None]
        srho, sp = (ss.spearmanr(xs, ys) if len(set(ys)) > 1 else (float("nan"), float("nan")))
        tests.append({
            "claim": f"{label} changes with album order", "stat": f"album ρ {rho:+.2f}, song ρ {srho:+.2f}",
            "p": p, "kind": "trend", "family": "trends", "key": key, "rho": rho,
            "method": f"Spearman ρ of album means vs release order, {'exact over all orders' if exact else 'Monte Carlo over 20,000 orders'}; song-level ρ p {fmt_p(sp)}"})
        return rows, rho, p, srho, sp

    res: Dict[str, object] = {"artist": corpus.artist, "albums": albums, "per_song": per_song}

    if A >= 3:
        res["pronouns"] = {}
        for g in PRONOUNS:
            rows, rho, p, srho, sp = trend(f"pron_{g}", f"“{g}” rate")
            res["pronouns"][g] = {"rows": rows, "rho": rho, "p": p, "song_rho": srho, "song_p": sp}
        res["repetition"] = dict(zip(("rows", "rho", "p", "song_rho", "song_p"), trend("repetition_pct", "Share of repeated lines")))
        res["mattr"] = dict(zip(("rows", "rho", "p", "song_rho", "song_p"), trend("mattr", "Word variety (MATTR, all lines)")))
        res["mattr_u"] = dict(zip(("rows", "rho", "p", "song_rho", "song_p"), trend("mattr_u", "Word variety (MATTR, repeats removed)")))
    else:
        res["pronouns"] = {g: {"rows": by_album(f"pron_{g}")} for g in PRONOUNS}
        res["repetition"] = {"rows": by_album("repetition_pct")}
        res["mattr"] = {"rows": by_album("mattr")}
        res["mattr_u"] = {"rows": by_album("mattr_u")}

    # Album differences: Kruskal-Wallis per measure, BH over the family
    def kruskal(key):
        groups = [[r[key] for r in per_song if r["order"] == i and r[key] is not None] for i in range(A)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            return float("nan"), float("nan")
        try:
            h = ss.kruskal(*groups)
            return float(h.statistic), float(h.pvalue)
        except ValueError:
            return float("nan"), float("nan")

    th_rows, th_p = [], []
    for th in THEMES:
        h, p = kruskal(f"theme_{th}")
        th_p.append(p if np.isfinite(p) else 1.0)
        th_rows.append({"theme": th, "rows": by_album(f"theme_{th}"), "H": h, "p": p,
                        "overall": float(np.mean([r[f"theme_{th}"] for r in per_song]))})
    for r, q in zip(th_rows, bh(th_p)):
        r["q"] = q
        tests.append({"claim": f"Theme “{r['theme']}” differs between albums", "stat": f"H {r['H']:.1f}",
                      "p": r["p"], "family": "themes", "method": "Kruskal-Wallis over songs; q: Benjamini-Hochberg over the 12 themes"})
    res["themes"] = th_rows

    sent = {}
    for key, label in (("vader", "VADER"), ("afinn", "AFINN")):
        h, p = kruskal(key)
        sent[label] = {"rows": by_album(key), "H": h, "p": p}
        tests.append({"claim": f"Sentiment ({label}) differs between albums", "stat": f"H {h:.1f}", "p": p,
                      "family": "sentiment", "method": "Kruskal-Wallis over songs; q: BH over the two lexicons"})
    rho_sent = ss.spearmanr([r["vader"] for r in per_song], [r["afinn"] for r in per_song]).statistic
    sent["agreement"] = float(rho_sent)
    res["sentiment"] = sent

    # Sentiment arc through an average song (deciles of line position)
    arc = defaultdict(list)
    first_last = []
    for d in S:
        L = d["lines"]
        if len(L) < 6:
            continue
        sc = [vader.polarity_scores(ln)["compound"] for ln in L]
        per_dec = defaultdict(list)
        for i, v in enumerate(sc):
            per_dec[min(9, int(10 * i / len(L)))].append(v)
        for k, v in per_dec.items():
            arc[k].append(float(np.mean(v)))
        third = len(sc) // 3
        first_last.append((np.mean(sc[:third]), np.mean(sc[-third:])))
    res["arc"] = [{"x": 10 * k + 5, **dict(zip(("y", "lo", "hi"), bootstrap_ci(arc[k])))} for k in sorted(arc)]
    if len(first_last) >= 6:
        f, l = zip(*first_last)
        try:
            w = ss.wilcoxon(f, l)
            wp = float(w.pvalue)
        except ValueError:
            wp = 1.0
        res["arc_test"] = {"first": float(np.mean(f)), "last": float(np.mean(l)), "p": wp}
        tests.append({"claim": "Songs end happier or sadder than they start", "stat": f"first third {np.mean(f):+.3f}, last third {np.mean(l):+.3f}",
                      "p": wp, "family": "arc", "method": "Wilcoxon signed-rank, first vs last third of each song (VADER)"})

    # Words
    stop = lyr.STOPWORDS | FILLER | {w for ws in PRONOUNS.values() for w in ws} | {"am", "would", "not", "will"}
    cw_all = Counter(w for d in S for w in d["l"] if w not in stop and len(w) > 1 and w.isalpha())
    cw_u = Counter(w for d in S for w in d["lu"] if w not in stop and len(w) > 1 and w.isalpha())
    song_sets = [set(d["l"]) for d in S]
    top_content = []
    for w, c in cw_all.most_common(40):
        per = [(sum(1 for x in d["l"] if x == w), d["song"].title) for d in S]
        best = max(per)
        top_content.append({"w": w, "all": c, "uniq": cw_u.get(w, 0), "songs": sum(w in st for st in song_sets),
                            "song": best[1], "song_n": best[0]})
    res["top_content"] = top_content
    res["top_all"] = [{"w": w, "n": c} for w, c in form_counts.most_common(30)]
    album_top = []
    for i in range(A):
        c = Counter(w for d in S if d["song"].order == i for w in d["l"] if w in cw_all)
        album_top.append({"album": alb_name[i], "words": c.most_common(8)})
    res["album_top"] = album_top
    if top_content:
        t0 = top_content[0]
        facts.update(top_word=t0["w"], top_word_n=t0["all"], top_word_songs=t0["songs"],
                     top_word_pct=t0["songs"] / n)

    # word lookup: every content lemma used at least twice
    lookup = {}
    for w, c in cw_all.items():
        if c >= 2:
            per_alb = [0] * A
            for d in S:
                per_alb[d["song"].order] += sum(1 for x in d["l"] if x == w)
            lookup[w] = {"total": c, "songs": sum(w in st for st in song_sets), "albums": per_alb}
    res["lookup"] = lookup

    # Distributions
    res["hist_len"] = sorted(Counter(min(len(w), 15) for w in all_tokens).items())
    ff = Counter(min(c, 20) for c in form_counts.values())
    res["hist_freq"] = sorted(ff.items())
    res["hist_line"] = sorted(Counter(min(len(lyr.tokenize(ln)), 20) for d in S for ln in d["lines"]).items())

    # Keyness vs English (wordfreq)
    from wordfreq import top_n_list, word_frequency
    N = len(all_tokens)
    key_rows = []
    for w, c in form_counts.items():
        if c < 6 or w in stop or not w.isalpha():
            continue
        nsongs = sum(1 for d in S if w in d["t"])
        if nsongs < 3:
            continue
        f = word_frequency(w, "en")
        if f <= 0:
            continue
        key_rows.append({"w": w, "n": c, "songs": nsongs, "per": round((c / N) / f, 1)})
    key_rows.sort(key=lambda r: -r["per"])
    res["keyness"] = key_rows[:25]
    common = [w for w in top_n_list("en", 1500) if w.isalpha() and len(w) > 2 and w not in stop]
    checked = common[:300]
    never = [w for w in checked if form_counts.get(w, 0) == 0 and cw_all.get(w, 0) == 0]
    res["never"] = never[:48]
    facts.update(common_checked=len(checked), common_used=len(checked) - len(never))

    # Head to head
    vs = []
    for name, a_set, b_set in PAIRS:
        a_n = sum(1 for d in S for w in d["l"] if w in a_set)
        b_n = sum(1 for d in S for w in d["l"] if w in b_set)
        a_lead = sum(1 for d in S if sum(w in a_set for w in d["l"]) > sum(w in b_set for w in d["l"]))
        b_lead = sum(1 for d in S if sum(w in b_set for w in d["l"]) > sum(w in a_set for w in d["l"]))
        p = ss.binomtest(a_lead, a_lead + b_lead).pvalue if a_lead + b_lead else 1.0
        vs.append({"pair": name, "a": a_n, "b": b_n, "a_songs": a_lead, "b_songs": b_lead, "p": float(p)})
        # Shown as a chart only. As a scorecard test it "held" for nearly every one of 20 artists
        # ("love" beats "hate" everywhere), so it carried no information; the Atlas compares
        # each artist with the others instead.
    res["versus"] = vs

    # Self-similarity patterns (word identity replaced by per-song integer ids)
    ssim = []
    for d in S:
        ids, m = [], {}
        for w in d["t"][:400]:
            ids.append(m.setdefault(w, len(m)))
        ssim.append({"title": d["song"].title, "ids": ids})
    res["selfsim"] = ssim

    # Similarity, classifier, map
    res.update(_similarity(S, A, alb_name, alb_year, tests, stop))
    # Laws, rarefaction, phrases, network
    res.update(_laws(S, A, alb_name, tests))
    res["phrases"] = _phrases(S, alb_name)
    res["network"] = _network(S, stop, tests)

    res["tests"] = score(tests)
    res["facts"] = facts
    res["headline"] = headline(res)
    # Comparable profile for the cross-artist Atlas: word *counts* only, never text or order.
    res["profile"] = {"forms": dict(form_counts), "content": dict(cw_all), "tokens": len(all_tokens),
                      "years": [min(alb_year), max(alb_year)] if alb_year else None}
    res["unofficial_ignored"] = list(getattr(corpus, "unofficial", []))
    res["coverage"] = [{"album": s.album, "track": s.title, "status": s.status}
                       for s in corpus.songs if s.status in ("missing", "duplicate", "instrumental", "no_tracklist")]
    res["ok"] = True
    return mask_slurs(res)


SLURS = re.compile(r"\bnigg(a|as|er|ers|az)\b")


def _mask(w: str) -> str:
    return SLURS.sub(lambda m: "n" + "*" * (len(m.group(0)) - 2) + m.group(0)[-1], w)


def mask_slurs(res: Dict) -> Dict:
    """Racial slurs are counted like any word but shown masked (n****s) on the public page."""
    for r in res.get("top_content", []) + res.get("top_all", []) + res.get("keyness", []):
        r["w"] = _mask(r["w"])
    for a in res.get("album_top", []):
        a["words"] = [(_mask(w), c) for w, c in a["words"]]
    res["lookup"] = {_mask(w): v for w, v in res.get("lookup", {}).items()}
    if "profile" in res:
        for k in ("forms", "content"):
            res["profile"][k] = {_mask(w): c for w, c in res["profile"][k].items()}
    for r in res.get("phrases", []):
        r["p"] = _mask(r["p"])
    nw = res.get("network", {})
    for n in nw.get("nodes", []):
        n["id"] = _mask(n["id"])
    for e in nw.get("edges", []):
        e["a"], e["b"] = _mask(e["a"]), _mask(e["b"])
    if "hub" in nw:
        nw["hub"] = _mask(nw["hub"])
    f = res.get("facts", {})
    if "top_word" in f:
        f["top_word"] = _mask(f["top_word"])
    if "headline" in res:
        res["headline"] = {k: _mask(v) for k, v in res["headline"].items()}
    return res


def _similarity(S, A, alb_name, alb_year, tests, stop):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(w for w in d["l"] if w not in stop and w.isalpha()) for d in S]
    out = {}
    try:
        X = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True).fit_transform(docs).toarray()
    except ValueError:
        return out
    C = X @ X.T
    n = len(S)
    order = np.array([d["song"].order for d in S])
    year = np.array([d["song"].year for d in S])
    # album x album mean similarity
    sim = np.zeros((A, A))
    for i in range(A):
        for j in range(A):
            mask = np.outer(order == i, order == j)
            if i == j:
                mask &= ~np.eye(n, dtype=bool)
            sim[i, j] = C[mask].mean() if mask.any() else np.nan
    out["album_sim"] = {"names": alb_name, "z": sim.tolist()}
    # cohesion per album vs label shuffles
    rng = np.random.default_rng(RNG_SEED)
    offdiag = ~np.eye(n, dtype=bool)

    def within(o, i):
        m = np.outer(o == i, o == i) & offdiag
        return C[m].mean() if m.any() else np.nan

    coh, coh_p = [], []
    perms = [rng.permutation(order) for _ in range(500)]
    for i in range(A):
        if (order == i).sum() < 3:
            continue
        obs = within(order, i)
        null = np.array([within(p, i) for p in perms])
        p = (1 + np.sum(null >= obs)) / (1 + len(null))
        coh.append({"album": alb_name[i], "within": float(obs), "null": float(np.nanmean(null))})
        coh_p.append(p)
    for c, p, q in zip(coh, coh_p, bh(coh_p)):
        c["p"], c["q"] = float(p), float(q)
        tests.append({"claim": f"“{c['album']}” songs share more vocabulary than random songs",
                      "stat": f"within {c['within']:.3f} vs {c['null']:.3f}", "p": float(p), "family": "cohesion",
                      "method": "Mean TF-IDF cosine within the album vs 500 label shuffles; q: BH over albums"})
    out["cohesion"] = coh
    # Mantel: time distance vs lyric distance
    iu = np.triu_indices(n, 1)
    dist = (1 - C)[iu]
    dy = np.abs(year[:, None] - year[None, :])[iu]
    if len(set(dy)) > 1:
        r_obs = ss.spearmanr(dy, dist).statistic
        null = []
        for _ in range(499):
            yp = rng.permutation(year)
            null.append(ss.spearmanr(np.abs(yp[:, None] - yp[None, :])[iu], dist).statistic)
        p = (1 + np.sum(np.abs(null) >= abs(r_obs))) / 500
        out["mantel"] = {"rho": float(r_obs), "p": float(p)}
        tests.append({"claim": "Songs further apart in time are further apart in vocabulary",
                      "stat": f"Mantel ρ {r_obs:+.3f}", "p": float(p), "family": "mantel",
                      "method": "Spearman between year gap and TF-IDF cosine distance, 499 permutations of years"})
    # Map: classical MDS (PCoA) of cosine distance + nearest neighbour links
    D = 1 - C
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J
    ev, evec = np.linalg.eigh(B)
    idx = np.argsort(ev)[::-1]
    ev, evec = ev[idx], evec[:, idx]
    pos = ev[ev > 0]
    coords = evec[:, :2] * np.sqrt(np.maximum(ev[:2], 0))
    out["map_var"] = float(pos[:2].sum() / pos.sum()) if pos.sum() > 0 else 0.0
    Cn = C.copy()
    np.fill_diagonal(Cn, -1)
    nn = Cn.argmax(axis=1)
    out["map"] = [{"title": d["song"].title, "album": d["song"].album, "year": d["song"].year, "order": d["song"].order,
                   "x": float(coords[i, 0]), "y": float(coords[i, 1]), "near": S[nn[i]]["song"].title,
                   "near_sim": round(float(C[i, nn[i]]), 3)} for i, d in enumerate(S)]
    out.update(_classifier(S, A, alb_name, tests))
    return out


def _classifier(S, A, alb_name, tests):
    """Leave-one-song-out nearest centroid with Burrows' Delta on function-word rates."""
    order = np.array([d["song"].order for d in S])
    keep_alb = [i for i in range(A) if (order == i).sum() >= 3]
    if len(keep_alb) < 2:
        return {}
    mask = np.isin(order, keep_alb)
    Ss = [d for d, m in zip(S, mask) if m]
    y = order[mask]
    func_vocab = [w for w, _ in Counter(w for d in Ss for w in d["t"] if w in lyr.STOPWORDS or w in
                                        {x for v in PRONOUNS.values() for x in v}).most_common(100)]
    pron = {x for v in PRONOUNS.values() for x in v}

    def feats(vocab):
        M = np.array([[Counter(d["t"])[w] / max(len(d["t"]), 1) for w in vocab] for d in Ss])
        sd = M.std(axis=0)
        sd[sd == 0] = 1
        return (M - M.mean(axis=0)) / sd

    def loo(Z, labels):
        pred = []
        for i in range(len(labels)):
            m = np.ones(len(labels), bool)
            m[i] = False
            best, bd = None, np.inf
            for c in np.unique(labels[m]):
                cen = Z[m & (labels == c)].mean(axis=0)
                dd = np.abs(Z[i] - cen).mean()
                if dd < bd:
                    best, bd = c, dd
            pred.append(best)
        return np.array(pred)

    rng = np.random.default_rng(RNG_SEED)
    out = {}
    content_vocab = [w for w, _ in Counter(w for d in Ss for w in d["l"] if w not in lyr.STOPWORDS and w.isalpha()).most_common(300)]
    for name, vocab in (("function", func_vocab), ("nopron", [w for w in func_vocab if w not in pron]),
                        ("content", content_vocab)):
        if len(vocab) < 5:
            continue
        Z = feats(vocab)
        pred = loo(Z, y)
        acc = float(np.mean(pred == y))
        null = [float(np.mean(loo(Z, perm) == perm)) for perm in (rng.permutation(y) for _ in range(100))]
        p = (1 + sum(v >= acc for v in null)) / (1 + len(null))
        out[f"cl_{name}"] = {"acc": acc, "chance": float(np.mean(null)), "p": float(p)}
        if name == "function":
            labs = sorted(set(y))
            out["confusion"] = {"names": [alb_name[i] for i in labs],
                                "z": [[int(np.sum((y == a) & (pred == b))) for b in labs] for a in labs]}
            out["cl_within1"] = float(np.mean(np.abs(pred - y) <= 1))
        label = {"function": "the 100 most common function words", "nopron": "function words without pronouns",
                 "content": "the 300 most common content words"}[name]
        tests.append({"claim": f"A song's album can be identified from {label}", "stat": f"accuracy {acc:.0%} vs chance {np.mean(null):.0%}",
                      "p": float(p), "family": "classifier",
                      "method": "Leave-one-song-out nearest centroid (Burrows' Delta), 100 label permutations; q: BH over the 3 feature sets"})
    return out


def _laws(S, A, alb_name, tests):
    out = {}
    tokens = [w for d in S for w in d["t"]]
    c = np.array(sorted(Counter(tokens).values(), reverse=True), float)
    ranks = np.arange(1, len(c) + 1)
    k = min(len(c), 1000)
    slope, intercept, r, _, _ = ss.linregress(np.log10(ranks[:k]), np.log10(c[:k]))
    alpha = 1 + len(c) / np.sum(np.log(c / 0.5))
    pts = np.unique(np.round(np.logspace(0, np.log10(len(c)), 120)).astype(int)) - 1
    out["zipf"] = {"slope": float(slope), "r2": float(r ** 2), "alpha": float(alpha),
                   "points": [[int(ranks[i]), float(c[i])] for i in pts]}
    tests.append({"claim": "Word frequencies follow Zipf's law", "stat": f"slope {slope:.2f}, R² {r ** 2:.3f}",
                  "p": float("nan"), "family": "fit", "method": "OLS on log rank vs log frequency, top 1,000 forms; a description, not a test"})
    # Heaps: vocabulary growth in career order vs shuffled song orders
    vocab = {w: i for i, w in enumerate(set(tokens))}
    song_ids = [np.array([vocab[w] for w in d["t"]]) for d in S]

    def curve(seq_list):
        seq = np.concatenate(seq_list)
        first = np.zeros(len(seq), bool)
        _, idx = np.unique(seq, return_index=True)
        first[idx] = True
        return np.cumsum(first)

    obs = curve(song_ids)
    Ntot = len(obs)
    grid = np.unique(np.linspace(1, Ntot, 150).astype(int)) - 1
    rng = np.random.default_rng(RNG_SEED)
    shuf = np.array([curve([song_ids[i] for i in rng.permutation(len(S))])[grid] for _ in range(1000)])
    mean = shuf.mean(axis=0)
    D_obs = np.max(np.abs(obs[grid] - mean))
    D_null = np.max(np.abs(shuf - mean), axis=1)
    p = (1 + np.sum(D_null >= D_obs)) / (1 + len(D_null))
    beta = ss.linregress(np.log(grid + 1), np.log(obs[grid])).slope
    out["heaps"] = {"beta": float(beta), "p": float(p),
                    "points": [{"x": int(g + 1), "y": int(obs[g]), "lo": float(np.percentile(shuf[:, k], 2.5)),
                                "hi": float(np.percentile(shuf[:, k], 97.5))} for k, g in enumerate(grid)]}
    tests.append({"claim": "Vocabulary growth in career order departs from random song orders",
                  "stat": f"Heaps β {beta:.2f}", "p": float(p), "family": "heaps",
                  "method": "Max deviation from the mean of 1,000 shuffled song orders"})
    # Rarefaction: expected distinct words in an equal-size sample from each album
    sizes = [sum(len(d["t"]) for d in S if d["song"].order == i) for i in range(A)]
    m = min([s for s in sizes if s > 0], default=0)
    rare = []
    if m >= 50:
        for i in range(A):
            cnt = np.array(list(Counter(w for d in S if d["song"].order == i for w in d["t"]).values()), float)
            Ni = cnt.sum()
            if Ni < m:
                continue
            lg = math.lgamma
            ev = sum(1 - math.exp(lg(Ni - ci + 1) - lg(Ni - ci - m + 1) - lg(Ni + 1) + lg(Ni - m + 1))
                     if Ni - ci >= m else 1.0 for ci in cnt)
            rare.append({"album": alb_name[i], "expected": float(ev)})
    out["rarefaction"] = {"sample": m, "rows": rare}
    return out


def _phrases(S, alb_name, nmin=3, nmax=5):
    found = defaultdict(set)
    for d in S:
        for ln in d["ulines"]:
            w = [x.replace("'", "") for x in lyr.tokenize(ln)]
            for k in range(nmin, nmax + 1):
                for i in range(len(w) - k + 1):
                    found[" ".join(w[i:i + k])].add(id(d))
    by_id = {id(d): d for d in S}
    rows = []
    for p, ids in found.items():
        albs = {by_id[i]["song"].order for i in ids}
        if len(ids) >= 3 and len(albs) >= 2:
            if all(t in lyr.STOPWORDS for t in p.split()):
                continue
            rows.append({"p": p, "songs": len(ids), "albums": len(albs), "first": alb_name[min(albs)],
                         "last": alb_name[max(albs)], "_ids": frozenset(ids)})
    rows.sort(key=lambda r: (-r["songs"], -len(r["p"])))
    kept = []
    for r in rows:  # drop a phrase contained in a kept longer phrase with the same songs
        if any(r["p"] in k["p"] and r["_ids"] == k["_ids"] for k in kept):
            continue
        kept.append(r)
    kept.sort(key=lambda r: (-r["songs"], -r["albums"]))
    for r in kept:
        r.pop("_ids")
    return kept[:25]


def _network(S, stop, tests, window=3, top=120, n_shuffle=30):
    import networkx as nx
    lines = []  # (song index, set of content lemmas)
    for si, d in enumerate(S):
        for ln in d["ulines"]:
            lines.append((si, {lemma(w.replace("'", "")) for w in lyr.tokenize(ln)}))
    freq = Counter(w for _, ws in lines for w in ws if w not in stop and w.isalpha() and len(w) > 2)
    vocab = {w for w, _ in freq.most_common(top) if freq[w] >= 5}
    lines = [(si, ws & vocab) for si, ws in lines]

    def count(seq):
        cnt, songs = Counter(), defaultdict(set)
        for i in range(len(seq)):
            si = seq[i][0]
            win = set()
            for j in range(i, min(i + window, len(seq))):
                if seq[j][0] != si:
                    break
                win |= seq[j][1]
            for a, b in itertools.combinations(sorted(win), 2):
                cnt[(a, b)] += 1
                songs[(a, b)].add(si)
        return cnt, songs

    obs, songs = count(lines)
    rng = np.random.default_rng(RNG_SEED)
    words_only = [ws for _, ws in lines]
    sids = [si for si, _ in lines]
    exp = Counter()
    for _ in range(n_shuffle):
        perm = rng.permutation(len(words_only))
        c, _ = count([(sids[k], words_only[p]) for k, p in enumerate(perm)])
        exp.update(c)
    edges, tested = [], 0
    for pair, c in obs.items():
        if c < 3 or len(songs[pair]) < 2:
            continue
        tested += 1
        mu = exp[pair] / n_shuffle
        if c >= 3 * max(mu, 1e-9):
            p = float(ss.poisson.sf(c - 1, max(mu, 1e-9)))
            edges.append({"a": pair[0], "b": pair[1], "n": c, "lift": round(c / max(mu, 0.05), 1), "p": p})
    for e in edges:
        e["strict"] = bool(e["p"] * max(tested, 1) < 0.05)  # Bonferroni over all pairs tested
    if not edges:
        return {"nodes": [], "edges": [], "tested": tested}
    G = nx.Graph()
    G.add_edges_from((e["a"], e["b"]) for e in edges)
    pos = nx.spring_layout(G, seed=RNG_SEED)
    deg = dict(G.degree())
    from networkx.algorithms.community import greedy_modularity_communities, modularity
    Q = modularity(G, greedy_modularity_communities(G)) if G.number_of_edges() else 0.0
    nullQ = []
    for k in range(20):
        H = G.copy()
        try:
            nx.double_edge_swap(H, nswap=2 * H.number_of_edges(), max_tries=20 * H.number_of_edges() + 100, seed=k)
        except nx.NetworkXException:
            pass
        nullQ.append(modularity(H, greedy_modularity_communities(H)) if H.number_of_edges() else 0.0)
    pQ = (1 + sum(q >= Q for q in nullQ)) / (1 + len(nullQ))
    tests.append({"claim": "The word network splits into distinct communities", "stat": f"modularity {Q:.2f} vs {np.mean(nullQ):.2f} rewired",
                  "p": float(pQ), "family": "network", "method": "Greedy modularity vs 20 degree-preserving rewirings"})
    hub = max(deg, key=deg.get)
    return {"nodes": [{"id": w, "x": float(pos[w][0]), "y": float(pos[w][1]), "f": freq[w], "deg": deg[w]} for w in G.nodes],
            "edges": edges, "tested": tested, "strict": sum(e["strict"] for e in edges), "Q": float(Q),
            "Q_null": float(np.mean(nullQ)), "Q_p": float(pQ), "hub": hub, "hub_deg": deg[hub]}


def headline(res) -> Dict[str, str]:
    """Pick the strongest trend that holds and phrase it; fall back to the top word."""
    a = res["artist"]
    cands = [t for t in res["tests"] if t.get("kind") == "trend" and t["result"] == "Held"]
    if cands:
        t = min(cands, key=lambda t: t["p"])
        key = t["key"]
        rows = ([r for r in res["pronouns"][key[5:]]["rows"]] if key.startswith("pron_") else res[{"repetition_pct": "repetition", "mattr": "mattr", "mattr_u": "mattr_u"}[key]]["rows"])
        first, last = rows[0], rows[-1]
        up = t["rho"] > 0
        if key.startswith("pron_"):
            g = key[5:]
            title = f"{a} {'started' if up else 'stopped'} saying “{g}”" if g == "I" else f"{a}'s lyrics turned {'toward' if up else 'away from'} “{g}”"
            lede = (f"“{g}” words went from {first['mean']:.1f} per 100 on {first['album']} to {last['mean']:.1f} on "
                    f"{last['album']} (album ρ {t['rho']:+.2f}, exact p {fmt_p(t['p'])}).")
        elif key == "repetition_pct":
            title = f"{a} {'repeats more' if up else 'repeats less'} with every album"
            lede = (f"Repeated lines were {first['mean']:.0f}% of the average song on {first['album']} and "
                    f"{last['mean']:.0f}% on {last['album']} (album ρ {t['rho']:+.2f}, p {fmt_p(t['p'])}).")
        else:
            title = f"{a}'s vocabulary got {'richer' if up else 'narrower'} over the years"
            lede = (f"Word variety (MATTR) went from {first['mean']:.3f} on {first['album']} to {last['mean']:.3f} on "
                    f"{last['album']} (album ρ {t['rho']:+.2f}, p {fmt_p(t['p'])}).")
        return {"title": title, "lede": lede}
    f = res["facts"]
    if "top_word" in f:
        return {"title": f"“{f['top_word']}” is {a}'s word",
                "lede": f"It appears {f['top_word_n']} times, in {f['top_word_pct']:.0%} of {f['songs']} songs. "
                        "No trend across albums passed the tests below."}
    return {"title": f"{a}, in words", "lede": ""}
