"""The Atlas: every analysed artist compared on the same measures.

Inputs are the per-artist ``summary.json`` files written by ``insights.artist.run``
(word counts and per-song scores only; no lyrics text).

Methods
- Vocabulary: exact rarefaction, the expected number of distinct words in a random sample of
  the same number of words from each artist (the smallest catalogue sets the sample size).
- Per-song measures (pronoun rates, repetition, MATTR, sentiment, themes): artist mean with a
  95% bootstrap interval over songs; Kruskal-Wallis across artists.
- Distinctive words: weighted log-odds ratio with an informative Dirichlet prior (Monroe,
  Colaresi & Quinn 2008, "Fightin' Words"), each artist against all the others; z > 1.96 kept.
- Similarity: cosine between artists' TF-IDF-weighted content-word vectors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats as ss

from .. import dashboard as D
from ..dashboard import Page, _layout, card, esc, tiles
from ..pipeline import slug
from .textstats import THEMES, bootstrap_ci, fmt_p

MEASURES = [
    # key, label, unit/format, higher-is phrase
    ("pron_I", "“I, me, my” per 100 words", ".1f", "most self-focused"),
    ("pron_we", "“we, us, our” per 100 words", ".1f", "most collective"),
    ("pron_you", "“you, your” per 100 words", ".1f", "most addressed to “you”"),
    ("repetition_pct", "% of words in repeated lines", ".0f", "most repetitive"),
    ("mattr_u", "word variety (MATTR, repeats removed)", ".3f", "most varied wording"),
    ("vader", "sentiment (VADER mean line score)", ".3f", "brightest"),
    ("theme_death & pain", "death & pain words per 1,000", ".1f", "most death & pain"),
    ("theme_love", "love words per 1,000", ".1f", "most love"),
    ("theme_sky & cosmos", "sky & cosmos words per 1,000", ".1f", "most cosmic"),
    ("theme_conflict", "conflict words per 1,000", ".1f", "most conflict"),
]


def load(root: Path) -> List[Dict]:
    out = []
    for f in sorted(Path(root).glob("*/summary.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("profile") and d.get("source") != "synthetic demo" and d["artist"] != "The Demo Whales":
            d["_slug"] = f.parent.name
            out.append(d)
    return out


def rarefy(counts: np.ndarray, m: int) -> float:
    """Expected distinct types in a sample of m tokens without replacement (exact)."""
    N = counts.sum()
    if m >= N:
        return float(len(counts))
    lg = math.lgamma
    base = lg(N - m + 1) - lg(N + 1)
    return float(sum(1 - math.exp(lg(N - c + 1) - lg(N - c - m + 1) + base) if N - c >= m else 1.0 for c in counts))


def fightin_words(counts: List[Dict[str, int]], alpha0: float = 1000.0, top: int = 12, min_total: int = 10):
    vocab = sorted({w for c in counts for w in c})
    idx = {w: i for i, w in enumerate(vocab)}
    Y = np.zeros((len(counts), len(vocab)))
    for i, c in enumerate(counts):
        for w, n in c.items():
            Y[i, idx[w]] = n
    tot = Y.sum(axis=0)
    keep = tot >= min_total
    Y, tot = Y[:, keep], tot[keep]
    vocab = [w for w, k in zip(vocab, keep) if k]
    alpha = alpha0 * tot / tot.sum()
    a0 = alpha.sum()
    out = []
    for i in range(len(counts)):
        yi, yj = Y[i], tot - Y[i]
        ni, nj = yi.sum(), yj.sum()
        d = (np.log((yi + alpha) / (ni + a0 - yi - alpha)) - np.log((yj + alpha) / (nj + a0 - yj - alpha)))
        z = d / np.sqrt(1 / (yi + alpha) + 1 / (yj + alpha))
        order = np.argsort(-z)
        out.append([(vocab[j], round(float(z[j]), 1), int(yi[j])) for j in order[:top] if z[j] > 1.96])
    return out


def compute(arts: List[Dict]) -> Dict:
    names = [a["artist"] for a in arts]
    res: Dict = {"artists": names, "slugs": [a["_slug"] for a in arts]}
    tokens = [a["profile"]["tokens"] for a in arts]
    m = int(min(tokens))
    res["rarefaction"] = {"sample": m, "values": [rarefy(np.array(list(a["profile"]["forms"].values()), float), m) for a in arts]}
    res["sizes"] = [{"songs": a["facts"]["songs"], "albums": len(a["albums"]), "tokens": a["profile"]["tokens"],
                     "years": a["profile"].get("years")} for a in arts]
    meas = {}
    for key, label, fmt, phrase in MEASURES:
        rows, groups = [], []
        for a in arts:
            vals = [r[key] for r in a["per_song"] if r.get(key) is not None]
            mu, lo, hi = bootstrap_ci(vals)
            rows.append({"artist": a["artist"], "mean": mu, "lo": lo, "hi": hi, "n": len(vals)})
            groups.append(vals)
        try:
            p = float(ss.kruskal(*[g for g in groups if len(g) >= 2]).pvalue)
        except ValueError:
            p = float("nan")
        order = sorted(range(len(rows)), key=lambda i: -rows[i]["mean"] if np.isfinite(rows[i]["mean"]) else 1e9)
        top = rows[order[0]]
        # is the leader clearly ahead? its interval must not overlap the runner-up's
        clear = len(order) > 1 and top["lo"] > rows[order[1]]["hi"]
        meas[key] = {"label": label, "fmt": fmt, "phrase": phrase, "rows": rows, "p": p, "order": order, "clear_leader": bool(clear)}
    res["measures"] = meas
    res["distinctive"] = fightin_words([a["profile"]["content"] for a in arts])
    # similarity
    vocab = sorted({w for a in arts for w in a["profile"]["content"]})
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(arts), len(vocab)))
    for i, a in enumerate(arts):
        for w, c in a["profile"]["content"].items():
            X[i, vi[w]] = 1 + math.log(c)
    df = (X > 0).sum(axis=0)
    X *= np.log((1 + len(arts)) / (1 + df)) + 1
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = X @ X.T
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    Dm = np.clip(1 - C, 0, None)
    np.fill_diagonal(Dm, 0)
    order = leaves_list(linkage(squareform(Dm, checks=False), "average")) if len(arts) > 2 else list(range(len(arts)))
    res["similarity"] = {"z": C.tolist(), "order": [int(i) for i in order]}
    Cn = C.copy()
    np.fill_diagonal(Cn, -1)
    res["nearest"] = [[(names[j], round(float(C[i, j]), 3)) for j in np.argsort(-Cn[i])[:min(3, len(arts) - 1)]] for i in range(len(arts))]
    # word explorer: rate per 10,000 words for the most used content words
    total = {}
    for a in arts:
        for w, c in a["profile"]["content"].items():
            total[w] = total.get(w, 0) + c
    words = [w for w, _ in sorted(total.items(), key=lambda kv: -kv[1])[:4000]]
    res["explorer"] = {w: [round(1e4 * a["profile"]["content"].get(w, 0) / a["profile"]["tokens"], 2) for a in arts] for w in words}
    # per-artist ranks, for the box on each artist page
    ranks = {}
    for i, a in enumerate(arts):
        r = {}
        for key, mm in meas.items():
            pos = mm["order"].index(i) + 1
            r[key] = {"rank": pos, "of": len(arts), "label": mm["label"], "phrase": mm["phrase"]}
        vals = res["rarefaction"]["values"]
        r["vocabulary"] = {"rank": int(sorted(vals, reverse=True).index(vals[i]) + 1), "of": len(arts),
                           "label": f"distinct words per {m:,}", "phrase": "richest vocabulary"}
        ranks[a["_slug"]] = {"artist": a["artist"], "ranks": r, "nearest": res["nearest"][i],
                             "distinctive": [w for w, _, _ in res["distinctive"][i][:6]]}
    res["ranks"] = ranks
    return res


def _dots(rows, order, fmt, slugs, xtitle):
    ys = [rows[i]["artist"] for i in order][::-1]
    pick = order[::-1]
    return ([{"type": "scatter", "mode": "markers", "y": ys, "x": [rows[i]["mean"] for i in pick],
              "error_x": {"type": "data", "symmetric": False, "array": [rows[i]["hi"] - rows[i]["mean"] for i in pick],
                          "arrayminus": [rows[i]["mean"] - rows[i]["lo"] for i in pick], "color": "@s1", "thickness": 2, "width": 0},
              "marker": {"size": 9, "color": "@s1", "line": {"color": "@surface", "width": 2}},
              "customdata": [[rows[i]["lo"], rows[i]["hi"], rows[i]["n"]] for i in pick],
              "hovertemplate": f"%{{y}}: %{{x:{fmt}}} (95% CI %{{customdata[0]:{fmt}}}–%{{customdata[1]:{fmt}}}, %{{customdata[2]}} songs)<extra></extra>"}],
            _layout(height=max(300, 24 * len(rows) + 70), margin={"l": 140}, xaxis={"title": xtitle}, yaxis={"automargin": True}))


def render(res: Dict, out_path: Path, inline_js: bool = False) -> Path:
    P = Page()
    names = res["artists"]
    n = len(names)
    sizes = res["sizes"]
    rr = res["rarefaction"]
    v = rr["values"]
    vo = sorted(range(n), key=lambda i: -v[i])
    meas = res["measures"]
    lead = []
    for key, mm in meas.items():
        i = mm["order"][0]
        lead.append(f"<div class='theme'><div class='note'>{esc(mm['phrase'])}</div><b>{esc(names[i])}</b>"
                    f"<div class='note'>{format(mm['rows'][i]['mean'], mm['fmt'])} · "
                    f"{'clear lead' if mm['clear_leader'] else ('tied with #2' if mm['rows'][i]['mean'] == mm['rows'][mm['order'][1]]['mean'] else 'interval overlaps #2')}</div></div>")
    hero = (f"<section id='top'><h2 class='hero'>{n} artists, {sum(s['songs'] for s in sizes):,} songs, one set of tests</h2>"
            f"<p class='lede big'>Every artist is measured the same way: the same albums rule, the same word lists, the same "
            f"statistics. Richest vocabulary: <b>{esc(names[vo[0]])}</b>, with {v[vo[0]]:,.0f} distinct words in a random "
            f"{rr['sample']:,}-word sample, against {v[vo[-1]]:,.0f} for {esc(names[vo[-1]])}.</p>"
            + tiles([("Artists", n, ""), ("Songs", f"{sum(s['songs'] for s in sizes):,}", ""),
                     ("Words analysed", f"{sum(s['tokens'] for s in sizes):,}", ""),
                     ("Studio albums", sum(s["albums"] for s in sizes), "")])
            + card("Leaders", f"<div class='themes'>{''.join(lead)}</div>",
                   "“Clear lead” means the leader's 95% interval does not overlap the runner-up's.", wide=True)
            + "</section>")
    P.parts.append(hero)
    P.nav.append(("top", "Leaders"))

    # Vocabulary
    body = card("Distinct words in an equal-size sample", P.fig("x_vocab", [{
        "type": "bar", "orientation": "h", "y": [names[i] for i in vo][::-1], "x": [round(v[i]) for i in vo][::-1],
        "marker": {"color": "@s3"}, "customdata": [[sizes[i]["tokens"], sizes[i]["songs"]] for i in vo][::-1],
        "hovertemplate": "%{y}: %{x:,} distinct words per " + f"{rr['sample']:,}" + " (catalogue: %{customdata[0]:,} words, %{customdata[1]} songs)<extra></extra>"}],
        _layout(height=max(300, 24 * n + 60), margin={"l": 140}, xaxis={"title": f"expected distinct words in {rr['sample']:,} words"}),
        (["Artist", "Distinct words", "Catalogue words", "Songs"], [(names[i], round(v[i]), sizes[i]["tokens"], sizes[i]["songs"]) for i in vo])),
        f"Exact rarefaction to {rr['sample']:,} words, the size of the smallest catalogue here, so long careers get no advantage. "
        "Counts word forms as sung, after expanding contractions.", wide=True)
    P.section("vocab", "Who has the biggest vocabulary", body)

    # Measures
    body = "<div class='grid2'>"
    for k, (key, mm) in enumerate(meas.items()):
        d, l = _dots(mm["rows"], mm["order"], mm["fmt"], res["slugs"], mm["label"])
        body += card(mm["label"][0].upper() + mm["label"][1:], P.fig(f"x_m{k}", d, l),
                     f"Artist mean over songs with 95% bootstrap interval. Differences between artists: Kruskal-Wallis p {fmt_p(mm['p'])}.")
    body += "</div>"
    P.section("measures", "Rankings", body)

    # Distinctive words
    cards = ""
    for i in sorted(range(n), key=lambda i: names[i].lower()):
        words = res["distinctive"][i]
        chips = "".join(f"<span class='chip' title='z {z}, {c} uses'>{esc(w)}</span>" for w, z, c in words[:10])
        near = ", ".join(f"{esc(a)}" for a, _ in res["nearest"][i])
        cards += (f"<div class='theme'><b><a href='../artists/{esc(res['slugs'][i])}/index.html'>{esc(names[i])}</a></b>"
                  f"<div class='chips' style='margin:6px 0'>{chips}</div><div class='note'>closest vocabulary: {near}</div></div>")
    P.section("distinct", "Each artist's signature words", card("Words each artist uses far more than the other "
              f"{n - 1}", f"<div class='themes'>{cards}</div>",
              "Weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008), each artist vs all others; "
              "words with z > 1.96, strongest first. Hover a word for its z-score.", wide=True))

    # Similarity
    o = res["similarity"]["order"]
    z = np.array(res["similarity"]["z"])[np.ix_(o, o)]
    np.fill_diagonal(z, np.nan)
    lab = [names[i] for i in o]
    P.section("similar", "Who sounds alike, on paper", card("Vocabulary similarity", P.fig("x_sim", [{
        "type": "heatmap", "z": np.round(z, 3).tolist(), "x": lab, "y": lab, "colorscale": D.SEQ, "xgap": 1, "ygap": 1,
        "colorbar": {"thickness": 10, "outlinewidth": 0}, "hovertemplate": "%{y} · %{x}: cosine %{z:.2f}<extra></extra>"}],
        _layout(height=max(420, 24 * n + 180), margin={"l": 140, "b": 140}, xaxis={"tickangle": -40, "gridcolor": "rgba(0,0,0,0)"},
                yaxis={"autorange": "reversed", "gridcolor": "rgba(0,0,0,0)"})),
        "Cosine similarity of TF-IDF-weighted content-word vectors, ordered by hierarchical clustering (average linkage), so "
        "blocks are groups of artists with shared vocabulary.", wide=True))

    # Explorer
    P.section("explorer", "Who says it most", card("Word explorer", "<input id='xq' type='search' value='love' aria-label='Word' "
              "style='font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text)'>"
              "<p class='note' id='xq-note'></p><div class='chart' id='x_explore'></div>"
              f"<script>window.EXPLORER={json.dumps({'artists': names, 'words': res['explorer']})};</script>",
              "Uses per 10,000 words sung, for the 4,000 most used content words (base forms). Try: fire, money, god, ocean, night.", wide=True))
    P.figs["x_explore"] = {"data": [], "layout": _layout()}

    # Method
    P.section("about", "Method and limits", card("", "<ul class='facts'>"
              "<li>Catalogue: MusicBrainz studio albums (no live, compilation or bootleg albums), at most 20 per artist, the most common official tracklist of each. Lyrics: LRCLIB. Songs with no lyrics found are listed on each artist page.</li>"
              "<li>Lyrics are read in memory during the build and never stored or published; this page holds word counts and scores only.</li>"
              "<li>Lyrics transcriptions differ: a chorus written once instead of four times lowers repetition. Measures that ignore repeated lines are marked.</li>"
              f"<li>Theme scores use fixed word lists ({', '.join(THEMES)}); other lists would give other numbers.</li>"
              "<li>Rankings show 95% intervals: artists whose intervals overlap are not reliably different.</li></ul>", wide=True))

    html = D._html("Lyric Atlas: artists compared", P, None, inline_js, "../index.html",
                   subtitle=f"{n} artists · {sum(s['songs'] for s in sizes):,} songs", back_label="← home")
    html = html.replace("</body>", "<style>h2.hero{font-size:clamp(26px,4.5vw,40px);line-height:1.1;margin:8px 0}"
                        "p.big{font-size:17px;max-width:76ch;color:var(--text2)}</style><script>" + _JS + "</script></body>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    (out_path.parent / "ranks.json").write_text(json.dumps(res["ranks"]), encoding="utf-8")
    return out_path


_JS = """
(function(){
  function draw(){
    const q=(document.getElementById('xq').value||'').trim().toLowerCase(); const pal=PAL[mode()];
    const d=EXPLORER.words[q]; const note=document.getElementById('xq-note');
    if(!d){note.textContent=q?('“'+q+'” is not among the 4,000 most used content words.'):'Type a word.'; Plotly.purge('x_explore'); return;}
    const idx=EXPLORER.artists.map((a,i)=>i).sort((a,b)=>d[a]-d[b]);
    note.textContent='“'+q+'”, uses per 10,000 words';
    Plotly.react('x_explore',[{type:'bar',orientation:'h',y:idx.map(i=>EXPLORER.artists[i]),x:idx.map(i=>d[i]),marker:{color:pal.s1},
      hovertemplate:'%{y}: %{x} per 10,000 words<extra></extra>'}],
      {height:Math.max(300,24*idx.length+60),margin:{l:140,r:20,t:8,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
       font:{color:pal.text2,family:'system-ui'},xaxis:{gridcolor:pal.grid,title:'per 10,000 words'},yaxis:{automargin:true}},
      {displayModeBar:false,responsive:true});
  }
  document.getElementById('xq').addEventListener('input',draw);
  document.getElementById('theme').addEventListener('click',draw);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
  draw();
})();
"""


def build(site_dir: str, inline_js: bool = False):
    arts = load(Path(site_dir) / "artists")
    if len(arts) < 2:
        return None
    res = compute(arts)
    return render(res, Path(site_dir) / "atlas" / "index.html", inline_js=inline_js)


__all__ = ["build", "compute", "render", "rarefy", "fightin_words", "slug"]
