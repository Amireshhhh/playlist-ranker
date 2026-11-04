"""Render an artist study (insights.artist.textstats.analyze) as a self-contained page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .. import dashboard as D
from ..dashboard import Page, _layout, bar_h, card, esc, tiles
from .textstats import THEMES, fmt_p


def _band_lines(series: List[tuple], fmt=".1f", height=340, ytitle=""):
    """series: [(name, color token, rows)] where rows have album, year, mean, lo, hi."""
    data = []
    for name, col, rows in series:
        rows = [r for r in rows if r["n"]]
        x = [f"{r['album']} ({r['year']})" for r in rows]
        data += [
            {"type": "scatter", "x": x, "y": [r["hi"] for r in rows], "mode": "lines", "line": {"width": 0},
             "hoverinfo": "skip", "showlegend": False},
            {"type": "scatter", "x": x, "y": [r["lo"] for r in rows], "mode": "lines", "line": {"width": 0},
             "fill": "tonexty", "fillcolor": "rgba(137,135,129,0.16)", "hoverinfo": "skip", "showlegend": False},
            {"type": "scatter", "x": x, "y": [r["mean"] for r in rows], "mode": "lines+markers", "name": name,
             "line": {"color": col, "width": 2}, "marker": {"size": 8, "color": col},
             "customdata": [[r["lo"], r["hi"], r["n"]] for r in rows],
             "hovertemplate": f"{name}: %{{y:{fmt}}} (95% CI %{{customdata[0]:{fmt}}} to %{{customdata[1]:{fmt}}}, %{{customdata[2]}} songs)<extra>%{{x}}</extra>"},
        ]
    return data, _layout(height=height, showlegend=len(series) > 1, legend={"orientation": "h", "y": 1.12, "font": {"color": "@text2"}},
                         xaxis={"tickangle": -25, "automargin": True}, yaxis={"title": ytitle}, margin={"b": 90})


def _dots_ci(rows, fmt=".2f", xtitle=""):
    rows = [r for r in rows if r["n"]]
    y = [f"{r['album']} ({r['year']})" for r in rows][::-1]
    m = [r["mean"] for r in rows][::-1]
    lo = [r["mean"] - r["lo"] for r in rows][::-1]
    hi = [r["hi"] - r["mean"] for r in rows][::-1]
    return ([{"type": "scatter", "mode": "markers", "x": m, "y": y,
              "error_x": {"type": "data", "symmetric": False, "array": hi, "arrayminus": lo, "color": "@s1", "thickness": 2, "width": 0},
              "marker": {"size": 10, "color": "@s1", "line": {"color": "@surface", "width": 2}},
              "hovertemplate": f"%{{y}}: %{{x:{fmt}}}<extra></extra>"}],
            _layout(height=max(240, 34 * len(rows) + 70), margin={"l": 200}, xaxis={"title": xtitle, "zeroline": True, "zerolinecolor": "@axis"},
                    yaxis={"automargin": True}))


def _heat(names_r, names_c, z, fmt=".2f", height=None, zmin=None, zmax=None):
    return ([{"type": "heatmap", "z": z, "x": names_c, "y": names_r, "colorscale": D.SEQ, "xgap": 2, "ygap": 2,
              "zmin": zmin, "zmax": zmax, "colorbar": {"thickness": 10, "outlinewidth": 0},
              "texttemplate": f"%{{z:{fmt}}}", "textfont": {"size": 10},
              "hovertemplate": f"%{{y}} · %{{x}}: %{{z:{fmt}}}<extra></extra>"}],
            _layout(height=height or max(300, 34 * len(names_r) + 150), margin={"l": 180, "b": 120},
                    xaxis={"tickangle": -35, "gridcolor": "rgba(0,0,0,0)"},
                    yaxis={"autorange": "reversed", "gridcolor": "rgba(0,0,0,0)"}))


def render(res: Dict, out_path: Path, inline_js: bool = False, back_href: Optional[str] = None) -> Path:
    P = Page()
    f = res["facts"]
    A = len(res["albums"])
    hl = res["headline"]
    tests = res["tests"]
    held = sum(t["result"] == "Held" for t in tests)

    # Hero ------------------------------------------------------------------
    P.parts.append(f"<section id='top'><h2 class='hero'>{esc(hl['title'])}</h2><p class='lede big'>{esc(hl['lede'])}</p>"
                   + tiles([("Songs with lyrics", f["songs"], ""), ("Studio albums", A, ""), ("Lines", f"{f['lines']:,}", ""),
                            ("Words", f"{f['tokens']:,}", ""), ("Distinct words", f"{f['types']:,}", ""),
                            ("Claims that held", f"{held} of {len([t for t in tests if t['result'] not in ('Fit', 'Not testable')])}", "")])
                   + card("How to read the proof", "<p>Every test treats a song as the unit, so a chorus repeated many times counts as one "
                          "piece of evidence. <b>p</b> is the chance of a result this extreme if the pattern were noise. <b>q</b> is p "
                          "corrected (Benjamini-Hochberg) for the other tests in the same family. <b>Held</b> means q &lt; 0.05; "
                          "<b>Partly</b> means p &lt; 0.05 that does not survive the correction. Contractions are expanded, so "
                          "“I'm” counts as “I am”. With n albums the smallest possible exact p for an album trend is 2/n!.</p>", wide=True)
                   + "</section>")
    P.nav.append(("top", "Summary"))

    # Voice -----------------------------------------------------------------
    pr = res["pronouns"]
    d, l = _band_lines([("I, me, my", "@s2", pr["I"]["rows"]), ("we, us, our", "@s1", pr["we"]["rows"]),
                        ("you, your", "@s3", pr["you"]["rows"])], ytitle="per 100 words")
    note = "Rate per song on lines with in-song repeats removed. Bands: 95% bootstrap intervals over songs. "
    if "rho" in pr["I"]:
        note += " ".join(f"“{g}”: album ρ {pr[g]['rho']:+.2f} (exact p {fmt_p(pr[g]['p'])}), song ρ {pr[g]['song_rho']:+.2f} (p {fmt_p(pr[g]['song_p'])})."
                         for g in ("I", "we", "you") if np.isfinite(pr[g].get("rho", np.nan)))
    body = card("Pronoun use per 100 words, by album", P.fig("a_pron", d, l), note, wide=True)
    P.section("voice", "Who is speaking", body)

    # Words -----------------------------------------------------------------
    tc = res["top_content"][:30]
    body = "<div class='grid2'>"
    body += card("Top content words", P.fig("a_top", [
        {"type": "bar", "orientation": "h", "y": [r["w"] for r in tc][::-1], "x": [r["all"] for r in tc][::-1], "name": "all uses",
         "marker": {"color": "@grid"}, "customdata": [[r["songs"], r["song"], r["song_n"]] for r in tc][::-1],
         "hovertemplate": "%{y}: %{x} uses in %{customdata[0]} songs, most in %{customdata[1]} (%{customdata[2]})<extra></extra>"},
        {"type": "bar", "orientation": "h", "y": [r["w"] for r in tc][::-1], "x": [r["uniq"] for r in tc][::-1], "name": "repeated lines removed",
         "marker": {"color": "@s1"}, "hovertemplate": "%{y}: %{x} without repeats<extra></extra>"}],
        _layout(barmode="overlay", height=max(260, 20 * len(tc) + 60), margin={"l": 110}, showlegend=True,
                legend={"orientation": "h", "y": 1.06, "font": {"color": "@text2"}}, bargap=0.25),
        (["Word", "All uses", "Repeats removed", "Songs"], [(r["w"], r["all"], r["uniq"], r["songs"]) for r in tc])),
        "Lemmas (simplemma), so “whales” counts as “whale”. Function words and pronouns removed.")
    ta = res["top_all"]
    d, l = bar_h([r["w"] for r in ta], [r["n"] for r in ta], color="@muted", hover="%{y}: %{x}<extra></extra>")
    body += card("All words, nothing removed", P.fig("a_topall", d, l), "Raw counts of every word form.")
    body += "</div>"
    mult = "<div class='themes'>"
    mx = max((c for a in res["album_top"] for _, c in a["words"]), default=1)
    for i, a in enumerate(res["album_top"]):
        if not a["words"]:
            continue
        d, l = bar_h([w for w, _ in a["words"]], [c for _, c in a["words"]], hover="%{y}: %{x}<extra></extra>")
        l["xaxis"]["range"] = [0, mx * 1.1]
        l["margin"] = {"l": 90, "r": 10, "t": 4, "b": 24}
        mult += f"<div class='theme'><b>{esc(a['album'])}</b>{P.fig(f'a_alb{i}', d, l)}</div>"
    body += card("Top words on each album", mult + "</div>", "Same scale on every album.", wide=True)
    body += "<div class='grid2'>"
    for fid, key, title, xt in (("a_hlen", "hist_len", "Word length", "letters (15 = 15 or more)"),
                                ("a_hfreq", "hist_freq", "How often words repeat", "times used (20 = 20 or more)"),
                                ("a_hline", "hist_line", "Words per line", "words (20 = 20 or more)")):
        xs, ys = zip(*res[key]) if res[key] else ([], [])
        body += card(title, P.fig(fid, [{"type": "bar", "x": list(xs), "y": list(ys), "marker": {"color": "@s1"},
                                         "hovertemplate": "%{x}: %{y:,}<extra></extra>"}],
                                  _layout(height=240, xaxis={"title": xt}, bargap=0.15)),
                     f"{f['hapax']:,} word forms ({f['hapax_pct']:.0%} of the vocabulary) appear exactly once." if key == "hist_freq" else "")
    body += "</div>"
    lookup_js = json.dumps({"words": res["lookup"], "albums": [a["title"] for a in res["albums"]]})
    default = res["top_content"][0]["w"] if res["top_content"] else ""
    body += card("Look up any word", f"<input id='wq' type='search' value='{esc(default)}' aria-label='Word' "
                 "style='font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text)'>"
                 "<p class='note' id='wq-note'></p><div class='chart' id='a_lookup'></div>"
                 f"<script>window.LOOKUP={lookup_js};</script>", "Base forms of content words used at least twice.", wide=True)
    P.figs["a_lookup"] = {"data": [], "layout": _layout(height=280)}
    P.section("words", "Words", body)

    # What makes the vocabulary theirs -------------------------------------------------
    k = res["keyness"][:20]
    body = "<div class='grid2'>"
    if k:
        d, l = bar_h([r["w"] for r in k], [r["per"] for r in k], hover="%{y}: %{x:,}× the English rate<extra></extra>",
                     text=[f"{r['per']:,.0f}×" for r in k])
        body += card("Most overused compared with English", P.fig("a_key", d, l, (["Word", "Uses", "Songs", "× English"], [(r["w"], r["n"], r["songs"], r["per"]) for r in k])),
                     "Rate in the lyrics divided by the rate in everyday English (wordfreq). At least 6 uses in 3 songs.")
    chips = "".join(f"<span class='chip'>{esc(w)}</span>" for w in res["never"])
    body += card("Common English words never sung", f"<div class='chips'>{chips}</div>",
                 f"Of the {f['common_checked']} most common English content words, {f['common_used']} appear at least once.")
    body += "</div>"
    vs = res["versus"]
    if vs:
        names = [v["pair"] for v in vs][::-1]
        body += card("Head to head", P.fig("a_vs", [
            {"type": "bar", "orientation": "h", "y": names, "x": [-v["a"] for v in vs][::-1], "name": "first word",
             "marker": {"color": "@s1"}, "customdata": [[v["a"], v["a_songs"], v["p"]] for v in vs][::-1],
             "hovertemplate": "%{y}: first side %{customdata[0]} uses, leads in %{customdata[1]} songs (sign test p %{customdata[2]:.3f})<extra></extra>"},
            {"type": "bar", "orientation": "h", "y": names, "x": [v["b"] for v in vs][::-1], "name": "second word",
             "marker": {"color": "@s2"}, "customdata": [[v["b"], v["b_songs"]] for v in vs][::-1],
             "hovertemplate": "%{y}: second side %{customdata[0]} uses, leads in %{customdata[1]} songs<extra></extra>"}],
            _layout(barmode="relative", height=max(240, 40 * len(vs) + 60), margin={"l": 120},
                    xaxis={"title": "uses (left: first word, right: second word)", "zeroline": True, "zerolinecolor": "@axis"}),
            (["Pair", "First", "Second", "Songs led (first)", "Songs led (second)", "Sign test p"],
             [(v["pair"], v["a"], v["b"], v["a_songs"], v["b_songs"], fmt_p(v["p"])) for v in vs])),
            "Lemma groups, e.g. Life = life, live, living, alive. Sign test over songs where one side leads.", wide=True)
    P.section("vocab", f"What makes the vocabulary {res['artist']}", body)

    # Repetition ------------------------------------------------------------
    body = "<div class='grid2'>"
    rp = res["repetition"]
    d, l = _dots_ci(rp["rows"], fmt=".1f", xtitle="% of words in repeated lines")
    body += card("Share of words in repeated lines", P.fig("a_rep", d, l),
                 ("Share of a song's words removed when duplicate lines are dropped. " +
                  (f"Album ρ {rp['rho']:+.2f}, exact p {fmt_p(rp['p'])}. " if "rho" in rp else "") +
                  "Depends on the transcription: a file that writes a chorus once scores lower than one that writes it out."))
    d, l = _band_lines([("all lines", "@muted", res["mattr"]["rows"]), ("repeats removed", "@s1", res["mattr_u"]["rows"])], fmt=".3f",
                       ytitle="MATTR (50-word window)")
    body += card("Word variety in a moving window", P.fig("a_mattr", d, l),
                 "Moving-average type-token ratio, independent of song length. Bands: 95% bootstrap intervals.")
    body += "</div>"
    opts = "".join(f"<option>{esc(s['title'])}</option>" for s in res["selfsim"])
    body += card("Self-similarity map of any song", f"<select id='sssel' style='font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--border);"
                 f"background:var(--surface);color:var(--text)'>{opts}</select><div class='chart' id='a_ss' style='max-width:520px'></div>"
                 f"<script>window.SELFSIM={json.dumps({s['title']: s['ids'] for s in res['selfsim']})};</script>",
                 "Each axis is the song, word by word. A dot means the same word at those two positions: diagonal stripes are "
                 "repeated phrases, solid squares are chants. Only the match pattern is stored, no words.", wide=True)
    P.figs["a_ss"] = {"data": [], "layout": _layout(height=480)}
    P.section("repeat", "Repetition and variety", body)

    # Themes ----------------------------------------------------------------
    th = res["themes"]
    alb_labels = [f"{a['title']} ({a['year']})" for a in res["albums"]]
    z = [[r["mean"] if r["n"] else None for r in t["rows"]] for t in th]
    names = [f"{t['theme']}{' *' if t.get('q', 1) < 0.05 else ''}" for t in th]
    d, l = _heat(names, alb_labels, z, fmt=".1f")
    body = card("Theme words per 1,000 words", P.fig("a_themes", d, l),
                "Generic hand-built word lists (below), on lines with repeats removed. * = differs between albums "
                "(Kruskal-Wallis, BH-corrected q < 0.05). Other lists could give other rates.", wide=True)
    body += card("The theme word lists", "<div class='scroll'><table><thead><tr><th>Theme</th><th>Words counted</th><th>q</th></tr></thead><tbody>" +
                 "".join(f"<tr><td>{esc(t['theme'])}</td><td>{esc(THEMES[t['theme']])}</td><td class='n'>{fmt_p(t.get('q', float('nan')))}</td></tr>" for t in th) +
                 "</tbody></table></div>", wide=True)
    P.section("themes", "Themes", body)

    # Mood ------------------------------------------------------------------
    se = res["sentiment"]
    body = "<div class='grid2'>"
    d, l = _dots_ci(se["VADER"]["rows"], fmt=".3f", xtitle="mean line score (−1 to +1)")
    body += card("VADER, mean line score", P.fig("a_vader", d, l), f"Album differences: Kruskal-Wallis p {fmt_p(se['VADER']['p'])}.")
    d, l = _dots_ci(se["AFINN"]["rows"], fmt=".1f", xtitle="points per 100 words")
    body += card("AFINN, points per 100 words", P.fig("a_afinn", d, l),
                 f"Album differences: p {fmt_p(se['AFINN']['p'])}. The two lexicons agree on song scores at ρ {se['agreement']:+.2f}.")
    ps = sorted(res["per_song"], key=lambda r: r["afinn"])
    rows = ps[:7] + ps[-7:] if len(ps) > 14 else ps
    body += card("Darkest and brightest songs", "<div class='scroll'><table><thead><tr><th>Song</th><th>Album</th><th>AFINN</th><th>VADER</th></tr></thead><tbody>" +
                 "".join(f"<tr><td>{esc(r['title'])}</td><td>{esc(r['album'])}</td><td class='n'>{r['afinn']:+.1f}</td><td class='n'>{r['vader']:+.3f}</td></tr>" for r in rows) +
                 "</tbody></table></div>", "Sorted by AFINN. Lexicons score words, not meaning: “death is just an illusion” scores negative.")
    arc = res["arc"]
    if arc:
        x = [a["x"] for a in arc]
        body += card("Mood through an average song", P.fig("a_arc", [
            {"type": "scatter", "x": x, "y": [a["hi"] for a in arc], "mode": "lines", "line": {"width": 0}, "hoverinfo": "skip"},
            {"type": "scatter", "x": x, "y": [a["lo"] for a in arc], "mode": "lines", "line": {"width": 0}, "fill": "tonexty",
             "fillcolor": "rgba(137,135,129,0.16)", "hoverinfo": "skip"},
            {"type": "scatter", "x": x, "y": [a["y"] for a in arc], "mode": "lines+markers", "line": {"color": "@s1", "width": 2},
             "marker": {"size": 7}, "hovertemplate": "%{x}% through the song: %{y:.3f}<extra></extra>"}],
            _layout(height=280, xaxis={"title": "position in song (%)"}, yaxis={"title": "VADER", "zeroline": True, "zerolinecolor": "@axis"})),
            (f"First third {res['arc_test']['first']:+.3f}, last third {res['arc_test']['last']:+.3f}, Wilcoxon p {fmt_p(res['arc_test']['p'])}."
             if "arc_test" in res else ""))
    body += "</div>"
    P.section("mood", "Mood", body)

    # Identity --------------------------------------------------------------
    body = "<div class='grid2'>"
    if "confusion" in res:
        c = res["confusion"]
        d, l = _heat(c["names"], c["names"], c["z"], fmt="d")
        l["xaxis"]["title"] = "predicted album"
        l["yaxis"]["title"] = "true album"
        cf, cn, cc = res.get("cl_function", {}), res.get("cl_nopron", {}), res.get("cl_content", {})
        body += card("True album vs predicted album", P.fig("a_conf", d, l),
                     f"Function words: {cf.get('acc', 0):.0%} correct vs {cf.get('chance', 0):.0%} by chance (permutation p {fmt_p(cf.get('p'))}). "
                     f"Without pronouns: {cn.get('acc', 0):.0%}. Content words: {cc.get('acc', 0):.0%}. "
                     f"{res.get('cl_within1', 0):.0%} of guesses land within one album. Leave-one-song-out nearest centroid, Burrows' Delta.")
    if "album_sim" in res:
        s = res["album_sim"]
        lab = [f"{n}" for n in s["names"]]
        d, l = _heat(lab, lab, np.round(np.array(s["z"], dtype=float), 3).tolist(), fmt=".3f")
        mt = res.get("mantel")
        body += card("Mean lyric similarity between albums", P.fig("a_sim", d, l),
                     "TF-IDF cosine between songs, averaged by album pair. " +
                     (f"Songs drift apart with time: Mantel ρ {mt['rho']:+.3f}, p {fmt_p(mt['p'])}." if mt else ""))
    body += "</div>"
    if res.get("map"):
        mp = res["map"]
        byt = {m["title"]: m for m in mp}
        ex, ey = [], []
        for m in mp:
            o = byt[m["near"]]
            ex += [m["x"], o["x"], None]
            ey += [m["y"], o["y"], None]
        body += card("Map of all songs", P.fig("a_map", [
            {"type": "scatter", "x": ex, "y": ey, "mode": "lines", "line": {"color": "@axis", "width": 1}, "hoverinfo": "skip"},
            {"type": "scatter", "mode": "markers", "x": [m["x"] for m in mp], "y": [m["y"] for m in mp],
             "customdata": [[m["title"], m["album"], m["year"], m["near"], m["near_sim"]] for m in mp],
             "marker": {"size": 10, "color": [m["year"] for m in mp], "colorscale": D.SEQ, "showscale": True,
                        "colorbar": {"title": {"text": "year"}, "thickness": 10, "outlinewidth": 0},
                        "line": {"color": "@surface", "width": 1.5}},
             "hovertemplate": "%{customdata[0]}<br>%{customdata[1]}, %{customdata[2]}<br>nearest: %{customdata[3]} (%{customdata[4]})<extra></extra>"}],
            _layout(height=480, xaxis={"visible": False}, yaxis={"visible": False})),
            f"Classical MDS (PCoA) of TF-IDF cosine distance; lines join each song to its nearest neighbour. The two axes hold "
            f"{res.get('map_var', 0):.0%} of the variance, so trust the links more than the layout.", wide=True)
    P.section("identity", "Do albums have a fingerprint?", body)

    # Network ---------------------------------------------------------------
    nw = res["network"]
    if nw.get("nodes"):
        pos = {n["id"]: n for n in nw["nodes"]}
        tr = []
        for strict, col, w in ((False, "@grid", 1), (True, "@s2", 2.5)):
            ex, ey = [], []
            for e in nw["edges"]:
                if e["strict"] == strict:
                    ex += [pos[e["a"]]["x"], pos[e["b"]]["x"], None]
                    ey += [pos[e["a"]]["y"], pos[e["b"]]["y"], None]
            tr.append({"type": "scatter", "x": ex, "y": ey, "mode": "lines", "line": {"color": col, "width": w}, "hoverinfo": "skip"})
        nodes = nw["nodes"]
        big = sorted(nodes, key=lambda n: -n["deg"])[:25]
        tr.append({"type": "scatter", "mode": "markers+text", "x": [n["x"] for n in nodes], "y": [n["y"] for n in nodes],
                   "text": [n["id"] if n in big else "" for n in nodes], "textposition": "top center",
                   "textfont": {"size": 11, "color": "@text2"},
                   "marker": {"size": [6 + 2.2 * np.sqrt(n["f"]) for n in nodes], "color": ["@s2" if n["id"] == nw["hub"] else "@s1" for n in nodes],
                              "line": {"color": "@surface", "width": 1.5}},
                   "customdata": [[n["id"], n["f"], n["deg"]] for n in nodes],
                   "hovertemplate": "%{customdata[0]}: %{customdata[1]} lines, %{customdata[2]} links<extra></extra>"})
        P.section("network", f"“{nw['hub']}” is the hub of the word network", card("Words that keep company", P.fig("a_net", tr, _layout(
            height=560, xaxis={"visible": False}, yaxis={"visible": False})),
            f"A link joins two words seen within 3 consecutive lines at least 3 times, in at least 2 songs, and at least 3× more often "
            f"than with lines shuffled (30 shuffles). {len(nw['edges'])} links from {nw['tested']} pairs tested; {nw['strict']} survive "
            f"Bonferroni correction (orange). Modularity {nw['Q']:.2f} vs {nw['Q_null']:.2f} for rewired networks (p {fmt_p(nw['Q_p'])}). "
            f"“{nw['hub']}” has {nw['hub_deg']} links. Exploratory.", wide=True))

    # Laws ------------------------------------------------------------------
    body = "<div class='grid2'>"
    z = res["zipf"]
    pts = z["points"]
    body += card("Zipf: rank vs frequency", P.fig("a_zipf", [
        {"type": "scatter", "mode": "lines", "x": [p[0] for p in pts], "y": [p[1] for p in pts], "line": {"color": "@s1", "width": 2},
         "hovertemplate": "rank %{x}: %{y} uses<extra></extra>"},
        {"type": "scatter", "mode": "lines", "x": [1, pts[-1][0]], "y": [pts[0][1], pts[0][1] / pts[-1][0]],
         "line": {"color": "@muted", "dash": "dot", "width": 1.5}, "hoverinfo": "skip"}],
        _layout(height=280, xaxis={"type": "log", "title": "word rank"}, yaxis={"type": "log", "title": "uses"})),
        f"Log-log. Slope {z['slope']:.2f} (dotted: ideal −1), R² {z['r2']:.3f}. Discrete power-law MLE α {z['alpha']:.2f}.")
    hp = res["heaps"]["points"]
    body += card("Heaps: vocabulary growth", P.fig("a_heaps", [
        {"type": "scatter", "x": [p["x"] for p in hp], "y": [p["hi"] for p in hp], "mode": "lines", "line": {"width": 0}, "hoverinfo": "skip"},
        {"type": "scatter", "x": [p["x"] for p in hp], "y": [p["lo"] for p in hp], "mode": "lines", "line": {"width": 0}, "fill": "tonexty",
         "fillcolor": "rgba(137,135,129,0.2)", "hoverinfo": "skip"},
        {"type": "scatter", "x": [p["x"] for p in hp], "y": [p["y"] for p in hp], "mode": "lines", "line": {"color": "@s1", "width": 2},
         "hovertemplate": "%{x:,} words sung: %{y:,} distinct<extra></extra>"}],
        _layout(height=280, xaxis={"title": "words sung so far (career order)"}, yaxis={"title": "distinct words"})),
        f"Band: 95% range of 1,000 shuffled song orders. Heaps β {res['heaps']['beta']:.2f}. Career order vs shuffles: p {fmt_p(res['heaps']['p'])} "
        "(max-deviation test). A significant dip below the band would mean the writer was running out of new words.")
    rr = res["rarefaction"]
    if rr["rows"]:
        rows = sorted(rr["rows"], key=lambda r: -r["expected"])
        d, l = bar_h([r["album"] for r in rows], [round(r["expected"]) for r in rows], color="@s3", hover="%{y}: %{x} distinct words<extra></extra>")
        body += card("Distinct words at equal length", P.fig("a_rare", d, l),
                     f"Exact rarefaction: expected distinct words in a random sample of {rr['sample']:,} words from each album, "
                     "which removes the advantage of longer albums.")
    body += "</div>"
    ph = res["phrases"]
    if ph:
        body += card("Phrases that return across albums", "<div class='scroll'><table><thead><tr><th>Phrase</th><th>Songs</th><th>Albums</th><th>First</th><th>Last</th></tr></thead><tbody>" +
                     "".join(f"<tr><td>“{esc(r['p'])}”</td><td class='n'>{r['songs']}</td><td class='n'>{r['albums']}</td><td>{esc(r['first'])}</td><td>{esc(r['last'])}</td></tr>" for r in ph) +
                     "</tbody></table></div>", "3-5 word sequences in at least 3 songs on at least 2 albums.", wide=True)
    P.section("laws", "Language laws", body)

    # Scorecard -------------------------------------------------------------
    rows = "".join(
        f"<tr><td>{esc(t['claim'])}</td><td class='res res-{t['result'].split()[0].lower()}'>{esc(t['result'])}</td><td>{esc(t['stat'])}</td>"
        f"<td class='n'>{fmt_p(t['p'])}</td><td class='n'>{fmt_p(t.get('q', float('nan')))}</td><td class='note'>{esc(t['method'])}</td></tr>"
        for t in sorted(tests, key=lambda t: (["Held", "Partly", "Failed", "Fit", "Not testable"].index(t["result"]), t["p"] if np.isfinite(t["p"]) else 9)))
    P.section("scorecard", "Everything tested, including what failed", card("", "<div class='scroll' style='max-height:none'><table class='sortable'><thead><tr><th>Claim</th><th>Result</th>"
              f"<th>Effect</th><th>p</th><th>q</th><th>Method</th></tr></thead><tbody>{rows}</tbody></table></div>", wide=True))

    # Coverage --------------------------------------------------------------
    cov = res["coverage"]
    status = {"missing": "no lyrics found", "duplicate": "same title earlier (counted once)", "instrumental": "instrumental"}
    body = card("Tracks not analysed", ("<div class='scroll'><table><thead><tr><th>Album</th><th>Track</th><th>Why</th></tr></thead><tbody>" +
                "".join(f"<tr><td>{esc(c['album'])}</td><td>{esc(c['track'])}</td><td>{esc(status[c['status']])}</td></tr>" for c in cov) +
                "</tbody></table></div>") if cov else "<p>Every track on the studio albums had lyrics.</p>",
                "Studio albums and tracklists from MusicBrainz (primary type Album, no Live/Compilation/Soundtrack; the most common "
                "official tracklist of each). Lyrics from LRCLIB. Lemmas: simplemma. English baseline: wordfreq. Sentiment: VADER, AFINN.", wide=True)
    P.section("coverage", "What the analysis covers", body)

    extra_js = _ARTIST_JS
    html = D._html(f"{res['artist']}: lyrics study", P, None, inline_js, back_href,
                   subtitle=f"{f['songs']} songs on {A} studio albums ({res['albums'][0]['year']}–{res['albums'][-1]['year']})",
                   back_label="← all dashboards")
    html = html.replace("</body>", f"<style>{_ARTIST_CSS}</style><script>{extra_js}</script></body>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


_ARTIST_CSS = """
h2.hero{font-size:clamp(26px,4.5vw,40px);line-height:1.1;margin:8px 0 8px;letter-spacing:-.01em}
p.big{font-size:17px;max-width:72ch;color:var(--text2)}
td.res{font-weight:600}.res-held{color:#0ca30c}.res-partly{color:#c98500}.res-failed{color:#d03b3b}.res-fit,.res-not{color:var(--muted)}
"""

_ARTIST_JS = """
(function(){
  const p=()=>PAL[mode()];
  function lookup(){
    const q=(document.getElementById('wq').value||'').trim().toLowerCase();
    const d=LOOKUP.words[q]; const note=document.getElementById('wq-note');
    if(!d){note.textContent=q?('“'+q+'” is not among the content words used twice or more.'):'Type a word.';
      Plotly.purge('a_lookup'); return;}
    note.textContent='“'+q+'”: '+d.total+' uses in '+d.songs+' songs';
    const pal=p();
    Plotly.react('a_lookup',[{type:'bar',x:LOOKUP.albums,y:d.albums,marker:{color:pal.s1},hovertemplate:'%{x}: %{y}<extra></extra>'}],
      {height:280,margin:{l:40,r:10,t:8,b:90},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
       font:{color:pal.text2,family:'system-ui'},xaxis:{tickangle:-25,gridcolor:pal.grid},yaxis:{gridcolor:pal.grid,title:'uses'}},
      {displayModeBar:false,responsive:true});
  }
  function selfsim(){
    const t=document.getElementById('sssel').value; const ids=SELFSIM[t]||[]; const n=ids.length; const pal=p();
    const x=[],y=[];
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)if(ids[i]===ids[j]){x.push(i);y.push(j);}
    Plotly.react('a_ss',[{type:'scattergl',mode:'markers',x:x,y:y,marker:{size:Math.max(1.5,360/Math.max(n,1)),color:pal.s1},hoverinfo:'skip'}],
      {height:480,width:Math.min(520,document.getElementById('a_ss').clientWidth||480),margin:{l:30,r:10,t:10,b:30},
       paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:pal.text2},
       xaxis:{range:[-1,n],gridcolor:pal.grid,title:'word position'},yaxis:{range:[n,-1],gridcolor:pal.grid,scaleanchor:'x'}},
      {displayModeBar:false});
  }
  document.getElementById('wq').addEventListener('input',lookup);
  document.getElementById('sssel').addEventListener('change',selfsim);
  const redraw=()=>{lookup();selfsim();};
  document.getElementById('theme').addEventListener('click',redraw);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',redraw);
  redraw();
})();
"""
