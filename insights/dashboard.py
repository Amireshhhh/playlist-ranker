"""Render the analysis of one playlist (and an index per user) as a self-contained HTML page.

Charts are Plotly figures serialised to JSON. Colours inside the figures are written as
tokens ('@s1', '@text', '@grid', ...) and resolved in the browser from CSS custom
properties, so light and dark mode each get their own validated palette (see the
'dataviz' reference palette) and switching theme re-renders the charts.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import lyrics as lyr
from .models import Playlist

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
SERIES = [f"@s{i}" for i in range(1, 9)]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "hexagon"]

LIGHT = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4",
         "s6": "#008300", "s7": "#6250d6", "s8": "#e34948", "surface": "#fcfcfb", "page": "#f9f9f7",
         "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "neutral": "#f0efec", "seq1": "#cde2fb", "seq2": "#86b6ef", "seq3": "#3987e5", "seq4": "#1c5cab",
         "seq5": "#0d366b"}
DARK = {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500", "s5": "#d55181",
        "s6": "#008300", "s7": "#9085e9", "s8": "#e66767", "surface": "#1a1a19", "page": "#0d0d0d",
        "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "neutral": "#383835", "seq1": "#184f95", "seq2": "#256abf", "seq3": "#3987e5", "seq4": "#86b6ef",
        "seq5": "#cde2fb"}
SEQ = [[0, "@seq1"], [0.25, "@seq2"], [0.5, "@seq3"], [0.75, "@seq4"], [1, "@seq5"]]
DIVERGING = [[0, "@s8"], [0.5, "@neutral"], [1, "@s1"]]

esc = html.escape


def fmt_duration(ms: Optional[int]) -> str:
    if not ms:
        return "n/a"
    s = int(round(ms / 1000))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{m}:{s:02d}"


def _layout(**kw) -> Dict:
    base = {"margin": {"l": 48, "r": 16, "t": 8, "b": 40}, "height": 320, "showlegend": False,
            "hoverlabel": {"bgcolor": "@surface", "font": {"color": "@text"}, "bordercolor": "@axis"},
            "xaxis": {"gridcolor": "@grid", "linecolor": "@axis", "zeroline": False},
            "yaxis": {"gridcolor": "@grid", "linecolor": "@axis", "zeroline": False}}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


class Page:
    def __init__(self):
        self.figs: Dict[str, Dict] = {}
        self.parts: List[str] = []
        self.nav: List[tuple] = []

    def fig(self, fid: str, data: List[Dict], layout: Dict, table: Optional[tuple] = None) -> str:
        self.figs[fid] = {"data": data, "layout": layout}
        out = f'<div class="chart" id="{fid}" role="img"></div>'
        if table:
            head, rows = table
            out += "<details class='data'><summary>Data table</summary><table><thead><tr>" + \
                   "".join(f"<th>{esc(str(h))}</th>" for h in head) + "</tr></thead><tbody>" + \
                   "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>" for r in rows) + \
                   "</tbody></table></details>"
        return out

    def section(self, sid: str, title: str, body: str, lede: str = ""):
        self.nav.append((sid, title))
        lede_html = f"<p class='lede'>{lede}</p>" if lede else ""
        self.parts.append(f"<section id='{sid}'><h2>{esc(title)}</h2>{lede_html}{body}</section>")


def card(title: str, body: str, note: str = "", wide: bool = False) -> str:
    n = f"<p class='note'>{note}</p>" if note else ""
    return f"<div class='card{' wide' if wide else ''}'><h3>{esc(title)}</h3>{body}{n}</div>"


def tiles(items: List[tuple]) -> str:
    return "<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='tv'>{esc(str(v))}</div><div class='tl'>{esc(l)}</div>"
        + (f"<div class='ts'>{esc(s)}</div>" if s else "") + "</div>" for l, v, s in items) + "</div>"


def wordcloud_imgs(counts: Counter) -> str:
    """Two transparent PNGs (light ink / dark ink); CSS shows the one matching the theme."""
    if len(counts) < 5:
        return ""
    from wordcloud import WordCloud
    imgs = []
    for theme, pal in (("light", LIGHT), ("dark", DARK)):
        cols = [pal["s1"], pal["s7"], pal["s3"], pal["text2"]]

        def color(word, font_size, position, orientation, random_state=None, **kw):
            return cols[hash(word) % len(cols)]

        wc = WordCloud(width=900, height=420, background_color=None, mode="RGBA", max_words=120,
                       color_func=color, random_state=0, prefer_horizontal=0.9, collocations=False)
        wc.generate_from_frequencies(dict(counts.most_common(200)))
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG", optimize=True)
        imgs.append(f"<img class='wc wc-{theme}' alt='Word cloud of the most frequent lyric words' "
                    f"src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>")
    return "".join(imgs)


def bar_h(labels, values, color="@s1", height=None, hover="%{y}: %{x}<extra></extra>", text=None):
    labels, values = list(labels)[::-1], list(values)[::-1]
    return ([{"type": "bar", "orientation": "h", "x": values, "y": labels, "marker": {"color": color},
              "hovertemplate": hover, "text": text[::-1] if text else None, "textposition": "outside",
              "cliponaxis": False}],
            _layout(height=height or max(220, 22 * len(labels) + 40), margin={"l": 160, "r": 40},
                    yaxis={"automargin": True, "gridcolor": "rgba(0,0,0,0)"}, bargap=0.25))


# ----------------------------------------------------------------------------- summary

def personality(pl, st, lyr_rep, mood, flow, ms, clusters) -> List[str]:
    """Plain-language facts, each computed from the numbers shown elsewhere on the page."""
    out = []
    n = st["n_tracks"]
    out.append(f"{n} tracks by {st['n_artists']} artists"
               + (f", {fmt_duration(st['total_ms'])} of music." if st["total_ms"] else "."))
    if st["top_artists"]:
        a, c = st["top_artists"][0]
        share = c / n
        if share >= 0.25:
            out.append(f"{a} dominates: {c} of {n} tracks ({share:.0%}).")
        elif st["max_entropy_bits"]:
            out.append(f"Artist variety is {st['artist_entropy_bits'] / st['max_entropy_bits']:.0%} of the maximum "
                       f"possible for {n} tracks; the most frequent artist is {a} ({c} tracks).")
    if st["median_year"]:
        dec = max(st["decades"].items(), key=lambda kv: kv[1])
        out.append(f"Median release year {st['median_year']}; the {dec[0]}s contribute the most songs ({dec[1]}).")
    if mood and mood["quadrants"]:
        q, c = max(mood["quadrants"].items(), key=lambda kv: kv[1])
        out.append(f"Mood: {c} of {sum(mood['quadrants'].values())} placed songs land in '{q}' "
                   f"(axes: {mood['basis'][0]} x {mood['basis'][1]}).")
    if lyr_rep:
        w, c = lyr_rep.word_counts.most_common(1)[0]
        out.append(f"The most used lyric word is '{w}' ({c} times across {len(lyr_rep.songs)} songs with lyrics).")
        pr = max(lyr_rep.pronouns.items(), key=lambda kv: kv[1]) if lyr_rep.pronouns else None
        if pr and pr[1]:
            out.append(f"Lyrics address '{pr[0]}' most ({pr[1]} uses).")
        reps = [s for s in lyr_rep.songs if s.repetitiveness is not None]
        if reps:
            r = max(reps, key=lambda s: s.repetitiveness)
            out.append(f"Most repetitive lyrics: '{r.title}' by {r.artist} "
                       f"(zlib shrinks it by {r.repetitiveness:.0%}).")
    if ms:
        out.append(f"Mainstream score {ms['score']:.2f} on a 0-1 scale ({ms['source']}); deepest cut: "
                   f"'{ms['deepest_cut'].title}' by {ms['deepest_cut'].artist}.")
    if flow and flow["mean_jump_current"] > flow["mean_jump_suggested"]:
        out.append(f"Tempo flow: the average BPM jump between neighbours is {flow['mean_jump_current']:.1f}; "
                   f"the suggested order below brings it to {flow['mean_jump_suggested']:.1f}.")
    if clusters and clusters.silhouette is not None:
        out.append(f"The songs split best into {clusters.k} groups (silhouette {clusters.silhouette:.2f}, "
                   f"features: {', '.join(clusters.blocks_used)}).")
    return out


# ----------------------------------------------------------------------------- page

def render_playlist(pl: Playlist, analysis: Dict, out_path: Path, inline_js: bool = False,
                    user_index_href: Optional[str] = None) -> Path:
    st, lr, cl, ag = analysis["stats"], analysis["lyrics"], analysis["clusters"], analysis["artist_graph"]
    mood, flow, ms, cov, recs = analysis["mood"], analysis["flow"], analysis["mainstream"], analysis["coverage"], \
        analysis["recommendations"]
    P = Page()
    T = pl.tracks

    # Overview ---------------------------------------------------------------
    kp = [("Tracks", st["n_tracks"], f"{st['n_unique_tracks']} unique"),
          ("Listening time", fmt_duration(st["total_ms"]),
           f"avg {fmt_duration(st['mean_track_ms'])}" if st["mean_track_ms"] else ""),
          ("Artists", st["n_artists"], f"{st['one_hit_artists']} appear once"),
          ("Albums", st["n_albums"], "")]
    if st["median_year"]:
        kp.append(("Median year", st["median_year"], f"{min(st['years'])}–{max(st['years'])}"))
    if st["explicit_share"] is not None:
        kp.append(("Explicit", f"{st['explicit_share']:.0%}", ""))
    if ms:
        kp.append(("Mainstream score", f"{ms['score']:.2f}", ms["source"]))
    if lr:
        kp.append(("Lyric vocabulary", f"{lr.vocabulary:,}", f"{lr.total_words:,} words sung"))
    facts = personality(pl, st, lr, mood, flow, ms, cl)
    cov_rows = [(k.replace("_", " "), v) for k, v in cov.items()]
    cov_html = "<div class='chips'>" + "".join(
        f"<span class='chip'>{esc(k)}: <b>{v}</b></span>" for k, v in cov_rows) + "</div>"
    P.section("overview", "Overview", tiles(kp) + card("In numbers", "<ul class='facts'>" +
              "".join(f"<li>{esc(f)}</li>" for f in facts) + "</ul>", wide=True) +
              card("Data coverage", cov_html, "How many tracks or artists each free source returned data for. "
                   "Sections without data are left out.", wide=True))

    # Artists ----------------------------------------------------------------
    top = st["top_artists"][:15]
    body = "<div class='grid2'>"
    d, l = bar_h([a for a, _ in top], [c for _, c in top], hover="%{y}: %{x} tracks<extra></extra>")
    body += card("Top artists", P.fig("f_artists", d, l, (["Artist", "Tracks"], top)))
    lor = st["lorenz"]
    xs = list(np.linspace(0, 1, len(lor)))
    body += card("Artist concentration (Lorenz curve)", P.fig("f_lorenz", [
        {"type": "scatter", "x": [0, 1], "y": [0, 1], "mode": "lines", "line": {"color": "@muted", "dash": "dot", "width": 2},
         "hoverinfo": "skip", "name": "equal share"},
        {"type": "scatter", "x": xs, "y": lor, "mode": "lines+markers", "line": {"color": "@s1", "width": 2},
         "marker": {"size": 8}, "name": "this playlist",
         "hovertemplate": "least-frequent %{x:.0%} of artists hold %{y:.0%} of tracks<extra></extra>"}],
        _layout(xaxis={"tickformat": ".0%", "title": "share of artists (least frequent first)"},
                yaxis={"tickformat": ".0%", "title": "share of tracks"})),
        f"Gini {st['gini']:.2f} (0 = every artist equally often). Entropy {st['artist_entropy_bits']:.2f} of "
        f"{st['max_entropy_bits']:.2f} bits max. Same-artist neighbours: {st['repeat_rate']:.0%} of adjacent pairs. "
        "These are the playlist-health metrics the ranker optimises.")
    body += "</div>"
    if ag:
        ex, ey = [], []
        for a, b, w in ag.edges:
            ex += [ag.coords[a, 0], ag.coords[b, 0], None]
            ey += [ag.coords[a, 1], ag.coords[b, 1], None]
        size = [10 + 26 * math.sqrt(c / max(ag.counts)) for c in ag.counts]
        label_idx = set(np.argsort(ag.counts)[::-1][:12])
        body += card("Artist neighbourhood (k-nearest-neighbour graph)", P.fig("f_graph", [
            {"type": "scatter", "x": ex, "y": ey, "mode": "lines", "line": {"color": "@axis", "width": 1},
             "hoverinfo": "skip"},
            {"type": "scatter", "x": ag.coords[:, 0].tolist(), "y": ag.coords[:, 1].tolist(), "mode": "markers+text",
             "text": [n if i in label_idx else "" for i, n in enumerate(ag.names)], "textposition": "top center",
             "textfont": {"color": "@text2", "size": 11}, "customdata": [[n, c] for n, c in zip(ag.names, ag.counts)],
             "marker": {"size": size, "color": "@s1", "line": {"color": "@surface", "width": 2}},
             "hovertemplate": "%{customdata[0]}: %{customdata[1]} tracks<extra></extra>"}],
            _layout(height=460, xaxis={"visible": False}, yaxis={"visible": False})),
            "Each artist links to its 3 most similar playlist artists (cosine similarity of tag, lyric and sound "
            "vectors), plus links from Last.fm/Deezer 'similar artists'. Node size = number of tracks. "
            "Layout: force-directed.", wide=True)
    infos = [pl.artists.get(a) for a, _ in st["top_artists"][:8]]
    cards = ""
    for a in [i for i in infos if i and (i.bio or i.image_url or i.country)]:
        img = f"<img src='{esc(a.image_url)}' alt='' loading='lazy'>" if a.image_url else ""
        meta = " · ".join(x for x in [a.kind or "", a.country or "", f"since {a.begin_year}" if a.begin_year else ""] if x)
        bio = esc((a.bio or "")[:280] + ("…" if a.bio and len(a.bio) > 280 else ""))
        src = f" <a href='{esc(a.wiki_url)}'>Wikipedia</a> (CC BY-SA)" if a.wiki_url else ""
        tags = ", ".join(list(a.tags)[:4])
        cards += (f"<div class='artist'>{img}<div><b>{esc(a.name)}</b><div class='note'>{esc(meta)}</div>"
                  f"<div class='note'>{esc(tags)}</div><p>{bio}{src}</p></div></div>")
    if cards:
        body += card("Artist profiles", f"<div class='artists'>{cards}</div>", wide=True)
    P.section("artists", "Artists", body)

    # Geography --------------------------------------------------------------
    countries = Counter()
    for t in T:
        info = pl.artists.get(t.artist)
        if info and info.country:
            countries[info.country] += 1
    if countries:
        try:
            import pycountry
            iso3 = {c: (pycountry.countries.get(alpha_2=c) or type("x", (), {"alpha_3": None})).alpha_3 for c in countries}
            name = {c: getattr(pycountry.countries.get(alpha_2=c), "name", c) for c in countries}
        except ImportError:
            iso3, name = {}, {c: c for c in countries}
        body = ""
        locs = [(iso3.get(c), name[c], n) for c, n in countries.items() if iso3.get(c)]
        if locs:
            body += card("Tracks by artist origin", P.fig("f_map", [{
                "type": "choropleth", "locations": [x[0] for x in locs], "z": [x[2] for x in locs],
                "text": [x[1] for x in locs], "colorscale": SEQ, "marker": {"line": {"color": "@surface", "width": 0.5}},
                "colorbar": {"title": {"text": "tracks"}, "outlinewidth": 0, "thickness": 12},
                "hovertemplate": "%{text}: %{z} tracks<extra></extra>"}],
                _layout(height=380, margin={"l": 0, "r": 0, "t": 0, "b": 0},
                        geo={"showframe": False, "showcoastlines": False, "bgcolor": "rgba(0,0,0,0)",
                             "landcolor": "@neutral", "showland": True, "projection": {"type": "natural earth"}})),
                "Country of each track's primary artist, from MusicBrainz.", wide=True)
        top_c = countries.most_common(10)
        d, l = bar_h([name[c] for c, _ in top_c], [n for _, n in top_c], hover="%{y}: %{x} tracks<extra></extra>")
        body += card("Top countries", P.fig("f_countries", d, l, (["Country", "Tracks"], [(name[c], n) for c, n in top_c])))
        P.section("geo", "Where the music comes from", body)

    # Time -------------------------------------------------------------------
    body = "<div class='grid2'>"
    if st["decades"]:
        dec = st["decades"]
        body += card("Songs per decade", P.fig("f_decades", [{
            "type": "bar", "x": [f"{d}s" for d in dec], "y": list(dec.values()), "marker": {"color": "@s1"},
            "hovertemplate": "%{x}: %{y} songs<extra></extra>"}], _layout(bargap=0.25),
            (["Decade", "Songs"], [(f"{d}s", c) for d, c in dec.items()])),
            f"Oldest: '{st['oldest'].title}' ({st['oldest'].year}). Newest: '{st['newest'].title}' ({st['newest'].year}).")
        pos = [(i + 1, t.year, t.title, t.artist) for i, t in enumerate(T) if t.year]
        body += card("Release year along the playlist", P.fig("f_yearpos", [{
            "type": "scatter", "mode": "markers", "x": [p[0] for p in pos], "y": [p[1] for p in pos],
            "customdata": [[p[2], p[3]] for p in pos], "marker": {"color": "@s1", "size": 8, "line": {"color": "@surface", "width": 1}},
            "hovertemplate": "#%{x} %{customdata[0]} – %{customdata[1]} (%{y})<extra></extra>"}],
            _layout(xaxis={"title": "position in playlist"}, yaxis={"title": "release year"})))
    if st.get("added"):
        added = st["added"]
        body += card("How the playlist grew", P.fig("f_growth", [{
            "type": "scatter", "mode": "lines", "x": [d.isoformat() for d in added], "y": list(range(1, len(added) + 1)),
            "line": {"color": "@s1", "width": 2, "shape": "hv"}, "hovertemplate": "%{x|%b %d, %Y}: %{y} songs<extra></extra>"}],
            _layout(yaxis={"title": "songs"})), f"Built over {st['added_span_days']} days.")
        wh = st["added_by_weekday_hour"]
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        z = [[wh.get((d, h), 0) for h in range(24)] for d in range(7)]
        body += card("When songs get added (UTC)", P.fig("f_when", [{
            "type": "heatmap", "z": z, "x": list(range(24)), "y": days, "colorscale": SEQ, "xgap": 2, "ygap": 2,
            "colorbar": {"thickness": 10, "outlinewidth": 0}, "hovertemplate": "%{y} %{x}:00 – %{z} songs<extra></extra>"}],
            _layout(xaxis={"title": "hour", "dtick": 3, "gridcolor": "rgba(0,0,0,0)"},
                    yaxis={"autorange": "reversed", "gridcolor": "rgba(0,0,0,0)"})))
    body += "</div>"
    if body != "<div class='grid2'></div>":
        P.section("time", "Time", body)

    # Lyrics -----------------------------------------------------------------
    if lr:
        body = card("Word cloud", wordcloud_imgs(lr.word_counts),
                    f"{len(lr.songs)} songs with lyrics from LRCLIB. Stop words, filler ('oh', 'yeah') and the artists' "
                    "own names are removed.", wide=True)
        body += "<div class='grid2'>"
        top_w = lr.word_counts.most_common(25)
        d, l = bar_h([w for w, _ in top_w], [c for _, c in top_w], hover="'%{y}': %{x} times<extra></extra>")
        body += card("Top 25 words", P.fig("f_words", d, l, (["Word", "Count"], top_w)))
        top_b = [b for b in lr.bigrams.most_common(15) if b[1] > 1]
        if top_b:
            d, l = bar_h([b for b, _ in top_b], [c for _, c in top_b], color="@s7",
                         hover="'%{y}': %{x} times<extra></extra>")
            body += card("Top phrases (word pairs)", P.fig("f_bigrams", d, l, (["Phrase", "Count"], top_b)))
        if lr.pronouns and sum(lr.pronouns.values()):
            pr = sorted(lr.pronouns.items(), key=lambda kv: -kv[1])
            tot = sum(v for _, v in pr)
            d, l = bar_h([p for p, _ in pr], [v / tot for _, v in pr], hover="%{y}: %{x:.0%}<extra></extra>",
                         text=[f"{v / tot:.0%}" for _, v in pr])
            l["xaxis"]["tickformat"] = ".0%"
            body += card("Who the songs are about (pronouns)", P.fig("f_pron", d, l, (["Pronoun group", "Uses"], pr)))
        ss = sorted(lr.songs, key=lambda s: s.index)
        comp = [s.sentiment["compound"] for s in ss]
        w = 5
        roll = [float(np.mean(comp[max(0, i - w // 2): i + w // 2 + 1])) for i in range(len(comp))]
        body += card("Emotional arc", P.fig("f_arc", [
            {"type": "scatter", "mode": "markers", "x": [s.index + 1 for s in ss], "y": comp,
             "customdata": [[s.title, s.artist] for s in ss], "marker": {"color": "@muted", "size": 8},
             "hovertemplate": "#%{x} %{customdata[0]} – %{customdata[1]}: %{y:.2f}<extra></extra>"},
            {"type": "scatter", "mode": "lines", "x": [s.index + 1 for s in ss], "y": roll,
             "line": {"color": "@s1", "width": 2, "shape": "spline"}, "hoverinfo": "skip"}],
            _layout(xaxis={"title": "position in playlist"}, yaxis={"title": "sentiment", "range": [-1, 1],
                    "zeroline": True, "zerolinecolor": "@axis"})),
            "VADER sentiment per song (mean over lines, -1 negative to +1 positive) with a 5-song moving average.")
        body += "</div>"
        if lr.themes:
            th = "".join(f"<div class='theme'><b>Theme {t['theme']}</b> · {len(t['songs'])} songs"
                         f"<div class='words'>{esc(', '.join(t['words']))}</div>"
                         f"<div class='note'>{esc('; '.join(t['songs'][:5]))}{'…' if len(t['songs']) > 5 else ''}</div></div>"
                         for t in lr.themes)
            body += card("Lyric themes (NMF topic model)", f"<div class='themes'>{th}</div>",
                         "Non-negative matrix factorisation of the TF-IDF word matrix; each song is assigned to its "
                         "strongest theme. Themes are word groups, named by you, not by the model.", wide=True)
        sim = lyr.similarity_matrix(lr)
        if sim is not None and 3 <= len(ss) <= 60:
            labels = [f"{s.title[:22]} ({s.artist[:12]})" for s in lr.songs]
            body += card("Which songs share vocabulary", P.fig("f_sim", [{
                "type": "heatmap", "z": np.round(sim, 3).tolist(), "x": labels, "y": labels, "colorscale": SEQ,
                "zmin": 0, "zmax": 1, "colorbar": {"thickness": 10, "outlinewidth": 0},
                "hovertemplate": "%{y}<br>%{x}<br>cosine %{z:.2f}<extra></extra>"}],
                _layout(height=max(380, 14 * len(labels) + 160), margin={"l": 200, "b": 160},
                        xaxis={"tickfont": {"size": 9}, "gridcolor": "rgba(0,0,0,0)"},
                        yaxis={"tickfont": {"size": 9}, "autorange": "reversed", "gridcolor": "rgba(0,0,0,0)"})),
                "Cosine similarity of TF-IDF lyric vectors.", wide=True)
        rows = "".join(
            f"<tr><td>{s.index + 1}</td><td>{esc(s.title)}</td><td>{esc(s.artist)}</td><td class='n'>{s.n_words}</td>"
            f"<td class='n'>{'' if s.mattr is None else f'{s.mattr:.2f}'}</td>"
            f"<td class='n'>{'' if s.repetitiveness is None else f'{s.repetitiveness:.0%}'}</td>"
            f"<td class='n'>{s.sentiment['compound']:+.2f}</td><td>{esc(', '.join(w for w, _ in s.distinctive[:5]))}</td></tr>"
            for s in sorted(lr.songs, key=lambda s: s.index))
        body += card("Song by song", "<div class='scroll'><table class='sortable'><thead><tr><th>#</th><th>Song</th>"
                     "<th>Artist</th><th>Words</th><th>Vocabulary (MATTR)</th><th>Repetitive</th><th>Sentiment</th>"
                     f"<th>Distinctive words (TF-IDF)</th></tr></thead><tbody>{rows}</tbody></table></div>",
                     "MATTR = moving-average type-token ratio over 50-word windows (higher = richer vocabulary; "
                     "blank under 50 words). Repetitive = how much zlib compresses the lyrics. Click a header to sort.",
                     wide=True)
        ba = [(a, v) for a, v in lr.by_artist.items() if not math.isnan(v["mattr"])]
        if len(ba) >= 3:
            ba.sort(key=lambda kv: -kv[1]["mattr"])
            names_b = [a for a, _ in ba[:15]][::-1]
            vals_b = [round(v["mattr"], 3) for _, v in ba[:15]][::-1]
            # Dot plot, not bars: the differences live far from zero, and a bar must start at zero.
            body += card("Richest vocabulary by artist", P.fig("f_vocab", [{
                "type": "scatter", "mode": "markers", "x": vals_b, "y": names_b,
                "marker": {"color": "@s3", "size": 10, "line": {"color": "@surface", "width": 2}},
                "hovertemplate": "%{y}: MATTR %{x:.3f}<extra></extra>"}],
                _layout(height=max(220, 22 * len(names_b) + 60), margin={"l": 160},
                        xaxis={"title": "MATTR (higher = richer vocabulary)"}, yaxis={"automargin": True})),
                "Mean MATTR of each artist's songs with at least 50 words.")
        P.section("lyrics", "Lyrics", body)

    # Sound ------------------------------------------------------------------
    body = "<div class='grid2'>"
    tempos = [t.tempo for t in T if t.tempo]
    if len(tempos) >= 3:
        body += card("Tempo distribution", P.fig("f_bpm", [{
            "type": "histogram", "x": tempos, "xbins": {"size": 10}, "marker": {"color": "@s1"},
            "hovertemplate": "%{x} BPM: %{y} songs<extra></extra>"}], _layout(xaxis={"title": "BPM"}, bargap=0.08)),
            f"Median {np.median(tempos):.0f} BPM. Source: Deezer track BPM, else audio features.")
    if flow:
        cur = flow["track_indices"]
        sug = flow["suggested_order"]
        b_of = {i: T[i].tempo for i in cur}
        body += card("Tempo flow: current vs suggested order", P.fig("f_flow", [
            {"type": "scatter", "mode": "lines+markers", "name": "current order", "x": list(range(1, len(cur) + 1)),
             "y": [b_of[i] for i in cur], "line": {"color": "@muted", "width": 2}, "marker": {"size": 6},
             "customdata": [T[i].title for i in cur], "hovertemplate": "%{customdata}: %{y:.0f} BPM<extra>current</extra>"},
            {"type": "scatter", "mode": "lines+markers", "name": "suggested order", "x": list(range(1, len(sug) + 1)),
             "y": [b_of[i] for i in sug], "line": {"color": "@s1", "width": 2}, "marker": {"size": 6},
             "customdata": [T[i].title for i in sug], "hovertemplate": "%{customdata}: %{y:.0f} BPM<extra>suggested</extra>"}],
            _layout(showlegend=True, legend={"orientation": "h", "y": 1.12, "font": {"color": "@text2"}},
                    xaxis={"title": "step"}, yaxis={"title": "BPM"})),
            f"Mean BPM jump {flow['mean_jump_current']:.1f} now, {flow['mean_jump_suggested']:.1f} in the suggested "
            "order (half/double time counts as a match).")
    if mood:
        q = mood["points"]
        body += card("Mood map", P.fig("f_mood", [{
            "type": "scatter", "mode": "markers", "x": [p[1] for p in q], "y": [p[2] for p in q],
            "customdata": [[T[p[0]].title, T[p[0]].artist] for p in q],
            "marker": {"size": 10, "color": "@s1", "opacity": 0.85, "line": {"color": "@surface", "width": 2}},
            "hovertemplate": "%{customdata[0]} – %{customdata[1]}<extra></extra>"}],
            _layout(height=380, xaxis={"title": mood["basis"][0], "range": [0, 1]}, yaxis={"title": mood["basis"][1], "range": [0, 1]},
                    shapes=[{"type": "line", "x0": 0.5, "x1": 0.5, "y0": 0, "y1": 1, "line": {"color": "@axis", "width": 1}},
                            {"type": "line", "x0": 0, "x1": 1, "y0": 0.5, "y1": 0.5, "line": {"color": "@axis", "width": 1}}],
                    annotations=[{"x": x, "y": y, "text": t, "showarrow": False, "font": {"color": "@muted", "size": 11}}
                                 for x, y, t in ((0.25, 0.97, "angry / tense"), (0.75, 0.97, "happy / energetic"),
                                                 (0.25, 0.03, "sad / melancholic"), (0.75, 0.03, "calm / content"))])),
            "Quadrants of Russell's circumplex model of affect: " +
            ", ".join(f"{k} {v}" for k, v in sorted(mood["quadrants"].items(), key=lambda kv: -kv[1])) + ".")
    feats = ["energy", "danceability", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"]
    fv = {f: [t.audio[f] for t in T if f in t.audio] for f in feats}
    fv = {f: v for f, v in fv.items() if len(v) >= 3}
    if len(fv) >= 3:
        body += card("Audio fingerprint", P.fig("f_radar", [{
            "type": "scatterpolar", "r": [float(np.mean(v)) for v in fv.values()] + [float(np.mean(list(fv.values())[0]))],
            "theta": list(fv) + [list(fv)[0]], "fill": "toself", "line": {"color": "@s1", "width": 2},
            "fillcolor": "rgba(57,135,229,0.18)", "hovertemplate": "%{theta}: %{r:.2f}<extra></extra>"}],
            _layout(height=360, margin={"t": 36, "b": 36, "l": 72, "r": 72},
                    polar={"bgcolor": "rgba(0,0,0,0)", "radialaxis": {"range": [0, 1], "gridcolor": "@grid",
                    "angle": 90, "tickangle": 90, "tickfont": {"size": 9, "color": "@muted"},
                    "linecolor": "@axis"}, "angularaxis": {"gridcolor": "@grid", "linecolor": "@axis"}}),
            (["Feature", "Mean"], [(f, f"{np.mean(v):.2f}") for f, v in fv.items()])),
            "Playlist means of Spotify-style audio features (Exportify export or ReccoBeats).")
    body += "</div>"
    if body != "<div class='grid2'></div>":
        P.section("sound", "Sound", body)

    # Clusters ---------------------------------------------------------------
    if cl:
        traces = []
        for k, s in enumerate(cl.summaries):
            m = s["members"]
            traces.append({"type": "scatter", "mode": "markers", "name": f"{k + 1}. {s['name']}",
                           "x": cl.coords[m, 0].tolist(), "y": cl.coords[m, 1].tolist(),
                           "customdata": [[T[i].title, T[i].artist] for i in m],
                           "marker": {"size": 10, "color": SERIES[k % 8], "symbol": SYMBOLS[k % 8],
                                      "line": {"color": "@surface", "width": 2}},
                           "hovertemplate": "%{customdata[0]} – %{customdata[1]}<extra>" + esc(s["name"]) + "</extra>"})
        ev = cl.explained_variance
        body = card("Song map (PCA of the song vectors)", P.fig("f_clusters", traces, _layout(
            height=440, showlegend=True, legend={"font": {"color": "@text2"}},
            xaxis={"title": f"component 1 ({ev[0]:.0%} of variance)" if ev else "", "zeroline": False},
            yaxis={"title": f"component 2 ({ev[1]:.0%} of variance)" if len(ev) > 1 else ""})),
            f"k-means with k chosen by silhouette (k = {cl.k}" +
            (f", silhouette {cl.silhouette:.2f}" if cl.silhouette is not None else "") +
            f"). Features: {', '.join(cl.blocks_used)}. Groups are named after their top tags.", wide=True)
        grid = ""
        for k, s in enumerate(cl.summaries):
            meta = " · ".join(x for x in [f"{s['size']} songs",
                                          f"~{s['median_bpm']:.0f} BPM" if s["median_bpm"] else "",
                                          f"median {s['median_year']}" if s["median_year"] else ""] if x)
            grid += (f"<div class='theme'><span class='sw' style='--c:var(--s{k % 8 + 1})'></span><b>{k + 1}. {esc(s['name'])}</b>"
                     f"<div class='note'>{esc(meta)}</div><div class='words'>{esc(', '.join(s['top_artists']))}</div>"
                     + (f"<div class='note'>words: {esc(', '.join(s['words']))}</div>" if s["words"] else "") + "</div>")
        body += card("Groups", f"<div class='themes'>{grid}</div>", wide=True)
        P.section("clusters", "Song groups", body)

    # Recommendations --------------------------------------------------------
    body = ""

    def rec_table(rows):
        out = "<div class='scroll'><table><thead><tr><th></th><th>Song</th><th>Artist</th><th>Why</th><th>Preview</th></tr></thead><tbody>"
        for r in rows:
            img = f"<img src='{esc(r['cover'])}' alt='' width='40' height='40' loading='lazy'>" if r.get("cover") else ""
            title = f"<a href='{esc(r['link'])}'>{esc(r['title'])}</a>" if r.get("link") else esc(r["title"])
            prev = f"<audio controls preload='none' src='{esc(r['preview'])}'></audio>" if r.get("preview") else ""
            out += f"<tr><td>{img}</td><td>{title}</td><td>{esc(r['artist'])}</td><td>{esc(r['reason'])}</td><td>{prev}</td></tr>"
        return out + "</tbody></table></div>"

    if recs.get("similar"):
        body += card("New artists for this playlist", rec_table(recs["similar"]),
                     "Similar-artist votes weighted by how often each of your artists appears, minus a popularity "
                     f"term (discovery = {recs.get('discovery', 0.3)}), re-ranked for artist diversity, at most "
                     "2 songs per artist. 30-second previews from Deezer.", wide=True)
    if recs.get("deep_cuts"):
        body += card("More from artists you already have", rec_table(recs["deep_cuts"]), wide=True)
    if recs.get("collaborative"):
        c = recs["collaborative"]
        rows = "".join(f"<tr><td>{i + 1}</td><td>{('<a href=' + repr(r['link']) + '>' + esc(r['track_uri']) + '</a>') if r['link'] else esc(r['track_uri'])}"
                       f"</td><td>{esc(r['artist'])}</td><td class='n'>{r['score']:.3f}</td></tr>" for i, r in enumerate(c["rows"]))
        body += card("Playlist-continuation model (this repo's ranker)", "<div class='scroll'><table><thead><tr><th>#</th>"
                     f"<th>Track</th><th>Artist</th><th>Score</th></tr></thead><tbody>{rows}</tbody></table></div>",
                     f"{c['matched_tracks']} playlist tracks found in the training catalog; candidates by co-occurrence, "
                     "scored by the trained MLP, re-ranked with fatigue 0.5 / diversity 0.5.", wide=True)
    if body:
        P.section("recs", "Recommendations", body)

    # Methods ----------------------------------------------------------------
    P.section("about", "Sources and methods", card("", (
        "<ul class='facts'><li>Lyrics: LRCLIB (free, no key). Lyrics text is analysed locally and never shown here.</li>"
        "<li>Tempo, gain, popularity rank, previews, album genres, related artists: Deezer public API (free, no key).</li>"
        "<li>Artist origin, type, start year, community tags: MusicBrainz (free, no key).</li>"
        "<li>Artist tags and similar artists: Last.fm (free key, non-commercial use) when LASTFM_API_KEY is set.</li>"
        "<li>Audio features: Exportify CSV columns or ReccoBeats (free, no key).</li>"
        "<li>Artist summaries and images: Wikipedia REST API (text CC BY-SA).</li>"
        "<li>Sentiment: VADER. Topics: NMF on TF-IDF. Groups: k-means on combined tag/lyric/sound vectors, "
        "k by silhouette. Artist graph: cosine kNN (k = 3). Health metrics: <code>evaluation/metrics.py</code>.</li></ul>"),
        wide=True))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_html(f"{pl.name}", P, pl, inline_js, user_index_href), encoding="utf-8")
    return out_path


# ----------------------------------------------------------------------------- user index

def render_user_index(user: str, entries: List[Dict], overlap: Optional[Dict], combined: Dict,
                      out_path: Path, inline_js: bool = False) -> Path:
    P = Page()
    cards = ""
    for e in entries:
        st = e["stats"]
        cards += (f"<a class='plcard' href='{esc(e['href'])}'><b>{esc(e['name'])}</b>"
                  f"<div class='note'>{st['n_tracks']} tracks · {st['n_artists']} artists · {fmt_duration(st['total_ms'])}</div>"
                  + (f"<div class='note'>median year {st['median_year']}</div>" if st["median_year"] else "") + "</a>")
    k = [("Playlists", len(entries), ""), ("Tracks", combined["n_tracks"], f"{combined['n_unique_tracks']} unique"),
         ("Artists", combined["n_artists"], ""), ("Listening time", fmt_duration(combined["total_ms"]), "")]
    P.section("playlists", "Playlists", tiles(k) + f"<div class='plcards'>{cards}</div>")
    body = "<div class='grid2'>"
    top = combined["top_artists"][:15]
    d, l = bar_h([a for a, _ in top], [c for _, c in top], hover="%{y}: %{x} tracks<extra></extra>")
    body += card("Top artists across all playlists", P.fig("u_artists", d, l, (["Artist", "Tracks"], top)))
    if combined.get("word_counts"):
        tw = combined["word_counts"].most_common(20)
        d, l = bar_h([w for w, _ in tw], [c for _, c in tw], color="@s7", hover="'%{y}': %{x}<extra></extra>")
        body += card("Your most-sung words", P.fig("u_words", d, l, (["Word", "Count"], tw)))
    body += "</div>"
    if overlap and len(overlap["names"]) >= 2:
        for key, title in (("artists", "Shared artists between playlists"), ("tracks", "Shared tracks between playlists")):
            body += card(title + " (Jaccard)", P.fig(f"u_ov_{key}", [{
                "type": "heatmap", "z": np.round(overlap[key], 3).tolist(), "x": overlap["names"], "y": overlap["names"],
                "colorscale": SEQ, "zmin": 0, "zmax": 1, "xgap": 2, "ygap": 2, "colorbar": {"thickness": 10, "outlinewidth": 0},
                "hovertemplate": "%{y} ∩ %{x}: %{z:.0%}<extra></extra>"}],
                _layout(height=max(260, 40 * len(overlap["names"]) + 110), margin={"l": 160, "b": 70},
                        xaxis={"gridcolor": "rgba(0,0,0,0)"}, yaxis={"autorange": "reversed", "gridcolor": "rgba(0,0,0,0)"})))
    P.section("taste", "Taste across playlists", body)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_html(f"{user}'s playlists", P, None, inline_js, None), encoding="utf-8")
    return out_path


# ----------------------------------------------------------------------------- html shell

CSS = """
:root{color-scheme:light;--page:%(page)s;--surface:%(surface)s;--text:%(text)s;--text2:%(text2)s;--muted:%(muted)s;
--grid:%(grid)s;--axis:%(axis)s;--border:rgba(11,11,11,.10);%(series)s}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:%(dpage)s;
--surface:%(dsurface)s;--text:%(dtext)s;--text2:%(dtext2)s;--muted:%(dmuted)s;--grid:%(dgrid)s;--axis:%(daxis)s;
--border:rgba(255,255,255,.10);%(dseries)s}}
:root[data-theme="dark"]{color-scheme:dark;--page:%(dpage)s;--surface:%(dsurface)s;--text:%(dtext)s;--text2:%(dtext2)s;
--muted:%(dmuted)s;--grid:%(dgrid)s;--axis:%(daxis)s;--border:rgba(255,255,255,.10);%(dseries)s}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:28px 16px 8px;max-width:1180px;margin:0 auto}
header h1{margin:0 0 4px;font-size:28px;letter-spacing:-.01em}
header .sub{color:var(--text2)}
nav{position:sticky;top:0;z-index:5;background:var(--page);border-bottom:1px solid var(--border)}
nav .in{max-width:1180px;margin:0 auto;padding:8px 16px;display:flex;gap:4px;overflow-x:auto;align-items:center}
nav a{color:var(--text2);text-decoration:none;padding:4px 10px;border-radius:999px;white-space:nowrap;font-size:14px}
nav a:hover{background:var(--surface);color:var(--text)}
nav button{margin-left:auto;background:none;border:1px solid var(--border);color:var(--text2);border-radius:999px;padding:4px 10px;cursor:pointer;font:inherit;font-size:13px}
main{max-width:1180px;margin:0 auto;padding:0 16px 48px}
section{margin-top:28px}
h2{font-size:20px;margin:0 0 12px}
h3{font-size:14px;margin:0 0 8px;color:var(--text2);font-weight:600}
.lede{color:var(--text2);margin-top:-6px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;min-width:0;margin-bottom:16px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,440px),1fr));gap:16px}
.grid2 .card{margin-bottom:0}.grid2{margin-bottom:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px}
.tv{font-size:26px;font-weight:600;line-height:1.15}.tl{color:var(--text2);font-size:13px}.ts{color:var(--muted);font-size:12px}
.note{color:var(--muted);font-size:12.5px;margin:6px 0 0}
.facts{margin:0;padding-left:18px}.facts li{margin:3px 0}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{border:1px solid var(--border);border-radius:999px;padding:2px 10px;font-size:12.5px;color:var(--text2)}
.chart{width:100%%;min-height:200px}
.wc{width:100%%;height:auto;display:block}.wc-dark{display:none}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .wc-light{display:none}:root:not([data-theme="light"]) .wc-dark{display:block}}
:root[data-theme="dark"] .wc-light{display:none}:root[data-theme="dark"] .wc-dark{display:block}
:root[data-theme="light"] .wc-light{display:block}:root[data-theme="light"] .wc-dark{display:none}
table{border-collapse:collapse;width:100%%;font-size:13.5px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:middle}
th{color:var(--text2);font-weight:600;position:sticky;top:0;background:var(--surface)}
td.n{text-align:right;font-variant-numeric:tabular-nums}
table.sortable th{cursor:pointer}
.scroll{max-height:520px;overflow:auto}
details.data{margin-top:8px}details.data summary{color:var(--muted);font-size:12.5px;cursor:pointer}
.themes{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.theme{border:1px solid var(--border);border-radius:10px;padding:10px}.words{margin-top:4px}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;background:var(--c);margin-right:6px}
.artists{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,320px),1fr));gap:12px}
.artist{display:flex;gap:12px}.artist img{width:72px;height:72px;object-fit:cover;border-radius:8px;flex:none}
.artist p{margin:6px 0 0;font-size:13px;color:var(--text2)}
a{color:var(--s1)}
audio{height:32px;max-width:220px}
.plcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.plcard{display:block;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px;color:var(--text);text-decoration:none}
.plcard:hover{border-color:var(--s1)}
footer{color:var(--muted);font-size:12px;text-align:center;padding:24px 16px}
"""

JS = """
const FIGS = %(figs)s;
const PAL = {light: %(light)s, dark: %(dark)s};
function mode(){const t=document.documentElement.dataset.theme; if(t) return t;
  return matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}
function resolve(v,p){ if(typeof v==='string' && v[0]==='@') return p[v.slice(1)] ?? v;
  if(Array.isArray(v)) return v.map(x=>resolve(x,p));
  if(v && typeof v==='object'){const o={}; for(const k in v) o[k]=resolve(v[k],p); return o;} return v;}
function draw(){ const p=PAL[mode()];
  for(const id in FIGS){ const el=document.getElementById(id); if(!el) continue;
    const f=resolve(FIGS[id],p);
    f.layout.paper_bgcolor='rgba(0,0,0,0)'; f.layout.plot_bgcolor='rgba(0,0,0,0)';
    f.layout.font={family:'system-ui,-apple-system,Segoe UI,sans-serif',color:p.text2,size:12};
    Plotly.react(el,f.data,f.layout,{displayModeBar:false,responsive:true}); } }
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
document.getElementById('theme').addEventListener('click',()=>{
  document.documentElement.dataset.theme = mode()==='dark'?'light':'dark'; draw();});
document.querySelectorAll('table.sortable').forEach(t=>t.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
  const b=t.tBodies[0], rows=[...b.rows], asc=th.dataset.asc!=='1'; th.dataset.asc=asc?'1':'0';
  const val=r=>{const s=r.cells[i].textContent.replace('%%','').trim(); const n=parseFloat(s); return isNaN(n)?s.toLowerCase():n;};
  rows.sort((a,c)=>{const x=val(a),y=val(c); return (x>y?1:x<y?-1:0)*(asc?1:-1);}); rows.forEach(r=>b.appendChild(r));})));
draw();
"""


def _series_css(p: Dict) -> str:
    return "".join(f"--s{i}:{p[f's{i}']};" for i in range(1, 9))


def _html(title: str, P: Page, pl: Optional[Playlist], inline_js: bool, back_href: Optional[str]) -> str:
    css = CSS % {**LIGHT, **{f"d{k}": v for k, v in DARK.items()}, "series": _series_css(LIGHT), "dseries": _series_css(DARK)}
    if inline_js:
        from plotly.offline import get_plotlyjs
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_tag = f"<script src='{PLOTLY_CDN}' charset='utf-8'></script>"
    js = JS % {"figs": json.dumps(P.figs, default=_json_default), "light": json.dumps(LIGHT), "dark": json.dumps(DARK)}
    nav = "".join(f"<a href='#{sid}'>{esc(t)}</a>" for sid, t in P.nav)
    back = f"<a href='{esc(back_href)}'>← all playlists</a>" if back_href else ""
    sub = (f"{esc(pl.owner)} · {len(pl.tracks)} tracks · source {esc(pl.source)}" if pl else "Playlist dashboards")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{css}</style>{plotly_tag}</head><body>"
            f"<header>{back}<h1>{esc(title)}</h1><div class='sub'>{sub}</div></header>"
            f"<nav><div class='in'>{nav}<button id='theme' type='button'>Light / dark</button></div></nav>"
            f"<main>{''.join(P.parts)}</main><footer>Generated {now} by playlist-ranker insights</footer>"
            f"<script>{js}</script></body></html>")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
