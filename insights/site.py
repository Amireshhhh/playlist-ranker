"""Home page of the published site: links every artist page, the Atlas and the playlist dashboards."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

esc = html.escape
NAME = "Lyric Atlas"  # working product name

CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:rgba(11,11,11,.1);--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:rgba(255,255,255,.1);--acc:#3987e5}}
:root[data-theme=dark]{color-scheme:dark;--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:rgba(255,255,255,.1);--acc:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:40px 16px 64px}
h1{font-size:clamp(34px,6vw,60px);line-height:1.02;letter-spacing:-.02em;margin:0 0 14px}
h2{font-size:22px;margin:44px 0 12px}
.lede{font-size:clamp(17px,2.2vw,20px);color:var(--ink2);max-width:62ch;margin:0}
.cta{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0 8px}
.btn{display:inline-block;padding:10px 16px;border-radius:999px;text-decoration:none;font-weight:600;border:1px solid var(--line);color:var(--ink)}
.btn.primary{background:var(--acc);border-color:var(--acc);color:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,250px),1fr));gap:12px}
a.c{display:block;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;color:var(--ink);text-decoration:none}
a.c:hover{border-color:var(--acc)}
a.c b{display:block;font-size:17px}.s{color:var(--ink2);font-size:14px;margin-top:4px}.m{color:var(--muted);font-size:12.5px;margin-top:8px}
.stats{display:flex;flex-wrap:wrap;gap:28px;margin:28px 0 0}.stats div b{display:block;font-size:28px}.stats span{color:var(--ink2);font-size:14px}
.promise{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
code{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1px 6px;font-size:14px}
footer{color:var(--muted);font-size:13px;margin-top:48px}
"""


def build(site: str, commit: str = "local") -> Path:
    root = Path(site)
    artists = []
    for f in sorted((root / "artists").glob("*/summary.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        artists.append({"slug": f.parent.name, "artist": d["artist"], "songs": d["facts"]["songs"], "albums": len(d["albums"]),
                        "years": (d["albums"][0]["year"], d["albums"][-1]["year"]) if d["albums"] else None,
                        "headline": d["headline"]["title"],
                        "held": sum(t["result"] == "Held" for t in d["tests"]), "demo": d["artist"] == "The Demo Whales"})
    real = [a for a in artists if not a["demo"]]
    real.sort(key=lambda a: a["artist"].lower())
    songs = sum(a["songs"] for a in real)
    has_atlas = (root / "atlas" / "index.html").exists()
    users = sorted(p.parent for p in root.glob("*/index.html") if p.parent.name not in ("artists", "atlas"))

    def acard(a):
        y = f"{a['years'][0]}–{a['years'][1]}" if a["years"] else ""
        return (f"<a class='c' href='artists/{esc(a['slug'])}/index.html'><b>{esc(a['artist'])}</b>"
                f"<div class='s'>{esc(a['headline'])}</div><div class='m'>{a['songs']} songs · {a['albums']} albums · {y} · "
                f"{a['held']} findings held</div></a>")

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{NAME}</title><meta name='description' content='What artists actually say: a statistical atlas of song lyrics.'>"
             f"<style>{CSS}</style></head><body><main>",
             f"<h1>What artists actually say.</h1><p class='lede'>{NAME} measures song lyrics the way a scientist would: every "
             "studio album, the same tests for every artist, and a public scorecard of what held and what failed.</p>",
             "<div class='cta'>" + ("<a class='btn primary' href='atlas/index.html'>Open the Atlas</a>" if has_atlas else "") +
             "<a class='btn' href='#artists'>Browse artists</a><a class='btn' href='#yours'>Analyse your playlist</a></div>",
             f"<div class='stats'><div><b>{len(real)}</b><span>artists</span></div><div><b>{songs:,}</b><span>songs analysed</span></div>"
             f"<div><b>0</b><span>lyrics stored</span></div></div>",
             f"<h2 id='artists'>Artists</h2><div class='grid'>{''.join(acard(a) for a in real)}</div>"]
    demo = [a for a in artists if a["demo"]]
    pl = "".join(f"<a class='c' href='{esc(u.name)}/index.html'><b>{esc(u.name.replace('-', ' ').title())}</b>"
                 "<div class='s'>Playlist dashboards</div></a>" for u in users)
    if demo:
        pl += ("<a class='c' href='artists/the-demo-whales/index.html'><b>Demo artist (synthetic)</b>"
               "<div class='s'>Invented lyrics with planted trends; the tests must find exactly those.</div></a>")
    parts.append(f"<h2>Playlists and demos</h2><div class='grid'>{pl}</div>")
    parts.append("<h2 id='yours'>Analyse your own playlist</h2><div class='promise'><p>Export a Spotify playlist with "
                 "<a href='https://exportify.app/'>Exportify</a> (CSV), or use any public Deezer playlist, then run:</p>"
                 "<p><code>pip install -r requirements-insights.txt</code><br><code>python -m insights analyze my_playlist.csv --user me</code></p>"
                 "<p>An artist study: <code>python -m insights artist \"Gojira\"</code>. Everything runs on your machine.</p></div>")
    parts.append("<h2>Our promise</h2><div class='promise'><p><b>We never store or publish lyrics.</b> Lyrics are read in memory "
                 "while a page is built and thrown away. Pages hold counts, scores and statistics only; the build fails if lyrics "
                 "text is found on disk. Every claim on an artist page comes with its test, p-value and method, including the ones that failed.</p></div>")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"<footer>Data: MusicBrainz (studio albums), LRCLIB (lyrics, read only), Deezer, Wikipedia. Built {now} from "
                 f"commit {esc(commit[:7])}. Source: <a href='https://github.com/Amireshhhh/playlist-ranker'>playlist-ranker</a>.</footer>"
                 "</main></body></html>")
    out = root / "index.html"
    out.write_text("".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    import os
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "site", os.environ.get("GITHUB_SHA", "local")))
