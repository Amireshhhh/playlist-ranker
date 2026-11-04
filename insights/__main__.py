"""Command line: ``python -m insights``.

Examples
  # Exportify CSVs (one or more playlists of one user), fetch free data, write dashboards
  python -m insights analyze exports/*.csv --user amir

  # A public Deezer playlist
  python -m insights analyze deezer:908622995 --user amir

  # One playlist of an MPD slice, plus the repo's trained ranker for recommendations
  python -m insights analyze data/mpd/mpd.slice.0-999.json --pid 7 --user mpd \
      --data-dir data/processed --checkpoint checkpoints/full_constraints/best_model.pt

  # Fully offline demo with synthetic data
  python -m insights demo
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for evaluation/ and models/

from insights import ingest, pipeline  # noqa: E402
from insights.sources import Sources  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m insights", description="Playlist analysis dashboards")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse playlists and write dashboards")
    a.add_argument("inputs", nargs="+", help=".csv (Exportify or title/artist), .json, or deezer:<id>")
    a.add_argument("--user", default="me")
    a.add_argument("--pid", type=int, help="playlist id inside an MPD slice JSON")
    a.add_argument("--name", help="playlist name (single input only)")
    a.add_argument("--out", default="reports")
    a.add_argument("--cache", default=".cache/insights")
    a.add_argument("--offline", action="store_true", help="use cached API responses only")
    a.add_argument("--no-enrich", action="store_true", help="skip all API calls")
    a.add_argument("--lastfm-key", help="Last.fm API key (or set LASTFM_API_KEY)")
    a.add_argument("--discovery", type=float, default=0.3, help="0 = popular picks, 1 = obscure picks")
    a.add_argument("--data-dir", help="processed MPD data, enables the trained-ranker recommendations")
    a.add_argument("--checkpoint", help="model checkpoint for --data-dir")
    a.add_argument("--inline-js", action="store_true", help="embed plotly.js (works without internet, +3.5 MB)")
    ar = sub.add_parser("artist", help="statistical lyrics study of one or more artists")
    ar.add_argument("names", nargs="+", help="artist names, e.g. Gojira 'Kendrick Lamar'")
    ar.add_argument("--out", default="reports")
    ar.add_argument("--cache", default=".cache/insights")
    ar.add_argument("--offline", action="store_true")
    ar.add_argument("--inline-js", action="store_true")
    ar.add_argument("--demo", action="store_true", help="synthetic artist with planted trends (offline)")
    d = sub.add_parser("demo", help="offline demo on synthetic data")
    d.add_argument("--out", default="reports")
    d.add_argument("--inline-js", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "artist":
        from insights.artist import run as artist_run
        if args.demo:
            from insights.artist import demo as artist_demo
            print(f"Artist page: {artist_run.build('demo', None, args.out, inline_js=args.inline_js, corpus=artist_demo.make())}")
            return 0
        src = Sources(cache_dir=args.cache, offline=args.offline)
        failed = 0
        for n in args.names:
            path = artist_run.build(n, src, args.out, inline_js=args.inline_js)
            print(f"Artist page: {path}" if path else f"Artist page: FAILED for {n}")
            failed += path is None
        logging.info("HTTP: %s", src.http.stats)
        return 1 if failed == len(args.names) else 0

    if args.cmd == "demo":
        from insights import demo
        path = demo.run(args.out, inline_js=args.inline_js)
        print(f"Dashboard: {path}")
        return 0

    src = None if args.no_enrich else Sources(cache_dir=args.cache, offline=args.offline, lastfm_key=args.lastfm_key)
    pls = [ingest.load(i, pid=args.pid, name=args.name if len(args.inputs) == 1 else None, owner=args.user,
                       deezer=src.deezer if src else None) for i in args.inputs]
    path = pipeline.build_user(args.user, pls, args.out, src=src, enrich_data=src is not None,
                               inline_js=args.inline_js, discovery=args.discovery,
                               data_dir=args.data_dir, checkpoint=args.checkpoint)
    if src:
        logging.info("HTTP: %s", src.http.stats)
    print(f"Dashboard: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
