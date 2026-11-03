# Playlist Insights

`insights/` turns a playlist into a dashboard: statistics, lyrics analysis, song clusters,
an artist similarity graph, mood and tempo views, and recommendations. Every data source is
free, and every model runs locally.

```bash
pip install -r requirements-insights.txt

python -m insights demo                               # offline, synthetic data
python -m insights analyze exports/*.csv --user amir  # your playlists (Exportify CSVs)
python -m insights analyze deezer:<playlist id> --user amir
```

Output: `reports/<user>/index.html` (all of that user's playlists) and
`reports/<user>/<playlist>/index.html` + `summary.json` (numbers only) per playlist.
The pages are static HTML, so they can be hosted for free on GitHub Pages.

## Getting a playlist in, for free

| Input | How | Notes |
|---|---|---|
| Exportify CSV | [exportify.app](https://exportify.app/) ([source](https://github.com/watsonbox/exportify)) → export a playlist or Liked Songs | Columns: track/artist/album URIs and names, release date, duration, explicit, popularity, ISRC, added at/by. Turn on "Include artists data" for **Artist Genres** and "Include audio features data" for danceability … tempo; both are picked up automatically. |
| Public Deezer playlist | `deezer:<id>` (the number in the playlist URL) | No account or key. |
| MPD slice | `data/mpd/mpd.slice.X-Y.json --pid N` | The dataset the ranker trains on; with `--data-dir/--checkpoint` the trained ranker also recommends. |
| Any CSV / JSON | columns `title`/`track name` and `artist` | Anything else is optional. |

The Spotify Web API itself is not used. Since February 2026, a Development Mode app needs an
owner with an active Spotify Premium subscription (11 Feb 2026 for new apps, 9 Mar 2026 for
existing ones), and the February 2026 changelog removed the `popularity` field from tracks,
albums and artists in Development Mode. That makes it neither free nor complete for this use.

## What the dashboard shows

| Section | Contents | Method |
|---|---|---|
| Overview | tracks, listening time, artists, albums, median year, explicit share, mainstream score, lyric vocabulary, a list of computed facts, data coverage per source | plain counts |
| Artists | top artists, Lorenz curve + Gini, entropy, repeat rate (the ranker's playlist-health metrics from `evaluation/metrics.py`), artist kNN graph, artist profiles | cosine kNN (k = 3) over mean song vectors + Last.fm/Deezer similar-artist edges; Fruchterman–Reingold layout |
| Where the music comes from | choropleth + top countries | MusicBrainz artist country |
| Time | songs per decade, release year along the playlist, cumulative growth, weekday × hour heatmap of when songs were added | Exportify `Added At` / Deezer `time_add` |
| Lyrics | word cloud, top words, top word pairs, pronoun focus (I / you / we / they), emotional arc, NMF themes, song-to-song lyric similarity, per-song table, vocabulary richness by artist | stop-word filtering, VADER sentiment, TF-IDF, NMF, MATTR (50-word window), zlib compression ratio as repetitiveness |
| Sound | tempo histogram, tempo flow now vs a suggested smoother order, mood map, audio fingerprint radar | half/double-time-aware BPM distance, nearest-neighbour tour; Russell's valence × arousal quadrants |
| Song groups | PCA map + group cards | k-means on [tags \| lyric SVD \| standardised sound] blocks (each L2-normalised and weighted), k = 2–8 picked by silhouette |
| Recommendations | new artists with 30-second previews, more from artists you have, and (MPD only) the trained ranker's continuation | similar-artist voting weighted by artist share, minus a `--discovery` popularity term, then the repo's greedy diversity re-ranker plus a cap of 2 songs per artist |

Lyrics text is analysed in memory and never written to the page or to `summary.json`; only
counts and scores are. A test checks this (`tests/test_insights.py::test_demo_dashboards`).

## Free data sources used

| Source | Key | Used for | Limits |
|---|---|---|---|
| LRCLIB | none | plain and synced lyrics, instrumental flag | none documented; `/api/get` matches duration within ±2 s, so the client falls back to `/api/search` |
| Deezer API | none for public data | BPM, gain, popularity rank, previews, covers, album genres, related artists, top tracks, public playlists | about 50 requests / 5 s (third-party figure); the client waits 0.12 s between calls |
| MusicBrainz | none, but a descriptive User-Agent is required | artist country, type, start year, tags | 1 request / s on average, enforced in the client |
| Last.fm | free key (`LASTFM_API_KEY`) | weighted artist tags, similar artists with match scores | non-commercial use only; commercial or research use needs Last.fm's permission |
| ReccoBeats | none | Spotify-style audio features for Spotify track IDs | coverage and reliability vary, by user reports |
| Wikipedia REST | none | artist summary and image | text is CC BY-SA and the page links the source |

All responses are cached on disk (`.cache/insights/`). Re-runs and `--offline` make no
network calls for data already fetched. API keys are left out of cache keys and files.

**Not yet tested against the live services.** The development environment had no outbound
access to these hosts. The clients follow each service's documented or widely used response
shapes and are unit-tested with recorded-shape fakes. Run one real playlist first and check the
coverage chips on the Overview.

## Ideas for the next level (not built, services not checked in this session)

- **Compute audio features yourself.** Run `librosa` or Essentia on Deezer's 30-second
  previews to get key, mode, tempo confidence, spectral brightness and loudness. This replaces
  Spotify's audio-features endpoint, which was deprecated for new apps on 2024-11-27. Free and
  local, and it removes the ReccoBeats dependency.
- **Listening history.** ListenBrainz (open, free) or Last.fm `user.getRecentTracks` would
  show what the user actually plays next to what the playlist contains.
- **Album-art palette.** Cover Art Archive (free, keyed by MusicBrainz release id) + k-means
  on pixels gives a colour fingerprint of the playlist.
- **Semantic lyric themes.** A local sentence-embedding model (e.g. `all-MiniLM-L6-v2` via
  `sentence-transformers`) instead of TF-IDF would let "ocean" and "sea" count as one theme.
  A local LLM (Ollama) could name the NMF themes.
- **Cross-platform links.** Odesli / song.link maps a track to Spotify, Apple Music, YouTube
  and others from one URL.
- **Hosting.** GitHub Pages for the static dashboards, or a Streamlit app if the dashboards
  should take input interactively.
