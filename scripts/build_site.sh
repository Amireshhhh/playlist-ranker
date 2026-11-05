#!/usr/bin/env bash
# Build the static site into ./site (used by .github/workflows/dashboards.yml).
set -uo pipefail
OUT=${1:-site}
rm -rf "$OUT" && mkdir -p "$OUT"

# 1. Synthetic playlist demo: always works, no network.
python -m insights demo --out "$OUT" || exit 1

# 2. A real playlist enriched from the free APIs. Failures here must not block the rest.
python -m insights analyze examples/heavy_and_more.csv --name "Heavy and More" --user example \
  --out "$OUT" --cache .cache/insights || echo "example playlist failed"

# 3. Synthetic artist with planted trends (offline; shows every section of an artist page).
python -m insights artist --demo x --out "$OUT" >/dev/null 2>&1 || true

# 4. Artist lyric studies (MusicBrainz discography + LRCLIB lyrics). One artist per line.
#    PREV (the previously published site, checked out by CI) lets unchanged analyses be reused.
PREV_ARG=()
[ -n "${PREV:-}" ] && [ -d "$PREV/artists" ] && PREV_ARG=(--prev "$PREV")
if [ -f examples/artists.txt ]; then
  while IFS= read -r a; do
    [ -z "${a// }" ] && continue
    python -m insights artist "$a" --out "$OUT" --cache .cache/insights "${PREV_ARG[@]}" || echo "artist failed: $a"
  done < examples/artists.txt
fi

# 5. Beyond music: US inaugural addresses (public domain, NLTK corpus).
SPEECH_DIR=.cache/speech-corpus
if [ ! -d "$SPEECH_DIR/inaugural" ]; then
  mkdir -p "$SPEECH_DIR" && curl -fsSL -o "$SPEECH_DIR/inaugural.zip" \
    https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/inaugural.zip \
    && unzip -q -o "$SPEECH_DIR/inaugural.zip" -d "$SPEECH_DIR" || echo "speech corpus download failed"
fi
[ -d "$SPEECH_DIR/inaugural" ] && { python -m insights speeches "$SPEECH_DIR/inaugural" --out "$OUT" || echo "speeches failed"; }

# 6. Cross-artist Atlas, the game, Mirror, then the home page.
python -m insights atlas --site "$OUT" || echo "atlas skipped"
python -m insights.play "$OUT" || echo "game skipped"
python -m insights.mirror "$OUT" || exit 1
python -m insights.site "$OUT" || exit 1
touch "$OUT/.nojekyll"
echo "Site built in $OUT"
