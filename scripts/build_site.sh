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
if [ -f examples/artists.txt ]; then
  while IFS= read -r a; do
    [ -z "${a// }" ] && continue
    python -m insights artist "$a" --out "$OUT" --cache .cache/insights || echo "artist failed: $a"
  done < examples/artists.txt
fi

# 5. Cross-artist Atlas, then the home page.
python -m insights atlas --site "$OUT" || echo "atlas skipped"
python -m insights.site "$OUT" || exit 1
touch "$OUT/.nojekyll"
echo "Site built in $OUT"
