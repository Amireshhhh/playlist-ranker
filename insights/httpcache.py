"""HTTP client with an on-disk cache and per-host rate limiting.

Every free API used here asks clients to be gentle (MusicBrainz: about 1 request/s and a
descriptive User-Agent). Responses are cached forever on disk, so re-running an analysis
or rebuilding a dashboard makes no network calls for data already fetched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse

logger = logging.getLogger(__name__)

DEFAULT_UA = "playlist-ranker-insights/0.1 (+https://github.com/Amireshhhh/playlist-ranker)"

# Minimum seconds between two requests to the same host.
MIN_INTERVAL = {
    "musicbrainz.org": 1.1,
    "ws.audioscrobbler.com": 0.25,
    "api.deezer.com": 0.12,     # stays under 50 requests / 5 s
    "lrclib.net": 0.5,          # returned HTTP 429 at 0.1 s spacing (observed in CI, Nov 2026)
    "api.reccobeats.com": 0.2,
    "en.wikipedia.org": 0.1,
}


# Responses from these hosts are kept in memory for one run only and never written to disk:
# LRCLIB responses contain full lyrics text, which this project must not store.
NO_DISK = {"lrclib.net"}


class OfflineError(RuntimeError):
    pass


class CachedHTTP:
    def __init__(self, cache_dir: str = ".cache/insights", offline: bool = False,
                 user_agent: str = DEFAULT_UA, timeout: float = 45.0, retries: int = 3):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.user_agent = os.environ.get("INSIGHTS_USER_AGENT", user_agent)
        self.timeout = timeout
        self.retries = retries
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._session = None
        self._memory: Dict[str, Any] = {}
        self.stats = {"hits": 0, "requests": 0, "retries": 0, "errors": 0}

    def _path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def _wait(self, host: str):
        gap = MIN_INTERVAL.get(host, 0.1)
        with self._lock:
            now = time.monotonic()
            delay = self._last.get(host, 0) + gap - now
            if delay > 0:
                time.sleep(delay)
            self._last[host] = time.monotonic()

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None,
                 cache_key_exclude: tuple = ("api_key",)) -> Optional[Any]:
        """GET a JSON document. Returns None on 404 / network failure / offline cache miss.

        Parameters named in ``cache_key_exclude`` (API keys) are left out of the cache key,
        so cached files never contain secrets and survive a key change.
        """
        params = {k: v for k, v in (params or {}).items() if v is not None}
        full = f"{url}?{urlencode(params)}" if params else url
        key_params = {k: v for k, v in params.items() if k not in cache_key_exclude}
        cache_url = f"{url}?{urlencode(key_params)}" if key_params else url
        path = self._path(cache_url)
        host = urlparse(url).hostname or ""
        if host in NO_DISK:
            if cache_url in self._memory:
                self.stats["hits"] += 1
                return self._memory[cache_url]
        elif path.exists():
            self.stats["hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))["body"]
        if self.offline:
            return None
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self.user_agent, "Accept": "application/json"})
        r = None
        for attempt in range(self.retries + 1):
            self._wait(host)
            self.stats["requests"] += 1
            try:
                r = self._session.get(full, timeout=self.timeout)
            except Exception as e:  # network errors: retry with backoff, then report (never raise)
                if attempt < self.retries:
                    self.stats["retries"] += 1
                    logger.warning("GET %s failed (%s), retrying", cache_url, type(e).__name__)
                    time.sleep(2.0 * 2 ** attempt)
                    continue
                self.stats["errors"] += 1
                logger.warning("GET %s failed: %s", cache_url, e)
                return None
            if r.status_code not in (429, 502, 503, 504) or attempt == self.retries:
                break
            # Rate limited / temporarily unavailable: honour Retry-After, else back off 2, 4, 8 s.
            try:
                delay = min(float(r.headers.get("Retry-After", "")), 30.0)
            except ValueError:
                delay = 2.0 * 2 ** attempt
            self.stats["retries"] += 1
            time.sleep(delay)
        if r.status_code == 404:
            body = None
        elif r.status_code == 200:
            try:
                body = r.json()
            except ValueError:
                self.stats["errors"] += 1
                return None
        else:
            # 429 / 5xx etc.: do not cache, so a later run retries.
            self.stats["errors"] += 1
            logger.warning("GET %s -> HTTP %s", cache_url, r.status_code)
            return None
        if host in NO_DISK:
            self._memory[cache_url] = body
            return body
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": cache_url, "fetched": time.time(), "body": body}), encoding="utf-8")
        return body
