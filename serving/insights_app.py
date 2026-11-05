"""Playlist analysis as a web service.

    uvicorn serving.insights_app:app --port 7860

A visitor pastes a playlist link (Deezer, YouTube / YouTube Music, or their own Spotify
playlist after logging in) or uploads an export file (Exportify CSV, Apple Music .txt/.xml,
JSON). The job runs in the background; the page polls for progress and then opens the
dashboard. Reports are deleted after ``JOB_TTL_HOURS``. Lyrics are never written to disk.

Environment
  YOUTUBE_API_KEY         enables YouTube / YouTube Music links (free key, Google Cloud console)
  SPOTIFY_CLIENT_ID       enables "Log in with Spotify" (Authorization Code + PKCE, no secret)
  SPOTIFY_REDIRECT_URI    must match the app's setting, e.g. https://<space>.hf.space/spotify/callback
  LASTFM_API_KEY          better tags and similar artists (optional)
  INSIGHTS_DATA_DIR       where reports and the metadata cache live (default /tmp/insights)
  MAX_TRACKS (300), MAX_UPLOAD_MB (3), JOB_TTL_HOURS (24), JOBS_PER_HOUR (10), WORKERS (2)
  INSIGHTS_OFFLINE=1      no network calls at all (tests)
"""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import os
import secrets
import shutil
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from insights import connectors, dashboard, enrich, ingest, pipeline  # noqa: E402
from insights.sources import Sources  # noqa: E402

logger = logging.getLogger("insights.app")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


DATA = Path(os.environ.get("INSIGHTS_DATA_DIR", "/tmp/insights"))
JOBS_DIR = DATA / "jobs"
MAX_TRACKS = _env_int("MAX_TRACKS", 300)
MAX_UPLOAD = _env_int("MAX_UPLOAD_MB", 3) * 1024 * 1024
TTL = _env_int("JOB_TTL_HOURS", 24) * 3600
PER_HOUR = _env_int("JOBS_PER_HOUR", 10)
OFFLINE = os.environ.get("INSIGHTS_OFFLINE") == "1"
YT_KEY = os.environ.get("YOUTUBE_API_KEY")
SP_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SP_REDIRECT = os.environ.get("SPOTIFY_REDIRECT_URI")

app = FastAPI(title="Playlist insights", docs_url=None, redoc_url=None)
jobs: Dict[str, Dict] = {}
sessions: Dict[str, Dict] = {}          # session id -> {"token", "expires", "verifier", "state"}
recent: Dict[str, deque] = defaultdict(deque)
lock = threading.Lock()
pool = ThreadPoolExecutor(max_workers=_env_int("WORKERS", 2))
_src: Optional[Sources] = None


def sources() -> Sources:
    global _src
    if _src is None:
        _src = Sources(cache_dir=str(DATA / "cache"), offline=OFFLINE)
    return _src


# ----------------------------------------------------------------------------- jobs

def _set(jid, **kw):
    with lock:
        jobs[jid].update(kw)


def _run(jid: str, kind: str, payload):
    try:
        src = sources()
        _set(jid, status="running", stage="reading the playlist")
        if kind == "file":
            pl = connectors.load_upload(*payload)
        else:
            service, pid = payload
            if OFFLINE:
                raise connectors.InputError("Links need network access, which is off on this server.")
            if service == "deezer":
                pl = ingest.load_deezer_playlist(int(pid), src.deezer)
            elif service == "youtube":
                pl = connectors.load_youtube(src.http, pid, YT_KEY, max_tracks=MAX_TRACKS)
            elif service == "spotify":
                pl = connectors.load_spotify(_spotify_get, pid, jobs[jid].get("_token"), max_tracks=MAX_TRACKS)
            else:
                raise connectors.InputError("Unsupported link.")
        if len(pl.tracks) > MAX_TRACKS:
            pl.tracks = pl.tracks[:MAX_TRACKS]
            _set(jid, note=f"Only the first {MAX_TRACKS} tracks were analysed.")
        _set(jid, stage=f"found {len(pl.tracks)} tracks; fetching lyrics and details", name=pl.name, tracks=len(pl.tracks))

        def progress(stage, i, n):
            _set(jid, stage=f"{stage}: {i}/{n}", progress={"stage": stage, "i": i, "n": n})

        if not OFFLINE:
            enrich.enrich(pl, src, progress=progress)
        _set(jid, stage="analysing")
        a = pipeline.analyze(pl, None if OFFLINE else src)
        out = JOBS_DIR / jid / "index.html"
        dashboard.render_playlist(pl, a, out, user_index_href="/")
        _set(jid, status="done", stage="done", finished=time.time())
    except connectors.InputError as e:
        _set(jid, status="error", stage="error", error=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("job %s failed", jid)
        _set(jid, status="error", stage="error", error=f"Something went wrong ({type(e).__name__}). Try again, or upload a file.")
    finally:
        with lock:
            jobs[jid].pop("_token", None)


def _janitor():
    while True:
        cutoff = time.time() - TTL
        with lock:
            old = [j for j, v in jobs.items() if v["created"] < cutoff]
            for j in old:
                jobs.pop(j, None)
            for s in [s for s, v in sessions.items() if v.get("expires", 0) < time.time() - 3600]:
                sessions.pop(s, None)
        for j in old:
            shutil.rmtree(JOBS_DIR / j, ignore_errors=True)
        if JOBS_DIR.exists():
            for d in JOBS_DIR.iterdir():
                if d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
        time.sleep(600)


threading.Thread(target=_janitor, daemon=True).start()


def _client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")


def _rate_ok(ip: str) -> bool:
    now = time.time()
    with lock:
        q = recent[ip]
        while q and q[0] < now - 3600:
            q.popleft()
        if len(q) >= PER_HOUR:
            return False
        q.append(now)
        return True


@app.post("/api/analyze")
async def analyze(request: Request, url: str = Form(""), file: Optional[UploadFile] = File(None)):
    if not _rate_ok(_client_ip(request)):
        raise HTTPException(429, f"At most {PER_HOUR} analyses per hour. Try again later.")
    token = None
    if file is not None and file.filename:
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"Files up to {MAX_UPLOAD // (1024 * 1024)} MB.")
        kind, payload = "file", (file.filename, data)
    elif url.strip():
        try:
            service, pid = connectors.parse_link(url)
        except connectors.InputError as e:
            raise HTTPException(400, str(e))
        if service == "youtube" and not YT_KEY:
            raise HTTPException(400, "YouTube links are not enabled on this server yet. Upload a file or use a Deezer link.")
        if service == "spotify":
            if not SP_ID:
                raise HTTPException(400, "Spotify login is not enabled on this server. Export the playlist with Exportify "
                                         "(exportify.app) and upload the CSV.")
            s = sessions.get(request.cookies.get("sid", ""))
            token = s.get("token") if s and s.get("expires", 0) > time.time() else None
            if not token:
                raise HTTPException(401, "Log in with Spotify first (button above).")
        kind, payload = "link", (service, pid)
    else:
        raise HTTPException(400, "Paste a playlist link or choose a file.")
    jid = secrets.token_urlsafe(9)
    with lock:
        jobs[jid] = {"status": "queued", "stage": "queued", "created": time.time(), "_token": token}
    pool.submit(_run, jid, kind, payload)
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "Unknown or expired job.")
    out = {k: v for k, v in j.items() if not k.startswith("_")}
    if j["status"] == "done":
        out["report"] = f"/r/{jid}"
    return out


@app.get("/r/{jid}")
def report(jid: str):
    if not jid.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(404)
    f = JOBS_DIR / jid / "index.html"
    if not f.exists():
        raise HTTPException(404, "This report has expired. Reports are kept for a day.")
    return FileResponse(f, media_type="text/html")


@app.get("/healthz")
def healthz():
    return {"ok": True, "offline": OFFLINE, "youtube": bool(YT_KEY), "spotify": bool(SP_ID and SP_REDIRECT),
            "jobs": len(jobs)}


# ----------------------------------------------------------------------------- Spotify login (PKCE)

def _spotify_get(url, params, token):
    import requests
    try:
        r = requests.get(url, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    except Exception:  # noqa: BLE001
        return 0, None
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, None


@app.get("/spotify/login")
def spotify_login(request: Request):
    if not (SP_ID and SP_REDIRECT):
        raise HTTPException(404, "Spotify login is not enabled on this server.")
    sid = request.cookies.get("sid") or secrets.token_urlsafe(18)
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    sessions[sid] = {"verifier": verifier, "state": state, "expires": 0}
    q = urlencode({"client_id": SP_ID, "response_type": "code", "redirect_uri": SP_REDIRECT, "state": state,
                   "scope": "playlist-read-private playlist-read-collaborative", "code_challenge_method": "S256",
                   "code_challenge": challenge})
    resp = RedirectResponse(f"https://accounts.spotify.com/authorize?{q}")
    resp.set_cookie("sid", sid, httponly=True, samesite="lax", secure=SP_REDIRECT.startswith("https"), max_age=3600)
    return resp


@app.get("/spotify/callback")
def spotify_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    s = sessions.get(request.cookies.get("sid", ""))
    if error or not s or not code or state != s.get("state"):
        return RedirectResponse("/?spotify=failed")
    import requests
    try:
        r = requests.post("https://accounts.spotify.com/api/token", timeout=30, data={
            "grant_type": "authorization_code", "code": code, "redirect_uri": SP_REDIRECT, "client_id": SP_ID,
            "code_verifier": s["verifier"]})
        tok = r.json() if r.status_code == 200 else {}
    except Exception:  # noqa: BLE001
        tok = {}
    if not tok.get("access_token"):
        return RedirectResponse("/?spotify=failed")
    s.update(token=tok["access_token"], expires=time.time() + int(tok.get("expires_in", 3600)) - 60)
    s.pop("verifier", None)
    return RedirectResponse("/?spotify=ok")


@app.get("/spotify/logout")
def spotify_logout(request: Request):
    sessions.pop(request.cookies.get("sid", ""), None)
    resp = RedirectResponse("/")
    resp.delete_cookie("sid")
    return resp


# ----------------------------------------------------------------------------- page

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    s = sessions.get(request.cookies.get("sid", ""))
    logged = bool(s and s.get("token") and s.get("expires", 0) > time.time())
    sp = ""
    if SP_ID and SP_REDIRECT:
        sp = ("<span class='ok'>Logged in with Spotify</span> <a href='/spotify/logout'>log out</a>" if logged else
              "<a class='btn' href='/spotify/login'>Log in with Spotify</a> <span class='note'>to analyse playlists you own</span>")
    links = ["Deezer"] + (["YouTube / YouTube Music"] if YT_KEY else []) + (["Spotify (your own playlists)"] if SP_ID else [])
    return PAGE.replace("__SPOTIFY__", sp).replace("__LINKS__", html.escape(", ".join(links))) \
        .replace("__MAX__", str(MAX_TRACKS)).replace("__TTL__", str(TTL // 3600))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analyse a playlist</title><style>
:root{color-scheme:light;--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:rgba(11,11,11,.1);--acc:#2a78d6;--bad:#d03b3b;--good:#0ca30c}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:rgba(255,255,255,.1);--acc:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:720px;margin:0 auto;padding:40px 16px 64px}h1{font-size:clamp(30px,6vw,46px);margin:0 0 8px;letter-spacing:-.02em}
.lede{color:var(--ink2);margin:0 0 22px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:14px}
input[type=url]{width:100%;font:inherit;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.or{color:var(--muted);text-align:center;margin:12px 0}.btn,button{display:inline-block;font:inherit;font-weight:600;padding:10px 16px;border-radius:999px;
border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer;text-decoration:none}button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
.note{color:var(--muted);font-size:13.5px}.ok{color:var(--good);font-weight:600}.err{color:var(--bad);font-weight:600}
.bar{height:8px;background:var(--line);border-radius:99px;overflow:hidden;margin:10px 0}.bar i{display:block;height:100%;width:0;background:var(--acc);transition:width .4s}
a{color:var(--acc)}ul{padding-left:18px}</style></head><body><main>
<h1>Analyse a playlist</h1>
<p class="lede">Paste a link or upload an export. You get a dashboard: lyrics and themes, artist map, mood, tempo flow, where the music comes from, and recommendations.</p>
<div class="card"><form id="f">
<p style="margin-top:0">__SPOTIFY__</p>
<label for="url"><b>Playlist link</b> <span class="note">(__LINKS__)</span></label>
<input id="url" name="url" type="url" placeholder="https://www.deezer.com/playlist/..." autocomplete="off">
<div class="or">or</div>
<label><b>Upload a file</b> <span class="note">Exportify CSV · Apple Music .txt or .xml · JSON · any CSV with title and artist</span></label><br>
<input id="file" name="file" type="file" accept=".csv,.txt,.tsv,.xml,.json"><br><br>
<button class="primary" type="submit">Analyse</button>
</form></div>
<div class="card" id="status" hidden><b id="st">Starting…</b><div class="bar"><i id="bar"></i></div><p class="note" id="detail"></p></div>
<div class="card"><b>Where to get a file</b><ul>
<li><b>Spotify:</b> <a href="https://exportify.app/">Exportify</a> exports any of your playlists as CSV (turn on artist genres and audio features in its settings).</li>
<li><b>Apple Music:</b> in the Music app, select the playlist, then File &gt; Library &gt; Export Playlist (text or XML).</li>
<li><b>Anything else:</b> a CSV or JSON with a title column and an artist column.</li></ul>
<p class="note">Up to __MAX__ tracks. Your report is private to its link and deleted after __TTL__ hours. Lyrics are read in memory to compute statistics
and never stored. Spotify only lets apps read playlists you own or collaborate on.</p></div>
</main><script>
const f=document.getElementById('f'),box=document.getElementById('status'),st=document.getElementById('st'),bar=document.getElementById('bar'),det=document.getElementById('detail');
const q=new URLSearchParams(location.search);if(q.get('spotify')==='failed'){box.hidden=false;st.className='err';st.textContent='Spotify login failed. Try again.'}
f.onsubmit=async e=>{e.preventDefault();const fd=new FormData(f);box.hidden=false;st.className='';st.textContent='Starting…';bar.style.width='3%';det.textContent='';
 const r=await fetch('/api/analyze',{method:'POST',body:fd});const b=await r.json().catch(()=>({}));
 if(!r.ok){st.className='err';st.textContent=b.detail||('Error '+r.status);bar.style.width='0';return}
 poll(b.job)};
async function poll(id){const r=await fetch('/api/jobs/'+id);const j=await r.json();
 st.textContent=j.status==='done'?'Done':j.status==='error'?'Could not analyse this playlist':(j.name?j.name+': ':'')+j.stage;
 if(j.progress){const w={deezer:30,lyrics:60,artists:90}[j.progress.stage]||50;bar.style.width=Math.min(95,w*j.progress.i/j.progress.n+5)+'%'}
 if(j.note)det.textContent=j.note;
 if(j.status==='done'){bar.style.width='100%';st.className='ok';location.href=j.report;return}
 if(j.status==='error'){st.className='err';det.textContent=j.error;bar.style.width='0';return}
 setTimeout(()=>poll(id),1500)}
</script></body></html>"""


def main():
    import uvicorn
    uvicorn.run("serving.insights_app:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
