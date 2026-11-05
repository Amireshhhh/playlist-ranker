"""'Whose words?': a daily guessing game built from the Atlas signature words.

Each round shows one artist's most distinctive words (Monroe et al. log-odds against all the
other artists) and four names; the wrong answers are that artist's nearest neighbours by
vocabulary, so a round is hard exactly when the artists really are alike. Same puzzle for
everyone on a given day (seeded by the date).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .dashboard import DARK, LIGHT

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Whose words?</title>
<style>
:root{color-scheme:light;--bg:__page__;--card:__surface__;--ink:__text__;--ink2:__text2__;--muted:__muted__;--line:rgba(11,11,11,.1);--acc:__s1__;--ok:#0ca30c;--bad:#d03b3b}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:__dpage__;--card:__dsurface__;--ink:__dtext__;--ink2:__dtext2__;--muted:__dmuted__;--line:rgba(255,255,255,.1);--acc:__ds1__}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:620px;margin:0 auto;padding:28px 16px 64px}a{color:var(--acc)}
h1{font-size:clamp(32px,7vw,48px);margin:8px 0 4px;letter-spacing:-.02em}.sub{color:var(--ink2);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px}
.words{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}.words span{font-size:20px;font-weight:600;padding:6px 14px;border-radius:999px;border:1px solid var(--line);background:var(--bg)}
.opts{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.opts button{font:inherit;font-weight:600;padding:14px 10px;border-radius:12px;border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
.opts button:hover{border-color:var(--acc)}.opts button.ok{background:var(--ok);border-color:var(--ok);color:#fff}.opts button.bad{background:var(--bad);border-color:var(--bad);color:#fff}
.meta{display:flex;justify-content:space-between;color:var(--muted);font-size:14px;margin-bottom:6px}
.next{margin-top:16px;font:inherit;font-weight:600;padding:10px 18px;border-radius:999px;border:0;background:var(--acc);color:#fff;cursor:pointer;display:none}
.hint{color:var(--ink2);font-size:14px;margin-top:12px;min-height:1.5em}.big{font-size:40px;font-weight:700}
</style></head><body><main>
<a href="../index.html">← home</a>
<h1>Whose words?</h1><p class="sub">Six words an artist uses far more than the other __N__ in the Atlas. Who is it? A new set every day.</p>
<div class="card" id="game"></div>
<p class="sub" style="margin-top:16px;font-size:13px">Words are each artist's strongest signature words (weighted log-odds vs all other artists). Wrong answers are the
artists with the most similar vocabulary. <a href="../atlas/index.html">See the Atlas</a>.</p>
</main><script>
const A=__DATA__, ROUNDS=Math.min(5,A.length);
function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const day=new Date().toISOString().slice(0,10), R=rng([...day].reduce((h,c)=>(h*31+c.charCodeAt(0))|0,7));
const shuffle=a=>{a=a.slice();for(let i=a.length-1;i>0;i--){const j=Math.floor(R()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a};
const pool=shuffle(A.filter(a=>a.words.length>=4)).slice(0,ROUNDS);
let k=0,score=0,marks=[];
const g=document.getElementById('game');
function el(t,txt,cls){const e=document.createElement(t);if(txt!==undefined)e.textContent=txt;if(cls)e.className=cls;return e}
function round(){
  g.replaceChildren();const a=pool[k];
  const m=el('div',undefined,'meta');m.append(el('span',`Round ${k+1} of ${pool.length}`),el('span',`Score ${score}`));g.appendChild(m);
  const w=el('div',undefined,'words');a.words.slice(0,6).forEach(x=>w.appendChild(el('span',x)));g.appendChild(w);
  const names=A.map(x=>x.artist).filter(n=>n!==a.artist);
  const near=a.nearest.filter(n=>n!==a.artist).slice(0,3);
  while(near.length<3){const n=names[Math.floor(R()*names.length)];if(!near.includes(n))near.push(n)}
  const opts=shuffle([a.artist,...near]),o=el('div',undefined,'opts'),hint=el('div','', 'hint'),nx=el('button',k+1<pool.length?'Next →':'See result','next');
  opts.forEach(n=>{const b=el('button',n);b.onclick=()=>{
    const right=n===a.artist;if(right)score++;marks.push(right?'🟩':'🟥');
    [...o.children].forEach(c=>{c.disabled=true;if(c.textContent===a.artist)c.classList.add('ok');else if(c===b)c.classList.add('bad')});
    hint.textContent=(right?'Yes! ':'It was '+a.artist+'. ')+'Closest vocabulary: '+a.nearest.join(', ')+'.';
    const link=el('a',' Open the full study →');link.href='../artists/'+a.slug+'/index.html';hint.appendChild(link);
    nx.style.display='inline-block'};o.appendChild(b)});
  nx.onclick=()=>{k++;k<pool.length?round():done()};
  g.append(o,hint,nx);
}
function done(){
  g.replaceChildren();g.appendChild(el('div',`${score} / ${pool.length}`,'big'));
  const share=`Whose words? ${day}\n${marks.join('')} ${score}/${pool.length}\n${location.href}`;
  let best=null;try{best=+localStorage.getItem('ww-best')||0;if(score>best)localStorage.setItem('ww-best',score)}catch(e){}
  g.appendChild(el('p',best!==null?`Your best: ${Math.max(best,score)} / ${pool.length}. Come back tomorrow for a new set.`:'Come back tomorrow for a new set.','hint'));
  const b=el('button','Copy result','next');b.style.display='inline-block';
  b.onclick=()=>{navigator.clipboard&&navigator.clipboard.writeText(share).then(()=>b.textContent='Copied!',()=>b.textContent='Copy failed')};g.appendChild(b);
}
round();
</script></body></html>"""


def build(site: str) -> Optional[Path]:
    ranks_f = Path(site) / "atlas" / "ranks.json"
    if not ranks_f.exists():
        return None
    ranks = json.loads(ranks_f.read_text(encoding="utf-8"))
    data = [{"slug": s, "artist": r["artist"], "words": r["distinctive"], "nearest": [n for n, _ in r["nearest"]]}
            for s, r in ranks.items()]
    if len(data) < 4:
        return None
    rep = {"__DATA__": json.dumps(data), "__N__": str(len(data) - 1)}
    rep.update({f"__{k}__": v for k, v in LIGHT.items()})
    rep.update({f"__d{k}__": v for k, v in DARK.items()})
    html = PAGE
    for k in sorted(rep, key=len, reverse=True):
        html = html.replace(k, rep[k])
    out = Path(site) / "play" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "site"))
