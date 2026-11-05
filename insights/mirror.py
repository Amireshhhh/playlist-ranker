"""Mirror: the same language measures on your own writing, computed in your browser.

Writes one static page. Text the visitor adds never leaves their browser: the page makes no
network request with it (the only request is the chart library from its CDN), and stores
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from .dashboard import DARK, LIGHT, PLOTLY_CDN
from .lyrics import STOPWORDS

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Mirror</title>
<script src="__PLOTLY__" charset="utf-8"></script>
<style>
:root{color-scheme:light;--bg:__page__;--card:__surface__;--ink:__text__;--ink2:__text2__;--muted:__muted__;--grid:__grid__;--line:rgba(11,11,11,.1);--acc:__s1__}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:__dpage__;--card:__dsurface__;--ink:__dtext__;--ink2:__dtext2__;--muted:__dmuted__;--grid:__dgrid__;--line:rgba(255,255,255,.1);--acc:__ds1__}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}a{color:var(--acc)}
h1{font-size:clamp(30px,5vw,46px);margin:8px 0 6px;letter-spacing:-.01em}h2{font-size:19px;margin:28px 0 10px}
.lede{color:var(--ink2);max-width:66ch;font-size:17px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px;min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px}
textarea{width:100%;min-height:180px;font:14px/1.5 ui-monospace,monospace;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:10px}
button,label.btn{font:inherit;font-weight:600;padding:8px 14px;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}
.note{color:var(--muted);font-size:13px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}.tile b{display:block;font-size:24px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.lock{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:13px;color:var(--ink2)}
#out{display:none}.chart{width:100%;min-height:260px}.scroll{max-height:420px;overflow:auto}
</style></head><body><main>
<a href="../index.html">← home</a>
<h1>Mirror</h1>
<p class="lede">The same measures the Atlas uses on artists, run on your own writing: journal entries, essays, letters, posts.
Watch your voice change over time: how often you write “I” versus “we”, how varied your words are, what you keep coming back to.</p>
<p><span class="lock">🔒 Runs entirely in your browser. Your text is never uploaded or saved.</span></p>
<div class="card">
<p style="margin-top:0"><b>Add text.</b> Choose several .txt or .md files (one per entry; names that start with a date such as
<code>2024-03-01</code> are sorted by it), or paste below with a line containing only <code>---</code> between entries.</p>
<textarea id="paste" placeholder="First entry...&#10;---&#10;Second entry..."></textarea>
<div class="row"><label class="btn">Choose files<input id="files" type="file" multiple accept=".txt,.md,text/plain" hidden></label>
<button class="primary" id="go">Analyse</button><button id="sample">Try a sample</button><span class="note" id="status"></span></div>
</div>
<div id="out">
<div class="tiles" id="tiles"></div>
<div class="card" id="finding"></div>
<div class="grid">
<div class="card"><h2 style="margin-top:0">Who you write about</h2><div class="chart" id="c_pron"></div><p class="note">Per 100 words in each entry. “I” = I, me, my, mine, myself.</p></div>
<div class="card"><h2 style="margin-top:0">Word variety</h2><div class="chart" id="c_mattr"></div><p class="note">Moving-average type-token ratio over 50-word windows, which does not depend on length. Blank under 50 words.</p></div>
<div class="card"><h2 style="margin-top:0">Your most used words</h2><div class="chart" id="c_top"></div><p class="note">Function words removed.</p></div>
<div class="card"><h2 style="margin-top:0">Words that rose and fell</h2><div class="scroll"><table id="t_shift"></table></div><p class="note">Rate per 1,000 words in your later half vs your earlier half (words used 5+ times).</p></div>
</div>
<div class="card"><h2 style="margin-top:0">Entry by entry</h2><div class="scroll"><table id="t_docs"></table></div></div>
</div>
<p class="note">Method: tokens are letter sequences with contractions expanded (“I'm” → “I am”). The trend test is Spearman ρ between entry order and
the entry's rate, with a permutation p over 10,000 shuffles of the order. Nothing here is a diagnosis of anything.</p>
</main>
<script>
const STOP=new Set(__STOP__), PAL={light:__LIGHT__,dark:__DARK__};
const PRON={I:new Set(['i','me','my','mine','myself']),we:new Set(['we','us','our','ours','ourselves']),you:new Set(['you','your','yours','yourself','yourselves'])};
const pal=()=>PAL[matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'];
function expand(t){return t.toLowerCase().replace(/’/g,"'").replace(/\bi'm\b/g,'i am').replace(/\b(\w+)'re\b/g,'$1 are').replace(/\b(\w+)'ve\b/g,'$1 have')
 .replace(/\b(\w+)'ll\b/g,'$1 will').replace(/\b(\w+)'d\b/g,'$1 would').replace(/\bcan't\b/g,'can not').replace(/\bwon't\b/g,'will not').replace(/\b(\w+)n't\b/g,'$1 not')}
const tok=t=>(expand(t).match(/\p{L}+(?:'\p{L}+)*/gu)||[]).map(w=>w.replace(/'/g,''));
function mattr(t,w=50){if(t.length<w)return null;const c=new Map();for(let i=0;i<w;i++)c.set(t[i],(c.get(t[i])||0)+1);let s=c.size;
 for(let i=w;i<t.length;i++){const o=t[i-w];c.set(o,c.get(o)-1);if(!c.get(o))c.delete(o);c.set(t[i],(c.get(t[i])||0)+1);s+=c.size}return s/(t.length-w+1)/w}
function rank(a){const o=a.map((v,i)=>[v,i]).sort((x,y)=>x[0]-y[0]);const r=Array(a.length);let i=0;while(i<o.length){let j=i;while(j+1<o.length&&o[j+1][0]===o[i][0])j++;
 for(let k=i;k<=j;k++)r[o[k][1]]=(i+j)/2+1;i=j+1}return r}
function pearson(x,y){const n=x.length,mx=x.reduce((a,b)=>a+b)/n,my=y.reduce((a,b)=>a+b)/n;let sxy=0,sx=0,sy=0;for(let i=0;i<n;i++){sxy+=(x[i]-mx)*(y[i]-my);sx+=(x[i]-mx)**2;sy+=(y[i]-my)**2}return sx&&sy?sxy/Math.sqrt(sx*sy):0}
function spearmanPerm(y,n=10000){const x=rank(y.map((_,i)=>i)),ry=rank(y),rho=pearson(x,ry);let hit=0;const p=x.slice();
 for(let k=0;k<n;k++){for(let i=p.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[p[i],p[j]]=[p[j],p[i]]}if(Math.abs(pearson(p,ry))>=Math.abs(rho)-1e-12)hit++}
 return {rho,p:(hit+1)/(n+1)}}
function docsFromPaste(t){return t.split(/^\s*---\s*$/m).map((s,i)=>({name:'Entry '+(i+1),text:s.trim()})).filter(d=>d.text)}
async function docsFromFiles(fl){const out=[];for(const f of fl)out.push({name:f.name.replace(/\.(txt|md)$/i,''),text:await f.text()});
 out.sort((a,b)=>a.name.localeCompare(b.name));return out}
function el(tag,txt){const e=document.createElement(tag);if(txt!==undefined)e.textContent=txt;return e}
function table(id,head,rows){const t=document.getElementById(id);t.replaceChildren();const h=t.createTHead().insertRow();head.forEach(x=>h.appendChild(el('th',x)));
 const b=t.createTBody();rows.forEach(r=>{const tr=b.insertRow();r.forEach((c,i)=>{const td=tr.insertCell();td.textContent=c;if(i>0&&typeof c!=='string')td.className='n'})})}
function plot(id,data,layout){const p=pal();Plotly.react(id,data,Object.assign({height:280,margin:{l:44,r:12,t:8,b:44},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
 font:{color:p.text2,family:'system-ui'},xaxis:{gridcolor:p.grid},yaxis:{gridcolor:p.grid},legend:{orientation:'h',y:1.15}},layout),{displayModeBar:false,responsive:true})}
function analyse(docs){
 if(!docs.length){document.getElementById('status').textContent='Add some text first.';return}
 const p=pal(),D=docs.map(d=>{const t=tok(d.text);const lines=d.text.split(/\n+/).map(s=>s.trim().toLowerCase()).filter(Boolean);
  const n=Math.max(t.length,1);const r={name:d.name,words:t.length,t};for(const g in PRON)r[g]=100*t.filter(w=>PRON[g].has(w)).length/n;
  r.mattr=mattr(t);r.rep=lines.length?1-new Set(lines).size/lines.length:0;
  const sents=d.text.split(/[.!?]+\s/).filter(s=>s.trim());r.sent=t.length/Math.max(sents.length,1);return r});
 const all=D.flatMap(d=>d.t),cw=new Map();all.forEach(w=>{if(!STOP.has(w)&&w.length>2)cw.set(w,(cw.get(w)||0)+1)});
 const top=[...cw].sort((a,b)=>b[1]-a[1]).slice(0,20);
 document.getElementById('out').style.display='block';
 const tiles=document.getElementById('tiles');tiles.replaceChildren();
 [['Entries',D.length],['Words',all.length.toLocaleString()],['Distinct words',new Set(all).size.toLocaleString()],
  ['“I” per 100',(100*all.filter(w=>PRON.I.has(w)).length/Math.max(all.length,1)).toFixed(1)]].forEach(([l,v])=>{const d=el('div');d.className='tile';d.appendChild(el('b',v));d.appendChild(el('span',l));tiles.appendChild(d)});
 const f=document.getElementById('finding');f.replaceChildren();f.appendChild(el('h2','What changed'));
 if(D.length>=5){const ul=el('ul');for(const g of ['I','we','you']){const s=spearmanPerm(D.map(d=>d[g]));const first=D.slice(0,Math.ceil(D.length/3)),last=D.slice(-Math.ceil(D.length/3));
   const m=a=>(a.reduce((x,d)=>x+d[g],0)/a.length).toFixed(1);
   ul.appendChild(el('li',`“${g}”: ${m(first)} per 100 words in your first third of entries, ${m(last)} in your last third. Trend ρ ${s.rho>=0?'+':''}${s.rho.toFixed(2)}, permutation p ${s.p<0.001?'<0.001':s.p.toFixed(3)} — ${s.p<0.05?'a real trend':'not distinguishable from chance'}.`))}
  f.appendChild(ul)}else f.appendChild(el('p','Add at least 5 entries to test for trends over time.'));
 const x=D.map(d=>d.name);
 plot('c_pron',[['I',p.s2],['we',p.s1],['you',p.s3]].map(([g,c])=>({type:'scatter',mode:'lines+markers',name:g,x,y:D.map(d=>d[g]),line:{color:c,width:2},marker:{size:7}})),{xaxis:{gridcolor:p.grid,tickangle:-30,automargin:true},yaxis:{gridcolor:p.grid,title:'per 100 words'}});
 plot('c_mattr',[{type:'scatter',mode:'lines+markers',x,y:D.map(d=>d.mattr),line:{color:p.s1,width:2},marker:{size:7},connectgaps:false}],{showlegend:false,xaxis:{gridcolor:p.grid,tickangle:-30,automargin:true},yaxis:{gridcolor:p.grid,title:'MATTR'}});
 plot('c_top',[{type:'bar',orientation:'h',y:top.map(t=>t[0]).reverse(),x:top.map(t=>t[1]).reverse(),marker:{color:p.s1}}],{height:Math.max(260,22*top.length+40),margin:{l:100,r:12,t:8,b:30},showlegend:false});
 const half=Math.floor(D.length/2),A=D.slice(0,half).flatMap(d=>d.t),B=D.slice(half).flatMap(d=>d.t),ca=new Map(),cb=new Map();
 A.forEach(w=>ca.set(w,(ca.get(w)||0)+1));B.forEach(w=>cb.set(w,(cb.get(w)||0)+1));
 const shift=[...cw].filter(([w,c])=>c>=5).map(([w])=>[w,1000*(ca.get(w)||0)/Math.max(A.length,1),1000*(cb.get(w)||0)/Math.max(B.length,1)]).map(r=>[...r,r[2]-r[1]]).sort((a,b)=>Math.abs(b[3])-Math.abs(a[3])).slice(0,20);
 table('t_shift',['Word','Earlier','Later','Change'],shift.map(r=>[r[0],+r[1].toFixed(1),+r[2].toFixed(1),(r[3]>0?'+':'')+r[3].toFixed(1)]));
 table('t_docs',['Entry','Words','I','we','you','MATTR','Repeated lines','Words/sentence'],D.map(d=>[d.name,d.words,+d.I.toFixed(1),+d.we.toFixed(1),+d.you.toFixed(1),d.mattr===null?'':+d.mattr.toFixed(3),Math.round(100*d.rep)+'%',+d.sent.toFixed(1)]));
 document.getElementById('status').textContent='Done. Nothing was uploaded.';
}
document.getElementById('go').onclick=async()=>{const fl=document.getElementById('files').files;analyse(fl.length?await docsFromFiles(fl):docsFromPaste(document.getElementById('paste').value))};
document.getElementById('files').onchange=e=>{document.getElementById('status').textContent=e.target.files.length+' file(s) chosen'};
document.getElementById('sample').onclick=()=>{const S=__SAMPLE__;document.getElementById('paste').value=S.join('\n---\n');analyse(docsFromPaste(S.join('\n---\n')))};
</script></body></html>"""


def _sample():
    """Invented diary with a planted shift from 'I' to 'we', so the trend test has something to find."""
    import random
    rng = random.Random(3)
    topics = "work morning coffee city river book friends garden rain music walk plan".split()
    out = []
    for k in range(10):
        p_i = 0.22 - 0.018 * k
        p_we = 0.02 + 0.018 * k
        words = []
        for _ in range(140):
            r = rng.random()
            words.append(rng.choice(["i", "my", "me"]) if r < p_i else rng.choice(["we", "our", "us"]) if r < p_i + p_we
                         else rng.choice(topics + ["the", "a", "and", "to", "of", "was", "today"]))
        out.append(f"Day {k + 1}. " + " ".join(words) + ".")
    return out


def build(site: str) -> Path:
    rep = {"__PLOTLY__": PLOTLY_CDN, "__STOP__": json.dumps(sorted(STOPWORDS)), "__LIGHT__": json.dumps(LIGHT),
           "__DARK__": json.dumps(DARK), "__SAMPLE__": json.dumps(_sample())}
    for k, v in LIGHT.items():
        rep[f"__{k}__"] = v
    for k, v in DARK.items():
        rep[f"__d{k}__"] = v
    html = PAGE
    for k in sorted(rep, key=len, reverse=True):   # longest first: __dsurface__ before __surface__
        html = html.replace(k, rep[k])
    out = Path(site) / "mirror" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "site"))
