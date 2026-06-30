# ui/map_html.py

def get_map_html(image_src: str) -> str:
    """
    Returns the full interactive map HTML with the given image_src
    (either a data URL for local PNG, or a remote URL) pre-loaded.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tagoloan River Basin — Interactive Map</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
:root{{--p:rgba(14,20,30,0.93);--a:#D32F2F;--t:#DDE4EC;--m:#6B7D8D;--b:rgba(255,255,255,0.08);--g:#43A047;--bl:#1E88E5;--y:#FFB300}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{overflow:hidden;background:#111820;font-family:'DM Sans',sans-serif;color:var(--t)}}
#upload-screen{{position:fixed;inset:0;background:#111820;display:flex;align-items:center;justify-content:center;z-index:200}}
#upload-screen.gone{{display:none}}
.ubox{{text-align:center;max-width:440px;padding:40px}}
.ubox h1{{font-size:22px;font-weight:700;color:#fff;margin-bottom:6px}}
.ubox .sub{{font-size:12px;color:var(--a);font-weight:600;text-transform:uppercase;letter-spacing:2px;margin-bottom:24px}}
.ubox p{{font-size:13px;color:var(--m);line-height:1.6;margin-bottom:28px}}
.drop{{border:2px dashed rgba(255,255,255,.12);border-radius:16px;padding:48px 32px;cursor:pointer;transition:all .2s;position:relative}}
.drop:hover,.drop.over{{border-color:var(--a);background:rgba(211,47,47,.05)}}
.drop i{{font-size:36px;color:var(--m);margin-bottom:14px;display:block;transition:color .2s}}
.drop:hover i{{color:var(--a)}}
.drop .dt{{font-size:14px;font-weight:600;color:var(--t);margin-bottom:4px}}
.drop .dh{{font-size:11px;color:var(--m)}}
.drop input{{position:absolute;inset:0;opacity:0;cursor:pointer}}
#viewport{{position:fixed;inset:0;overflow:hidden;cursor:grab;display:none}}
#viewport.on{{display:block}}
#viewport:active{{cursor:grabbing}}
#mw{{position:absolute;top:0;left:0;transform-origin:0 0;will-change:transform}}
#mw img{{display:block;width:100%;height:100%;pointer-events:none;-webkit-user-drag:none}}
.st{{position:absolute;width:30px;height:30px;border-radius:50%;border:2.5px solid #fff;transform:translate(-50%,-50%);cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.4);z-index:2;transition:filter .15s}}
.st:hover{{filter:brightness(1.25);z-index:3}}
.st.sel{{z-index:4}}
.st.sel::after{{content:'';position:absolute;inset:-6px;border-radius:50%;border:2px solid;animation:pu 1.5s ease-in-out infinite}}
.st.ed{{cursor:move;filter:brightness(1.3) drop-shadow(0 0 6px rgba(255,255,255,.4))}}
@keyframes pu{{0%,100%{{opacity:.9;transform:scale(1)}}50%{{opacity:.35;transform:scale(1.4)}}}}
.sl{{position:absolute;left:50%;bottom:calc(100% + 3px);transform:translateX(-50%);white-space:nowrap;font-size:8.5px;font-weight:600;color:#fff;background:rgba(14,20,30,.88);padding:2px 6px;border-radius:3px;pointer-events:none;opacity:0;transition:opacity .12s}}
.st:hover .sl,.st.sel .sl{{opacity:1}}
#tb{{position:fixed;top:14px;right:14px;display:none;flex-direction:column;gap:4px;z-index:10}}
#tb.on{{display:flex}}
.tbtn{{width:36px;height:36px;border:none;border-radius:8px;background:var(--p);color:var(--t);font-size:13px;cursor:pointer;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(12px);border:1px solid var(--b);transition:background .15s,color .15s}}
.tbtn:hover{{background:rgba(211,47,47,.2);color:#fff}}
.tbtn.on{{background:rgba(211,47,47,.25);color:var(--a)}}
.dlbtn{{width:auto;padding:0 14px;font-size:11px;font-weight:700;gap:7px;background:rgba(67,160,71,.15)!important;border-color:rgba(67,160,71,.3)!important;color:#81C784!important}}
.dlbtn:hover{{background:rgba(67,160,71,.3)!important}}
.tsep{{height:1px;background:var(--b);margin:2px 4px}}
#ttl{{position:fixed;top:14px;left:14px;z-index:10;background:var(--p);border:1px solid var(--b);border-radius:10px;padding:10px 14px;backdrop-filter:blur(12px);max-width:300px;display:none}}
#ttl.on{{display:block}}
#ttl .ag{{font-size:8px;text-transform:uppercase;letter-spacing:2px;color:var(--a);font-weight:700}}
#ttl h1{{font-size:14px;font-weight:700;color:#fff;margin-top:2px;line-height:1.2}}
#ttl .mt{{font-size:9.5px;color:var(--m);margin-top:5px;line-height:1.4}}
#mm{{position:fixed;bottom:14px;left:14px;z-index:10;border-radius:8px;overflow:hidden;border:1px solid var(--b);background:var(--p);backdrop-filter:blur(12px);padding:6px;cursor:pointer;display:none}}
#mm.on{{display:block}}
#mmi{{display:block;border-radius:4px;background-size:cover;background-position:center;pointer-events:none}}
#mmv{{position:absolute;border:1.5px solid var(--a);background:rgba(211,47,47,.1);border-radius:2px;pointer-events:none}}
#sp{{position:fixed;bottom:14px;right:14px;width:265px;background:var(--p);border:1px solid var(--b);border-radius:10px;padding:14px;z-index:12;backdrop-filter:blur(12px);transform:translateY(8px);opacity:0;pointer-events:none;transition:all .2s}}
#sp.show{{transform:translateY(0);opacity:1;pointer-events:auto}}
.sph{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.spi{{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff;flex-shrink:0}}
.spn{{font-weight:700;font-size:12px;color:#fff}}.spt{{font-size:9px;color:var(--m)}}
.spc{{margin-left:auto;background:none;border:none;color:var(--m);cursor:pointer;font-size:13px;padding:4px}}.spc:hover{{color:#fff}}
.spr{{display:flex;justify-content:space-between;font-size:10.5px;padding:3px 0;border-top:1px solid rgba(255,255,255,.04)}}
.spr .l{{color:var(--m)}}.spr .v{{color:var(--t);font-weight:500;text-align:right;max-width:145px}}
#cb{{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);font-size:9px;color:rgba(255,255,255,.3);z-index:10;pointer-events:none;font-variant-numeric:tabular-nums;display:none}}
#cb.on{{display:block}}
#tt{{position:fixed;pointer-events:none;background:rgba(14,20,30,.92);border:1px solid var(--b);border-radius:6px;padding:6px 10px;font-size:10px;z-index:20;opacity:0;transition:opacity .1s;backdrop-filter:blur(8px);max-width:220px}}
#tt .tn{{font-weight:600;color:#fff;margin-bottom:1px}}
#tt .ts{{color:var(--m)}}
#hint{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--p);border:1px solid var(--b);border-radius:10px;padding:14px 22px;z-index:15;text-align:center;backdrop-filter:blur(12px);transition:opacity .8s;pointer-events:none}}
#hint.off{{opacity:0}}
#hint p{{font-size:11px;color:var(--t);line-height:1.6}}
#hint .hi{{color:var(--a);font-weight:600}}
#hint .hk{{display:inline-block;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:4px;padding:1px 5px;font-size:9px;font-weight:600;margin:0 1px}}
#eb{{position:fixed;top:56px;right:14px;background:rgba(255,152,0,.15);border:1px solid rgba(255,152,0,.3);border-radius:8px;padding:8px 14px;z-index:10;font-size:10px;color:#FFB300;opacity:0;pointer-events:none;transition:opacity .2s;backdrop-filter:blur(8px)}}
#eb.show{{opacity:1;pointer-events:auto}}
#toast{{position:fixed;top:60px;left:50%;transform:translateX(-50%) translateY(-20px);background:var(--p);border:1px solid var(--b);border-radius:8px;padding:10px 20px;font-size:11px;z-index:50;opacity:0;transition:all .3s;pointer-events:none;backdrop-filter:blur(12px);white-space:nowrap;max-width:92vw;text-align:center}}
#toast.show{{opacity:1;transform:translateX(-50%) translateY(0)}}
#toast.ok{{border-color:rgba(67,160,71,.3);color:#81C784}}
#toast.no{{border-color:rgba(255,152,0,.3);color:#FFB74D}}
@media(max-width:768px){{
  #ttl{{max-width:200px;padding:8px 10px}}#ttl h1{{font-size:12px}}#ttl .mt{{display:none}}
  #sp{{width:220px}}#mm{{display:none!important}}#hint{{max-width:90vw;padding:10px 14px}}
  .dlbtn{{font-size:10px;padding:0 8px}}
}}
</style>
</head>
<body>

<div id="upload-screen">
  <div class="ubox">
    <div class="sub">DOST-PAGASA</div>
    <h1>Tagoloan River Basin<br>Interactive Map Maker</h1>
    <p>Upload your hazard map image below. It will be turned into a fully interactive map with pan, zoom, and clickable monitoring stations — then downloadable as a single file.</p>
    <label class="drop" id="drop-zone">
      <i class="fas fa-cloud-upload-alt"></i>
      <div class="dt">Drop your map image here</div>
      <div class="dh">or click to browse — JPG, PNG, WebP</div>
      <input type="file" id="fi" accept="image/*">
    </label>
  </div>
</div>

<div id="viewport">
  <div id="mw"><img id="mi" alt="Map" draggable="false"></div>
</div>

<div id="ttl">
  <div class="ag">DOST-PAGASA</div>
  <h1>Tagoloan River Basin Hazard Map</h1>
  <div class="mt">Basin Area: 1,704 km² — Misamis Oriental, Bukidnon, Agusan del Sur<br>May 2024</div>
</div>

<div id="tb">
  <button class="tbtn dlbtn" id="b-dl" title="Download"><i class="fas fa-download"></i> Download</button>
  <div class="tsep"></div>
  <button class="tbtn" id="b-zi" title="Zoom In"><i class="fas fa-plus"></i></button>
  <button class="tbtn" id="b-zo" title="Zoom Out"><i class="fas fa-minus"></i></button>
  <button class="tbtn" id="b-fit" title="Fit View"><i class="fas fa-expand"></i></button>
  <div class="tsep"></div>
  <button class="tbtn" id="b-edit" title="Move Stations (E)"><i class="fas fa-arrows-alt"></i></button>
  <div class="tsep"></div>
  <button class="tbtn" id="b-sta" title="Toggle Stations"><i class="fas fa-map-marker-alt" style="font-size:11px"></i></button>
</div>

<div id="mm"><div id="mmi"></div><div id="mmv"></div></div>
<div id="sp">
  <div class="sph"><div class="spi" id="s-ic"><i class="fas fa-tint"></i></div><div><div class="spn" id="s-nm">--</div><div class="spt" id="s-tp">--</div></div><button class="spc" id="s-cl"><i class="fas fa-times"></i></button></div>
  <div class="spr"><span class="l">Elevation</span><span class="v" id="s-el">--</span></div>
  <div class="spr"><span class="l">Details</span><span class="v" id="s-in">--</span></div>
  <div class="spr"><span class="l">Status</span><span class="v" id="s-st" style="color:var(--g)">Active</span></div>
  <div class="spr"><span class="l">Coordinates</span><span class="v" id="s-co">--</span></div>
  <div class="spr"><span class="l">Position</span><span class="v" id="s-mp">--</span></div>
</div>
<div id="cb"></div>
<div id="tt"><div class="tn" id="t-tn">--</div><div class="ts" id="t-ts">--</div></div>
<div id="hint"><p><span class="hi">Interactive Map</span><br>Drag to pan · Scroll to zoom · Click stations<br><span class="hk">+</span><span class="hk">-</span> Zoom · <span class="hk">R</span> Reset · <span class="hk">E</span> Move stations · <span class="hk">D</span> Download</p></div>
<div id="eb"><i class="fas fa-info-circle"></i> Drag station markers to reposition</div>
<div id="toast"></div>

<script>
(function(){{
  var img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = '{image_src}';
  img.onload = function(){{
    document.getElementById('mi').src = img.src;
  }};
  img.onerror = function(){{
    var fallback = new Image();
    fallback.src = 'https://pubfiles.pagasa.dost.gov.ph/pagasaweb/images/basins/tagoloan-river-basin.jpg';
    fallback.onload = function(){{ document.getElementById('mi').src = fallback.src; }};
  }};
}})();

var WW = 2200;
var ST = [
  {{id:0,nm:'Malaybalay Rain Gauge',tp:'rainfall',x:49,y:16,el:'1,280 m',inf:'Annual: 2,650mm | Installed: 2018',co:'8.15\u00B0N, 125.08\u00B0E'}},
  {{id:1,nm:'Impasug-ong Rain Gauge',tp:'rainfall',x:34,y:21,el:'1,400 m',inf:'Annual: 2,900mm | Installed: 2017',co:'8.42\u00B0N, 124.78\u00B0E'}},
  {{id:2,nm:'Sumilao Rain Gauge',tp:'rainfall',x:29,y:34,el:'1,150 m',inf:'Annual: 2,380mm | Installed: 2020',co:'8.38\u00B0N, 124.95\u00B0E'}},
  {{id:3,nm:'Lantapan Rain Gauge',tp:'rainfall',x:63,y:19,el:'1,350 m',inf:'Annual: 2,800mm | Installed: 2019',co:'8.10\u00B0N, 125.28\u00B0E'}},
  {{id:4,nm:'Manolo Fortich WL Station',tp:'waterlevel',x:47,y:39,el:'890 m',inf:'Alert: 3.2m | Warning: 4.1m',co:'8.35\u00B0N, 125.07\u00B0E'}},
  {{id:5,nm:'Libona Weather Station',tp:'weather',x:55,y:51,el:'760 m',inf:'Temp: 24.5\u00B0C | Humidity: 82%',co:'8.33\u00B0N, 125.15\u00B0E'}},
  {{id:6,nm:'Baungon WL Station',tp:'waterlevel',x:44,y:59,el:'620 m',inf:'Alert: 2.8m | Warning: 3.5m',co:'8.32\u00B0N, 125.05\u00B0E'}},
  {{id:7,nm:'Tagoloan WL Station',tp:'waterlevel',x:50,y:73,el:'180 m',inf:'Alert: 2.5m | Warning: 3.0m',co:'8.54\u00B0N, 125.08\u00B0E'}},
  {{id:8,nm:'CDO Weather Station',tp:'weather',x:53,y:86,el:'45 m',inf:'Temp: 28.2\u00B0C | Humidity: 78%',co:'8.48\u00B0N, 124.64\u00B0E'}},
  {{id:9,nm:'Villanueva WL Station',tp:'waterlevel',x:48,y:94,el:'20 m',inf:'Alert: 2.0m | Warning: 2.8m',co:'8.56\u00B0N, 125.01\u00B0E'}}
];
var SC={{rainfall:'#43A047',waterlevel:'#1E88E5',weather:'#FFB300'}};
var SI={{rainfall:'fa-cloud-rain',waterlevel:'fa-water',weather:'fa-cloud-sun'}};
var SN={{rainfall:'Rainfall Gauge',waterlevel:'Water Level Station',weather:'Weather Station'}};

var zm=1,ox=0,oy=0,bs=1,ia=1,drag=false,ds=[0,0],dox=0,doy=0;
var edit=false,staVis=true,ready=false,sel=-1,hov=-1;
var dStn=null,dOff=[0,0],pD=0,pM=[0,0],sEls=[];
var imgDU=null;

var vp=document.getElementById('viewport'),mw=document.getElementById('mw'),mi=document.getElementById('mi');
var mm=document.getElementById('mm'),mmi=document.getElementById('mmi'),mmv=document.getElementById('mmv');
var sp=document.getElementById('sp'),tt=document.getElementById('tt'),cb=document.getElementById('cb');
var hint=document.getElementById('hint'),eb=document.getElementById('eb'),toast=document.getElementById('toast');
var dropZone=document.getElementById('drop-zone'),fi=document.getElementById('fi');

function toast(m,t){{toast.textContent=m;toast.className='show '+(t||'');clearTimeout(toast._t);toast._t=setTimeout(function(){{toast.className=''}},4000)}}

dropZone.addEventListener('dragover',function(e){{e.preventDefault();dropZone.classList.add('over')}});
dropZone.addEventListener('dragleave',function(){{dropZone.classList.remove('over')}});
dropZone.addEventListener('drop',function(e){{e.preventDefault();dropZone.classList.remove('over');if(e.dataTransfer.files[0])loadFile(e.dataTransfer.files[0])}});
fi.addEventListener('change',function(e){{if(e.target.files[0])loadFile(e.target.files[0])}});

function loadFile(file){{
  if(!file||!file.type.startsWith('image/')){{toast('Please select an image file.','no');return}}
  var r=new FileReader();
  r.onload=function(e){{imgDU=e.target.result;mi.src=imgDU;}};
  r.readAsDataURL(file);
}}

mi.onload=function(){{
  var w=mi.naturalWidth,h=mi.naturalHeight;
  if(w<10||h<10)return;
  ia=w/h;
  mw.style.width=WW+'px';mw.style.height=Math.round(WW/ia)+'px';
  cbs();fit();setupMM();makeStations();
  document.getElementById('upload-screen').classList.add('gone');
  vp.classList.add('on');document.getElementById('tb').classList.add('on');
  document.getElementById('ttl').classList.add('on');mm.classList.add('on');
  cb.classList.add('on');
  ready=true;updMM();
  setTimeout(function(){{hint.classList.add('off')}},4000);
}};

window.addEventListener('resize',function(){{if(ready){{cbs();updTx();updMM()}}}});
function cbs(){{var vw=innerWidth,vh=innerHeight;bs=Math.min(vw/WW,vh/(WW/ia))*.94}}
function updTx(){{mw.style.transform='translate('+ox+'px,'+oy+'px) scale('+(bs*zm)+')'}}
function fit(){{var vw=innerWidth,vh=innerHeight,mh=WW/ia;ox=(vw-WW*bs)/2;oy=(vh-mh*bs)/2;zm=1;updTx();updMM()}}
function zAt(cx,cy,f){{var os=bs*zm,nz=Math.max(.3,Math.min(25,zm*f)),ns=bs*nz,mx=(cx-ox)/os,my=(cy-oy)/os;zm=nz;ox=cx-mx*ns;oy=cy-my*ns;updTx();updMM()}}

function setupMM(){{var w=150,h=Math.round(w/ia);mm.style.width=(w+12)+'px';mm.style.height=(h+12)+'px';mmi.style.width=w+'px';mmi.style.height=h+'px';mmi.style.backgroundImage='url('+mi.src+')';mmi.style.backgroundSize='cover'}}
function updMM(){{if(!ready)return;var s=bs*zm,mw2=WW,mh=WW/ia,x1=-ox/s,y1=-oy/s,x2=x1+innerWidth/s,y2=y1+innerHeight/s;var l=Math.max(0,x1/mw2*100),t=Math.max(0,y1/mh*100),w=Math.min(100,(x2-x1)/mw2*100),h=Math.min(100,(y2-y1)/mh*100);mmv.style.left=l+'%';mmv.style.top=t+'%';mmv.style.width=w+'%';mmv.style.height=h+'%';mmv.style.display=(w>=98&&h>=98)?'none':'block'}}
mm.addEventListener('click',function(e){{if(!ready)return;var r=mmi.getBoundingClientRect(),mx=(e.clientX-r.left)/r.width,my=(e.clientY-r.top)/r.height,s=bs*zm;ox=innerWidth/2-mx*WW*s;oy=innerHeight/2-my*(WW/ia)*s;updTx();updMM()}});

function ldPos(){{try{{var s=localStorage.getItem('tstp');if(s)JSON.parse(s).forEach(function(p){{if(ST[p.id]){{ST[p.id].x=p.x;ST[p.id].y=p.y}}}})}}catch(e){{}}}}
function svPos(){{try{{localStorage.setItem('tstp',JSON.stringify(ST.map(function(s){{return{{id:s.id,x:+s.x.toFixed(2),y:+s.y.toFixed(2)}}}})))}}catch(e){{}}}}

function makeStations(){{
  ldPos();
  ST.forEach(function(s){{
    var el=document.createElement('div');el.className='st';
    el.style.left=s.x+'%';el.style.top=s.y+'%';
    el.style.background=SC[s.tp];el.setAttribute('data-id',s.id);
    var st=document.createElement('style');
    st.textContent='.st[data-id="'+s.id+'"].sel::after{{border-color:'+SC[s.tp]+'}}';
    document.head.appendChild(st);
    var lb=document.createElement('div');lb.className='sl';lb.textContent=s.nm;
    el.appendChild(lb);
    el.addEventListener('mousedown',function(e){{stnDown(e,s.id)}});
    el.addEventListener('touchstart',function(e){{stnTDown(e,s.id)}},{{passive:false}});
    mw.appendChild(el);sEls[s.id]=el;
  }});
}}

function selSt(id){{
  if(sel>=0&&sEls[sel])sEls[sel].classList.remove('sel');
  sel=id;
  if(id>=0){{var s=ST[id];sEls[id].classList.add('sel');
    document.getElementById('s-ic').style.background=SC[s.tp];
    document.getElementById('s-ic').innerHTML='<i class="fas '+SI[s.tp]+'"></i>';
    document.getElementById('s-nm').textContent=s.nm;
    document.getElementById('s-tp').textContent=SN[s.tp];
    document.getElementById('s-el').textContent=s.el;
    document.getElementById('s-in').textContent=s.inf;
    document.getElementById('s-co').textContent=s.co;
    document.getElementById('s-mp').textContent=s.x.toFixed(1)+'%, '+s.y.toFixed(1)+'%';
    sp.classList.add('show');
  }}else sp.classList.remove('show');
}}

function stnDown(e,id){{
  e.stopPropagation();
  if(edit){{e.preventDefault();dStn=id;var s=ST[id],s0=bs*zm;
    dOff=[e.clientX-(ox+s.x/100*WW*s0),e.clientY-(oy+s.y/100*(WW/ia)*s0)];
    sEls[id].classList.add('ed');
  }}else{{
    selSt(id);var s=ST[id],ts=bs*Math.max(zm,2);
    ox=innerWidth/2-s.x/100*WW*ts;oy=innerHeight/2-s.y/100*(WW/ia)*ts;
    zm=ts/bs;updTx();updMM();
  }}
}}
function stnTDown(e,id){{
  e.stopPropagation();
  if(edit&&e.touches.length===1){{e.preventDefault();dStn=id;var s=ST[id],s0=bs*zm;
    dOff=[e.touches[0].clientX-(ox+s.x/100*WW*s0),e.touches[0].clientY-(oy+s.y/100*(WW/ia)*s0)];
    sEls[id].classList.add('ed');
  }}else if(!edit)selSt(id);
}}

vp.addEventListener('mousedown',function(e){{if(dStn!==null)return;drag=true;ds=[e.clientX,e.clientY];dox=ox;doy=oy}});
window.addEventListener('mousemove',function(e){{
  if(dStn!==null){{var s0=bs*zm,mw2=WW,mh=WW/ia;
    ST[dStn].x=Math.max(0,Math.min(100,((e.clientX-dOff[0]-ox)/s0/mw2)*100));
    ST[dStn].y=Math.max(0,Math.min(100,((e.clientY-dOff[1]-oy)/s0/mh)*100));
    sEls[dStn].style.left=ST[dStn].x+'%';sEls[dStn].style.top=ST[dStn].y+'%';
    if(sel===dStn)document.getElementById('s-mp').textContent=ST[dStn].x.toFixed(1)+'%, '+ST[dStn].y.toFixed(1)+'%';
    return}}
  if(drag){{ox=dox+(e.clientX-ds[0]);oy=doy+(e.clientY-ds[1]);updTx();updMM()}}
  if(!ready)return;hov=-1;
  if(staVis){{var s0=bs*zm,mw2=WW,mh=WW/ia;
    for(var i=0;i<ST.length;i++){{var s=ST[i],px=ox+s.x/100*mw2*s0,py=oy+s.y/100*mh*s0;
      if(Math.hypot(e.clientX-px,e.clientY-py)<18){{hov=s.id;break}}}}}}
  if(hov>=0){{var s=ST[hov];document.getElementById('t-tn').textContent=s.nm;
    document.getElementById('t-ts').textContent=SN[s.tp]+(edit?' \u2014 drag to move':' \u2014 click for details');
    tt.style.opacity='1';tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY-10)+'px';
    vp.style.cursor=edit?'move':'pointer';
  }}else{{tt.style.opacity='0';vp.style.cursor=drag?'grabbing':'grab'}}
  var s0=bs*zm;
  cb.textContent='X: '+((e.clientX-ox)/s0/WW*100).toFixed(1)+'%  Y: '+((e.clientY-oy)/s0/(WW/ia)*100).toFixed(1)+'%  |  Zoom: '+zm.toFixed(2)+'x';
}});
window.addEventListener('mouseup',function(e){{
  if(dStn!==null){{sEls[dStn].classList.remove('ed');svPos();dStn=null;return}}
  if(drag&&Math.hypot(e.clientX-ds[0],e.clientY-ds[1])<4&&!e.target.closest('.st'))selSt(-1);
  drag=false}});
vp.addEventListener('wheel',function(e){{e.preventDefault();zAt(e.clientX,e.clientY,e.deltaY<0?1.15:.87)}},{{passive:false}});
vp.addEventListener('dblclick',function(e){{zAt(e.clientX,e.clientY,2)}});
vp.addEventListener('touchstart',function(e){{
  if(dStn!==null)return;
  if(e.touches.length===1){{drag=true;ds=[e.touches[0].clientX,e.touches[0].clientY];dox=ox;doy=oy}}
  else if(e.touches.length===2){{drag=false;pD=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);pM=[(e.touches[0].clientX+e.touches[1].clientX)/2,(e.touches[0].clientY+e.touches[1].clientY)/2]}}
}},{{passive:false}});
vp.addEventListener('touchmove',function(e){{e.preventDefault();
  if(dStn!==null&&e.touches.length===1){{var s0=bs*zm,mw2=WW,mh=WW/ia;
    ST[dStn].x=Math.max(0,Math.min(100,((e.touches[0].clientX-dOff[0]-ox)/s0/mw2)*100));
    ST[dStn].y=Math.max(0,Math.min(100,((e.touches[0].clientY-dOff[1]-oy)/s0/mh)*100));
    sEls[dStn].style.left=ST[dStn].x+'%';sEls[dStn].style.top=ST[dStn].y+'%';return}}
  if(e.touches.length===1&&drag){{ox=dox+(e.touches[0].clientX-ds[0]);oy=doy+(e.touches[0].clientY-ds[1]);updTx();updMM()}}
  else if(e.touches.length===2){{var nd=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);zAt(pM[0],pM[1],nd/pD);pD=nd}}
}},{{passive:false}});
vp.addEventListener('touchend',function(e){{
  if(dStn!==null&&e.touches.length===0){{sEls[dStn].classList.remove('ed');svPos();dStn=null;return}}
  if(e.touches.length===0)drag=false}});
window.addEventListener('keydown',function(e){{
  switch(e.key){{
    case'+':case'=':zAt(innerWidth/2,innerHeight/2,1.25);break;
    case'-':case'_':zAt(innerWidth/2,innerHeight/2,.8);break;
    case'ArrowLeft':ox+=40;updTx();updMM();break;
    case'ArrowRight':ox-=40;updTx();updMM();break;
    case'ArrowUp':oy+=40;updTx();updMM();break;
    case'ArrowDown':oy-=40;updTx();updMM();break;
    case'r':case'R':fit();break;
    case'e':case'E':togEdit();break;
    case'd':case'D':dlMap();break;
    case'Escape':selSt(-1);if(edit)togEdit();break;
  }}
}});
document.getElementById('b-zi').addEventListener('click',function(){{zAt(innerWidth/2,innerHeight/2,1.4)}});
document.getElementById('b-zo').addEventListener('click',function(){{zAt(innerWidth/2,innerHeight/2,.71)}});
document.getElementById('b-fit').addEventListener('click',fit);
document.getElementById('b-edit').addEventListener('click',togEdit);
document.getElementById('b-sta').addEventListener('click',function(){{
  staVis=!staVis;document.getElementById('b-sta').classList.toggle('on',!staVis);
  sEls.forEach(function(el){{el.style.display=staVis?'':'none'}});if(!staVis)selSt(-1)}});
document.getElementById('s-cl').addEventListener('click',function(){{selSt(-1)}});
vp.addEventListener('contextmenu',function(e){{e.preventDefault()}});

function togEdit(){{edit=!edit;document.getElementById('b-edit').classList.toggle('on',edit);eb.classList.toggle('show',edit);sEls.forEach(function(el){{el.style.cursor=edit?'move':'pointer'}})}}

document.getElementById('b-dl').addEventListener('click',dlMap);

function dlMap(){{
  if(!imgDU){{toast('No image to embed. This should not happen.','no');return}}
  toast('Building download...','ok');
  setTimeout(function(){{
    try{{
      var html='<!DOCTYPE html>\n'+document.documentElement.outerHTML;
      var pos=JSON.stringify(ST.map(function(s){{return{{id:s.id,x:+s.x.toFixed(2),y:+s.y.toFixed(2)}}}})).replace(/'/g,"\\'");
      var preload="<script>try{{localStorage.setItem('tstp','"+pos+"')}}catch(e){{}}<\/script>";
      var upScreen=document.getElementById('upload-screen').outerHTML;
      var newUpScreen=upScreen.replace('id="upload-screen"','id="upload-screen" class="gone"');
      html=html.replace(upScreen,newUpScreen);
      var inject="<script>document.addEventListener('DOMContentLoaded',function(){{document.getElementById('mi').src='"+imgDU+"'}});<\/script>";
      html=html.replace('</head>',preload+inject+'</head>');
      html=html.replace('vp.classList.add(\'on\')','vp.classList.add(\'on\')');
      html=html.replace('document.getElementById(\'tb\').classList.add(\'on\')','document.getElementById(\'tb\').classList.add(\'on\')');
      html=html.replace('document.getElementById(\'ttl\').classList.add(\'on\')','document.getElementById(\'ttl\').classList.add(\'on\')');
      html=html.replace('mm.classList.add(\'on\')','mm.classList.add(\'on\')');
      html=html.replace('cb.classList.add(\'on\')','cb.classList.add(\'on\')');
      var blob=new Blob([html],{{type:'text/html;charset=utf-8'}});
      var url=URL.createObjectURL(blob);
      var a=document.createElement('a');a.href=url;
      a.download='Tagoloan_River_Basin_Interactive_Map.html';
      a.style.display='none';document.body.appendChild(a);a.click();
      setTimeout(function(){{document.body.removeChild(a);URL.revokeObjectURL(url)}},200);
      toast('Downloaded — works offline, no internet needed','ok');
    }}catch(err){{toast('Download failed: '+err.message,'no')}}
  }},150);
}}
</script>
</body>
</html>"""
