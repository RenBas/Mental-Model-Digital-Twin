# ui/map_html.py

def get_map_html(image_src: str) -> str:
    """
    Returns interactive map HTML with only the Download button.
    No title box, no minimap, no extra toolbar buttons, no hint, no edit banner.
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
#viewport{{position:fixed;inset:0;overflow:hidden;cursor:grab;display:block}}
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
#dl-btn{{position:fixed;top:14px;right:14px;z-index:10;width:auto;padding:0 14px;height:36px;border:none;border-radius:8px;background:rgba(67,160,71,.15);border:1px solid rgba(67,160,71,.3);color:#81C784;font-size:11px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:7px;backdrop-filter:blur(12px);transition:background .15s}}
#dl-btn:hover{{background:rgba(67,160,71,.3)}}
#sp{{position:fixed;bottom:14px;right:14px;width:265px;background:var(--p);border:1px solid var(--b);border-radius:10px;padding:14px;z-index:12;backdrop-filter:blur(12px);transform:translateY(8px);opacity:0;pointer-events:none;transition:all .2s}}
#sp.show{{transform:translateY(0);opacity:1;pointer-events:auto}}
.sph{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.spi{{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff;flex-shrink:0}}
.spn{{font-weight:700;font-size:12px;color:#fff}}.spt{{font-size:9px;color:var(--m)}}
.spc{{margin-left:auto;background:none;border:none;color:var(--m);cursor:pointer;font-size:13px;padding:4px}}.spc:hover{{color:#fff}}
.spr{{display:flex;justify-content:space-between;font-size:10.5px;padding:3px 0;border-top:1px solid rgba(255,255,255,.04)}}
.spr .l{{color:var(--m)}}.spr .v{{color:var(--t);font-weight:500;text-align:right;max-width:145px}}
#cb{{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);font-size:9px;color:rgba(255,255,255,.3);z-index:10;pointer-events:none;font-variant-numeric:tabular-nums;display:block}}
#tt{{position:fixed;pointer-events:none;background:rgba(14,20,30,.92);border:1px solid var(--b);border-radius:6px;padding:6px 10px;font-size:10px;z-index:20;opacity:0;transition:opacity .1s;backdrop-filter:blur(8px);max-width:220px}}
#tt .tn{{font-weight:600;color:#fff;margin-bottom:1px}}
#tt .ts{{color:var(--m)}}
@media(max-width:768px){{
  #sp{{width:220px}}
}}
</style>
</head>
<body>

<div id="viewport">
  <div id="mw"><img id="mi" alt="Map" draggable="false"></div>
</div>

<button id="dl-btn"><i class="fas fa-download"></i> Download</button>

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

<script>
var mi = document.getElementById('mi');
mi.src = '{image_src}';
mi.onload = function(){{
  var w = mi.naturalWidth, h = mi.naturalHeight;
  if (w < 10 || h < 10) return;
  var WW = 2200, ia = w / h;
  var mw = document.getElementById('mw');
  mw.style.width = WW + 'px';
  mw.style.height = Math.round(WW / ia) + 'px';
  cbs();
  fit();
  makeStations();
  ready = true;
  updMM();
}};

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

var vp=document.getElementById('viewport'),mw=document.getElementById('mw'),mm=document.getElementById('mm'),mmi=document.getElementById('mmi'),mmv=document.getElementById('mmv');
var sp=document.getElementById('sp'),tt=document.getElementById('tt'),cb=document.getElementById('cb');
var hint=document.getElementById('hint'),eb=document.getElementById('eb'),toast=document.getElementById('toast');

function toast(m,t){{toast.textContent=m;toast.className='show '+(t||'');clearTimeout(toast._t);toast._t=setTimeout(function(){{toast.className=''}},4000)}}

function cbs(){{var vw=innerWidth,vh=innerHeight;bs=Math.min(vw/WW,vh/(WW/ia))*.94}}
function updTx(){{mw.style.transform='translate('+ox+'px,'+oy+'px) scale('+(bs*zm)+')'}}
function fit(){{var vw=innerWidth,vh=innerHeight,mh=WW/ia;ox=(vw-WW*bs)/2;oy=(vh-mh*bs)/2;zm=1;updTx()}}
function zAt(cx,cy,f){{var os=bs*zm,nz=Math.max(.3,Math.min(25,zm*f)),ns=bs*nz,mx=(cx-ox)/os,my=(cy-oy)/os;zm=nz;ox=cx-mx*ns;oy=cy-my*ns;updTx()}}

function makeStations(){{
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
    zm=ts/bs;updTx();
  }}
}}

vp.addEventListener('mousedown',function(e){{if(dStn!==null)return;drag=true;ds=[e.clientX,e.clientY];dox=ox;doy=oy}});
window.addEventListener('mousemove',function(e){{
  if(dStn!==null){{var s0=bs*zm,mw2=WW,mh=WW/ia;
    ST[dStn].x=Math.max(0,Math.min(100,((e.clientX-dOff[0]-ox)/s0/mw2)*100));
    ST[dStn].y=Math.max(0,Math.min(100,((e.clientY-dOff[1]-oy)/s0/mh)*100));
    sEls[dStn].style.left=ST[dStn].x+'%';sEls[dStn].style.top=ST[dStn].y+'%';
    return}}
  if(drag){{ox=dox+(e.clientX-ds[0]);oy=doy+(e.clientY-ds[1]);updTx()}}
  if(!ready)return;hov=-1;
  if(staVis){{var s0=bs*zm,mw2=WW,mh=WW/ia;
    for(var i=0;i<ST.length;i++){{var s=ST[i],px=ox+s.x/100*mw2*s0,py=oy+s.y/100*mh*s0;
      if(Math.hypot(e.clientX-px,e.clientY-py)<18){{hov=s.id;break}}}}}}
  if(hov>=0){{var s=ST[hov];document.getElementById('t-tn').textContent=s.nm;
    document.getElementById('t-ts').textContent=SN[s.tp];
    tt.style.opacity='1';tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY-10)+'px';
    vp.style.cursor='pointer';
  }}else{{tt.style.opacity='0';vp.style.cursor=drag?'grabbing':'grab'}}
  var s0=bs*zm;
  cb.textContent='X: '+((e.clientX-ox)/s0/WW*100).toFixed(1)+'%  Y: '+((e.clientY-oy)/s0/(WW/ia)*100).toFixed(1)+'%  |  Zoom: '+zm.toFixed(2)+'x';
}});
window.addEventListener('mouseup',function(e){{
  if(dStn!==null){{sEls[dStn].classList.remove('ed');dStn=null;return}}
  if(drag&&Math.hypot(e.clientX-ds[0],e.clientY-ds[1])<4&&!e.target.closest('.st'))selSt(-1);
  drag=false}});
vp.addEventListener('wheel',function(e){{e.preventDefault();zAt(e.clientX,e.clientY,e.deltaY<0?1.15:.87)}},{{passive:false}});
vp.addEventListener('dblclick',function(e){{zAt(e.clientX,e.clientY,2)}});
window.addEventListener('keydown',function(e){{
  switch(e.key){{
    case'+':case'=':zAt(innerWidth/2,innerHeight/2,1.25);break;
    case'-':case'_':zAt(innerWidth/2,innerHeight/2,.8);break;
    case'ArrowLeft':ox+=40;updTx();break;
    case'ArrowRight':ox-=40;updTx();break;
    case'ArrowUp':oy+=40;updTx();break;
    case'ArrowDown':oy-=40;updTx();break;
    case'r':case'R':fit();break;
    case'Escape':selSt(-1);break;
  }}
}});
document.getElementById('s-cl').addEventListener('click',function(){{selSt(-1)}});
vp.addEventListener('contextmenu',function(e){{e.preventDefault()}});

document.getElementById('dl-btn').addEventListener('click',function(){{
  if(!mi.src){{toast('No map loaded.','no');return}}
  var canvas=document.createElement('canvas');
  var ctx=canvas.getContext('2d');
  var img=new Image();img.crossOrigin='anonymous';img.src=mi.src;
  img.onload=function(){{
    canvas.width=img.width;canvas.height=img.height;
    ctx.drawImage(img,0,0);
    var link=document.createElement('a');link.download='Tagoloan_River_Basin_Map.png';
    link.href=canvas.toDataURL('image/png');
    link.click();
    toast('Map downloaded as PNG','ok');
  }};
}});
</script>
</body>
</html>"""
