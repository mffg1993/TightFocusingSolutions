/* Browser-only reconstruction from the sampled complex electric field. */
(function(){
"use strict";
const P=window.POLARIZATION_DATA, gd=document.getElementById("polarization-scenes");
const $=id=>document.getElementById(id);
const keys=["linear","positive","negative","alpha","beta","gamma"];
const add=(a,b)=>a.map((v,i)=>v+b[i]), sub=(a,b)=>a.map((v,i)=>v-b[i]), mul=(a,s)=>a.map(v=>v*s);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0), norm=a=>Math.hypot(...a);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>mul(a,1/Math.max(norm(a),1e-30));
const basis=a=>{let k=0;for(let i=1;i<3;i++)if(Math.abs(a[i])<Math.abs(a[k]))k=i;return [0,1,2].map(i=>i===k?1:0)};
const colors={linear:"#bfc4cb",positive:"#e64760",negative:"#24b4ef",alpha:"#20ae57",beta:"#ff941f",gamma:"#d928cf"};
const group=()=>({x:[],y:[],z:[],i:[],j:[],k:[]});
function append(g,verts,faces){const off=g.x.length;for(const v of verts){g.x.push(v[0]);g.y.push(v[1]);g.z.push(v[2])}for(const f of faces){g.i.push(f[0]+off);g.j.push(f[1]+off);g.k.push(f[2]+off)}}
function tube(g,p,r,closed=false){
 const N=p.length,S=10,V=[],F=[];let ref=null;
 for(let i=0;i<N;i++){
  const t=unit(sub(p[closed?(i+1)%N:Math.min(i+1,N-1)],p[closed?(i+N-1)%N:Math.max(i-1,0)]));
  if(!ref)ref=basis(t);let n=sub(ref,mul(t,dot(ref,t)));
  if(norm(n)<1e-8){ref=basis(t);n=sub(ref,mul(t,dot(ref,t)))}
  n=unit(n);ref=n;const b=cross(t,n);
  for(let k=0;k<S;k++){const a=2*Math.PI*k/S;V.push(add(p[i],mul(add(mul(n,Math.cos(a)),mul(b,Math.sin(a))),r)))}
 }
 for(let i=0;i<(closed?N:N-1);i++){const j=(i+1)%N;for(let k=0;k<S;k++){const l=(k+1)%S;F.push([i*S+k,j*S+k,j*S+l],[i*S+k,j*S+l,i*S+l])}}
 if(!closed)for(let k=1;k<S-1;k++)F.push([0,k+1,k],[(N-1)*S,(N-1)*S+k,(N-1)*S+k+1]);
 append(g,V,F);
}
function arrow(g,c,d,r){
 const l=norm(d);if(l<1e-12)return;
 const neck=add(c,mul(d,.72)),a=unit(d),n=unit(cross(a,basis(a))),b=cross(a,n);
 tube(g,[c,neck],Math.min(r,.10*l));
 const V=[],F=[],S=10,R=Math.min(2.1*r,.22*l);
 for(let k=0;k<S;k++){const t=2*Math.PI*k/S;V.push(add(neck,mul(add(mul(n,Math.cos(t)),mul(b,Math.sin(t))),R)))}
 V.push(add(c,d),neck);for(let k=0;k<S;k++)F.push([k,(k+1)%S,S],[k,S+1,(k+1)%S]);append(g,V,F);
}
function decompose(re,im){
 const amp=Math.hypot(...re,...im);if(!amp)return null;
 const a=mul(re,1/amp),b=mul(im,1/amp);
 const qr=dot(a,a)-dot(b,b),qi=2*dot(a,b),q=Math.min(1,Math.hypot(qr,qi));
 const chi=.5*Math.atan2(qi,qr),c=Math.cos(chi),s=Math.sin(chi);
 return {amp,alpha:add(mul(a,c),mul(b,s)),beta:add(mul(b,c),mul(a,-s)),
   gamma:mul(cross(a,b),2),ell:Math.sqrt(Math.max(0,1-q)/(1+q)),circular:q<=1e-8,a,b};
}
function range(axis,lo,hi){return axis.map((v,i)=>v>=lo-1e-10&&v<=hi+1e-10?i:-1).filter(i=>i>=0)}
function sites(indices,n){const pad=indices.length>=n+4?2:0;return Array.from({length:n},(_,i)=>Math.round(indices[pad]+i*(indices[indices.length-1-pad]-indices[pad])/(n-1)))}
const fields={xmin:"x minimum",xmax:"x maximum",ymin:"y minimum",ymax:"y maximum",density:"Ellipses per side",thickness:"Tube radius / spacing",scale:"Glyph size / spacing",opacity:"Intensity opacity",cutoff:"Minimum field amplitude",linear:"Linear ellipticity threshold"};
const state=()=>Object.fromEntries(Object.keys(fields).map(k=>[k,Number($(k).value)]));
function visibility(k){const e=document.querySelector('input[data-key="'+k+'"]');return !e||e.checked}
let busy=false,pending=false,initial=true,syncing=false;
async function rebuild(){
 if(busy){pending=true;return}busy=true;
 try{
 const s=state(),xs=range(P.x,s.xmin,s.xmax),ys=range(P.y,s.ymin,s.ymax);
 const n=Math.min(s.density,xs.length,ys.length);
 if(s.xmin>=s.xmax||s.ymin>=s.ymax||n<2){$("slider-status").textContent="Select increasing limits containing at least two grid samples.";return}
 const ix=sites(xs,n),iy=sites(ys,n),pitch=Math.min(P.x[ix[n-1]]-P.x[ix[0]],P.y[iy[n-1]]-P.y[iy[0]])/(n-1);
 const radius=s.thickness*pitch,groups=Object.fromEntries(keys.map(k=>[k,group()]));
 const zcenter=(s.scale+ s.thickness+.02)*pitch;
 let count=0;
 for(const j of iy)for(const i of ix){
  const k=j*P.x.length+i,re=P.re.map(v=>v[k]),im=P.im.map(v=>v[k]),d=decompose(re,im);
  if(!d||d.amp<s.cutoff)continue;count++;
  const center=[P.x[i],P.y[j],zcenter],ell=d.ell,g=d.gamma;
  let sign=g[2];if(Math.abs(sign)<.05*norm(g))sign=Math.abs(g[0])>=Math.abs(g[1])?g[0]:g[1];
  const key=ell<s.linear?"linear":sign>0?"positive":"negative";
  if(ell<1e-5){const v=mul(d.alpha,s.scale*pitch);tube(groups[key],[sub(center,v),add(center,v)],radius)}
  else{
   const M=32,points=Array.from({length:M},(_,k)=>{const t=2*Math.PI*k/M;return add(center,mul(add(mul(d.a,Math.cos(t)),mul(d.b,Math.sin(t))),s.scale*pitch))});
   tube(groups[key],points,radius,true);
  }
  if(!d.circular)for(const name of ["alpha","beta"]){
   const v=mul(d[name],s.scale*pitch);if(norm(v)>1e-9)tube(groups[name],[sub(center,v),add(center,v)],.45*radius);
  }
  const v=mul(g,s.scale*pitch);if(norm(v)>1e-9)arrow(groups.gamma,center,v,.45*radius);
 }
 const xx=xs.filter((_,k)=>k%Math.max(1,Math.ceil(xs.length/120))===0||k===xs.length-1),yy=ys.filter((_,k)=>k%Math.max(1,Math.ceil(ys.length/120))===0||k===ys.length-1);
 const I=yy.map(j=>xx.map(i=>P.I[j*P.x.length+i])),zero=yy.map(()=>xx.map(()=>0));
 const data=[],indices={};
 for(let k=1;k<=2;k++){const key="intensity"+k;indices[key]=data.length;data.push({type:"surface",scene:k===1?"scene":"scene2",x:xx.map(i=>P.x[i]),y:yy.map(i=>P.y[i]),z:zero,surfacecolor:I,colorscale:"Viridis",cmin:0,cmax:1,opacity:s.opacity,visible:visibility(key),showscale:true,showlegend:false,colorbar:{x:1.01,y:k===1?.77:.23,len:.34,thickness:14,title:{text:"I/Imax"}},lighting:{ambient:1,diffuse:0,specular:0},hoverinfo:"skip"})}
 for(const key of keys){indices[key]=data.length;data.push({...groups[key],type:"mesh3d",scene:["alpha","beta","gamma"].includes(key)?"scene2":"scene",color:colors[key],visible:visibility(key),showlegend:false,hoverinfo:"skip",lighting:{ambient:.65,diffuse:.35,specular:0,roughness:1},flatshading:false})}
 const layout=JSON.parse(JSON.stringify(P.layout)),zmax=Math.max((2*s.scale+2*s.thickness+.1)*pitch,1.1*pitch);
 for(const key of ["scene","scene2"]){const scene=layout[key];scene.xaxis.range=[s.xmin,s.xmax];scene.yaxis.range=[s.ymin,s.ymax];scene.zaxis.range=[-.1*pitch,zmax];scene.aspectratio={x:s.xmax-s.xmin,y:s.ymax-s.ymin,z:zmax+.1*pitch};
  if(!initial&&gd.layout?.[key]?.camera)scene.camera=gd.layout[key].camera;
 }
 await Plotly.react(gd,data,layout,{responsive:true,displaylogo:false});
 window.polarizationTraceIndices=indices;initial=false;
 $("slider-status").textContent=count+" / "+(n*n)+" glyph locations shown"+(n<s.density?" (density limited by sampled grid)":"");
 }catch(err){$("slider-status").textContent="Update failed: "+err.message;console.error(err)}finally{busy=false;if(pending){pending=false;rebuild()}}
}
let timer;function schedule(){clearTimeout(timer);timer=setTimeout(rebuild,180)}
for(const k of Object.keys(fields)){$(k).addEventListener("input",()=>{$(k+"-value").textContent=$(k).value;schedule()})}
document.querySelectorAll("input[data-key]").forEach(box=>box.addEventListener("change",()=>{const i=window.polarizationTraceIndices?.[box.dataset.key];if(i!==undefined)Plotly.restyle(gd,{visible:box.checked},[i]);else schedule()}));
$("reset-controls").onclick=()=>{for(const k of Object.keys(fields)){$(k).value=P.defaults[k];$(k+"-value").textContent=$(k).value}document.querySelectorAll("input[data-key]").forEach(e=>e.checked=true);$("sync-cameras").checked=true;initial=true;schedule()};
function syncCamera(evt){if(syncing||!$("sync-cameras").checked)return;for(const key of ["scene","scene2"]){if(Object.keys(evt).some(k=>k===key+".camera"||k.startsWith(key+".camera."))){const camera=evt[key+".camera"]||gd.layout[key].camera;syncing=true;Plotly.relayout(gd,{[(key==="scene"?"scene2":"scene")+".camera"]:camera}).finally(()=>{syncing=false});break}}}
function start(){if(typeof Plotly==="undefined"||typeof gd.on!=="function"){setTimeout(start,100);return}gd.on("plotly_relayout",syncCamera);rebuild()}
start();
})();