/* Circular-contour inspection of the exported complex vector field. */
(function(){
'use strict';
const P=window.CONTOUR_DATA, gd=document.getElementById('contour-scenes');
const $=id=>document.getElementById(id);
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
const labels={alpha:'Major axis α',beta:'Minor axis β',gamma:'Directed normal γ'};
function interval(axis,v){let lo=0,hi=axis.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(axis[mid]<=v)lo=mid;else hi=mid}return Math.min(lo,axis.length-2)}
function sample(x,y){
 const i=interval(P.x,x),j=interval(P.y,y),a=(x-P.x[i])/(P.x[i+1]-P.x[i]),b=(y-P.y[j])/(P.y[j+1]-P.y[j]);
 const ids=[j*P.x.length+i,j*P.x.length+i+1,(j+1)*P.x.length+i,(j+1)*P.x.length+i+1],weights=[(1-a)*(1-b),a*(1-b),(1-a)*b,a*b];
 return [P.re,P.im].map(parts=>parts.map(values=>ids.reduce((sum,k,n)=>sum+values[k]*weights[n],0)));
}
function ring(radius){
 const N=P.angular_samples,points=[],values=[],phase=[];let maxstep=0,minI=Infinity,minQ=Infinity;
 for(let j=0;j<=N;j++){
  const t=2*Math.PI*j/N,p=[P.center[0]+radius*Math.cos(t),P.center[1]+radius*Math.sin(t),0];
  const [re,im]=j===N?[values[0].re,values[0].im]:sample(p[0],p[1]);
  const I=dot(re,re)+dot(im,im),qr=dot(re,re)-dot(im,im),qi=2*dot(re,im),q=Math.hypot(qr,qi)/Math.max(I,1e-30);
  const raw=Math.atan2(qi,qr);let psi=raw;
  if(j){let delta=raw-Math.atan2(2*dot(values[j-1].re,values[j-1].im),dot(values[j-1].re,values[j-1].re)-dot(values[j-1].im,values[j-1].im));
   delta=Math.atan2(Math.sin(delta),Math.cos(delta));maxstep=Math.max(maxstep,Math.abs(delta));psi=phase[j-1]+delta;
  }
  const a=Math.cos(psi/2),b=Math.sin(psi/2),amp=Math.sqrt(Math.max(I,1e-30));
  const alpha=mul(add(mul(re,a),mul(im,b)),1/amp),beta=mul(sub(mul(im,a),mul(re,b)),1/amp),gamma=mul(cross(re,im),2/Math.max(I,1e-30));
  points.push(p);phase.push(psi);values.push({re,im,I,q,alpha,beta,gamma});minI=Math.min(minI,I);minQ=Math.min(minQ,q);
 }
 const winding=Math.round((phase[N]-phase[0])/(2*Math.PI));
 return {points,values,winding,minI,minQ,maxstep};
}
function direction(v,key){
 const d=v[key];if(v.I<1e-10||norm(d)<1e-7||(key!=='gamma'&&v.q<1e-6))return null;return unit(d);
}
function mesh(g,color,scene){return {...g,type:'mesh3d',scene,color,showlegend:false,hoverinfo:'skip',lighting:{ambient:.8,diffuse:.45,specular:.05}}}
function curve(points,scene,color='#64748b',width=4){return {type:'scatter3d',scene,mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),line:{color,width},showlegend:false,hoverinfo:'skip'}}
function state(){return {radius:Number($('radius').value),count:Number($('count').value),length:Number($('length').value),thickness:Number($('thickness').value),angle:Number($('angle').value),quantity:$('quantity').value}}
let busy=false,pending=false;
async function render(){
 if(busy){pending=true;return}busy=true;
 try{
 const s=state(),R=ring(s.radius),key=s.quantity,N=P.angular_samples,arrows=group(),selected=group(),ellipses={positive:group(),negative:group(),linear:group()};
 let shown=0;
 const axisValid=key==='gamma'||(R.minQ>=1e-6&&R.minI>=1e-10&&R.maxstep<Math.PI/2);
 for(let i=0;i<s.count;i++){
  const j=Math.round(i*N/s.count),d=direction(R.values[j],key);if(!d||!axisValid)continue;
  const p=R.points[j];if(key==='gamma')arrow(arrows,p,mul(d,s.length),s.thickness);
  else arrow(arrows,sub(p,mul(d,s.length)),mul(d,2*s.length),s.thickness);
  shown++;
 }
 const contextCount=Math.min(24,s.count),ellipseScale=Math.min(.075,.20*s.radius),ellipseRadius=Math.min(.0018,.2*ellipseScale);
 for(let i=0;i<contextCount;i++){
  const j=Math.round(i*N/contextCount),v=R.values[j],p=R.points[j];if(v.I<1e-10)continue;
  const d=decompose(v.re,v.im),name=d.ell<.03?'linear':(d.gamma[2]>=0?'positive':'negative');
  if(d.ell<1e-5){const dir=mul(unit(d.alpha),ellipseScale);tube(ellipses[name],[sub(p,dir),add(p,dir)],ellipseRadius);}
  else{const pts=Array.from({length:32},(_,k)=>{const t=2*Math.PI*k/32;return add(p,mul(add(mul(d.a,Math.cos(t)),mul(d.b,Math.sin(t))),ellipseScale))});tube(ellipses[name],pts,ellipseRadius,true)}
 }
 const index=Math.round(s.angle*N/360),point=R.points[index],d=direction(R.values[index],key);
 if(d&&axisValid){if(key==='gamma')arrow(selected,point,mul(d,s.length),1.8*s.thickness);else arrow(selected,sub(point,mul(d,s.length)),mul(d,2*s.length),1.8*s.thickness);}
 const data=[curve(R.points,'scene'),curve(R.points,'scene2')];
 for(const name of ['linear','positive','negative'])data.push(mesh(ellipses[name],colors[name],'scene'));
 data.push(mesh(arrows,colors[key],'scene2'));data.push(mesh(selected,'#111827','scene2'));
 for(const scene of ['scene','scene2'])data.push({type:'scatter3d',scene,mode:'markers',x:[point[0]],y:[point[1]],z:[point[2]],marker:{color:'#111827',size:5},showlegend:false,name:'Highlighted position'});
 if($('background').checked){
  const step=Math.max(1,Math.ceil(P.x.length/65)),xs=P.x.filter((_,i)=>i%step===0),ys=P.y.filter((_,i)=>i%step===0);
  const intens=ys.map(y=>xs.map(x=>{const e=sample(x,y);return dot(e[0],e[0])+dot(e[1],e[1])}));
  for(const scene of ['scene','scene2'])data.push({type:'surface',scene,x:xs,y:ys,z:ys.map(()=>xs.map(()=>0)),surfacecolor:intens,colorscale:'Viridis',cmin:0,cmax:1,opacity:.3,showscale:false,hoverinfo:'skip',showlegend:false});
 }
 const extent=s.radius+Math.max(s.length*1.6,.10),axis=(title,range)=>({title:{text:title},range,showbackground:false,showspikes:false,zeroline:false});
 const scene=()=>({xaxis:axis('x / λ₀',[P.center[0]-extent,P.center[0]+extent]),yaxis:axis('y / λ₀',[P.center[1]-extent,P.center[1]+extent]),zaxis:axis('z / λ₀',[-extent,extent]),aspectmode:'cube',camera:{eye:{x:1.25,y:-1.65,z:1.1}},domain:{x:[0,.46],y:[0,1]},uirevision:'ring-camera'});
 const layout={height:640,margin:{l:0,r:0,t:55,b:5},scene:scene(),scene2:scene(),showlegend:false,uirevision:'ring-camera',paper_bgcolor:'white',annotations:[{text:'Ellipse context on the circle',xref:'paper',yref:'paper',x:.23,y:1.06,showarrow:false},{text:labels[key]+' — '+shown+' arrows',xref:'paper',yref:'paper',x:.77,y:1.06,showarrow:false}]};
 layout.scene2.domain.x=[.54,1];
 for(const name of ['scene','scene2'])if(gd.layout?.[name]?.camera)layout[name].camera=gd.layout[name].camera;
 await Plotly.react(gd,data,layout,{responsive:true,displaylogo:false});
 const first=direction(R.values[0],key),last=direction(R.values[N],key),closure=first&&last?dot(first,last):null;
 $('quantity-title').textContent=labels[key]+' at radius '+s.radius.toFixed(2)+' λ₀';$('quantity-title').style.color=colors[key];
 $('count-status').textContent=shown+' / '+s.count+' arrows';$('winding-status').textContent='Q winding: '+(R.minQ>=1e-6?R.winding:'undefined on this contour');
 $('closure-status').textContent=closure===null?'Direction undefined':(closure<-.99?'Axis endpoint reverses after one turn':closure>.99?'Direction closes after one turn':'Inspect closure');
 $('warning').textContent=!axisValid?'Axis directions are undefined or undersampled on this contour. Change the radius.':shown<s.count?'Undefined vectors are omitted on this circle.':'';
 $('selected-values').textContent=d&&axisValid?'Highlighted '+s.angle+'°: unit direction (x, y, z) = ('+d.map(v=>v.toFixed(4)).join(', ')+')':'Highlighted direction is undefined.';
 window.contourDiagnostics={...s,shown,winding:R.winding,minQ:R.minQ,minI:R.minI,closure,first,last,selected:d,axisValid};
 }catch(e){$('warning').textContent='Could not update this contour: '+e.message;console.error(e)}finally{busy=false;if(pending){pending=false;render()}}
}
let timer;function schedule(){clearTimeout(timer);timer=setTimeout(render,100)}
for(const key of ['radius','count','length','thickness','angle']){
 $(key+'-value').textContent=$(key).value;
 $(key).addEventListener('input',()=>{$(key+'-value').textContent=$(key).value;schedule()});
}
$('quantity').addEventListener('change',schedule);$('background').addEventListener('change',schedule);
$('reset').onclick=()=>{for(const [key,value]of Object.entries(P.defaults)){$(key).value=value;if($(key+'-value'))$(key+'-value').textContent=value}$('background').checked=false;schedule()};
render();
})();
