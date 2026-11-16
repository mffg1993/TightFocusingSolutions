"""Self-contained interactive polarization atlas (Plotly/WebGL, no VTK server).

The static PyVista renderer is unchanged. This viewer uses the same field,
sampling convention, normalized ellipse geometry and polarization_axes helper.
Only one trace per color/group is sent to the browser, reducing actor overhead.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from html import escape
from pathlib import Path
from .mathtubes import polarization_axes


def _tube_mesh(points, radius, closed=False, sides=12):
    """Compact triangulated tube; transport a local frame along the curve."""
    p=np.asarray(points,float)
    tangent=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0) if closed else np.gradient(p,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-15)
    # Choose a stable initial normal, then project the previous one at each step.
    normals=[]
    ref=np.eye(3)[np.argmin(abs(tangent[0]))]
    for t in tangent:
        n=ref-np.dot(ref,t)*t
        if np.linalg.norm(n)<1e-8:
            ref=np.eye(3)[np.argmin(abs(t))]
            n=ref-np.dot(ref,t)*t
        n/=np.linalg.norm(n)
        normals.append(n)
        ref=n
    n=np.array(normals)
    b=np.cross(tangent,n)
    phi=np.arange(sides)*2*np.pi/sides
    q=p[:,None,:]+radius*(n[:,None,:]*np.cos(phi)[None,:,None]+b[:,None,:]*np.sin(phi)[None,:,None])
    faces=[]
    for i in range(len(p) if closed else len(p)-1):
        j=(i+1)%len(p)
        for k in range(sides):
            l=(k+1)%sides
            faces.extend([(i*sides+k,j*sides+k,j*sides+l),(i*sides+k,j*sides+l,i*sides+l)])
    if not closed:
        for k in range(1,sides-1):
            faces.extend([(0,k+1,k),((len(p)-1)*sides,(len(p)-1)*sides+k,(len(p)-1)*sides+k+1)])
    return q.reshape(-1,3),np.asarray(faces,int)


def _arrow_mesh(center,d,shaft_radius):
    """Triangulated straight shaft and conical arrowhead."""
    length=np.linalg.norm(d)
    axis=d/length
    base=np.asarray(center)
    neck=base+.72*d
    verts,faces=_tube_mesh([base,neck],shaft_radius)
    ref=np.eye(3)[np.argmin(abs(axis))]
    n=np.cross(axis,ref); n/=np.linalg.norm(n)
    b=np.cross(axis,n)
    phi=np.arange(12)*2*np.pi/12
    radius=min(2.1*shaft_radius,.22*length)
    ring=neck+radius*(np.cos(phi)[:,None]*n+np.sin(phi)[:,None]*b)
    q=np.vstack((ring,base+d,neck))
    f=np.array([(k,(k+1)%12,12) for k in range(12)]+[(k,13,(k+1)%12) for k in range(12)])
    return np.vstack((verts,q)),np.vstack((faces,f+len(verts)))


def export_polarization_dashboard(field,x,y,html_path,*,title="",ngrid=8,roi=1.5,
                                  xlim=None,ylim=None,tube_radius_fraction=.065,
                                  auto_roi=True,roi_threshold=.05,
                                  minimum_relative_amplitude=.03,linear_threshold=.03,
                                  intensity_threshold=.3, **unused):
    """Write an offline HTML dashboard; all checkboxes act locally in the browser."""
    e=np.asarray(field,complex); x=np.asarray(x,float); y=np.asarray(y,float)
    if e.shape!=(3,len(y),len(x)) or not np.isfinite(e).all():
        raise ValueError("field must be finite with shape (3, len(y), len(x))")
    if ngrid<2 or tube_radius_fraction<=0: raise ValueError("invalid glyph geometry")
    if not 0<roi_threshold<=1 or not 0<=minimum_relative_amplitude<1 or not 0<=linear_threshold<1 or not 0<=intensity_threshold<=1:
        raise ValueError("invalid thresholds")
    I=np.sum(abs(e)**2,axis=0); peak=float(I.max())
    if peak<=0: raise ValueError("field is identically zero")
    I/=peak
    iy_sig,ix_sig=np.nonzero(I>=roi_threshold)
    def bounds(axis,idx,explicit):
        if explicit is not None: return tuple(map(float,explicit))
        if not auto_roi: return (-float(roi),float(roi))
        lo,hi=axis[idx.min()],axis[idx.max()]
        step=np.min(np.diff(axis)); mid=(lo+hi)/2
        half=max((hi-lo)/2+max(.18*(hi-lo),3*step),(ngrid+4)*step/2)
        return max(axis[0],-roi,mid-half),min(axis[-1],roi,mid+half)
    xlim=bounds(x,ix_sig,xlim); ylim=bounds(y,iy_sig,ylim)
    if len(xlim)!=2 or len(ylim)!=2 or not np.isfinite((*xlim,*ylim)).all() or xlim[0]>=xlim[1] or ylim[0]>=ylim[1]:
        raise ValueError("invalid limits")
    # The explorer uses a single origin-centered square viewport.
    limit_max=float(min(-x[0],x[-1],-y[0],y[-1]))
    if limit_max<=0:
        raise ValueError("The shared x/y limit requires a grid spanning the origin")
    limit=min(limit_max,max(abs(v) for v in (*xlim,*ylim)))
    xlim=ylim=(-limit,limit)
    xs=np.flatnonzero((x>=xlim[0])&(x<=xlim[1]))
    ys=np.flatnonzero((y>=ylim[0])&(y<=ylim[1]))
    if min(len(xs),len(ys))<ngrid+2: raise ValueError("not enough samples in requested region")
    ix=np.linspace(xs[0]+2,xs[-1]-2,ngrid).round().astype(int)
    iy=np.linspace(ys[0]+2,ys[-1]-2,ngrid).round().astype(int)
    pitch=min(x[ix[-1]]-x[ix[0]],y[iy[-1]]-y[iy[0]])/(ngrid-1)
    rad=tube_radius_fraction*pitch
    normal_display_scale=.44*pitch  # Fixed length for the unit normal arrows.
    zcenter=max((.44+tube_radius_fraction+.02)*pitch,normal_display_scale+.5*rad)
    groups={k:[] for k in ("linear","positive","negative","alpha","beta","gamma")}
    t=np.linspace(0,2*np.pi,40,endpoint=False)
    for j in iy:
        for i in ix:
            v=e[:,j,i]; amp=np.linalg.norm(v)
            if amp==0 or I[j,i]<intensity_threshold: continue
            center=np.array([x[i],y[j],zcenter])
            u,s,_=np.linalg.svd(np.column_stack((v.real,v.imag)),full_matrices=False)
            ellipticity=s[1]/s[0]
            spin=np.imag(np.cross(v.conj(),v))/amp**2
            sign=spin[2]
            if abs(sign)<.05*np.linalg.norm(spin): sign=spin[np.argmax(abs(spin[:2]))]
            key="linear" if ellipticity<linear_threshold else ("positive" if sign>0 else "negative")
            if ellipticity<1e-5:
                d=u[:,0]*s[0]*.44*pitch/amp
                groups[key].append(_tube_mesh([center-d,center+d],rad))
            else:
                p=center+.44*pitch/amp*(v.real[None,:]*np.cos(t)[:,None]+v.imag[None,:]*np.sin(t)[:,None])
                groups[key].append(_tube_mesh(p,rad,closed=True))
            alpha,beta,gamma=polarization_axes(v)
            for name,d in (("alpha",alpha),("beta",beta),("gamma",gamma)):
                if not np.isfinite(d).all() or np.linalg.norm(d)<1e-8: continue
                if name=="gamma":
                    groups[name].append(_arrow_mesh(center,normal_display_scale*d/np.linalg.norm(d),.45*rad))
                else:
                    d=.44*pitch*d
                    groups[name].append(_arrow_mesh(center-d,2*d,.45*rad))

    fig=make_subplots(rows=2,cols=1,specs=[[{"type":"scene"}],[{"type":"scene"}]],
                      subplot_titles=("Polarization ellipses","Polarization axes"),vertical_spacing=.08)
    indices={}
    # Keep WebGL data bounded even when the underlying field uses a fine grid.
    xx=xs[::max(1,len(xs)//120)]; yy=ys[::max(1,len(ys)//120)]
    xx=np.unique(np.r_[xx,xs[-1]]); yy=np.unique(np.r_[yy,ys[-1]])
    for col in (1,2):
        indices[f"intensity{col}"]=[len(fig.data)]
        fig.add_trace(go.Surface(x=x[xx],y=y[yy],z=np.zeros((len(yy),len(xx))),
                                 surfacecolor=I[np.ix_(yy,xx)],colorscale="Viridis",
                                 cmin=0,cmax=1,showscale=True,showlegend=False,
                                 colorbar=dict(x=1.01,y=.77 if col==1 else .23,len=.34,thickness=14,title="I/Imax"),
                                 lighting=dict(ambient=1,diffuse=0,specular=0),
                                 hoverinfo="skip",name="Intensity"),row=col,col=1)
    colors={"linear":"#bfc4cb","positive":"#e64760","negative":"#24b4ef",
            "alpha":"#20ae57","beta":"#ff941f","gamma":"#d928cf"}
    labels={"linear":"Linear","positive":"Positive","negative":"Negative",
            "alpha":"Major α","beta":"Minor β","gamma":"Normal γ"}
    for key,meshes in groups.items():
        indices[key]=[]
        if not meshes: continue
        points=[]; faces=[]; offset=0
        for v,f in meshes:
            points.append(v); faces.append(f+offset); offset+=len(v)
        v=np.concatenate(points); f=np.concatenate(faces)
        indices[key]=[len(fig.data)]
        col=2 if key in ("alpha","beta","gamma") else 1
        fig.add_trace(go.Mesh3d(x=v[:,0].astype("float32"),y=v[:,1].astype("float32"),
                                z=v[:,2].astype("float32"),i=f[:,0],j=f[:,1],k=f[:,2],
                                color=colors[key],name=labels[key],showlegend=False,hoverinfo="skip",
                                lighting=dict(ambient=.65,diffuse=.35,specular=0,roughness=1),
                                flatshading=False),row=col,col=1)
    # Explicit ranges keep the camera and wavelength rulers stable when traces hide.
    zmax=zcenter+max(.44*pitch,normal_display_scale)+2*rad+.02*pitch
    cam=dict(eye=dict(x=-1.0,y=.12,z=1.6),up=dict(x=0,y=1,z=0),
             projection=dict(type="orthographic"))
    axis_style=dict(showgrid=False,zeroline=False,showbackground=False,showspikes=False,
                    showline=True,linecolor="#737373",ticks="outside",nticks=5)
    scene=dict(xaxis=dict(axis_style,title="x / λ₀",range=list(xlim)),
               yaxis=dict(axis_style,title="y / λ₀",range=list(ylim)),
               zaxis=dict(visible=False,range=[-0.1*pitch,zmax]),
               aspectmode="manual",aspectratio=dict(x=(xlim[1]-xlim[0]),
                   y=(ylim[1]-ylim[0]),z=zmax+.1*pitch),
               camera=cam,bgcolor="white",dragmode="orbit",uirevision="keep-camera")
    fig.update_layout(scene=scene,scene2=scene,margin=dict(l=15,r=65,t=45,b=10),
                      paper_bgcolor="white",height=1250,showlegend=False)
    fragment=fig.to_html(full_html=False,include_plotlyjs=True,div_id="polarization-scenes",
                         config={"responsive":True,"displaylogo":False})
    import json
    controls=[]
    for heading,keys in (("Ellipses",["intensity1"]),("Axes",["intensity2","alpha","beta","gamma"])):
        entries=[]
        for k in keys:
            label="Intensity map" if k.startswith("intensity") else labels[k]
            col=colors.get(k,"#555555")
            entries.append(f'<label><input type="checkbox" data-key="{k}" checked> <span style="color:{col}">{label}</span></label>')
        controls.append(f'<section><strong>{heading}</strong> {" ".join(entries)}</section>')
    # Normalize once to peak amplitude before serializing. All browser operations
    # are on the original sampled complex field (not on precomputed glyphs).
    defaults=dict(limit=limit,
                  density=int(ngrid),thickness=float(tube_radius_fraction),
                  scale=.44,opacity=1.,cutoff=float(intensity_threshold),
                  linear=float(linear_threshold))
    settings=[
        ("limit","Shared x/y half-width (λ₀)",min(limit,max(np.min(np.diff(x)),np.min(np.diff(y)))),limit_max,.005),
        ("density","Ellipses per side",2,max(20,ngrid),1),
        ("thickness","Tube radius / spacing",.005,max(.15,tube_radius_fraction),.005),
        ("scale","Glyph size / spacing",.10,.75,.01),
        ("opacity","Intensity opacity",0,1,.05),
        ("cutoff","Minimum relative intensity (I/Imax)",0,1,.01),
        ("linear","Linear ellipticity threshold",0,max(.3,linear_threshold),.005)]
    sliders=[]
    for key,label,lo,hi,step in settings:
        v=defaults[key]
        value_label=f"{v:.2f}" if key=="cutoff" else f"{v:.4g}"
        sliders.append(f'<label class="slider"><span>{label} <output id="{key}-value">{value_label}</output></span><input type="range" id="{key}" min="{lo:.12g}" max="{hi:.12g}" step="{step}" value="{v:.12g}"></label>')
    layout=fig.layout.to_plotly_json()
    normalized=e/np.sqrt(peak)
    data=dict(x=x.tolist(),y=y.tolist(),I=I.ravel().tolist(),
              re=normalized.real.reshape(3,-1).tolist(),im=normalized.imag.reshape(3,-1).tolist(),
              defaults=defaults,layout=layout,normal_display_scale=normal_display_scale)
    from plotly.utils import PlotlyJSONEncoder
    payload=json.dumps(data,cls=PlotlyJSONEncoder,separators=(",",":"),allow_nan=False)
    js=Path(__file__).with_name("dashboard_controls.js").read_text(encoding="utf-8")
    html=('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
          +escape(title or "Polarization explorer")+'</title><style>'
          'body{font:15px system-ui,sans-serif;margin:18px auto;padding:0 18px;max-width:1100px;color:#202124;background:white}'
          'h2{margin:0 0 10px}.controls{display:grid;grid-template-columns:1fr;gap:8px}'
          '.controls section,.slider-panel{border:1px solid #ddd;border-radius:8px;padding:12px}'
          '.controls section{display:flex;gap:12px;align-items:center;flex-wrap:wrap}'
          '.sliders{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 24px}'
          '.slider{display:flex;flex-direction:column;gap:4px}.slider span{display:flex;justify-content:space-between;gap:6px}'
          'output{font-variant-numeric:tabular-nums;color:#555}input[type=range]{width:100%;accent-color:#4669a9}'
          'label{cursor:pointer}button{padding:6px 12px;cursor:pointer}small{display:block;color:#555;margin-top:10px}'
          '.actions{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:12px}'
          '@media(max-width:620px){.sliders{grid-template-columns:1fr}}</style></head><body><h2>'
          +escape(title or "Polarization explorer")+'</h2><div class="slider-panel"><div class="sliders">'
          +''.join(sliders)+'</div><div class="actions"><label><input type="checkbox" id="sync-cameras" checked> Synchronize cameras</label>'
          '<button id="reset-controls">Reset defaults</button><span id="slider-status" role="status"></span></div></div>'
          '<div class="controls">'+''.join(controls)+'</div>'+fragment
          +'<small>Red/blue: signed spin convention; gray: nearly linear. Major/minor arrowheads indicate a chosen positive endpoint (the simultaneous sign is phase-convention dependent); both axes are masked at circular states. The magenta arrow is a unit normal oriented along normalized electric spin, with a constant display length independent of sliders. No arrow is shown when the spin is zero (linear polarization). Arrow length does not represent spin magnitude. Glyph height is a display offset. All sliders resample the exported numerical field; no optical propagation is recalculated.</small>'
          +'<script>window.POLARIZATION_DATA='+payload+';</script><script>'+js+'</script></body></html>')
    Path(html_path).write_text(html,encoding="utf-8")
    return Path(html_path)
