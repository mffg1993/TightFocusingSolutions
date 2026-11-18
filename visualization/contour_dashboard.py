"""Offline vector/ellipse inspection on circular focal-plane contours."""
from pathlib import Path
from html import escape
import json
import numpy as np
from plotly.offline import get_plotlyjs


def export_contour_dashboard(field,x,y,html_path,*,center=(0.,0.),radius=.4,
                             count=96,quantity='alpha',arrow_length=.13):
    """Export the complex field; sliders resample it bilinearly on a circle.

    Principal-axis signs use the continuously unwrapped phase of E dot E around
    a dense contour independent of the displayed arrow count. The two ellipse
    axes remain directors; gamma is the directed electric-spin normal.
    """
    e=np.asarray(field,complex);x=np.asarray(x,float);y=np.asarray(y,float)
    center=np.asarray(center,float)
    if e.shape!=(3,len(y),len(x)) or not np.isfinite(e).all():
        raise ValueError('Expected finite complex field with shape (3,y,x)')
    if min(len(x),len(y))<3 or np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):
        raise ValueError('Coordinates must be strictly increasing')
    if center.shape!=(2,) or not np.isfinite(center).all():raise ValueError('Invalid circle center')
    maximum=float(min(center[0]-x[0],x[-1]-center[0],center[1]-y[0],y[-1]-center[1]))
    maximum=np.floor((maximum-1e-8)*100)/100
    if maximum<=.03 or not .03<=radius<=maximum:raise ValueError('Circle must lie inside the exported field')
    if quantity not in ('alpha','beta','gamma'):raise ValueError('Unknown vector quantity')
    if not 16<=count<=256 or not .03<=arrow_length<=.3:raise ValueError('Invalid arrow geometry')
    peak=float(np.sum(abs(e)**2,axis=0).max())
    if peak<=0:raise ValueError('Zero field')
    e=e/np.sqrt(peak)
    payload=dict(x=x.tolist(),y=y.tolist(),re=e.real.reshape(3,-1).tolist(),
        im=e.imag.reshape(3,-1).tolist(),center=center.tolist(),maximum=maximum,
        defaults=dict(radius=radius,count=count,quantity=quantity,length=arrow_length,thickness=.007,angle=0),
        angular_samples=1440)
    js=Path(__file__).with_name('contour_controls.js').read_text(encoding='utf-8')
    options=''.join(f'<option value="{key}"'+(' selected' if key==quantity else '')+f'>{label}</option>'
                    for key,label in [('alpha','Major axis α'),('beta','Minor axis β'),('gamma','Normal γ')])
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lemon — polarization vectors on a circle</title><style>
body{font:15px system-ui,sans-serif;max-width:1250px;margin:20px auto;padding:0 18px;color:#202124;background:white}
h1{font-size:24px;margin-bottom:8px}.controls{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:15px;padding:18px;border:1px solid #ddd;border-radius:10px}
label{display:flex;flex-direction:column;gap:6px}label.check{flex-direction:row;align-items:center}input[type=range]{width:100%;accent-color:#256945}select,button{font:inherit;padding:8px;border:1px solid #bbb;border-radius:5px;background:white}button{cursor:pointer}
label span{display:flex;justify-content:space-between}output{font-variant-numeric:tabular-nums;color:#4b5563}.summary{padding:13px 0;display:flex;gap:20px;flex-wrap:wrap;font-variant-numeric:tabular-nums}#quantity-title{font-weight:700;font-size:20px;margin:20px 0 0}#warning{color:#9a3412;min-height:1.5em}.notes{color:#4b5563;line-height:1.5;border-top:1px solid #ddd;padding-top:12px}#selected-values{font-family:ui-monospace,monospace}.plot{height:660px}
@media(max-width:700px){.controls{grid-template-columns:1fr 1fr}.plot{height:600px}}
</style></head><body><h1>Polarization on a circular contour</h1>
<p>Move the radius, add arrows around the circle, and choose the vector you want to inspect.</p>
<div role="tablist" style="display:flex;gap:10px;margin:18px 0">
<button role="tab" id="tab-polarization" aria-selected="true">Polarization</button>
<button role="tab" id="tab-animation" aria-selected="false">Time animation</button>
<button role="tab" id="tab-instant" aria-selected="false">Instantaneous E(t)</button></div>
<div class="controls">
<label>Ellipse colors<select id="color-mode"><option value="handedness">Left / Right / Linear</option><option value="phase">Optical-cycle phase (HSV)</option></select></label>
<label><span>Time t / T <output id="time-value">0</output></span><input id="time" type="range" min="0" max="1" step="0.005" value="0"></label>
<button id="play">Play / Pause time evolution</button>
<label class="check"><input type="checkbox" id="left-ellipses" checked>Left: ellipse tubes</label>
<label class="check"><input type="checkbox" id="left-alpha">Left: major α</label>
<label class="check"><input type="checkbox" id="left-beta">Left: minor β</label>
<label class="check"><input type="checkbox" id="left-gamma">Left: normal γ</label>
<label class="check"><input type="checkbox" id="left-electric">Left: instantaneous E(t)</label>
<label>Selected quantity<select id="quantity">OPTIONS</select></label>
<label><span>Circle radius / λ₀ <output id="radius-value"></output></span><input id="radius" type="range" min="0.03" max="MAXIMUM" step="0.01" value="RADIUS"></label>
<label><span>Arrows around the circle <output id="count-value"></output></span><input id="count" type="range" min="16" max="256" step="8" value="COUNT"></label>
<label><span>Arrow length scale / λ₀ <output id="length-value"></output></span><input id="length" type="range" min="0.03" max="0.30" step="0.01" value="LENGTH"></label>
<label><span>Arrow thickness / λ₀ <output id="thickness-value"></output></span><input id="thickness" type="range" min="0.001" max="0.012" step="0.001" value="0.007"></label>
<label><span>Highlighted position / degrees <output id="angle-value"></output></span><input id="angle" type="range" min="0" max="360" step="1" value="0"></label>
<label class="check"><input type="checkbox" id="background" checked>Show intensity plane</label><button id="reset">Reset controls</button>
</div><div style="margin-top:12px">Handedness: <span style="color:#e64760">● L</span> · <span style="color:#24b4ef">● R</span> · <span style="color:#9298a1">● Linear</span> &nbsp; Phase t/T: 0 <span style="display:inline-block;width:160px;height:10px;background:linear-gradient(to right,red,yellow,lime,cyan,blue,magenta,red)"></span> 1</div><div id="quantity-title"></div><div class="summary"><span id="count-status"></span><span id="winding-status"></span><span id="closure-status"></span></div>
<div id="warning" role="status"></div><div id="contour-scenes" class="plot"></div><div id="selected-values"></div>
<div class="notes"><p><b>Color and time conventions.</b> Handedness uses the notebook-02 signed-spin convention relative to +z (red: L, blue: R, gray: nearly linear); for nearly transverse spin the dominant transverse component supplies the same fallback as the atlas. This is a stated display convention for tilted ellipses. HSV colors mark optical-cycle phase along each ellipse (red at t/T=0). Time arrows follow Re[E exp(−iωt)] = Re(E) cos(ωt) + Im(E) sin(ωt), retaining local ellipse eccentricity. Their size is locally normalized by sqrt(I), as for the ellipse glyphs. The animation and fixed-time tabs use the same formula. The intensity plane and projected sampling circle are shifted down only for display; the physical sampled circle is z=0.</p><p><b>Reading the arrows.</b> All centers lie on the selected circle in the focal plane, z=0. Left: sparse local 3D ellipses for context. Right: the selected vector at the requested density. x and y are transverse; z is longitudinal. Arrow length is a display scale and does not shrink when density increases. The black arrow and matching dot mark the highlighted position.</p>
<p>Major α (green) and minor β (orange) are unoriented ellipse axes. Their arrowheads mark an endpoint chosen continuously along this loop; a sign reversal after one turn is visible at the seam. The normal γ (magenta) is directed along electric spin and closes with the same sign wherever it is defined. Major/minor axes are omitted at circular states; normals are omitted at linear states.</p>
<p>All arrows show full physical 3D directions, with no radial–axial projection or added twist. Data are bilinearly interpolated from the exported complex focal field, then the axes are calculated. Numerical contour diagnostics here refer to that interpolation; use the notebook's direct quadrature for quantitative confirmation. The circle center is fixed at CENTER in units of λ₀.</p></div>
<script>PLOTLY</script><script>window.CONTOUR_DATA=PAYLOAD;</script><script>CONTROLS</script></body></html>'''
    for key,value in [('OPTIONS',options),('MAXIMUM',str(maximum)),('RADIUS',str(radius)),
                      ('COUNT',str(count)),('LENGTH',str(arrow_length)),
                      ('CENTER',escape(str(tuple(center)))),('PLOTLY',get_plotlyjs()),
                      ('PAYLOAD',json.dumps(payload,separators=(',',':'),allow_nan=False)),('CONTROLS',js)]:
        html=html.replace(key,value)
    path=Path(html_path);path.write_text(html,encoding='utf-8');return path
