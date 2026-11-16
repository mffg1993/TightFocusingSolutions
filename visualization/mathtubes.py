"""MathTubes-inspired scientific visualization for the tight-focusing atlas.

Adapted from the user-provided MathTubes_Colab.ipynb (MathTubes v0.1.0).
The Mathematica-style tube construction, parametric curves, closed-loop handling
and PBR lighting are preserved. The additional plot_field_ellipses() helper
creates the ellipse centerlines from complex focused electric fields.
This module does NOT calculate the optical field.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import numpy as np
import pyvista as pv

__version__ = "0.1.0-atlas"

def _points(points: Any, closed: bool = False) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(p) < (3 if closed else 2):
        raise ValueError("not enough curve points")
    if not np.isfinite(p).all():
        raise ValueError("curve coordinates must be finite")
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    p = p[np.r_[True, lengths > 1e-12]]
    if closed and len(p) > 2 and np.linalg.norm(p[0]-p[-1]) < 1e-12:
        p = p[:-1]
    if len(p) < (3 if closed else 2):
        raise ValueError("curve degenerates to too few distinct points")
    return p

def curve_mesh(points: Any, *, closed: bool = False) -> pv.PolyData:
    """Connect user coordinates into a VTK polyline."""
    return pv.lines_from_points(_points(points, closed=closed), close=closed)

def tube(points: Any, radius: float = 0.1, *, closed: bool = False,
         sides: int = 32, cap: bool | None = None) -> pv.PolyData:
    """Mathematica Tube[points, radius] equivalent (from MathTubes)."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a positive finite number")
    if sides < 3:
        raise ValueError("sides must be at least 3")
    if cap is None:
        cap = not closed
    return curve_mesh(points, closed=closed).tube(
        radius=float(radius), n_sides=int(sides), capping=bool(cap))

def parametric_points(function: Callable[[np.ndarray], Any],
                      tmin: float = 0, tmax: float = 2*np.pi,
                      *, samples: int = 600, closed: bool = False) -> np.ndarray:
    if samples < (3 if closed else 2):
        raise ValueError("samples is too small")
    if not np.isfinite([tmin, tmax]).all() or tmax <= tmin:
        raise ValueError("require finite tmin < tmax")
    t = np.linspace(tmin, tmax, samples, endpoint=not closed)
    result = np.asarray(function(t), dtype=float)
    if result.shape == (3, samples):
        result = result.T
    if result.shape != (samples, 3):
        raise ValueError("function(t) must return (N,3) or (3,N) coordinates")
    return _points(result, closed=closed)

def parametric_tube(function: Callable[[np.ndarray], Any],
                    tmin: float = 0, tmax: float = 2*np.pi,
                    *, samples: int = 600, radius: float = 0.1,
                    closed: bool = False, sides: int = 32) -> pv.PolyData:
    return tube(parametric_points(function, tmin, tmax,
                                 samples=samples, closed=closed),
                radius=radius, closed=closed, sides=sides)

def plot_tubes(curves: Sequence[Any], *, radius: float = 0.1,
               closed: bool | Sequence[bool] = False,
               colors: Sequence[str] | None = None,
               background: str = "white", size: tuple[int,int] = (1100,900),
               metallic: float = 0.4, roughness: float = 0.3,
               sides: int = 40, show_axes: bool = False,
               camera: str | tuple = "iso", zoom: float = 1.15,
               screenshot: str | None = None,
               show: bool = True, interactive: bool = False) -> pv.Plotter:
    """MathTubes' familiar multi-curve plotter with VTK PBR materials."""
    if isinstance(closed, bool):
        flags = [closed]*len(curves)
    else:
        flags = list(closed)
        if len(flags) != len(curves):
            raise ValueError("closed must contain one flag per curve")
    palette = list(colors or ["#d4a017", "#2388c7", "#ce4f4f", "#42a88a"])
    if not palette:
        raise ValueError("colors cannot be empty")
    pl = pv.Plotter(off_screen=not interactive, notebook=interactive,
                    window_size=size)
    pl.set_background(background)
    for k, obj in enumerate(curves):
        mesh = obj if isinstance(obj, pv.PolyData) else tube(
            obj, radius, closed=flags[k], sides=sides)
        pl.add_mesh(mesh, color=palette[k%len(palette)], smooth_shading=True,
                    pbr=True, metallic=metallic, roughness=roughness)
    pl.enable_anti_aliasing("ssaa")
    pl.enable_lightkit()
    if show_axes: pl.add_axes()
    pl.camera_position = camera
    pl.camera.zoom(zoom)
    if screenshot is not None:
        pl.screenshot(screenshot, return_img=False)
    if show:
        pl.show(jupyter_backend="client" if interactive else "static")
    return pl

def trefoil(samples: int = 900) -> np.ndarray:
    t = np.linspace(0,2*np.pi,samples,endpoint=False)
    return np.column_stack(((2+np.cos(3*t))*np.cos(2*t),
                            (2+np.cos(3*t))*np.sin(2*t),
                            np.sin(3*t)))

def torus_knot(p: int = 2, q: int = 3, *, R: float = 2.,
               r: float = .7, samples: int = 1000) -> np.ndarray:
    t=np.linspace(0,2*np.pi,samples,endpoint=False)
    return np.column_stack(((R+r*np.cos(q*t))*np.cos(p*t),
                            (R+r*np.cos(q*t))*np.sin(p*t),
                            r*np.sin(q*t)))

def ellipse_points(field: np.ndarray, center: tuple[float,float,float],
                   scale: float, samples: int = 72) -> np.ndarray:
    """Sample the true 3D optical polarization ellipse for exp(-i omega t).

    Field must contain complex Ex,Ey,Ez, not a transverse projection.
    `scale` affects ONLY visualization, not field or measured intensity.
    """
    e=np.asarray(field, dtype=complex)
    if e.shape!=(3,) or not np.isfinite(e).all():
        raise ValueError("field must contain three finite complex components")
    if not np.isfinite(scale) or scale<=0:
        raise ValueError("scale must be positive")
    t=np.linspace(0,2*np.pi,samples,endpoint=False)
    return np.asarray(center)[None,:] + scale*(
        e.real[None,:]*np.cos(t[:,None])+
        e.imag[None,:]*np.sin(t[:,None]))

def polarization_axes(field, *, circular_tolerance=1e-8):
    """Return normalized semiaxes alpha, beta and spin normal gamma.

    Component axis is first (3, ...). Convention exp(-i omega t).
    alpha/beta are directors and undefined at circular states; gamma is
    Im(conj(E) cross E)/|E|^2 = 2 alpha cross beta away from circularity.
    Zero-field sites are NaN. Reference: Bliokh, Alonso & Dennis (2019),
    doi:10.1088/1361-6633/ab4415, Eq. (18).
    """
    e = np.asarray(field, dtype=complex)
    if e.ndim < 1 or e.shape[0] != 3:
        raise ValueError("field must have shape (3, ...)")
    intensity = np.sum(abs(e)**2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = e / np.sqrt(intensity)
    q = np.sum(u*u, axis=0)
    rectified = u*np.exp(-0.5j*np.angle(q))
    alpha, beta = rectified.real.copy(), rectified.imag.copy()
    circular = abs(q) <= circular_tolerance
    alpha = np.where(circular[None,...], np.nan, alpha)
    beta = np.where(circular[None,...], np.nan, beta)
    gamma = np.imag(np.cross(u.conj(),u,axisa=0,axisb=0,axisc=0))
    return alpha, beta, gamma


def plot_polarization_axes(field, x, y, *, component="all", **kwargs):
    """Plot alpha/beta directors and/or the directed gamma spin vector."""
    return plot_field_ellipses(field,x,y,glyph_mode=component,**kwargs)


def plot_field_ellipses(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                        *, ngrid: int = 8, roi: float = 1.5,
                        xlim: tuple[float, float] | None = None,
                        ylim: tuple[float, float] | None = None,
                        tube_radius_fraction: float = 0.065,
                        minimum_relative_amplitude: float = 0.03,
                        auto_roi: bool = True,
                        roi_threshold: float = 0.05,
                        linear_threshold: float = 0.03,
                        glyph_mode: str = 'ellipse',
                        screenshot: str | None = None,
                        off_screen: bool = True, size: tuple[int,int] = (1100,900),
                        show: bool = True) -> pv.Plotter:
    """Plot normalized focal intensity with an ngrid-by-ngrid 3D ellipse grid.

    The ellipses are produced from the full complex vector E=(Ex,Ey,Ez).
    Ellipse sizes are *locally normalized for legibility*. Consequently,
    ellipse size must NOT be interpreted as measured field amplitude.
    The continuous background carries the intensity information.
    Gray marks nearly linear polarization. Red/blue mark positive/negative
    gamma_z, falling back to the dominant transverse component at near-zero
    gamma_z. The fallback is a display convention, not helicity.
    Sites below minimum_relative_amplitude times the global peak amplitude
    are omitted. Unspecified limits are automatically selected from the
    roi_threshold fraction of peak intensity, with padding, capped by roi.
    Explicit xlim/ylim override auto framing. Fields are not recalculated.
    """
    e=np.asarray(field)
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if e.shape!=(3,len(y),len(x)):
        raise ValueError("field shape must be (3, len(y), len(x))")
    if glyph_mode not in ('ellipse','all','alpha','beta','gamma'):
        raise ValueError('Unknown glyph_mode')
    if ngrid<2: raise ValueError("ngrid must be >=2")
    if tube_radius_fraction<=0: raise ValueError("tube_radius_fraction must be positive")
    if not np.all(np.isfinite(e)): raise ValueError("field is not finite")
    X,Y=np.meshgrid(x,y,indexing="xy")
    intensity=np.sum(np.abs(e)**2,axis=0)
    top=float(intensity.max())
    if top<=0: raise ValueError("field has zero intensity everywhere")
    intensity=intensity/top
    if not 0 < roi_threshold <= 1: raise ValueError("roi_threshold must lie in (0,1]")
    if not 0 <= minimum_relative_amplitude < 1: raise ValueError("minimum_relative_amplitude must lie in [0,1)")
    if not 0 <= linear_threshold < 1: raise ValueError("linear_threshold must lie in [0,1)")
    def auto_limits(axis, indices):
        lo, hi = axis[indices.min()], axis[indices.max()]
        margin = max(0.18*(hi-lo), 3*np.min(np.diff(axis)))
        # Keep enough samples to support the requested ellipse density.
        center = (lo+hi)/2
        half = max((hi-lo)/2+margin, (ngrid+4)*np.min(np.diff(axis))/2)
        return (max(float(axis[0]), -float(roi), center-half),
                min(float(axis[-1]), float(roi), center+half))
    iy_sig, ix_sig = np.nonzero(intensity >= roi_threshold)
    if xlim is None: xlim=auto_limits(x,ix_sig) if auto_roi else (-float(roi),float(roi))
    if ylim is None: ylim=auto_limits(y,iy_sig) if auto_roi else (-float(roi),float(roi))
    xlim=tuple(float(v) for v in xlim)
    ylim=tuple(float(v) for v in ylim)
    if (len(xlim)!=2 or len(ylim)!=2
        or not np.isfinite((*xlim,*ylim)).all()
        or xlim[0]>=xlim[1] or ylim[0]>=ylim[1]):
        raise ValueError("xlim and ylim must be (minimum, maximum) finite increasing pairs")
    xs=np.flatnonzero((x>=xlim[0]) & (x<=xlim[1]))
    ys=np.flatnonzero((y>=ylim[0]) & (y<=ylim[1]))
    if len(xs)<ngrid+2 or len(ys)<ngrid+2:
        raise ValueError(
            f"Requested region has only {len(xs)} x {len(ys)} samples. "
            f"Need >= {ngrid+2} per axis for ngrid={ngrid}; "
            "increase the region or recalculate the field with a finer grid.")
    sy=slice(ys[0],ys[-1]+1);sx=slice(xs[0],xs[-1]+1)
    grid=pv.StructuredGrid(X[sy,sx],Y[sy,sx],np.zeros_like(X[sy,sx]))
    grid.point_data["Intensity"]=intensity[sy,sx].ravel(order="F")
    # Independent renderers: the scalar bar cannot overlap the 3D scene.
    plot=pv.Plotter(shape=(1,2), border=False, off_screen=off_screen,window_size=size)
    plot.renderers[0].SetViewport(0.0,0.0,0.80,1.0)
    plot.renderers[1].SetViewport(0.80,0.0,1.0,1.0)
    plot.subplot(0,0)
    plot.set_background("white")
    # Do not use an automatic scalar bar here: its orientation and placement
    # can vary between PyVista themes and VTK/Colab versions.
    intensity_actor = plot.add_mesh(
        grid, scalars="Intensity", cmap="viridis", clim=(0, 1),
        opacity=1.0, lighting=False, show_scalar_bar=False)
    plot.subplot(0,1)
    plot.set_background("white")
    plot.add_scalar_bar(
        title="Intensity / max", mapper=intensity_actor.mapper,
        vertical=True, interactive=False,
        position_x=0.15, position_y=0.18,
        width=0.32, height=0.64,
        n_labels=5, fmt="%.2f",
        color="black", title_font_size=14, label_font_size=12,
        outline=False)
    # Explicit VTK enforcement protects against version/theme overrides.
    bar_actor = plot.scalar_bars["Intensity / max"]
    bar_actor.SetOrientationToVertical()
    bar_actor.SetPosition(0.15, 0.18)
    bar_actor.SetWidth(0.32)
    bar_actor.SetHeight(0.64)
    plot.subplot(0,0)
    plot.enable_anti_aliasing("ssaa")
    plot.enable_lightkit()
    inds_x=np.linspace(xs[0]+2,xs[-1]-2,ngrid).round().astype(int)
    inds_y=np.linspace(ys[0]+2,ys[-1]-2,ngrid).round().astype(int)
    pitch=min(x[inds_x[-1]]-x[inds_x[0]],
              y[inds_y[-1]]-y[inds_y[0]])/(ngrid-1)
    max_amp=np.sqrt(top)
    ellipse_radius=tube_radius_fraction*pitch
    for iy in inds_y:
        for ix in inds_x:
            vector=e[:,iy,ix]
            amp=np.linalg.norm(vector)
            if amp<minimum_relative_amplitude*max_amp:
                continue
            # Lift glyphs clear of the intensity plane: the whole ellipse is visible.
            # This is a graphical displacement, not an optical propagation distance.
            center=(x[ix],y[iy],(.46+tube_radius_fraction)*pitch)
            # The local 3D polarization ellipse is the image of a unit
            # circle under the real 3x2 matrix [Re(E), Im(E)]. Its singular
            # values are the major and minor semiaxes. If the minor axis
            # vanishes, the ellipse is a LINE, not a closed loop.
            U, singular_values, _ = np.linalg.svd(
                np.column_stack((vector.real,vector.imag)),
                full_matrices=False)
            scale=.44*pitch/amp
            if glyph_mode != "ellipse":
                aa, bb, gg = polarization_axes(vector)
                for name, direction, color in (("alpha",aa,"#e64760"),
                                                ("beta",bb,"#24b4ef"),
                                                ("gamma",gg,"#efb52e")):
                    if glyph_mode not in ("all",name): continue
                    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction)<1e-8:
                        continue
                    d=.40*pitch*direction
                    if name=="gamma":
                        mesh=pv.Arrow(start=center,direction=d,scale=float(np.linalg.norm(d)),
                                      tip_radius=.13,shaft_radius=.045)
                    else:
                        mesh=tube(np.stack((np.asarray(center)-d,np.asarray(center)+d)),
                                  radius=.45*ellipse_radius,closed=False,sides=24)
                    plot.add_mesh(mesh,color=color,smooth_shading=True,pbr=False,
                                  ambient=.65,diffuse=.35,specular=0,show_scalar_bar=False)
                continue
            if singular_values[1] < 1e-5*singular_values[0]:
                direction=U[:,0]*singular_values[0]*scale
                line=np.stack((np.asarray(center)-direction,
                               np.asarray(center)+direction))
                mesh=tube(line,radius=ellipse_radius,
                          closed=False,sides=24)
            else:
                path=ellipse_points(vector,center,scale=scale,samples=72)
                mesh=tube(path,radius=ellipse_radius,closed=True,sides=24)
            ellipticity=singular_values[1]/singular_values[0]
            spin=np.imag(np.cross(np.conj(vector),vector))/amp**2
            # Prefer the propagation axis. For almost transverse spin,
            # use its largest transverse component as an explicit display convention.
            component = spin[2]
            if abs(component) < .05*np.linalg.norm(spin):
                component = spin[np.argmax(np.abs(spin[:2]))]
            if ellipticity < linear_threshold:
                hue="#e6e6e6"
            else:
                hue="#e64760" if component>0 else "#24b4ef"
            # Matte color: no metallic environment reflections that can alter hue.
            plot.add_mesh(mesh, color=hue, smooth_shading=True,
                          pbr=False, ambient=0.60, diffuse=0.40,
                          specular=0.0, show_scalar_bar=False)
    # All positions have been expressed in vacuum-wavelength units.
    # Show actual wavelength coordinates on the focal plane. Set explicit
    # bounds so nearly-planar geometry does not create tiny/unreadable axes.
    # The z-coordinate is a display offset, not longitudinal propagation.
    bounds = (xlim[0], xlim[1], ylim[0], ylim[1],
              0.0, max(1.1, 0.92+2*tube_radius_fraction)*pitch)
    plot.show_bounds(bounds=bounds, xtitle="x / λ₀", ytitle="y / λ₀",
                     ztitle="z / λ₀", n_xlabels=5, n_ylabels=5,
                     n_zlabels=2, location="outer", all_edges=True,
                     color="black", show_zaxis=False)
    plot.add_axes(line_width=2)
    midpoint=((xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2, 0.0)
    span=max(xlim[1]-xlim[0],ylim[1]-ylim[0])
    # Keep y upright, and view the xy plane from +z and -x.
    # With this camera, the projection of +z points toward screen-right.
    plot.camera_position=[(midpoint[0]-1.65*span, midpoint[1]+0.20*span,
                           midpoint[2]+2.5*span), midpoint, (0,1,0)]
    plot.enable_parallel_projection()
    plot.reset_camera()
    plot.camera.zoom(0.87)
    if screenshot is not None: plot.screenshot(screenshot)
    if show: plot.show(jupyter_backend="static")
    return plot

__all__=["tube","curve_mesh","parametric_points","parametric_tube",
         "plot_tubes","trefoil","torus_knot","ellipse_points","plot_field_ellipses","polarization_axes","plot_polarization_axes"]
