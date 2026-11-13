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

def plot_field_ellipses(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                        *, ngrid: int = 8, roi: float = 2.5,
                        minimum_relative_amplitude: float = 1e-9,
                        screenshot: str | None = None,
                        off_screen: bool = True, size: tuple[int,int] = (1100,900),
                        show: bool = True) -> pv.Plotter:
    """Plot normalized focal intensity with an ngrid-by-ngrid 3D ellipse grid.

    The ellipses are produced from the full complex vector E=(Ex,Ey,Ez).
    Ellipse sizes are *locally normalized for legibility*. Consequently,
    ellipse size must NOT be interpreted as measured field amplitude.
    The continuous background carries the intensity information.
    Colors show sign(Im(Ex*conj(Ey))), a transverse handedness proxy only.
    Skip a site if the field nearly vanishes (polarization undefined).
    """
    e=np.asarray(field)
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if e.shape!=(3,len(y),len(x)):
        raise ValueError("field shape must be (3, len(y), len(x))")
    if ngrid<2: raise ValueError("ngrid must be >=2")
    if not np.all(np.isfinite(e)): raise ValueError("field is not finite")
    X,Y=np.meshgrid(x,y,indexing="xy")
    intensity=np.sum(np.abs(e)**2,axis=0)
    top=float(intensity.max())
    if top<=0: raise ValueError("field has zero intensity everywhere")
    intensity=intensity/top
    xs=np.flatnonzero(np.abs(x)<=roi)
    ys=np.flatnonzero(np.abs(y)<=roi)
    if len(xs)<ngrid+2 or len(ys)<ngrid+2:
        raise ValueError("ROI is too small for the requested ellipse grid")
    sy=slice(ys[0],ys[-1]+1);sx=slice(xs[0],xs[-1]+1)
    grid=pv.StructuredGrid(X[sy,sx],Y[sy,sx],np.zeros_like(X[sy,sx]))
    grid.point_data["Intensity"]=intensity[sy,sx].ravel(order="F")
    plot=pv.Plotter(off_screen=off_screen,window_size=size)
    plot.set_background("white")
    plot.add_mesh(grid,scalars="Intensity",cmap="viridis",
                  clim=(0,1),opacity=.83,show_scalar_bar=True)
    plot.enable_anti_aliasing("ssaa")
    plot.enable_lightkit()
    inds_x=np.linspace(xs[0]+2,xs[-1]-2,ngrid).round().astype(int)
    inds_y=np.linspace(ys[0]+2,ys[-1]-2,ngrid).round().astype(int)
    pitch=min(x[inds_x[-1]]-x[inds_x[0]],
              y[inds_y[-1]]-y[inds_y[0]])/(ngrid-1)
    max_amp=np.sqrt(top)
    ellipse_radius=.014*pitch
    for iy in inds_y:
        for ix in inds_x:
            vector=e[:,iy,ix]
            amp=np.linalg.norm(vector)
            if amp<minimum_relative_amplitude*max_amp:
                continue
            center=(x[ix],y[iy],.035*pitch)
            # The local 3D polarization ellipse is the image of a unit
            # circle under the real 3x2 matrix [Re(E), Im(E)]. Its singular
            # values are the major and minor semiaxes. If the minor axis
            # vanishes, the ellipse is a LINE, not a closed loop.
            U, singular_values, _ = np.linalg.svd(
                np.column_stack((vector.real,vector.imag)),
                full_matrices=False)
            scale=.36*pitch/amp
            if singular_values[1] < 1e-5*singular_values[0]:
                direction=U[:,0]*singular_values[0]*scale
                line=np.stack((np.asarray(center)-direction,
                               np.asarray(center)+direction))
                mesh=tube(line,radius=ellipse_radius,
                          closed=False,sides=24)
            else:
                path=ellipse_points(vector,center,scale=scale,samples=72)
                mesh=tube(path,radius=ellipse_radius,closed=True,sides=24)
            hand=np.imag(vector[0]*np.conj(vector[1]))
            hue="#d95f59" if hand>1e-9*amp*amp else (
                "#348fba" if hand<-1e-9*amp*amp else "#eeeeee")
            plot.add_mesh(mesh,color=hue,smooth_shading=True,
                          pbr=True,metallic=.35,roughness=.3,show_scalar_bar=False)
    plot.add_axes()
    plot.camera_position=[(7,-9,9),(0,0,0),(0,0,1)]
    plot.camera.zoom(1.15)
    if screenshot is not None: plot.screenshot(screenshot)
    if show: plot.show(jupyter_backend="static")
    return plot

__all__=["tube","curve_mesh","parametric_points","parametric_tube",
         "plot_tubes","trefoil","torus_knot","ellipse_points","plot_field_ellipses"]
