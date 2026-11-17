# TightFocusingSolutions

**Vectorial diffraction, tight focusing, and computational methods for structured light**

A research and educational repository for calculating the complex electromagnetic field produced by high-numerical-aperture (high-NA) focusing systems. The project starts from the **Richards–Wolf / Debye–Wolf diffraction integral**, develops the analytical azimuthal decomposition for Laguerre–Gaussian (LG) beams, and explores FFT-based approaches for computing arbitrary vectorial fields in three-dimensional focal volumes.

The objective is to build **physically transparent, independently validated solvers** for vectorial tight focusing.

> **Project status:** First collaborator-facing CPU integral atlas is available on `main`, with the full theory tutorial and an offline interactive HTML explorer. Historical notebooks remain available. FFT and GPU variants remain on development branches; GPU hardware validation is pending.

## What problem does this repository solve?

An incident Jones field at the objective pupil,

```text
      incident pupil                 high-NA objective                  focal volume
   Ex_in(xp, yp), Ey_in(xp, yp)  ───── Richards–Wolf ─────▶  Ex(x,y,z), Ey(x,y,z), Ez(x,y,z)
```

is transformed into a **three-component complex focused field**. Even when the incident beam is transverse (`Ez_in = 0`), a high-NA objective can generate a significant *longitudinal* component in the focal region. Accurate calculations therefore require vector diffraction rather than a scalar paraxial approximation.

For an ideal aplanatic lens in a homogeneous medium, one convenient convention is

```math
\mathbf E(\mathbf r)=
-\frac{ikf}{2\pi}\int_0^{\alpha}\!\int_0^{2\pi}
\sqrt{\cos\theta}\,\mathbf R(\theta,\phi)
\begin{pmatrix} E_x^{\mathrm{in}} \\ E_y^{\mathrm{in}}\end{pmatrix}
e^{ik\mathbf s\cdot\mathbf r}\sin\theta\,d\phi\,d\theta,
```

with `k = 2πn/λ₀`, `α = arcsin(NA/n)`, `s = (sinθ cosφ, sinθ sinφ, cosθ)`, and `R` describing the lens's polarization rotation. Global phase and amplitude conventions should be checked when comparing independent implementations.

The project explores two complementary numerical methods:

1. **Bessel-integral decomposition:** For a pupil with LG-like azimuthal factor `exp(iℓφ)`, integrate the azimuthal angle analytically using the Jacobi–Anger expansion, leaving **five one-dimensional polar-angle integrals**. This is the original thesis-based approach and a useful independent physical benchmark.
2. **2D angular-spectrum FFT:** Convert the Richards–Wolf integral to a Fourier integral over `(kₓ,kᵧ)`. A 2D inverse FFT computes an entire transverse focal plane; repeated or batched FFTs compute the volume at many `z` coordinates. This accepts an **arbitrary sampled complex pupil Jones field**, within the assumed lens model.

The methods should agree after matching the pupil envelope, polarization, numerical aperture, propagation phase, Fourier convention, coordinate sampling, and overall normalization.

## Repository organization

### Historical notebooks on `main`

| File | Purpose |
| --- | --- |
| [`TightFocusingLG.ipynb`](TightFocusingLG.ipynb) | Original Python calculations for tightly focused LG beams. |
| [`Tight_focusing_using_GPU.ipynb`](Tight_focusing_using_GPU.ipynb) | Historical notebook exploring a grid-based integral calculation. **Its name does not imply that the current code runs on a GPU**; it uses NumPy/SciPy and Python loops. |
| [`Tight FocusingLG.nb`](Tight%20FocusingLG.nb) | Original Wolfram Mathematica notebook. |
| [`Tight FocusingLG.pdf`](Tight%20FocusingLG.pdf) | PDF export of the Mathematica work. |

Historical files are retained for provenance. The tutorial notebooks provide documented derivations, clearer numerical conventions, and tests.

### Development branches

| Branch | Focus | Status |
| --- | --- | --- |
| [`feature/thesis-integral-decomposition`](../../tree/feature/thesis-integral-decomposition) | Thesis-based five Bessel-integral derivation, CPU quadrature, complex-field calculations, and independent Debye comparisons. | **CPU tutorial committed** |
| [`feature/fft-vectorial-focusing-tutorial`](../../tree/feature/fft-vectorial-focusing-tutorial) | Derivation of the Cartesian angular spectrum and a NumPy 2D-FFT vectorial solver, including multiplane examples and numerical checks. | **CPU tutorial committed** |
| [`feature/gpu-integral-decomposition`](../../tree/feature/gpu-integral-decomposition) | GPU acceleration of Bessel-integral evaluation over many radii/axial coordinates; Bessel-function support, precision, and validation need special attention. | **Branch created; implementation planned** |
| [`feature/gpu-fft-vectorial-focusing`](../../tree/feature/gpu-fft-vectorial-focusing) | GPU-resident angular spectrum, batched 2D inverse FFTs, memory-aware 3D propagation, and CPU/GPU benchmarks. | **Branch created; implementation planned** |

**Tutorial notebooks already available:**

- [Five-integral decomposition (CPU)](../../blob/feature/thesis-integral-decomposition/tutorials/01_Thesis_Bessel_Integral_Decomposition.ipynb)
- [Vectorial Richards–Wolf via 2D FFT (CPU)](../../blob/feature/fft-vectorial-focusing-tutorial/tutorials/02_Vectorial_Richards_Wolf_2D_FFT.ipynb)

The four branches are intentionally separate so numerical implementations can be developed and reviewed without overwriting historical code or introducing premature dependencies.


### Focal-plane polarization atlas (Gaussian, LG, cylindrical-vector, full-Poincaré)

**Recommended starting point:** [Open the main-branch atlas in Google Colab](https://colab.research.google.com/github/mffg1993/TightFocusingSolutions/blob/main/tutorials/02_Focal_Plane_Polarization_Atlas_PyVista.ipynb). It contains the detailed Richards–Wolf and five-integral derivation, commented NumPy/SciPy code, and 25 beam examples. Run the setup and definition cells, then the desired example. The default CPU runtime is sufficient.

The static PyVista image has an **Open interactive HTML** button linking to an offline Plotly/WebGL dashboard with vertically stacked ellipse and polarization-axis figures, intensity and axis checkboxes, visualization sliders, and linked cameras. The dashboard resamples the computed field; it does not solve the diffraction integral in the browser. The private cross-method benchmark is not included in the public repository.

[Detailed five-integral theory tutorial on main](tutorials/01_Thesis_Bessel_Integral_Decomposition.ipynb).

The main branch includes [`tutorials/02_Focal_Plane_Polarization_Atlas_PyVista.ipynb`](tutorials/02_Focal_Plane_Polarization_Atlas_PyVista.ipynb), using the **CPU integral** algorithm. Development branches retain their original `04_` atlas filenames and backend-specific implementations. The notebook generates all three complex electric-field components at `z=0`, six intensity/phase maps per example, and an **8×8 grid of 3D polarization ellipses** on an intensity background via PyVista. Included are Gaussian and LG `ℓ=±1` beams in six Jones states; four cylindrical-vector constructions; and star, monstar and lemon full-Poincaré input textures.

The polarization renderer is adapted from the supplied **MathTubes_Colab.ipynb**, now provided as [`visualization/mathtubes.py`](visualization/mathtubes.py). It uses Mathematica-style `tube()` meshes with PyVista lighting and an adapter that plots computed 3D polarization ellipses (or straight tubes for linear states) over the intensity map. Run the notebook from the repository root or its `tutorials/` directory.

The two GPU feature branches use **JAX** for the atlas calculation and explicitly check for an accelerator; these are teaching demonstrations, **not yet GPU benchmarks or replacements for a fully validated production GPU solver**. PyVista rendering requires an appropriate notebook graphics backend. Convergence and full-complex-field comparisons remain essential.

### Optical polarization Möbius strips (Tutorial 03)

[Open in Google Colab](https://colab.research.google.com/github/mffg1993/TightFocusingSolutions/blob/main/tutorials/03_Polarization_Mobius_Strips.ipynb) · [Notebook source](tutorials/03_Polarization_Mobius_Strips.ipynb).

A standalone tutorial on three-dimensional C-points, phase winding of `E·E`, nonorientable polarization-axis ribbons, interactive Plotly visualization, and OBJ mesh export. The star, monstar and lemon labels refer to the incident textures; the focused topology is tested independently.

### Polychromatic electric-field knots (Tutorial 04)

[Open in Google Colab](https://colab.research.google.com/github/mffg1993/TightFocusingSolutions/blob/main/tutorials/04_Polychromatic_Electric_Field_Knots.ipynb) · [Notebook source](tutorials/04_Polychromatic_Electric_Field_Knots.ipynb).

A standalone, editable reproduction tutorial based on Ferrer-Garcia *et al.*, *Physical Review Research* **3**, 033226 (2021). Independently focus three chromatic vector-vortex beams, reconstruct real instantaneous electric-field-tip trajectories, explore the published frequency presets, display local 3D knot candidates, optionally calculate knot determinants, and export the complex chromatic fields. This tutorial does not claim an exact reproduction of the published knot-density percentages.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/mffg1993/TightFocusingSolutions.git
cd TightFocusingSolutions
```

### 2. Select the method you want to study

**Five-integral / thesis method**

```bash
git switch feature/thesis-integral-decomposition
```

Open `tutorials/01_Thesis_Bessel_Integral_Decomposition.ipynb`. Both the detailed theory tutorial and the collaborator-facing atlas are also on `main`.

**Arbitrary-pupil FFT method**

```bash
git switch feature/fft-vectorial-focusing-tutorial
```

Open `tutorials/02_Vectorial_Richards_Wolf_2D_FFT.ipynb`.

### 3. Create a local environment

The CPU notebooks require Python, NumPy, SciPy, Matplotlib, and Jupyter. A minimal installation is:

```bash
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib jupyterlab pyvista vtk plotly "trame-pyvista[jupyter]"
jupyter lab
```

On Windows, activate with `.venv\Scripts\activate` instead.

**GPU requirements are separate.** Planned CUDA workflows will require a suitable NVIDIA GPU, matching drivers, and a compatible CuPy or JAX installation. GPU speedups must be measured rather than assumed, particularly for small arrays.

## Method 1 — Thesis-based integral decomposition

For an incident LG-type field with topological charge `ℓ`, the azimuthal integral can be evaluated exactly. The resulting field is expressed through five integrals containing Bessel functions with orders

```text
             ℓ,    ℓ+2,    ℓ−2,    ℓ+1,    ℓ−1.
```

These integral combinations reconstruct `Ex`, `Ey`, and `Ez` for arbitrary *uniform input Jones polarization* and a selected LG-type pupil envelope.

The CPU tutorial includes:

- The connection between the lens polarization matrix, the Jacobi–Anger expansion, and the five integrals.
- Complex field components at arbitrary `(x,y,z)`, including off-focus phase.
- A focal-plane intensity-and-phase example.
- An **independent two-angle Richards–Wolf quadrature** comparison.
- Convergence guidance and checks for earlier numerical pitfalls (angular step size, pupil-radius normalization, complex phase retention, array allocation).

The decomposition is particularly attractive for cylindrical-mode calculations where azimuthal integration can be done analytically. It is **not** itself a universal arbitrary-pupil solver.

## Method 2 — Cartesian angular-spectrum FFT

Under the same ideal-lens assumptions, the focusing integral can be written

```math
\mathbf E(x,y,z)=
\iint_{k_x^2+k_y^2\le (k_0\mathrm{NA})^2}
\mathbf A(k_x,k_y)
e^{i(k_x x+k_y y+k_z z)}\,dk_x\,dk_y,
\qquad k_z=\sqrt{k^2-k_x^2-k_y^2}.
```

Here `A(kₓ,kᵧ)` incorporates the vectorial polarization transformation, aplanatic apodization, **the Jacobian** from spherical angles to Cartesian wave numbers, and the pupil support. The critical physical point is that the angular-integral factor `√cosθ` becomes **`1/√cosθ` in the Cartesian angular-spectrum density**, owing to the change of variables.

For a fixed pupil spectrum, each axial plane follows from a 2D inverse FFT after multiplying by `exp(i k_z z)`. This provides a natural route to high-throughput 3D simulations.

The tutorial explains pupil-to-wave-vector mapping, frequency cutoffs, `fftshift`/`ifftshift`, FFT normalization, longitudinal polarization, axial propagation, memory requirements, and two distinct validation levels:

1. FFT versus the **same discrete direct sum**, checking implementation and conventions.
2. FFT versus independent **continuum Debye quadrature**, checking numerical approximation accuracy.

## Why GPU acceleration?

Many structured-light questions require the **complex vector field throughout a volume**, rather than only on the `z=0` plane. Computing dozens or hundreds of axial planes can be accelerated by reusing the angular spectrum and processing many phase-propagated planes in parallel.

For the FFT approach, the intended GPU workflow is:

```text
Arbitrary sampled pupil Jones field (Ex, Ey)
                   │
          High-NA lens operator
                   │
          One angular spectrum A
                   │
       GPU: multiply by exp(i kz z_j)
                   │
         GPU: batched inverse 2D FFT
                   │
    Complex focal volumes Ex, Ey, Ez
                   │
       Analysis / plotting / export
```

For the five-integral approach, the intended workflow is instead **batched radial/axial quadrature**, with GPU-capable Bessel evaluations or validated kernels. It may offer advantages for selected radii or limited spatial regions.

GPU performance depends on requested grid sizes, precision, GPU memory, and transfer overhead. The project aims to compare both approaches fairly, rather than assume one wins universally.

## Validation and scientific conventions

A numerical calculation is not validated by a plausible-looking intensity plot. Before interpreting phase singularities, longitudinal fields, or knots, check:

- **Complex fields, not just intensities:** compare amplitudes and phases of all three components.
- **Aplanatic polarization mapping:** the ray spectrum must obey `k · A = 0`.
- **Integration and FFT normalizations:** include the correct angular steps, Jacobians, and Fourier factors.
- **Sampling:** vary angular resolution, Fourier-grid size, spatial pitch, and computational window independently.
- **Defocus:** test `z ≠ 0`; prematurely applying `np.real` destroys essential phase information.
- **Topological robustness:** vortex positions and linking should be checked against numerical refinement and finite-volume boundaries.
- **Hardware:** synchronize GPU operations when timing asynchronous kernels; record device, software, precision, and data-transfer choices.

The repository currently models idealized monochromatic focusing in a homogeneous medium. Interfaces, multilayer objectives, aberrations, evanescent waves, broadband/polychromatic fields, and general nonlinear responses require separate physical extensions.



## Theoretical background and references

- B. Richards and E. Wolf, *Electromagnetic diffraction in optical systems, II. Structure of the image field in an aplanatic system*, Proceedings of the Royal Society A **253**, 358–379 (1959).
- M. F. Ferrer-García, *PhD thesis*, Chapter 2, Eqs. (2.2)–(2.10), and Appendix A (vectorial focusing of LG beams; numerical implementations).
- M. Leutenegger, R. Rao, R. A. Leitgeb, and T. Lasser, *Fast focus field calculations*, Optics Express **14**, 11277–11291 (2006), [DOI](https://doi.org/10.1364/OE.14.011277).
- J. Lin *et al.*, *Fast vectorial calculation of the volumetric focused field distribution by using a three-dimensional Fourier transform*, Optics Express **20**, 1060–1069 (2012), [DOI](https://doi.org/10.1364/OE.20.001060).
- A. G. Sedukhin, *Ultrafast vectorial calculation of tightly focused vortex and non-vortex fields in a meridional plane*, Optics Communications **558**, 130237 (2024), [DOI](https://doi.org/10.1016/j.optcom.2023.130237).
- M. F. Ferrer-García *et al.*, *Polychromatic electric field knots*, Physical Review Research **3**, 033226 (2021), [DOI](https://doi.org/10.1103/PhysRevResearch.3.033226).

## Contributions and use

Scientific corrections and benchmark examples are welcome. When proposing a numerical change, include the governing equations, parameter conventions, and **complex-field** regression comparisons, not only intensity plots.

**License:** No license file is currently provided in this repository. Until a license is added, do not assume the code is available for unrestricted redistribution or reuse.

---

*Maintained as an evolving collection of research-grade numerical methods and transparent teaching notebooks for vectorial structured-light focusing.*
