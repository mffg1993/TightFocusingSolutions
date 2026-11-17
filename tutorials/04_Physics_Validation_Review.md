# Tutorial 04: scientific review and validation

Review date: 2026-11-17. Reviewed starting notebook commit:
`f82845fef7366ac08f675ac96d055423fedbe221`.

Reference: M. F. Ferrer-García et al., *Polychromatic electric field knots*,
Physical Review Research **3**, 033226 (2021),
[published PDF](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.033226),
[DOI](https://doi.org/10.1103/PhysRevResearch.3.033226).
The final published text and Appendices A–B were reviewed. No separate
supplemental file was identified in the publisher listing; the arXiv PDF could
not be retrieved in this session. This review does not claim to have inspected
an additional unpublished supplement or the original numerical datasets.

## Findings

| Item | Assessment and action |
|---|---|
| Figure/equation numbering | Final journal Figure 3 shows monochromatic vector-vortex polarization; Figure 4 shows the knot maps/histograms. Vector vortices are Eq. (4), their three-colour superposition Eq. (5). Corrected notebook references. |
| Input modes | `(ell, delta)=(1,0),(-1,0),(-1,pi)` agrees with published Eq. (5). Retained. |
| Circular Jones convention | With `R=(1,-i)/sqrt(2)`, `L=(1,i)/sqrt(2)`, `(1,0)` is radial and `(1,pi)` is **-i times** the azimuthal unit vector. That intrinsic phase must be preserved. Added analytic checks. |
| Lens transport | Existing five-integral kernel agrees with direct integration of the first two columns of published Eq. (A2), including the negative longitudinal transport signs. No kernel sign correction was justified. |
| Longitudinal integrals | The existing extra `sin(theta)` factor is correct. Verified against direct two-angle integration and the vanishing longitudinal field of an azimuthal input. |
| Chromatic focusing | Each colour has its own wavenumber, Bessel arguments, axial phase and focusing prefactor. Existing implementation retains them. No per-colour focal normalization was added. |
| Temporal convention | Main-text Eq. (1) uses a negative time sign; Appendix A displays a positive one. Notebook consistently uses the Eq. (1) sign with explicitly specified spatial phasors. Uniform reversal traverses the same curve backwards; changing only selected phases is different. Added Figure 1 phasor-reconstruction and time-reversal tests. |
| Frequency presets | `(1,3,2)` and `(2,3,5)` encode the published ratios. Custom noninteger values were silently truncated by `int`; they are now rejected, as are duplicate colours and unknown preset names. Coprime positive integers remain required. |
| Length convention | Notebook uses the base-frequency vacuum wavelength. The paper uses the lowest optical frequency's wavelength. In air these differ by `min(harmonics)`; the `(2,3,5)` preset needs the factor of two. Documented conversion rather than silently changing existing coordinates. |
| Numerical aperture | Changed default from illustrative 0.85 to the published 0.95 in air. |
| Pupil profile | The shared `q**abs(ell)*exp(-(q/FILL)**2)` envelope is proportional to a common-waist p=0 LG profile for the three equal-absolute-charge modes. `FILL=1` is still an assumption; the reviewed publication does not give the waist/aperture ratio needed for exact numerical reproduction. Different absolute charges require explicit relative pupil normalization. |
| Integration accuracy | Added a direct Gauss–Legendre/periodic-azimuth oracle without Bessel reduction or interpolation. Added separate polar-quadrature and radial-interpolation convergence tables. |
| Preliminary strand diagnostic | Original finite-neighbour KD-tree search could return infinity after only inspecting local neighbours. Replaced by exhaustive nonlocal **vertex** distances with a fixed time-fraction exclusion. Clearly labelled as incapable of detecting segment crossings or certifying embedding. |
| Knot invariants | Existing determinant example remains optional and unvalidated as a classifier. It does not assign knot names or unknot status. No knot mapping was implemented, following the user's scope update. |
| Reproducibility | Added export schema, length/time/pupil conventions, lowest-optical wavelength and normalization metadata. Added regression tests and a sequential headless execution script. |

## Validation evidence

Commands from repository root:

```bash
python -m unittest discover -s tests -v
python tests/validate_tutorial04.py
```

The four regression tests pass. They cover both harmonic presets at `z=0` and
`z=0.31`, a nonunit reference wavelength and focal length, both vortex charges,
both incident Cartesian basis polarizations, weighted vector-vortex superpositions,
on-axis radial symmetry, azimuthal longitudinal cancellation, invalid harmonics,
and independent reconstruction of the published analytic Figure 1 trajectory.
These tests constrain complex fields, rather than fitting a global phase or
normalizing intensities until they agree.

At the default 81×81 grid, 401 polar and 401 radial nodes, the largest checked
component error is approximately `1.21e-4`, scaled by the largest reference
component in each basis case. Refining the direct oracle from 80×192 to 120×256
angular nodes changes the checked fields by approximately `1.2e-14`.
These are errors at selected probes, not global error bounds.

The separate convergence study at harmonic 3 gives:

| Samples | Polar quadrature error (probe radii evaluated exactly) | Radial interpolation error (801 polar nodes) |
|---|---:|---:|
| 101 | 9.96e-8 | 6.01e-4 |
| 201 | 6.19e-9 | 2.96e-4 |
| 401 | 3.86e-10 | 1.35e-4 |

The current probe errors are dominated by linear radial interpolation. Refining
only the polar quadrature therefore gives limited benefit at the default radial
resolution. Boundary regions in parameter space may need substantially tighter
field accuracy than these illustrative probes.

Every code cell executes sequentially in a headless IPython session, including
construction of plots/widgets and an NPZ export/reload. The optional pyknotid
import follows its documented unavailable-dependency path. A trial installation
of pyknotid 0.5.3 failed while building its `planarity` dependency because this
environment lacks `clang`; no successful invariant computation is claimed.

This is **not** fresh-Colab validation: browser rendering, widget callbacks,
Colab installation behaviour and download prompts remain to be checked there.
Ordinary Jupyter kernel execution was blocked by this environment's socket
restrictions, so the sequential harness uses IPython directly.

## Implementation sequence after this review

1. **Completed:** establish the published definitions and explicit conventions;
   correct the NA and references; reject lossy frequency input; independently
   verify complex focusing and temporal reconstruction.
2. **Completed:** separate diffraction-quadrature and interpolation errors;
   remove misleading confidence from sampled-vertex distances; record export
   provenance and maintain a repeatable notebook execution check.
3. **Next, without knot mapping:** exercise the unchanged standalone notebook in
   a fresh Colab runtime, including actual widget interactions and NPZ download.
   Add tube-based trajectory rendering with time colour, adjustable radius and
   shading, then HTML and image/mesh export with tested dependency versions.
4. **Optional numerical work:** evaluate selected points without interpolation
   or introduce a higher-order interpolant only after measuring complex-field
   error independently. Test additional axial planes, pupil filling and aperture
   values; do not infer topology from nonplanarity or a visually plausible curve.

Knot mapping is outside the current scope. If revisited, prerequisites include
closed-segment intersection and near-contact screening (including the closing
edge), time refinement tied to Fourier-curve curvature, robust projections,
normalized Alexander polynomials and invariant-collision handling. A determinant
of 1 is not an unknot certificate, 3 is not a unique trefoil identifier, and 5
is not a unique figure-eight identifier. Chirality would require additional
invariants and mirror controls. No such labels are produced by this change.
